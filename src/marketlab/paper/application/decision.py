from __future__ import annotations

from dataclasses import asdict
from datetime import date, timedelta
from typing import Any

import pandas as pd
from pandas.tseries.frequencies import to_offset

from marketlab.config import ExperimentConfig
from marketlab.data.market import load_symbol_frames
from marketlab.data.panel import build_market_panel
from marketlab.features.engineering import add_feature_set
from marketlab.models import (
    build_model_estimator,
    predict_direction_scores,
)
from marketlab.models.training import modeling_feature_columns
from marketlab.paper.alpaca import (
    AlpacaMarketDataProvider,
    AlpacaPaperBrokerClient,
)
from marketlab.paper.contracts import (
    PaperBroker,
    PaperDecisionRequest,
    PaperDecisionResult,
    PaperHistoryProvider,
)
from marketlab.paper.core import (
    APPROVAL_NOT_REQUIRED,
    APPROVAL_PENDING,
    CONSENSUS_POLICY,
    SUBMISSION_PENDING,
    SUBMISSION_SKIPPED,
    _iso_date,
    _local_now,
    _now_utc,
    _paper_model_names,
    _paper_symbol,
    validate_paper_trading_config,
)
from marketlab.paper.notifications import notify_paper_decision
from marketlab.paper.state import PaperStateStore
from marketlab.targets import add_forward_targets, build_rebalance_snapshots


def _build_alpaca_panel(
    config: ExperimentConfig,
    provider: PaperHistoryProvider | None = None,
) -> pd.DataFrame:
    frames = load_symbol_frames(
        config,
        provider=provider or AlpacaMarketDataProvider(),
        force_refresh=True,
    )
    return build_market_panel(frames)


def _is_trading_day(
    broker: PaperBroker,
    *,
    market_date: date,
) -> bool:
    calendar = broker.get_calendar(start_date=market_date, end_date=market_date)
    return any(item.get("date") == market_date.isoformat() for item in calendar)


def _next_trading_date(
    broker: PaperBroker,
    *,
    market_date: date,
) -> date:
    calendar = broker.get_calendar(
        start_date=market_date,
        end_date=market_date + timedelta(days=14),
    )
    future_dates = sorted(
        pd.to_datetime(item["date"]).date()
        for item in calendar
        if item.get("date")
    )
    for candidate in future_dates:
        if candidate > market_date:
            return candidate
    raise RuntimeError("The Alpaca calendar did not provide a future trading date for the paper decision.")


def _proposal_id(signal_date: str, effective_date: str, symbol: str) -> str:
    return f"{effective_date}-{symbol}-{signal_date}"


def _training_rows_for_latest_signal(
    labeled_dataset: pd.DataFrame,
    config: ExperimentConfig,
    latest_signal_date: pd.Timestamp,
) -> pd.DataFrame:
    label_cutoff = pd.Timestamp(latest_signal_date)
    for _ in range(max(0, config.evaluation.walk_forward.embargo_periods)):
        label_cutoff = pd.Timestamp(label_cutoff - to_offset(config.portfolio.ranking.rebalance_frequency))

    train_start = pd.Timestamp(latest_signal_date) - pd.DateOffset(
        years=config.evaluation.walk_forward.train_years
    )
    train_rows = labeled_dataset.loc[
        labeled_dataset["signal_date"].ge(train_start)
        & labeled_dataset["signal_date"].lt(label_cutoff)
        & labeled_dataset["target_end_date"].le(label_cutoff)
    ].copy()
    return train_rows.reset_index(drop=True)


def _latest_snapshot_row(
    featured_panel: pd.DataFrame,
    feature_columns: list[str],
    latest_signal_date: pd.Timestamp,
    effective_date: str,
) -> pd.DataFrame:
    latest_snapshot = featured_panel.loc[
        featured_panel["timestamp"] == latest_signal_date,
        ["symbol", "timestamp", *feature_columns],
    ].copy()
    latest_snapshot = latest_snapshot.rename(columns={"timestamp": "signal_date"})
    latest_snapshot["effective_date"] = effective_date
    latest_snapshot = latest_snapshot[
        ["symbol", "signal_date", "effective_date", *feature_columns]
    ].reset_index(drop=True)
    if len(latest_snapshot) != 1:
        raise RuntimeError("The Phase 7.1 paper decision path expects exactly one latest snapshot row.")
    return latest_snapshot


def _train_and_score_models(
    config: ExperimentConfig,
    *,
    train_rows: pd.DataFrame,
    latest_snapshot: pd.DataFrame,
    feature_columns: list[str],
) -> list[dict[str, Any]]:
    threshold = float(config.portfolio.ranking.min_score_threshold)
    rows: list[dict[str, Any]] = []
    target = train_rows["target"].astype(int)

    for model_name in _paper_model_names(config):
        definition, estimator = build_model_estimator(model_name, config.target.type)
        estimator.fit(train_rows[feature_columns], target)
        score = float(predict_direction_scores(estimator, latest_snapshot[feature_columns]).iloc[0])
        vote = "long" if score >= threshold else "cash"
        rows.append(
            {
                "model_name": model_name,
                "estimator_label": definition.estimator_label,
                "score": score,
                "vote": vote,
                "target_weight": 1.0 if vote == "long" else 0.0,
            }
        )

    return rows


def _proposal_consensus(
    config: ExperimentConfig,
    *,
    model_rows: list[dict[str, Any]],
) -> tuple[dict[str, Any], float]:
    long_vote_count = sum(1 for row in model_rows if row["vote"] == "long")
    cash_vote_count = len(model_rows) - long_vote_count
    consensus_rule = {
        "type": CONSENSUS_POLICY,
        "min_long_votes": int(config.paper.consensus_min_long_votes),
        "model_count": len(model_rows),
    }
    target_weight = 1.0 if long_vote_count >= config.paper.consensus_min_long_votes else 0.0
    return (
        {
            "decision_policy": CONSENSUS_POLICY,
            "consensus_rule": consensus_rule,
            "long_vote_count": long_vote_count,
            "cash_vote_count": cash_vote_count,
            "decision": "long" if target_weight > 0.0 else "cash",
            "target_weight": target_weight,
        },
        target_weight,
    )


def _reference_price_for_signal(
    featured_panel: pd.DataFrame,
    *,
    symbol: str,
    latest_signal_date: pd.Timestamp,
) -> float:
    latest_price_row = featured_panel.loc[
        (featured_panel["symbol"] == symbol)
        & (featured_panel["timestamp"] == latest_signal_date)
    ]
    if latest_price_row.empty:
        raise RuntimeError("The paper decision path could not resolve the latest reference price.")
    return float(latest_price_row.iloc[-1]["adj_close"])


class DecisionService:
    def __init__(self, config: ExperimentConfig) -> None:
        self._config = config

    def run(self, request: PaperDecisionRequest) -> PaperDecisionResult:
        config = self._config
        validate_paper_trading_config(config)
        paper_symbol = _paper_symbol(config)
        store = PaperStateStore(config)
        local_now = _local_now(config, request.now)
        broker_client = request.broker or AlpacaPaperBrokerClient()
        market_date = local_now.date()

        if not _is_trading_day(broker_client, market_date=market_date):
            status = {
                "event": "paper-decision",
                "status": SUBMISSION_SKIPPED,
                "reason": "non_trading_day",
                "market_date": market_date.isoformat(),
                "updated_at": _now_utc(request.now).isoformat(),
            }
            status_path = store.write_status(status)
            notify_paper_decision(
                config,
                store,
                outcome="non_trading_day",
                status=status,
                now=request.now,
                transport=request.notification_transport,
            )
            return PaperDecisionResult(
                status_path=str(status_path),
                status=status,
            )

        panel = _build_alpaca_panel(config, provider=request.provider)
        featured_panel = add_feature_set(panel=panel, **asdict(config.features))
        historical_snapshots = build_rebalance_snapshots(
            featured_panel,
            frequency=config.portfolio.ranking.rebalance_frequency,
        )
        if historical_snapshots.empty:
            raise RuntimeError("The paper decision path produced no rebalance snapshots.")

        latest_signal_date = pd.Timestamp(featured_panel["timestamp"].max())
        if latest_signal_date.date() != market_date:
            status = {
                "event": "paper-decision",
                "status": SUBMISSION_SKIPPED,
                "reason": "stale_signal_date",
                "market_date": market_date.isoformat(),
                "latest_signal_date": latest_signal_date.date().isoformat(),
                "updated_at": _now_utc(request.now).isoformat(),
            }
            status_path = store.write_status(status)
            notify_paper_decision(
                config,
                store,
                outcome="stale_signal_date",
                status=status,
                now=request.now,
                transport=request.notification_transport,
            )
            return PaperDecisionResult(
                status_path=str(status_path),
                status=status,
            )

        labeled_dataset = add_forward_targets(
            snapshots=historical_snapshots,
            panel=featured_panel,
            horizon_days=config.target.horizon_days,
            target_type=config.target.type,
        )
        if labeled_dataset.empty:
            raise RuntimeError("The paper decision path produced no labeled historical rows.")

        feature_columns = modeling_feature_columns(labeled_dataset)
        effective_date = _next_trading_date(
            broker_client,
            market_date=market_date,
        ).isoformat()
        latest_snapshot = _latest_snapshot_row(
            featured_panel,
            feature_columns,
            latest_signal_date,
            effective_date,
        )
        proposal_id = _proposal_id(
            signal_date=_iso_date(latest_signal_date),
            effective_date=effective_date,
            symbol=paper_symbol,
        )

        try:
            existing = store.load_proposal(proposal_id)
        except FileNotFoundError:
            existing = None
        if existing is not None:
            evidence_path = store.trade_evidence_path(existing["effective_date"])
            status = {
                "event": "paper-decision",
                "status": "existing_proposal",
                "proposal_id": proposal_id,
                "proposal_path": str(store.inbox_proposal_path(proposal_id)),
                "updated_at": _now_utc(request.now).isoformat(),
            }
            status_path = store.write_status(status)
            try:
                evidence = store.load_evidence(existing["effective_date"])
            except FileNotFoundError:
                evidence = None
            notify_paper_decision(
                config,
                store,
                outcome="existing_proposal",
                status=status,
                proposal=existing,
                now=request.now,
                transport=request.notification_transport,
            )
            return PaperDecisionResult(
                proposal_id=proposal_id,
                proposal_path=str(store.inbox_proposal_path(proposal_id)),
                evidence_path=str(evidence_path),
                status_path=str(status_path),
                status=status,
                proposal=existing,
                evidence=evidence,
            )

        train_rows = _training_rows_for_latest_signal(labeled_dataset, config, latest_signal_date)
        if len(train_rows) < max(1, config.evaluation.walk_forward.min_train_rows):
            raise RuntimeError(
                "The paper decision path does not have enough historical rows for the Phase 7.1 model set."
            )
        train_target = train_rows["target"].astype(int)
        train_positive_rate = float(train_target.mean())
        if train_positive_rate < float(config.evaluation.walk_forward.min_train_positive_rate):
            raise RuntimeError(
                "The paper decision path does not meet the configured minimum positive-rate floor."
            )
        if train_target.nunique() < 2:
            raise RuntimeError("The paper decision path needs both target classes in the training slice.")

        model_rows = _train_and_score_models(
            config,
            train_rows=train_rows,
            latest_snapshot=latest_snapshot,
            feature_columns=feature_columns,
        )
        consensus_summary, _ = _proposal_consensus(config, model_rows=model_rows)
        reference_price = _reference_price_for_signal(
            featured_panel,
            symbol=paper_symbol,
            latest_signal_date=latest_signal_date,
        )

        approval_status = (
            APPROVAL_NOT_REQUIRED
            if config.paper.execution_mode == "autonomous"
            else APPROVAL_PENDING
        )
        evidence = {
            "proposal_id": proposal_id,
            "experiment_name": config.experiment_name,
            "symbol": paper_symbol,
            "signal_date": _iso_date(latest_signal_date),
            "effective_date": effective_date,
            "feature_columns": feature_columns,
            "train_rows": int(len(train_rows)),
            "train_start": _iso_date(pd.Timestamp(train_rows["signal_date"].min())),
            "train_end": _iso_date(pd.Timestamp(train_rows["signal_date"].max())),
            "train_positive_rate": train_positive_rate,
            "min_score_threshold": float(config.portfolio.ranking.min_score_threshold),
            "reference_price": reference_price,
            "models": model_rows,
            **consensus_summary,
            "created_at": _now_utc(request.now).isoformat(),
        }
        evidence_path = store.save_evidence(evidence)

        proposal = {
            "proposal_id": proposal_id,
            "experiment_name": config.experiment_name,
            "symbol": paper_symbol,
            "signal_date": _iso_date(latest_signal_date),
            "effective_date": effective_date,
            "reference_price": reference_price,
            "execution_mode": config.paper.execution_mode,
            "approval_status": approval_status,
            "submission_status": SUBMISSION_PENDING,
            "min_score_threshold": float(config.portfolio.ranking.min_score_threshold),
            "train_rows": int(len(train_rows)),
            "train_start": evidence["train_start"],
            "train_end": evidence["train_end"],
            "train_positive_rate": train_positive_rate,
            "created_at": _now_utc(request.now).isoformat(),
            "data_provider": config.paper.data_provider,
            "broker": config.paper.broker,
            "evidence_path": str(evidence_path),
            **consensus_summary,
        }
        proposal_path = store.save_proposal(proposal)
        status = {
            "event": "paper-decision",
            "status": "proposal_created",
            "proposal_id": proposal_id,
            "proposal_path": str(proposal_path),
            "evidence_path": str(evidence_path),
            "updated_at": _now_utc(request.now).isoformat(),
        }
        status_path = store.write_status(status)
        notify_paper_decision(
            config,
            store,
            outcome="proposal_created",
            status=status,
            proposal=proposal,
            now=request.now,
            transport=request.notification_transport,
        )
        return PaperDecisionResult(
            proposal_id=proposal_id,
            proposal_path=str(proposal_path),
            evidence_path=str(evidence_path),
            status_path=str(status_path),
            status=status,
            proposal=proposal,
            evidence=evidence,
        )
