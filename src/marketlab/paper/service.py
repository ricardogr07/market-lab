from __future__ import annotations

import json
from dataclasses import asdict
from datetime import UTC, date, datetime, timedelta
from datetime import time as dt_time
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd
from pandas.tseries.frequencies import to_offset

from marketlab.config import ExperimentConfig
from marketlab.data.market import load_symbol_frames
from marketlab.data.panel import build_market_panel
from marketlab.features.engineering import add_feature_set
from marketlab.models import (
    build_model_estimator,
    predict_direction_scores,
    supported_model_names,
)
from marketlab.models.training import modeling_feature_columns
from marketlab.paper.alpaca import (
    AlpacaMarketDataProvider,
    AlpacaPaperBrokerClient,
)
from marketlab.targets import add_forward_targets, build_rebalance_snapshots

APPROVAL_PENDING = "pending"
APPROVAL_APPROVED = "approved"
APPROVAL_REJECTED = "rejected"
APPROVAL_NOT_REQUIRED = "not_required"
SUBMISSION_PENDING = "pending"
SUBMISSION_SUBMITTED = "submitted"
SUBMISSION_NOOP = "no_trade_required"
SUBMISSION_SKIPPED = "skipped"
CONSENSUS_POLICY = "consensus_vote"
TERMINAL_ORDER_STATUSES = {"canceled", "expired", "filled", "rejected"}


def _now_utc(now: datetime | None = None) -> datetime:
    if now is None:
        return datetime.now(UTC)
    if now.tzinfo is None:
        return now.replace(tzinfo=UTC)
    return now.astimezone(UTC)


def _local_now(config: ExperimentConfig, now: datetime | None = None) -> datetime:
    return _now_utc(now).astimezone(ZoneInfo(config.paper.schedule_timezone))


def _clock_value(clock_text: str) -> dt_time:
    hour, minute = (int(part) for part in clock_text.split(":"))
    return dt_time(hour=hour, minute=minute)


def _iso_date(value: pd.Timestamp | datetime | date | str) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, pd.Timestamp):
        return value.date().isoformat()
    if isinstance(value, datetime):
        return value.date().isoformat()
    return value.isoformat()


def _json_dump(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return path


def _json_load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _paper_symbol(config: ExperimentConfig) -> str:
    symbols = [str(symbol).strip() for symbol in config.data.symbols if str(symbol).strip() != ""]
    if len(symbols) != 1:
        raise RuntimeError(
            "Phase 7 paper trading requires exactly one configured symbol in data.symbols."
        )
    return symbols[0]


def _paper_model_names(config: ExperimentConfig) -> list[str]:
    configured = [spec.name for spec in config.models]
    if not configured:
        raise RuntimeError("Phase 7.1 paper trading requires configured models.")
    if len(set(configured)) != len(configured):
        raise RuntimeError("Phase 7.1 paper trading does not allow duplicate model names.")

    required_models = set(supported_model_names())
    configured_set = set(configured)
    if configured_set != required_models or len(configured) != len(required_models):
        required = ", ".join(sorted(required_models))
        raise RuntimeError(
            "Phase 7.1 paper trading requires config.models to include each supported "
            f"direction model exactly once: {required}."
        )
    if config.paper.consensus_min_long_votes > len(configured):
        raise RuntimeError(
            "paper.consensus_min_long_votes cannot exceed the number of configured paper models."
        )
    return configured


def validate_paper_trading_config(config: ExperimentConfig) -> None:
    if not config.paper.enabled:
        raise RuntimeError("paper.enabled must be true for Phase 7 paper-trading commands.")
    _paper_symbol(config)
    _paper_model_names(config)
    if config.data.interval != "1d":
        raise RuntimeError("Phase 7 paper trading requires data.interval='1d'.")
    if config.target.type != "direction":
        raise RuntimeError("Phase 7 paper trading requires target.type='direction'.")
    if config.target.horizon_days != 1:
        raise RuntimeError("Phase 7 paper trading requires target.horizon_days=1.")
    if config.portfolio.ranking.rebalance_frequency != "D":
        raise RuntimeError("Phase 7 paper trading requires portfolio.ranking.rebalance_frequency='D'.")
    if config.portfolio.ranking.mode != "long_only":
        raise RuntimeError("Phase 7 paper trading requires portfolio.ranking.mode='long_only'.")
    if config.portfolio.ranking.long_n != 1 or config.portfolio.ranking.short_n != 1:
        raise RuntimeError("Phase 7 paper trading requires long_n=1 and short_n=1.")


class PaperStateStore:
    def __init__(self, config: ExperimentConfig) -> None:
        self._config = config
        self.inbox_root = config.paper_approval_inbox_dir
        self.state_root = config.paper_state_dir
        self.trades_root = self.state_root / "trades"
        self.reports_root = self.state_root.parent / "reports"
        self.status_path = self.state_root / "status.json"
        for root in (self.inbox_root, self.state_root, self.trades_root, self.reports_root):
            root.mkdir(parents=True, exist_ok=True)

    def trade_dir(self, trade_date: str) -> Path:
        path = self.trades_root / trade_date
        path.mkdir(parents=True, exist_ok=True)
        return path

    def trade_proposal_path(self, trade_date: str) -> Path:
        return self.trade_dir(trade_date) / "proposal.json"

    def trade_evidence_path(self, trade_date: str) -> Path:
        return self.trade_dir(trade_date) / "evidence.json"

    def trade_approval_path(self, trade_date: str) -> Path:
        return self.trade_dir(trade_date) / "approval.json"

    def trade_submission_path(self, trade_date: str) -> Path:
        return self.trade_dir(trade_date) / "submission.json"

    def trade_account_snapshot_path(self, trade_date: str) -> Path:
        return self.trade_dir(trade_date) / "account_snapshot.json"

    def trade_order_preview_path(self, trade_date: str) -> Path:
        return self.trade_dir(trade_date) / "order_preview.json"

    def trade_order_status_path(self, trade_date: str) -> Path:
        return self.trade_dir(trade_date) / "order_status.json"

    def inbox_proposal_path(self, proposal_id: str) -> Path:
        return self.inbox_root / f"{proposal_id}.json"

    def report_dir(self, start_date: str, end_date: str) -> Path:
        path = self.reports_root / f"{start_date}_{end_date}"
        path.mkdir(parents=True, exist_ok=True)
        return path

    def write_status(self, payload: dict[str, Any]) -> Path:
        return _json_dump(self.status_path, payload)

    def read_status(self) -> dict[str, Any] | None:
        if not self.status_path.exists():
            return None
        return _json_load(self.status_path)

    def save_evidence(self, evidence: dict[str, Any]) -> Path:
        return _json_dump(self.trade_evidence_path(evidence["effective_date"]), evidence)

    def load_evidence(self, trade_date: str) -> dict[str, Any]:
        return _json_load(self.trade_evidence_path(trade_date))

    def save_proposal(self, proposal: dict[str, Any]) -> Path:
        trade_date = proposal["effective_date"]
        proposal_path = self.trade_proposal_path(trade_date)
        inbox_path = self.inbox_proposal_path(proposal["proposal_id"])
        _json_dump(proposal_path, proposal)
        _json_dump(inbox_path, proposal)
        return proposal_path

    def update_proposal(self, proposal: dict[str, Any]) -> Path:
        return self.save_proposal(proposal)

    def load_proposal(self, proposal_id: str) -> dict[str, Any]:
        path = self.inbox_proposal_path(proposal_id)
        if not path.exists():
            raise FileNotFoundError(f"Unknown proposal_id: {proposal_id}")
        return _json_load(path)

    def list_proposals(self) -> list[dict[str, Any]]:
        proposals = [_json_load(path) for path in sorted(self.inbox_root.glob("*.json"))]
        return sorted(
            proposals,
            key=lambda proposal: (
                proposal.get("effective_date", ""),
                proposal.get("created_at", ""),
                proposal.get("proposal_id", ""),
            ),
            reverse=True,
        )

    def latest_proposal(self) -> dict[str, Any] | None:
        proposals = self.list_proposals()
        if not proposals:
            return None
        return proposals[0]


def _build_alpaca_panel(
    config: ExperimentConfig,
    provider: AlpacaMarketDataProvider | None = None,
) -> pd.DataFrame:
    frames = load_symbol_frames(
        config,
        provider=provider or AlpacaMarketDataProvider(),
        force_refresh=True,
    )
    return build_market_panel(frames)


def _is_trading_day(
    broker: AlpacaPaperBrokerClient,
    *,
    market_date: date,
) -> bool:
    calendar = broker.get_calendar(start_date=market_date, end_date=market_date)
    return any(item.get("date") == market_date.isoformat() for item in calendar)


def _next_trading_date(
    broker: AlpacaPaperBrokerClient,
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


def run_paper_decision(
    config: ExperimentConfig,
    *,
    now: datetime | None = None,
    provider: AlpacaMarketDataProvider | None = None,
    broker: AlpacaPaperBrokerClient | None = None,
) -> dict[str, Any]:
    validate_paper_trading_config(config)
    paper_symbol = _paper_symbol(config)
    store = PaperStateStore(config)
    local_now = _local_now(config, now)
    broker_client = broker or AlpacaPaperBrokerClient()
    market_date = local_now.date()

    if not _is_trading_day(broker_client, market_date=market_date):
        status = {
            "event": "paper-decision",
            "status": SUBMISSION_SKIPPED,
            "reason": "non_trading_day",
            "market_date": market_date.isoformat(),
            "updated_at": _now_utc(now).isoformat(),
        }
        status_path = store.write_status(status)
        return {"status_path": str(status_path), "status": status}

    panel = _build_alpaca_panel(config, provider=provider)
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
            "updated_at": _now_utc(now).isoformat(),
        }
        status_path = store.write_status(status)
        return {"status_path": str(status_path), "status": status}

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
            "updated_at": _now_utc(now).isoformat(),
        }
        status_path = store.write_status(status)
        return {
            "proposal_id": proposal_id,
            "proposal_path": str(store.inbox_proposal_path(proposal_id)),
            "evidence_path": str(evidence_path),
            "status_path": str(status_path),
            "status": status,
        }

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
        "created_at": _now_utc(now).isoformat(),
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
        "created_at": _now_utc(now).isoformat(),
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
        "updated_at": _now_utc(now).isoformat(),
    }
    status_path = store.write_status(status)
    return {
        "proposal_id": proposal_id,
        "proposal_path": str(proposal_path),
        "evidence_path": str(evidence_path),
        "status_path": str(status_path),
        "status": status,
    }


def list_paper_proposals(config: ExperimentConfig) -> list[dict[str, Any]]:
    validate_paper_trading_config(config)
    return PaperStateStore(config).list_proposals()


def read_paper_proposal(
    config: ExperimentConfig,
    *,
    proposal_id: str,
) -> dict[str, Any]:
    validate_paper_trading_config(config)
    return PaperStateStore(config).load_proposal(proposal_id)


def read_paper_evidence(
    config: ExperimentConfig,
    *,
    proposal_id: str,
) -> dict[str, Any]:
    validate_paper_trading_config(config)
    store = PaperStateStore(config)
    proposal = store.load_proposal(proposal_id)
    return store.load_evidence(proposal["effective_date"])


def decide_paper_proposal(
    config: ExperimentConfig,
    *,
    proposal_id: str,
    decision: str,
    actor: str,
    rationale: str | None = None,
    provider: str | None = None,
    model: str | None = None,
    fallback_used: bool = False,
    fallback_reason: str | None = None,
    now: datetime | None = None,
) -> dict[str, Any]:
    validate_paper_trading_config(config)
    if decision not in {"approve", "reject"}:
        raise RuntimeError("paper-approve requires decision to be either approve or reject.")
    if actor not in {"agent", "manual"}:
        raise RuntimeError("paper-approve requires actor to be either agent or manual.")

    store = PaperStateStore(config)
    proposal = store.load_proposal(proposal_id)
    trade_date = proposal["effective_date"]
    submission_path = store.trade_submission_path(trade_date)
    if submission_path.exists():
        raise RuntimeError("Cannot change approval after paper-submit has already persisted state.")

    required_actor = None
    if config.paper.execution_mode == "agent_approval":
        required_actor = "agent"
    elif config.paper.execution_mode == "manual_approval":
        required_actor = "manual"
    else:
        raise RuntimeError("paper-approve is not used when execution_mode='autonomous'.")

    if actor != required_actor:
        raise RuntimeError(
            f"paper-approve for execution_mode='{config.paper.execution_mode}' requires actor='{required_actor}'."
        )

    approval_status = APPROVAL_APPROVED if decision == "approve" else APPROVAL_REJECTED
    approval_timestamp = _now_utc(now).isoformat()
    proposal["approval_status"] = approval_status
    proposal["approval_actor"] = actor
    proposal["approval_decision"] = decision
    proposal["approval_timestamp"] = approval_timestamp
    if rationale is not None:
        proposal["approval_rationale"] = rationale
    if provider is not None:
        proposal["approval_backend"] = provider
    if model is not None:
        proposal["approval_model"] = model
    proposal["approval_fallback_used"] = bool(fallback_used)
    if fallback_reason:
        proposal["approval_fallback_reason"] = fallback_reason
    proposal_path = store.update_proposal(proposal)
    approval_record = {
        "proposal_id": proposal_id,
        "trade_date": trade_date,
        "decision": decision,
        "approval_status": approval_status,
        "actor": actor,
        "timestamp": approval_timestamp,
        "provider": provider,
        "model": model,
        "fallback_used": bool(fallback_used),
        "fallback_reason": fallback_reason,
        "rationale": rationale,
    }
    approval_path = _json_dump(store.trade_approval_path(trade_date), approval_record)
    status = {
        "event": "paper-approve",
        "status": approval_status,
        "proposal_id": proposal_id,
        "proposal_path": str(proposal_path),
        "approval_path": str(approval_path),
        "updated_at": approval_timestamp,
    }
    status_path = store.write_status(status)
    return {
        "proposal_id": proposal_id,
        "proposal_path": str(proposal_path),
        "approval_path": str(approval_path),
        "status_path": str(status_path),
        "status": status,
    }


def _submission_gate_status(
    config: ExperimentConfig,
    proposal: dict[str, Any],
) -> tuple[str, str]:
    if config.paper.execution_mode == "autonomous":
        return "ready", ""

    approval_status = proposal.get("approval_status", APPROVAL_PENDING)
    if approval_status == APPROVAL_REJECTED:
        return SUBMISSION_SKIPPED, "rejected"
    if approval_status != APPROVAL_APPROVED:
        return SUBMISSION_SKIPPED, "missing_approval"

    required_actor = "agent" if config.paper.execution_mode == "agent_approval" else "manual"
    if proposal.get("approval_actor") != required_actor:
        return SUBMISSION_SKIPPED, "wrong_actor"
    return "ready", ""


def run_paper_submit(
    config: ExperimentConfig,
    *,
    now: datetime | None = None,
    broker: AlpacaPaperBrokerClient | None = None,
) -> dict[str, Any]:
    validate_paper_trading_config(config)
    paper_symbol = _paper_symbol(config)
    store = PaperStateStore(config)
    proposal = store.latest_proposal()
    if proposal is None:
        status = {
            "event": "paper-submit",
            "status": SUBMISSION_SKIPPED,
            "reason": "no_proposal",
            "updated_at": _now_utc(now).isoformat(),
        }
        status_path = store.write_status(status)
        return {"status_path": str(status_path), "status": status}

    local_now = _local_now(config, now)
    submission_clock = _clock_value(config.paper.submission_time)
    if local_now.time() < submission_clock:
        raise RuntimeError(
            "paper-submit is only allowed at or after "
            f"{config.paper.submission_time} {config.paper.schedule_timezone}."
        )

    trade_date = proposal["effective_date"]
    submission_path = store.trade_submission_path(trade_date)
    if submission_path.exists():
        submission = _json_load(submission_path)
        status = {
            "event": "paper-submit",
            "status": "existing_submission",
            "proposal_id": proposal["proposal_id"],
            "submission_path": str(submission_path),
            "updated_at": _now_utc(now).isoformat(),
        }
        status_path = store.write_status(status)
        return {
            "submission_path": str(submission_path),
            "status_path": str(status_path),
            "status": status,
            "submission": submission,
        }

    broker_client = broker or AlpacaPaperBrokerClient()
    gate_status, gate_reason = _submission_gate_status(config, proposal)
    if gate_status != "ready":
        submission = {
            "proposal_id": proposal["proposal_id"],
            "trade_date": trade_date,
            "status": gate_status,
            "reason": gate_reason,
            "updated_at": _now_utc(now).isoformat(),
        }
        _json_dump(submission_path, submission)
        status = {
            "event": "paper-submit",
            "status": gate_status,
            "reason": gate_reason,
            "proposal_id": proposal["proposal_id"],
            "submission_path": str(submission_path),
            "updated_at": _now_utc(now).isoformat(),
        }
        status_path = store.write_status(status)
        return {
            "submission_path": str(submission_path),
            "status_path": str(status_path),
            "status": status,
            "submission": submission,
        }

    account = broker_client.get_account()
    _json_dump(store.trade_account_snapshot_path(trade_date), account)
    position = broker_client.get_position(paper_symbol)
    current_qty = float(position["qty"]) if position is not None else 0.0
    equity = float(account["equity"])
    reference_price = float(proposal["reference_price"])
    desired_qty = 0.0
    if float(proposal["target_weight"]) > 0.0:
        desired_qty = equity / reference_price

    delta_qty = round(desired_qty - current_qty, 6)
    side = "buy" if delta_qty > 0.0 else "sell"
    order_preview = {
        "proposal_id": proposal["proposal_id"],
        "trade_date": trade_date,
        "symbol": paper_symbol,
        "equity": equity,
        "reference_price": reference_price,
        "current_qty": current_qty,
        "desired_qty": desired_qty,
        "delta_qty": delta_qty,
        "side": side if abs(delta_qty) > 0.0 else "none",
        "updated_at": _now_utc(now).isoformat(),
    }
    _json_dump(store.trade_order_preview_path(trade_date), order_preview)

    if abs(delta_qty) < 1e-6:
        submission = {
            "proposal_id": proposal["proposal_id"],
            "trade_date": trade_date,
            "status": SUBMISSION_NOOP,
            "reason": "already_at_target",
            "order_preview_path": str(store.trade_order_preview_path(trade_date)),
            "updated_at": _now_utc(now).isoformat(),
        }
        _json_dump(submission_path, submission)
        status = {
            "event": "paper-submit",
            "status": SUBMISSION_NOOP,
            "proposal_id": proposal["proposal_id"],
            "submission_path": str(submission_path),
            "updated_at": _now_utc(now).isoformat(),
        }
        status_path = store.write_status(status)
        return {
            "submission_path": str(submission_path),
            "status_path": str(status_path),
            "status": status,
            "submission": submission,
        }

    client_order_id = f"marketlab-{proposal['proposal_id']}".replace("_", "-")[:48]
    order = broker_client.submit_fractional_day_market_order(
        symbol=paper_symbol,
        qty=abs(delta_qty),
        side=side,
        client_order_id=client_order_id,
    )
    try:
        order_status = broker_client.get_order(str(order["id"]))
        poll_status = "observed"
    except RuntimeError as exc:
        order_status = {
            "id": order.get("id"),
            "client_order_id": order.get("client_order_id", client_order_id),
            "status": order.get("status", "unknown"),
            "poll_error": str(exc),
        }
        poll_status = "timeout"
    _json_dump(store.trade_order_status_path(trade_date), order_status)

    submission = {
        "proposal_id": proposal["proposal_id"],
        "trade_date": trade_date,
        "status": SUBMISSION_SUBMITTED,
        "order_id": order["id"],
        "client_order_id": order.get("client_order_id", client_order_id),
        "order_status": order_status.get("status", order.get("status", "unknown")),
        "poll_status": poll_status,
        "order_preview_path": str(store.trade_order_preview_path(trade_date)),
        "account_snapshot_path": str(store.trade_account_snapshot_path(trade_date)),
        "order_status_path": str(store.trade_order_status_path(trade_date)),
        "updated_at": _now_utc(now).isoformat(),
    }
    _json_dump(submission_path, submission)
    status = {
        "event": "paper-submit",
        "status": SUBMISSION_SUBMITTED,
        "proposal_id": proposal["proposal_id"],
        "submission_path": str(submission_path),
        "updated_at": _now_utc(now).isoformat(),
    }
    status_path = store.write_status(status)
    return {
        "submission_path": str(submission_path),
        "status_path": str(status_path),
        "status": status,
        "submission": submission,
    }


def get_paper_status(config: ExperimentConfig) -> dict[str, Any]:
    validate_paper_trading_config(config)
    store = PaperStateStore(config)
    latest_proposal = store.latest_proposal()
    proposals = store.list_proposals()
    pending_proposals = [
        proposal
        for proposal in proposals
        if proposal.get("approval_status", APPROVAL_PENDING) == APPROVAL_PENDING
    ]
    return {
        "status_path": str(store.status_path),
        "status": store.read_status(),
        "latest_proposal": latest_proposal,
        "pending_proposal_count": len(pending_proposals),
    }
