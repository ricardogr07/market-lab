from __future__ import annotations

import json
from datetime import timedelta
from pathlib import Path
from typing import Any

import pandas as pd

from marketlab.config import ExperimentConfig
from marketlab.paper.alpaca import AlpacaMarketDataProvider
from marketlab.paper.service import (
    PaperStateStore,
    _paper_symbol,
    validate_paper_trading_config,
)


def _json_load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _markdown_table(frame: pd.DataFrame) -> str:
    columns = list(frame.columns)
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join(["---"] * len(columns)) + " |"
    rows = [
        "| " + " | ".join(str(value) for value in row) + " |"
        for row in frame.itertuples(index=False, name=None)
    ]
    return "\n".join([header, separator, *rows])


def _trade_rows(store: PaperStateStore) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for trade_dir in sorted(store.trades_root.glob("*")):
        if not trade_dir.is_dir():
            continue
        proposal_path = trade_dir / "proposal.json"
        evidence_path = trade_dir / "evidence.json"
        if not proposal_path.exists() or not evidence_path.exists():
            continue
        proposal = _json_load(proposal_path)
        evidence = _json_load(evidence_path)
        approval_path = trade_dir / "approval.json"
        submission_path = trade_dir / "submission.json"
        approval = _json_load(approval_path) if approval_path.exists() else None
        submission = _json_load(submission_path) if submission_path.exists() else None
        rows.append(
            {
                "proposal": proposal,
                "evidence": evidence,
                "approval": approval,
                "submission": submission,
            }
        )
    return rows


def _load_price_frame(
    config: ExperimentConfig,
    *,
    symbol: str,
    start_date: str,
    end_date: str,
    provider: AlpacaMarketDataProvider | None = None,
) -> pd.DataFrame:
    slow_window = int(config.baselines.sma.slow_window)
    start_lookup = (pd.Timestamp(start_date) - timedelta(days=max(120, slow_window * 3))).date()
    frame = (provider or AlpacaMarketDataProvider()).download_symbol_history(
        symbol,
        start_lookup.isoformat(),
        end_date,
        config.data.interval,
    )
    price_frame = frame.rename(columns={"Date": "date", "Open": "open", "Close": "close"}).copy()
    price_frame["date"] = pd.to_datetime(price_frame["date"]).dt.normalize()
    price_frame["open"] = price_frame["open"].astype(float)
    price_frame["close"] = price_frame["close"].astype(float)
    price_frame = price_frame.loc[
        price_frame["date"].between(pd.Timestamp(start_date) - timedelta(days=120), pd.Timestamp(end_date))
    ].copy()
    price_frame["prev_close"] = price_frame["close"].shift(1)
    price_frame["overnight_return"] = (
        price_frame["open"] / price_frame["prev_close"]
    ).where(price_frame["prev_close"].notna(), 1.0) - 1.0
    price_frame["intraday_return"] = (price_frame["close"] / price_frame["open"]) - 1.0
    return price_frame.reset_index(drop=True)


def _replay_exposure(
    price_frame: pd.DataFrame,
    *,
    target_by_date: dict[str, float],
    default_exposure: float = 0.0,
) -> pd.Series:
    exposures: list[float] = []
    current = float(default_exposure)
    for timestamp in price_frame["date"]:
        key = timestamp.date().isoformat()
        if key in target_by_date:
            current = float(target_by_date[key])
        exposures.append(current)
    return pd.Series(exposures, index=price_frame.index, dtype=float)


def _strategy_frame(
    price_frame: pd.DataFrame,
    *,
    strategy_name: str,
    exposure: pd.Series,
) -> pd.DataFrame:
    current_exposure = exposure.astype(float)
    previous_exposure = current_exposure.shift(1).fillna(0.0)
    strategy_return = (
        (1.0 + previous_exposure * price_frame["overnight_return"])
        * (1.0 + current_exposure * price_frame["intraday_return"])
        - 1.0
    )
    equity = (1.0 + strategy_return).cumprod()
    return pd.DataFrame(
        {
            "date": price_frame["date"],
            "strategy": strategy_name,
            "exposure": current_exposure,
            "return": strategy_return,
            "equity": equity,
        }
    )


def _buy_hold_exposure(price_frame: pd.DataFrame) -> pd.Series:
    return pd.Series(1.0, index=price_frame.index, dtype=float)


def _sma_exposure(config: ExperimentConfig, price_frame: pd.DataFrame) -> pd.Series:
    fast_window = int(config.baselines.sma.fast_window)
    slow_window = int(config.baselines.sma.slow_window)
    fast = price_frame["close"].rolling(window=fast_window, min_periods=fast_window).mean()
    slow = price_frame["close"].rolling(window=slow_window, min_periods=slow_window).mean()
    signal = (fast > slow).astype(float).shift(1).fillna(0.0)
    return signal.astype(float)


def _realized_target_weight(submission: dict[str, Any] | None, proposal: dict[str, Any], current: float) -> float:
    if submission is None:
        return current
    status = str(submission.get("status", ""))
    if status == "no_trade_required":
        return float(proposal.get("target_weight", current))
    if status != "submitted":
        return current
    order_status = str(submission.get("order_status", "")).lower()
    if order_status in {"rejected", "canceled", "expired"}:
        return current
    return float(proposal.get("target_weight", current))


def _paper_target_maps(
    rows: list[dict[str, Any]],
) -> tuple[dict[str, float], dict[str, float], dict[str, dict[str, float]], list[dict[str, Any]]]:
    consensus_targets: dict[str, float] = {}
    realized_targets: dict[str, float] = {}
    model_targets: dict[str, dict[str, float]] = {}
    journal_rows: list[dict[str, Any]] = []
    current_realized = 0.0

    for row in sorted(rows, key=lambda item: item["proposal"]["effective_date"]):
        proposal = row["proposal"]
        evidence = row["evidence"]
        submission = row["submission"] or {}
        trade_date = str(proposal["effective_date"])
        consensus_targets[trade_date] = float(proposal["target_weight"])
        for model_row in evidence.get("models", []):
            model_targets.setdefault(model_row["model_name"], {})[trade_date] = float(
                model_row["target_weight"]
            )
        current_realized = _realized_target_weight(submission, proposal, current_realized)
        realized_targets[trade_date] = current_realized
        journal_rows.append(
            {
                "proposal_id": proposal["proposal_id"],
                "symbol": proposal["symbol"],
                "signal_date": proposal["signal_date"],
                "effective_date": trade_date,
                "consensus_decision": proposal["decision"],
                "target_weight": proposal["target_weight"],
                "long_vote_count": proposal["long_vote_count"],
                "cash_vote_count": proposal["cash_vote_count"],
                "approval_status": proposal.get("approval_status", ""),
                "approval_actor": proposal.get("approval_actor", ""),
                "approval_backend": proposal.get("approval_backend", ""),
                "approval_model": proposal.get("approval_model", ""),
                "approval_fallback_used": proposal.get("approval_fallback_used", False),
                "approval_fallback_reason": proposal.get("approval_fallback_reason", ""),
                "approval_rationale": proposal.get("approval_rationale", ""),
                "submission_status": submission.get("status", ""),
                "submission_reason": submission.get("reason", ""),
                "realized_target_weight": current_realized,
            }
        )
    return consensus_targets, realized_targets, model_targets, journal_rows


def _summary_frame(performance: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for strategy_name, strategy_frame in performance.groupby("strategy", sort=True):
        strategy_frame = strategy_frame.reset_index(drop=True)
        cumulative_return = float(strategy_frame["equity"].iloc[-1] - 1.0)
        periods = max(len(strategy_frame), 1)
        annualized_return = float(strategy_frame["equity"].iloc[-1] ** (252 / periods) - 1.0)
        rows.append(
            {
                "strategy": strategy_name,
                "cumulative_return": cumulative_return,
                "annualized_return": annualized_return,
                "avg_exposure": float(strategy_frame["exposure"].mean()),
                "days_long": int((strategy_frame["exposure"] > 0.0).sum()),
                "final_equity": float(strategy_frame["equity"].iloc[-1]),
            }
        )
    return pd.DataFrame(rows).sort_values(["cumulative_return", "strategy"], ascending=[False, True])


def _write_report_markdown(
    *,
    path: Path,
    config: ExperimentConfig,
    summary: pd.DataFrame,
    journal: pd.DataFrame,
    start_date: str,
    end_date: str,
) -> Path:
    lines = [
        f"# {config.experiment_name} paper report",
        "",
        "## Scope",
        "",
        f"- Symbol: {_paper_symbol(config)}",
        f"- Window: {start_date} to {end_date}",
        "- Strategy shape: single-ETF, long-or-cash, daily paper loop",
        "",
        "## Summary",
        "",
        _markdown_table(summary.round(6)),
        "",
        "## Decision Journal",
        "",
        _markdown_table(journal.round(6) if not journal.empty else journal),
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def run_paper_report(
    config: ExperimentConfig,
    *,
    start_date: str,
    end_date: str,
    provider: AlpacaMarketDataProvider | None = None,
) -> dict[str, Any]:
    validate_paper_trading_config(config)
    symbol = _paper_symbol(config)
    store = PaperStateStore(config)
    rows = _trade_rows(store)
    if not rows:
        raise RuntimeError("paper-report requires persisted proposal and evidence artifacts.")

    price_frame = _load_price_frame(
        config,
        symbol=symbol,
        start_date=start_date,
        end_date=end_date,
        provider=provider,
    )
    report_prices = price_frame.loc[
        price_frame["date"].between(pd.Timestamp(start_date), pd.Timestamp(end_date))
    ].reset_index(drop=True)
    if report_prices.empty:
        raise RuntimeError("paper-report found no price rows in the requested date range.")

    consensus_targets, realized_targets, model_targets, journal_rows = _paper_target_maps(rows)
    performance_frames = []

    def _windowed_strategy_frame(strategy_name: str, exposure: pd.Series) -> pd.DataFrame:
        full_frame = _strategy_frame(price_frame, strategy_name=strategy_name, exposure=exposure)
        return full_frame.loc[
            full_frame["date"].between(pd.Timestamp(start_date), pd.Timestamp(end_date))
        ].reset_index(drop=True)

    performance_frames.extend(
        [
            _windowed_strategy_frame(
                "paper_realized",
                _replay_exposure(price_frame, target_by_date=realized_targets),
            ),
            _windowed_strategy_frame(
                "consensus",
                _replay_exposure(price_frame, target_by_date=consensus_targets),
            ),
            _windowed_strategy_frame("buy_hold", _buy_hold_exposure(price_frame)),
        ]
    )
    if config.baselines.sma.enabled:
        performance_frames.append(
            _windowed_strategy_frame("sma", _sma_exposure(config, price_frame))
        )
    for model_name, target_map in sorted(model_targets.items()):
        performance_frames.append(
            _windowed_strategy_frame(
                f"model_{model_name}",
                _replay_exposure(price_frame, target_by_date=target_map),
            )
        )

    performance = pd.concat(performance_frames, ignore_index=True)
    summary = _summary_frame(performance).reset_index(drop=True)
    journal = pd.DataFrame(journal_rows).sort_values("effective_date").reset_index(drop=True)

    output_dir = store.report_dir(start_date, end_date)
    summary_path = output_dir / "summary.csv"
    journal_path = output_dir / "decision_journal.csv"
    performance_path = output_dir / "performance.csv"
    report_path = output_dir / "report.md"
    summary.to_csv(summary_path, index=False)
    journal.to_csv(journal_path, index=False)
    performance.to_csv(performance_path, index=False)
    _write_report_markdown(
        path=report_path,
        config=config,
        summary=summary,
        journal=journal,
        start_date=start_date,
        end_date=end_date,
    )
    return {
        "report_dir": str(output_dir),
        "summary_path": str(summary_path),
        "decision_journal_path": str(journal_path),
        "performance_path": str(performance_path),
        "report_path": str(report_path),
        "summary_preview": summary.head(10).to_dict(orient="records"),
    }
