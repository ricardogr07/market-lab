from __future__ import annotations

import math
import re
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any
from uuid import uuid4

import pandas as pd

from marketlab.data.panel import load_panel_csv
from marketlab.pipeline import prepare_data
from marketlab.shadow.contract import VerifiedShadowContract, verify_shadow_contract
from marketlab.shadow.decision import (
    ShadowDecisionEvaluation,
    ShadowDecisionEvaluator,
    ShadowDecisionRequest,
    ShadowDecisionResult,
    run_shadow_decision,
    shadow_bars_from_panel,
)
from marketlab.shadow.evaluator import NativeShadowDecisionEvaluator
from marketlab.shadow.evidence import (
    ShadowAttemptStore,
    ShadowDecisionEvidenceStore,
    ShadowEvidenceWrite,
    ShadowLabelEvidenceStore,
)
from marketlab.shadow.journal import ShadowDecisionJournal, canonical_fingerprint

PanelRefresher = Callable[[VerifiedShadowContract], pd.DataFrame]


@dataclass(frozen=True, slots=True)
class ShadowSchedulerResult:
    attempts: tuple[ShadowEvidenceWrite, ...]
    decision: ShadowDecisionResult | None
    decision_evidence: ShadowEvidenceWrite | None
    label_evidence: tuple[ShadowEvidenceWrite, ...]


def run_shadow_scheduler(
    config_path: str | Path,
    *,
    as_of: datetime,
    evaluator: ShadowDecisionEvaluator | None = None,
    panel_refresher: PanelRefresher | None = None,
    execution_id: str | None = None,
    journal: ShadowDecisionJournal | None = None,
    attempt_store: ShadowAttemptStore | None = None,
    decision_evidence_store: ShadowDecisionEvidenceStore | None = None,
    label_evidence_store: ShadowLabelEvidenceStore | None = None,
) -> ShadowSchedulerResult:
    contract = verify_shadow_contract(config_path)
    runtime = _as_utc(as_of)
    selected_journal = journal or ShadowDecisionJournal(
        contract.artifact_root / "decisions"
    )
    selected_attempts = attempt_store or ShadowAttemptStore(
        contract.artifact_root / "attempts"
    )
    selected_decision_evidence = (
        decision_evidence_store
        or ShadowDecisionEvidenceStore(contract.artifact_root / "evidence" / "decisions")
    )
    selected_label_evidence = label_evidence_store or ShadowLabelEvidenceStore(
        contract.artifact_root / "evidence" / "labels"
    )

    attempts: list[ShadowEvidenceWrite] = []
    for missed_date in _unaccounted_earlier_dates(
        contract=contract,
        as_of=runtime.date(),
        journal=selected_journal,
        attempt_store=selected_attempts,
    ):
        attempts.append(
            selected_attempts.write(
                _attempt_record(
                    contract=contract,
                    effective_date=missed_date,
                    attempt_id=f"missed-{missed_date.isoformat()}",
                    execution_id=f"missed-{missed_date.isoformat()}",
                    started_at=runtime,
                    completed_at=runtime,
                    outcome="missed",
                    reason="scheduled_date_was_not_evaluated_at_runtime",
                )
            )
        )

    decision: ShadowDecisionResult | None = None
    evidence_write: ShadowEvidenceWrite | None = None
    panel: pd.DataFrame | None = None
    current_date = runtime.date()
    if contract.protocol_start <= current_date <= contract.protocol_end:
        attempt_id = execution_id or uuid4().hex
        selected_evaluator = evaluator or NativeShadowDecisionEvaluator()
        captured: dict[str, ShadowDecisionEvaluation] = {}
        decision_path: str | None = None
        decision_fingerprint: str | None = None

        def _evaluate(context):
            evaluation = selected_evaluator(context)
            captured["evaluation"] = evaluation
            return evaluation

        try:
            panel = (panel_refresher or _refresh_panel)(contract)
            decision = run_shadow_decision(
                ShadowDecisionRequest(
                    contract=contract,
                    as_of=runtime,
                    bars=shadow_bars_from_panel(panel),
                ),
                evaluator=_evaluate,
                journal=selected_journal,
            )
            evaluation = captured["evaluation"]
            decision_path = str(decision.path)
            decision_fingerprint = str(decision.record["output_fingerprint"])
            outcome = str(decision.record["status"])
            if evaluation.status == "success":
                evidence_write = selected_decision_evidence.write(
                    _decision_evidence_record(
                        contract=contract,
                        decision=decision,
                        evaluation=evaluation,
                    )
                )
            attempts.append(
                selected_attempts.write(
                    _attempt_record(
                        contract=contract,
                        effective_date=current_date,
                        attempt_id=attempt_id,
                        execution_id=attempt_id,
                        started_at=runtime,
                        completed_at=runtime,
                        outcome=outcome,
                        decision_path=decision_path,
                        decision_fingerprint=decision_fingerprint,
                        reason=decision.record.get("reason"),
                    )
                )
            )
        except Exception as exc:
            attempts.append(
                selected_attempts.write(
                    _attempt_record(
                        contract=contract,
                        effective_date=current_date,
                        attempt_id=attempt_id,
                        execution_id=attempt_id,
                        started_at=runtime,
                        completed_at=datetime.now(UTC),
                        outcome="failed",
                        decision_path=decision_path,
                        decision_fingerprint=decision_fingerprint,
                        error_type=type(exc).__name__,
                        reason=_sanitized_reason(exc),
                    )
                )
            )
            raise

    if panel is None:
        panel = (panel_refresher or _refresh_panel)(contract)

    labels = _write_matured_labels(
        contract=contract,
        as_of=runtime,
        panel=panel,
        journal=selected_journal,
        decision_evidence_store=selected_decision_evidence,
        label_evidence_store=selected_label_evidence,
    )
    return ShadowSchedulerResult(
        attempts=tuple(attempts),
        decision=decision,
        decision_evidence=evidence_write,
        label_evidence=tuple(labels),
    )


def _refresh_panel(contract: VerifiedShadowContract) -> pd.DataFrame:
    _, panel_path = prepare_data(contract.config)
    return load_panel_csv(panel_path)


def _unaccounted_earlier_dates(
    *,
    contract: VerifiedShadowContract,
    as_of: date,
    journal: ShadowDecisionJournal,
    attempt_store: ShadowAttemptStore,
) -> list[date]:
    last_date = min(as_of - timedelta(days=1), contract.protocol_end)
    if last_date < contract.protocol_start:
        return []
    missing: list[date] = []
    current = contract.protocol_start
    while current <= last_date:
        if journal.read(current) is None and not attempt_store.list_for(current):
            missing.append(current)
        current += timedelta(days=1)
    return missing


def _attempt_record(
    *,
    contract: VerifiedShadowContract,
    effective_date: date,
    attempt_id: str,
    execution_id: str,
    started_at: datetime,
    completed_at: datetime,
    outcome: str,
    decision_path: str | None = None,
    decision_fingerprint: str | None = None,
    error_type: str | None = None,
    reason: object = None,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "attempt_id": attempt_id,
        "execution_id": execution_id,
        "scheduled_date": effective_date.isoformat(),
        "effective_date": effective_date.isoformat(),
        "started_at": _iso_utc(started_at),
        "completed_at": _iso_utc(completed_at),
        "outcome": outcome,
        "decision_path": decision_path,
        "decision_fingerprint": decision_fingerprint,
        "error_type": error_type,
        "reason": _sanitized_text(str(reason)) if reason is not None else None,
        **_contract_fields(contract),
    }


def _decision_evidence_record(
    *,
    contract: VerifiedShadowContract,
    decision: ShadowDecisionResult,
    evaluation: ShadowDecisionEvaluation,
) -> dict[str, Any]:
    diagnostics = dict(evaluation.input_payload)
    diagnostic_fingerprint = diagnostics.get("diagnostic_fingerprint")
    expected_fingerprint = canonical_fingerprint(
        {
            key: value
            for key, value in diagnostics.items()
            if key != "diagnostic_fingerprint"
        }
    )
    if diagnostic_fingerprint != expected_fingerprint:
        raise RuntimeError("Shadow evaluator diagnostic fingerprint is invalid.")
    return {
        "schema_version": 1,
        "effective_date": str(decision.record["effective_date"]),
        "decision_path": str(decision.path),
        "decision_fingerprint": str(decision.record["output_fingerprint"]),
        "raw_score": diagnostics.get("raw_score"),
        "selected_tier": diagnostics.get("selected_tier"),
        "selection_source": evaluation.selection_source,
        "fallback_mode": evaluation.fallback_mode,
        "regime_classification": diagnostics.get("regime_classification"),
        "gate_bull": _gate_bull(diagnostics),
        "input_cutoff": diagnostics.get("input_cutoff"),
        "diagnostic_fingerprint": diagnostic_fingerprint,
        "diagnostics": diagnostics,
        **_contract_fields(contract),
    }


def _write_matured_labels(
    *,
    contract: VerifiedShadowContract,
    as_of: datetime,
    panel: pd.DataFrame,
    journal: ShadowDecisionJournal,
    decision_evidence_store: ShadowDecisionEvidenceStore,
    label_evidence_store: ShadowLabelEvidenceStore,
) -> list[ShadowEvidenceWrite]:
    working = panel.copy()
    working["timestamp"] = pd.to_datetime(working["timestamp"], utc=True)
    working = working.sort_values("timestamp").drop_duplicates("timestamp")
    completed = working.loc[
        working["timestamp"] + pd.Timedelta(days=1) <= pd.Timestamp(as_of)
    ].reset_index(drop=True)
    date_positions = {
        timestamp.date(): index
        for index, timestamp in enumerate(completed["timestamp"])
    }
    decisions = sorted(journal.list(), key=lambda record: record["effective_date"])
    writes: list[ShadowEvidenceWrite] = []
    close_exposure = 0.0
    close_cash_weight = 1.0
    last_position: int | None = None
    for decision in decisions:
        effective_date = date.fromisoformat(str(decision["effective_date"]))
        if decision.get("status") != "success":
            continue
        position = date_positions.get(effective_date)
        if position is None:
            continue
        end_position = position + contract.maturity_lag_bars - 1
        if end_position >= len(completed):
            break
        existing = label_evidence_store.read(effective_date)
        if existing is not None:
            close_exposure = float(existing["closing_exposure"])
            close_cash_weight = float(existing["closing_cash_weight"])
            last_position = position
            continue
        decision_evidence = decision_evidence_store.read(effective_date)
        if decision_evidence is None:
            raise RuntimeError(
                f"Missing decision evidence for matured date {effective_date.isoformat()}."
            )
        if last_position is not None:
            for gap_position in range(last_position + 1, position):
                close_exposure, close_cash_weight = _advance_without_rebalance(
                    previous_adj_close=float(
                        completed.iloc[gap_position - 1]["adj_close"]
                    ),
                    adj_open=float(completed.iloc[gap_position]["adj_open"]),
                    adj_close=float(completed.iloc[gap_position]["adj_close"]),
                    close_exposure=close_exposure,
                    close_cash_weight=close_cash_weight,
                )
        path = completed.iloc[position : end_position + 1]
        record = _label_evidence_record(
            contract=contract,
            decision=decision,
            decision_evidence=decision_evidence,
            path=path,
            previous_adj_close=(
                float(completed.iloc[position - 1]["adj_close"])
                if position > 0 and last_position is not None
                else None
            ),
            close_exposure=close_exposure,
            close_cash_weight=close_cash_weight,
        )
        writes.append(label_evidence_store.write(record))
        close_exposure = float(record["closing_exposure"])
        close_cash_weight = float(record["closing_cash_weight"])
        last_position = position
    return writes


def _label_evidence_record(
    *,
    contract: VerifiedShadowContract,
    decision: dict[str, Any],
    decision_evidence: dict[str, Any],
    path: pd.DataFrame,
    previous_adj_close: float | None,
    close_exposure: float,
    close_cash_weight: float,
) -> dict[str, Any]:
    allocation = float(decision["target_allocation"])
    entry_adj_open = float(path.iloc[0]["adj_open"])
    effective_adj_close = float(path.iloc[0]["adj_close"])
    closes = [float(value) for value in path["adj_close"]]
    exit_adj_close = closes[-1]
    benchmark_return = (exit_adj_close / entry_adj_open) - 1.0
    overnight_asset_return = (
        (entry_adj_open / previous_adj_close) - 1.0
        if previous_adj_close is not None
        else 0.0
    )
    pre_open_exposure, pre_open_cash, overnight_return = _advance_weights(
        exposure=close_exposure,
        cash_weight=close_cash_weight,
        asset_return=overnight_asset_return,
    )
    turnover = abs(allocation - pre_open_exposure)
    intraday_asset_return = (effective_adj_close / entry_adj_open) - 1.0
    closing_exposure, closing_cash_weight, intraday_return = _advance_weights(
        exposure=allocation,
        cash_weight=1.0 - allocation,
        asset_return=intraday_asset_return,
    )
    daily_gross_return = (1.0 + overnight_return) * (1.0 + intraday_return) - 1.0
    base_cost_bps = float(contract.config.portfolio.costs.bps_per_trade)
    strategy_return = daily_gross_return - turnover * base_cost_bps / 10_000.0
    daily_benchmark_return = (
        (effective_adj_close / previous_adj_close) - 1.0
        if previous_adj_close is not None
        else intraday_asset_return
    )
    path_returns = [(value / entry_adj_open) - 1.0 for value in closes]
    forward_drawdown = min(0.0, min(path_returns))
    realized_volatility = float(
        pd.Series([entry_adj_open, *closes]).pct_change().dropna().std(ddof=0)
    )
    if not math.isfinite(realized_volatility):
        realized_volatility = 0.0
    utilities = {
        tier: _allocation_utility(
            tier=tier,
            forward_return=benchmark_return,
            forward_drawdown=forward_drawdown,
            realized_volatility=realized_volatility,
            cost_bps=base_cost_bps,
            contract=contract,
        )
        for tier in (0.0, 0.25, 0.50, 1.0)
    }
    realized_target_weight = max(
        utilities,
        key=lambda tier: (utilities[tier], -tier),
    )
    return {
        "schema_version": 1,
        "effective_date": str(decision["effective_date"]),
        "target_end_date": pd.Timestamp(path.iloc[-1]["timestamp"]).date().isoformat(),
        "decision_fingerprint": str(decision["output_fingerprint"]),
        "decision_evidence_fingerprint": str(
            decision_evidence["output_fingerprint"]
        ),
        "diagnostic_fingerprint": str(
            decision_evidence["diagnostic_fingerprint"]
        ),
        "entry_adj_open": entry_adj_open,
        "effective_adj_close": effective_adj_close,
        "previous_adj_close": previous_adj_close,
        "path_adj_closes": closes,
        "exit_adj_close": exit_adj_close,
        "benchmark_return": benchmark_return,
        "daily_benchmark_return": daily_benchmark_return,
        "overnight_asset_return": overnight_asset_return,
        "intraday_asset_return": intraday_asset_return,
        "daily_gross_return": daily_gross_return,
        "strategy_return": strategy_return,
        "realized_utility": utilities[allocation],
        "realized_target_weight": realized_target_weight,
        "forward_drawdown": forward_drawdown,
        "forward_realized_volatility": realized_volatility,
        "exposure": allocation,
        "pre_open_exposure": pre_open_exposure,
        "pre_open_cash_weight": pre_open_cash,
        "closing_exposure": closing_exposure,
        "closing_cash_weight": closing_cash_weight,
        "turnover": turnover,
        "cost_bps": base_cost_bps,
        "regime_classification": decision_evidence.get("regime_classification"),
        "gate_bull": bool(decision_evidence.get("gate_bull", False)),
        **_contract_fields(contract),
    }


def _advance_without_rebalance(
    *,
    previous_adj_close: float,
    adj_open: float,
    adj_close: float,
    close_exposure: float,
    close_cash_weight: float,
) -> tuple[float, float]:
    open_exposure, open_cash, _ = _advance_weights(
        exposure=close_exposure,
        cash_weight=close_cash_weight,
        asset_return=(adj_open / previous_adj_close) - 1.0,
    )
    next_exposure, next_cash, _ = _advance_weights(
        exposure=open_exposure,
        cash_weight=open_cash,
        asset_return=(adj_close / adj_open) - 1.0,
    )
    return next_exposure, next_cash


def _advance_weights(
    *,
    exposure: float,
    cash_weight: float,
    asset_return: float,
) -> tuple[float, float, float]:
    evolved_exposure = exposure * (1.0 + asset_return)
    portfolio_value = evolved_exposure + cash_weight
    if abs(portfolio_value) <= 1e-12:
        raise RuntimeError("Shadow label portfolio value collapsed to zero.")
    return (
        evolved_exposure / portfolio_value,
        cash_weight / portfolio_value,
        portfolio_value - 1.0,
    )


def _allocation_utility(
    *,
    tier: float,
    forward_return: float,
    forward_drawdown: float,
    realized_volatility: float,
    cost_bps: float,
    contract: VerifiedShadowContract,
) -> float:
    target = contract.config.target
    return (
        tier * forward_return
        - target.allocation_utility_drawdown_penalty
        * (tier**target.allocation_utility_risk_penalty_power)
        * abs(forward_drawdown)
        - target.allocation_utility_volatility_penalty
        * (tier**target.allocation_utility_risk_penalty_power)
        * realized_volatility
        - (cost_bps / 10_000.0) * tier
    )


def _contract_fields(contract: VerifiedShadowContract) -> dict[str, object]:
    return {
        "candidate_id": contract.candidate_id,
        "behavior_version": contract.behavior_version,
        "config_hash": contract.config_hash,
        "behavior_hash": contract.behavior_hash,
        "code_lock": contract.code_lock,
    }


def _gate_bull(diagnostics: dict[str, object]) -> bool:
    pipeline = diagnostics.get("pipeline")
    if not isinstance(pipeline, dict):
        return False
    prediction = pipeline.get("prediction")
    if not isinstance(prediction, dict):
        return False
    return bool(prediction.get("gate_bull", False))


def _sanitized_reason(exc: Exception) -> str:
    reason = _sanitized_text(str(exc))
    if not reason:
        return type(exc).__name__
    return reason


def _sanitized_text(value: str) -> str:
    reason = " ".join(value.split())
    reason = re.sub(
        r"(?i)\b(api[_-]?key|token|password|secret)\s*[:=]\s*\S+",
        r"\1=[redacted]",
        reason,
    )
    reason = re.sub(r"://[^/@\s]+@", "://[redacted]@", reason)
    return reason[:500]


def _as_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        raise ValueError("Shadow scheduler as_of must include an explicit timezone.")
    return value.astimezone(UTC)


def _iso_utc(value: datetime) -> str:
    return value.astimezone(UTC).isoformat().replace("+00:00", "Z")
