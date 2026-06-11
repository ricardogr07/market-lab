from __future__ import annotations

import math
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, replace
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any, Literal

import pandas as pd

from marketlab.data.panel import PANEL_COLUMNS
from marketlab.shadow.contract import VerifiedShadowContract, verify_shadow_contract
from marketlab.shadow.journal import (
    ShadowDecisionJournal,
    ShadowJournalError,
    ShadowJournalWrite,
    canonical_fingerprint,
)

ShadowDecisionStatus = Literal["success", "skipped", "failed"]
_ALLOWED_SELECTION_SOURCES = {
    "strict",
    "best_active_fallback",
    "regime_policy_fallback",
    "none",
}
_ALLOWED_FALLBACK_MODES = {
    "none",
    "best_active_fallback",
    "regime_policy_fallback",
}
_ALLOWED_ALLOCATIONS = (0.0, 0.25, 0.50, 1.0)
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")


class ShadowDecisionError(RuntimeError):
    """Raised when a shadow decision would violate the frozen protocol."""


@dataclass(frozen=True, slots=True)
class ShadowBar:
    symbol: str
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    adj_close: float
    adj_open: float
    adj_high: float
    adj_low: float

    def as_fingerprint_payload(self) -> dict[str, object]:
        return {
            "symbol": self.symbol,
            "timestamp": _iso_utc(self.timestamp),
            "open": self.open,
            "high": self.high,
            "low": self.low,
            "close": self.close,
            "volume": self.volume,
            "adj_close": self.adj_close,
            "adj_open": self.adj_open,
            "adj_high": self.adj_high,
            "adj_low": self.adj_low,
        }


@dataclass(frozen=True, slots=True)
class ShadowDecisionEvaluation:
    status: ShadowDecisionStatus
    selection_source: str
    fallback_mode: str
    target_allocation: float | None
    reason: str | None = None
    input_payload: Mapping[str, object] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ShadowDecisionRequest:
    contract: VerifiedShadowContract
    as_of: datetime
    bars: tuple[ShadowBar, ...]


@dataclass(frozen=True, slots=True)
class ShadowDecisionContext:
    contract: VerifiedShadowContract
    as_of: datetime
    completed_bars: tuple[ShadowBar, ...]
    signal_date: date
    effective_date: date
    matured_label_cutoff: date


ShadowDecisionEvaluator = Callable[[ShadowDecisionContext], ShadowDecisionEvaluation]


@dataclass(frozen=True, slots=True)
class ShadowDecisionResult:
    path: Path
    record: dict[str, Any]
    created: bool


def shadow_bars_from_panel(panel: pd.DataFrame) -> tuple[ShadowBar, ...]:
    missing = [column for column in PANEL_COLUMNS if column not in panel.columns]
    if missing:
        joined = ", ".join(missing)
        raise ShadowDecisionError(f"Shadow panel is missing required columns: {joined}")

    bars: list[ShadowBar] = []
    for row in panel.sort_values(["timestamp", "symbol"]).itertuples(index=False):
        bars.append(
            ShadowBar(
                symbol=str(row.symbol),
                timestamp=_as_utc_datetime(row.timestamp, label="bar timestamp"),
                open=_finite_float(row.open, label="open"),
                high=_finite_float(row.high, label="high"),
                low=_finite_float(row.low, label="low"),
                close=_finite_float(row.close, label="close"),
                volume=_finite_float(row.volume, label="volume"),
                adj_close=_finite_float(row.adj_close, label="adj_close"),
                adj_open=_finite_float(row.adj_open, label="adj_open"),
                adj_high=_finite_float(row.adj_high, label="adj_high"),
                adj_low=_finite_float(row.adj_low, label="adj_low"),
            )
        )
    return tuple(bars)


def run_shadow_decision(
    request: ShadowDecisionRequest,
    *,
    evaluator: ShadowDecisionEvaluator,
    journal: ShadowDecisionJournal | None = None,
) -> ShadowDecisionResult:
    contract = verify_shadow_contract(request.contract.config_path)
    _require_same_contract(request.contract, contract)
    as_of = _as_utc_datetime(request.as_of, label="as_of")
    completed_bars = _completed_bars(request.bars, as_of=as_of)
    signal_date, effective_date, matured_label_cutoff = _decision_dates(
        completed_bars,
        as_of=as_of,
        contract=contract,
    )
    context = ShadowDecisionContext(
        contract=contract,
        as_of=as_of,
        completed_bars=completed_bars,
        signal_date=signal_date,
        effective_date=effective_date,
        matured_label_cutoff=matured_label_cutoff,
    )
    evaluation = _validated_evaluation(evaluator(context))
    input_fingerprint = _input_fingerprint(
        contract=contract,
        as_of=as_of,
        completed_bars=completed_bars,
        evaluation=evaluation,
    )
    record: dict[str, Any] = {
        "schema_version": 1,
        "candidate_id": contract.candidate_id,
        "behavior_version": contract.behavior_version,
        "config_hash": contract.config_hash,
        "behavior_hash": contract.behavior_hash,
        "code_lock": contract.code_lock,
        "decision_timestamp": _iso_utc(as_of),
        "signal_date": signal_date.isoformat(),
        "effective_date": effective_date.isoformat(),
        "matured_label_cutoff": matured_label_cutoff.isoformat(),
        "selection_source": evaluation.selection_source,
        "fallback_mode": evaluation.fallback_mode,
        "target_allocation": evaluation.target_allocation,
        "input_fingerprint": input_fingerprint,
        "status": evaluation.status,
        "reason": evaluation.reason,
    }
    selected_journal = journal or ShadowDecisionJournal(
        contract.artifact_root / "decisions"
    )
    write = selected_journal.write(record)
    return _result_from_write(write)


def _result_from_write(write: ShadowJournalWrite) -> ShadowDecisionResult:
    return ShadowDecisionResult(
        path=write.path,
        record=write.record,
        created=write.created,
    )


def _require_same_contract(
    supplied: VerifiedShadowContract,
    verified: VerifiedShadowContract,
) -> None:
    fields = (
        "config_path",
        "candidate_id",
        "behavior_version",
        "protocol_start",
        "protocol_end",
        "earliest_final_evaluation",
        "maturity_lag_bars",
        "code_lock",
        "artifact_root",
        "config_hash",
        "behavior_hash",
    )
    for field_name in fields:
        if getattr(supplied, field_name) != getattr(verified, field_name):
            raise ShadowDecisionError(
                f"Supplied shadow contract field {field_name} differs from re-verification."
            )


def _completed_bars(
    bars: tuple[ShadowBar, ...],
    *,
    as_of: datetime,
) -> tuple[ShadowBar, ...]:
    if not bars:
        raise ShadowDecisionError("Shadow decision requires market bars.")

    normalized: list[ShadowBar] = []
    seen: set[tuple[str, datetime]] = set()
    for bar in bars:
        timestamp = _as_utc_datetime(bar.timestamp, label="bar timestamp")
        if timestamp.hour != 0 or timestamp.minute != 0 or timestamp.second != 0:
            raise ShadowDecisionError("Daily shadow bars must use midnight UTC timestamps.")
        if timestamp > as_of:
            raise ShadowDecisionError("Shadow bars must not be timestamped after the as-of cutoff.")
        if bar.symbol != "BTC-USD":
            raise ShadowDecisionError("Shadow decisions only accept BTC-USD bars.")
        key = (bar.symbol, timestamp)
        if key in seen:
            raise ShadowDecisionError("Shadow bars must not contain duplicate timestamps.")
        seen.add(key)
        validated_bar = replace(
            bar,
            timestamp=timestamp,
            open=_finite_float(bar.open, label="open"),
            high=_finite_float(bar.high, label="high"),
            low=_finite_float(bar.low, label="low"),
            close=_finite_float(bar.close, label="close"),
            volume=_finite_float(bar.volume, label="volume"),
            adj_close=_finite_float(bar.adj_close, label="adj_close"),
            adj_open=_finite_float(bar.adj_open, label="adj_open"),
            adj_high=_finite_float(bar.adj_high, label="adj_high"),
            adj_low=_finite_float(bar.adj_low, label="adj_low"),
        )
        if timestamp + timedelta(days=1) <= as_of:
            normalized.append(validated_bar)

    completed = tuple(sorted(normalized, key=lambda bar: bar.timestamp))
    if not completed:
        raise ShadowDecisionError("No completed daily bar is available at the as-of cutoff.")
    for previous, current in zip(completed, completed[1:], strict=False):
        if current.timestamp - previous.timestamp != timedelta(days=1):
            raise ShadowDecisionError(
                "Shadow BTC daily bars must be continuous through the as-of cutoff."
            )
    return completed


def _decision_dates(
    completed_bars: tuple[ShadowBar, ...],
    *,
    as_of: datetime,
    contract: VerifiedShadowContract,
) -> tuple[date, date, date]:
    signal_date = completed_bars[-1].timestamp.date()
    effective_date = signal_date + timedelta(days=1)
    if as_of.date() != effective_date:
        raise ShadowDecisionError(
            "The latest completed bar must be the immediately preceding UTC day; "
            "missed dates cannot be backfilled."
        )
    if not contract.protocol_start <= effective_date <= contract.protocol_end:
        raise ShadowDecisionError(
            "Shadow decision effective date is outside the frozen protocol window."
        )
    cutoff_index = len(completed_bars) - 1 - contract.maturity_lag_bars
    if cutoff_index < 0:
        raise ShadowDecisionError(
            "Shadow decision does not have enough completed bars for the maturity cutoff."
        )
    matured_label_cutoff = completed_bars[cutoff_index].timestamp.date()
    return signal_date, effective_date, matured_label_cutoff


def _validated_evaluation(
    evaluation: ShadowDecisionEvaluation,
) -> ShadowDecisionEvaluation:
    if evaluation.status not in {"success", "skipped", "failed"}:
        raise ShadowDecisionError(f"Unsupported shadow decision status: {evaluation.status!r}.")
    if evaluation.selection_source not in _ALLOWED_SELECTION_SOURCES:
        raise ShadowDecisionError(
            f"Unsupported shadow selection source: {evaluation.selection_source!r}."
        )
    if evaluation.fallback_mode not in _ALLOWED_FALLBACK_MODES:
        raise ShadowDecisionError(
            f"Unsupported shadow fallback mode: {evaluation.fallback_mode!r}."
        )

    expected_fallback = {
        "strict": "none",
        "best_active_fallback": "best_active_fallback",
        "regime_policy_fallback": "regime_policy_fallback",
        "none": "none",
    }[evaluation.selection_source]
    if evaluation.fallback_mode != expected_fallback:
        raise ShadowDecisionError(
            "Shadow fallback_mode must match the recorded selection_source."
        )

    if evaluation.status == "success":
        if evaluation.selection_source == "none":
            raise ShadowDecisionError("Successful shadow decisions require a selection source.")
        if evaluation.target_allocation is None:
            raise ShadowDecisionError("Successful shadow decisions require target_allocation.")
        target_allocation = _finite_float(
            evaluation.target_allocation,
            label="target_allocation",
        )
        if target_allocation not in _ALLOWED_ALLOCATIONS:
            raise ShadowDecisionError(
                "Shadow target_allocation must be one of 0.0, 0.25, 0.5, or 1.0."
            )
    else:
        if evaluation.selection_source != "none":
            raise ShadowDecisionError(
                "Skipped or failed shadow decisions must use selection_source='none'."
            )
        if evaluation.target_allocation is not None:
            raise ShadowDecisionError(
                "Skipped or failed shadow decisions must not set target_allocation."
            )
        if evaluation.reason is None or evaluation.reason.strip() == "":
            raise ShadowDecisionError(
                "Skipped or failed shadow decisions require an explicit reason."
            )

    try:
        canonical_fingerprint(dict(evaluation.input_payload))
    except (TypeError, ValueError, ShadowJournalError) as exc:
        raise ShadowDecisionError(
            "Shadow decision input_payload must be a deterministic JSON mapping."
        ) from exc
    return evaluation


def _input_fingerprint(
    *,
    contract: VerifiedShadowContract,
    as_of: datetime,
    completed_bars: tuple[ShadowBar, ...],
    evaluation: ShadowDecisionEvaluation,
) -> str:
    fingerprint = canonical_fingerprint(
        {
            "candidate_id": contract.candidate_id,
            "behavior_version": contract.behavior_version,
            "config_hash": contract.config_hash,
            "behavior_hash": contract.behavior_hash,
            "code_lock": contract.code_lock,
            "as_of": _iso_utc(as_of),
            "completed_bars": [
                bar.as_fingerprint_payload() for bar in completed_bars
            ],
            "evaluation_input": dict(evaluation.input_payload),
        }
    )
    if _SHA256_PATTERN.fullmatch(fingerprint) is None:
        raise ShadowDecisionError("Shadow input fingerprint must be SHA-256.")
    return fingerprint


def _as_utc_datetime(value: object, *, label: str) -> datetime:
    try:
        timestamp = pd.Timestamp(value)
    except (TypeError, ValueError) as exc:
        raise ShadowDecisionError(f"Shadow {label} must be a datetime.") from exc
    if timestamp.tzinfo is None:
        if label == "as_of":
            raise ShadowDecisionError("Shadow as_of must include an explicit timezone.")
        timestamp = timestamp.tz_localize(UTC)
    else:
        timestamp = timestamp.tz_convert(UTC)
    return timestamp.to_pydatetime()


def _finite_float(value: object, *, label: str) -> float:
    try:
        resolved = float(str(value))
    except (TypeError, ValueError) as exc:
        raise ShadowDecisionError(f"Shadow {label} must be numeric.") from exc
    if not math.isfinite(resolved):
        raise ShadowDecisionError(f"Shadow {label} must be finite.")
    return resolved


def _iso_utc(value: datetime) -> str:
    return value.astimezone(UTC).isoformat().replace("+00:00", "Z")
