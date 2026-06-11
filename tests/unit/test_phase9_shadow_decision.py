from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import cast

import pytest

from marketlab.shadow import (
    ShadowBar,
    ShadowContractError,
    ShadowDecisionError,
    ShadowDecisionEvaluation,
    ShadowDecisionJournal,
    ShadowDecisionRequest,
    ShadowDecisionStatus,
    ShadowJournalConflictError,
    ShadowJournalError,
    canonical_fingerprint,
    normalize_record_fingerprint,
    run_shadow_decision,
    verify_shadow_contract,
)

ROOT = Path(__file__).resolve().parents[2]
SHADOW_CONFIG = ROOT / "configs" / "experiment.btc_phase9_shadow_daily.yaml"
AS_OF = datetime(2026, 6, 11, 1, 15, tzinfo=UTC)


def _bars(
    *,
    symbol: str = "BTC-USD",
    end: datetime = datetime(2026, 6, 11, tzinfo=UTC),
    count: int = 72,
) -> tuple[ShadowBar, ...]:
    start = end - timedelta(days=count - 1)
    return tuple(
        ShadowBar(
            symbol=symbol,
            timestamp=start + timedelta(days=index),
            open=100.0 + index,
            high=101.0 + index,
            low=99.0 + index,
            close=100.5 + index,
            volume=1_000.0 + index,
            adj_close=100.5 + index,
            adj_open=100.0 + index,
            adj_high=101.0 + index,
            adj_low=99.0 + index,
        )
        for index in range(count)
    )


def _evaluation(
    *,
    status: str = "success",
    selection_source: str = "strict",
    fallback_mode: str = "none",
    target_allocation: float | None = 0.50,
    reason: str | None = None,
) -> ShadowDecisionEvaluation:
    return ShadowDecisionEvaluation(
        status=cast(ShadowDecisionStatus, status),
        selection_source=selection_source,
        fallback_mode=fallback_mode,
        target_allocation=target_allocation,
        reason=reason,
        input_payload={
            "selected_model": "hist_gradient_boosting",
            "score": 0.61,
            "validation": {"passed": True, "rows": 180},
        },
    )


def _request(
    *,
    bars: tuple[ShadowBar, ...] | None = None,
    as_of: datetime = AS_OF,
) -> ShadowDecisionRequest:
    return ShadowDecisionRequest(
        contract=verify_shadow_contract(SHADOW_CONFIG),
        as_of=as_of,
        bars=bars or _bars(),
    )


def _run(
    request: ShadowDecisionRequest,
    *,
    journal: ShadowDecisionJournal,
    evaluation: ShadowDecisionEvaluation | None = None,
):
    return run_shadow_decision(
        request,
        evaluator=lambda context: evaluation or _evaluation(),
        journal=journal,
    )


def test_shadow_decision_writes_stable_success_record(tmp_path: Path) -> None:
    journal = ShadowDecisionJournal(tmp_path / "decisions")

    result = _run(_request(), journal=journal)

    assert result.created is True
    assert result.path == tmp_path / "decisions" / "2026-06-11.json"
    assert result.record == journal.read(datetime(2026, 6, 11).date())
    assert result.record["candidate_id"] == "btc-phase9-shadow-v1"
    assert result.record["behavior_version"] == "btc-phase8-guarded-gate-v1"
    assert result.record["signal_date"] == "2026-06-10"
    assert result.record["effective_date"] == "2026-06-11"
    assert result.record["matured_label_cutoff"] == "2026-05-27"
    assert result.record["selection_source"] == "strict"
    assert result.record["fallback_mode"] == "none"
    assert result.record["target_allocation"] == 0.50
    assert result.record["status"] == "success"
    assert len(result.record["input_fingerprint"]) == 64
    assert result.record == normalize_record_fingerprint(result.record)


def test_identical_shadow_decision_rerun_is_idempotent(tmp_path: Path) -> None:
    journal = ShadowDecisionJournal(tmp_path / "decisions")
    request = _request()
    first = _run(request, journal=journal)
    original_bytes = first.path.read_bytes()

    second = _run(request, journal=journal)

    assert second.created is False
    assert second.path == first.path
    assert second.record == first.record
    assert second.path.read_bytes() == original_bytes


def test_conflicting_shadow_decision_preserves_original(tmp_path: Path) -> None:
    journal = ShadowDecisionJournal(tmp_path / "decisions")
    original = _run(_request(), journal=journal)
    original_bytes = original.path.read_bytes()

    with pytest.raises(ShadowJournalConflictError, match="original journal record"):
        _run(
            _request(),
            journal=journal,
            evaluation=_evaluation(target_allocation=1.0),
        )

    assert original.path.read_bytes() == original_bytes
    assert journal.read(datetime(2026, 6, 11).date()) == original.record


def test_shadow_decision_records_explicit_skip(tmp_path: Path) -> None:
    journal = ShadowDecisionJournal(tmp_path / "decisions")

    result = _run(
        _request(),
        journal=journal,
        evaluation=_evaluation(
            status="skipped",
            selection_source="none",
            fallback_mode="none",
            target_allocation=None,
            reason="no_valid_candidate",
        ),
    )

    assert result.record["status"] == "skipped"
    assert result.record["reason"] == "no_valid_candidate"
    assert result.record["target_allocation"] is None


def test_evaluator_receives_label_safe_decision_context(tmp_path: Path) -> None:
    journal = ShadowDecisionJournal(tmp_path / "decisions")
    captured = {}

    def _evaluate(context):
        captured["context"] = context
        return _evaluation()

    run_shadow_decision(_request(), evaluator=_evaluate, journal=journal)

    context = captured["context"]
    assert context.signal_date.isoformat() == "2026-06-10"
    assert context.effective_date.isoformat() == "2026-06-11"
    assert context.matured_label_cutoff.isoformat() == "2026-05-27"
    assert context.completed_bars[-1].timestamp.date() == context.signal_date
    assert all(
        bar.timestamp.date() <= context.signal_date for bar in context.completed_bars
    )


def test_shadow_journal_lists_records_by_effective_date(tmp_path: Path) -> None:
    journal = ShadowDecisionJournal(tmp_path / "decisions")
    first = normalize_record_fingerprint(
        {"effective_date": "2026-06-11", "status": "skipped"}
    )
    second = normalize_record_fingerprint(
        {"effective_date": "2026-06-12", "status": "failed"}
    )

    journal.write(second)
    journal.write(first)

    assert journal.list() == [first, second]


def test_record_fingerprint_is_independent_of_mapping_order() -> None:
    left = {"effective_date": "2026-06-11", "status": "success", "nested": {"b": 2, "a": 1}}
    right = {"nested": {"a": 1, "b": 2}, "status": "success", "effective_date": "2026-06-11"}

    assert canonical_fingerprint(left) == canonical_fingerprint(right)
    assert normalize_record_fingerprint(left)["output_fingerprint"] == (
        normalize_record_fingerprint(right)["output_fingerprint"]
    )


def test_shadow_journal_rejects_modified_record(tmp_path: Path) -> None:
    journal = ShadowDecisionJournal(tmp_path / "decisions")
    write = journal.write({"effective_date": "2026-06-11", "status": "success"})
    write.path.write_text(
        write.path.read_text(encoding="utf-8").replace('"status": "success"', '"status": "failed"'),
        encoding="utf-8",
    )

    with pytest.raises(ShadowJournalError, match="fingerprint"):
        journal.read(datetime(2026, 6, 11).date())


def test_shadow_journal_rejects_effective_date_path_mismatch(tmp_path: Path) -> None:
    journal = ShadowDecisionJournal(tmp_path / "decisions")
    write = journal.write({"effective_date": "2026-06-11", "status": "success"})
    mismatched_path = write.path.with_name("2026-06-12.json")
    write.path.rename(mismatched_path)

    with pytest.raises(ShadowJournalError, match="does not match its path"):
        journal.list()


@pytest.mark.parametrize(
    ("bars", "as_of", "message"),
    [
        (_bars(symbol="ETH-USD"), AS_OF, "only accept BTC-USD"),
        (_bars(end=datetime(2026, 6, 9, tzinfo=UTC)), AS_OF, "cannot be backfilled"),
        (
            _bars(end=datetime(2026, 6, 12, tzinfo=UTC)),
            AS_OF,
            "after the as-of cutoff",
        ),
        (_bars(count=10), AS_OF, "enough completed bars"),
        (_bars(), datetime(2026, 6, 11, 1, 15), "explicit timezone"),
    ],
)
def test_shadow_decision_rejects_timing_and_market_drift_before_write(
    tmp_path: Path,
    bars: tuple[ShadowBar, ...],
    as_of: datetime,
    message: str,
) -> None:
    journal = ShadowDecisionJournal(tmp_path / "decisions")

    with pytest.raises(ShadowDecisionError, match=message):
        _run(
            _request(bars=bars, as_of=as_of),
            journal=journal,
        )

    assert journal.list() == []


def test_shadow_decision_rejects_missing_daily_bar_before_write(tmp_path: Path) -> None:
    journal = ShadowDecisionJournal(tmp_path / "decisions")
    bars = _bars()
    bars_with_gap = bars[:20] + bars[21:]

    with pytest.raises(ShadowDecisionError, match="must be continuous"):
        _run(_request(bars=bars_with_gap), journal=journal)

    assert journal.list() == []


def test_shadow_decision_reverifies_contract_before_write(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    journal = ShadowDecisionJournal(tmp_path / "decisions")

    def _reject(path: str | Path) -> None:
        raise ShadowContractError(f"drift: {path}")

    monkeypatch.setattr("marketlab.shadow.decision.verify_shadow_contract", _reject)

    with pytest.raises(ShadowContractError, match="drift"):
        _run(_request(), journal=journal)

    assert journal.list() == []


def test_shadow_decision_rejects_supplied_contract_mismatch(tmp_path: Path) -> None:
    journal = ShadowDecisionJournal(tmp_path / "decisions")
    request = _request()
    changed_contract = replace(request.contract, behavior_hash="0" * 64)

    with pytest.raises(ShadowDecisionError, match="behavior_hash"):
        _run(
            replace(request, contract=changed_contract),
            journal=journal,
        )

    assert journal.list() == []


@pytest.mark.parametrize(
    "evaluation",
    [
        _evaluation(selection_source="strict", fallback_mode="regime_policy_fallback"),
        _evaluation(target_allocation=0.75),
        _evaluation(
            status="failed",
            selection_source="none",
            fallback_mode="none",
            target_allocation=None,
            reason=None,
        ),
    ],
)
def test_shadow_decision_rejects_invalid_evaluation_before_write(
    tmp_path: Path,
    evaluation: ShadowDecisionEvaluation,
) -> None:
    journal = ShadowDecisionJournal(tmp_path / "decisions")

    with pytest.raises(ShadowDecisionError):
        _run(_request(), journal=journal, evaluation=evaluation)

    assert journal.list() == []
