from __future__ import annotations

import math
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import pandas as pd
import pytest

from marketlab.shadow import (
    ShadowAttemptStore,
    ShadowDecisionEvaluation,
    ShadowDecisionEvidenceStore,
    ShadowDecisionJournal,
    ShadowLabelEvidenceStore,
    cli,
    run_shadow_scheduler,
)

ROOT = Path(__file__).resolve().parents[2]
SHADOW_CONFIG = ROOT / "configs" / "experiment.btc_phase9_shadow_daily.yaml"


def _panel(end: date, count: int = 500) -> pd.DataFrame:
    start = end - timedelta(days=count - 1)
    rows = []
    for index in range(count):
        close = 100.0 + (index * 0.03) + (4.0 * math.sin(index / 8.0))
        rows.append(
            {
                "symbol": "BTC-USD",
                "timestamp": pd.Timestamp(start + timedelta(days=index)),
                "open": close - 0.2,
                "high": close + 0.8,
                "low": close - 0.8,
                "close": close,
                "volume": 1_000.0 + (30.0 * math.cos(index / 6.0)),
                "adj_close": close,
                "adj_factor": 1.0,
                "adj_open": close - 0.2,
                "adj_high": close + 0.8,
                "adj_low": close - 0.8,
            }
        )
    return pd.DataFrame(rows)


def _evaluation(context) -> ShadowDecisionEvaluation:
    diagnostics = {
        "raw_score": 0.61,
        "selected_tier": 0.50,
        "selection_source": "strict",
        "fallback_mode": "none",
        "regime_classification": "bull",
        "input_cutoff": context.signal_date.isoformat(),
        "pipeline": {"prediction": {"gate_bull": True}},
    }
    from marketlab.shadow import canonical_fingerprint

    diagnostics["diagnostic_fingerprint"] = canonical_fingerprint(diagnostics)
    return ShadowDecisionEvaluation(
        status="success",
        selection_source="strict",
        fallback_mode="none",
        target_allocation=0.50,
        input_payload=diagnostics,
    )


def _stores(tmp_path: Path):
    return {
        "journal": ShadowDecisionJournal(tmp_path / "decisions"),
        "attempt_store": ShadowAttemptStore(tmp_path / "attempts"),
        "decision_evidence_store": ShadowDecisionEvidenceStore(
            tmp_path / "evidence" / "decisions"
        ),
        "label_evidence_store": ShadowLabelEvidenceStore(
            tmp_path / "evidence" / "labels"
        ),
    }


def test_scheduler_records_missed_dates_without_backfill(tmp_path: Path) -> None:
    stores = _stores(tmp_path)
    runtime = datetime(2026, 6, 5, 1, 15, tzinfo=UTC)

    result = run_shadow_scheduler(
        SHADOW_CONFIG,
        as_of=runtime,
        evaluator=_evaluation,
        panel_refresher=lambda contract: _panel(date(2026, 6, 4)),
        execution_id="execution-1",
        **stores,
    )

    outcomes = [write.record["outcome"] for write in result.attempts]
    assert outcomes == ["missed", "missed", "success"]
    assert stores["journal"].read(date(2026, 6, 3)) is None
    assert stores["journal"].read(date(2026, 6, 4)) is None
    assert stores["journal"].read(date(2026, 6, 5)) is not None


def test_scheduler_writes_linked_decision_evidence(tmp_path: Path) -> None:
    stores = _stores(tmp_path)
    runtime = datetime(2026, 6, 3, 1, 15, tzinfo=UTC)

    result = run_shadow_scheduler(
        SHADOW_CONFIG,
        as_of=runtime,
        evaluator=_evaluation,
        panel_refresher=lambda contract: _panel(date(2026, 6, 2)),
        execution_id="execution-1",
        **stores,
    )

    assert result.decision is not None
    assert result.decision_evidence is not None
    assert (
        result.decision_evidence.record["decision_fingerprint"]
        == result.decision.record["output_fingerprint"]
    )
    assert result.decision_evidence.record["raw_score"] == 0.61
    assert result.decision_evidence.record["gate_bull"] is True


def test_repeated_current_date_runs_add_attempts_but_reuse_decision(
    tmp_path: Path,
) -> None:
    stores = _stores(tmp_path)
    runtime = datetime(2026, 6, 3, 1, 15, tzinfo=UTC)
    kwargs = {
        "as_of": runtime,
        "evaluator": _evaluation,
        "panel_refresher": lambda contract: _panel(date(2026, 6, 2)),
        **stores,
    }

    first = run_shadow_scheduler(
        SHADOW_CONFIG,
        execution_id="execution-1",
        **kwargs,
    )
    second = run_shadow_scheduler(
        SHADOW_CONFIG,
        execution_id="execution-2",
        **kwargs,
    )

    assert first.decision is not None and first.decision.created is True
    assert second.decision is not None and second.decision.created is False
    assert len(stores["attempt_store"].list_for(date(2026, 6, 3))) == 2


def test_failed_scheduler_attempt_sanitizes_error_reason(tmp_path: Path) -> None:
    stores = _stores(tmp_path)

    def _fail(context):
        raise RuntimeError("token=abc123 secret: hidden")

    with pytest.raises(RuntimeError, match="token"):
        run_shadow_scheduler(
            SHADOW_CONFIG,
            as_of=datetime(2026, 6, 3, 1, 15, tzinfo=UTC),
            evaluator=_fail,
            panel_refresher=lambda contract: _panel(date(2026, 6, 2)),
            execution_id="execution-1",
            **stores,
        )

    attempt = stores["attempt_store"].read(date(2026, 6, 3), "execution-1")
    assert attempt is not None
    assert attempt["outcome"] == "failed"
    assert attempt["error_type"] == "RuntimeError"
    assert "abc123" not in attempt["reason"]
    assert "hidden" not in attempt["reason"]


def test_scheduler_records_panel_refresh_failure_as_failed_attempt(
    tmp_path: Path,
) -> None:
    stores = _stores(tmp_path)

    def _refresh_failure(contract):
        raise OSError("provider token=abc123 unavailable")

    with pytest.raises(OSError, match="provider"):
        run_shadow_scheduler(
            SHADOW_CONFIG,
            as_of=datetime(2026, 6, 3, 1, 15, tzinfo=UTC),
            evaluator=_evaluation,
            panel_refresher=_refresh_failure,
            execution_id="execution-1",
            **stores,
        )

    attempt = stores["attempt_store"].read(date(2026, 6, 3), "execution-1")
    assert attempt is not None
    assert attempt["outcome"] == "failed"
    assert attempt["error_type"] == "OSError"
    assert "abc123" not in attempt["reason"]


def test_scheduler_records_failed_attempt_when_decision_evidence_write_fails(
    tmp_path: Path,
) -> None:
    stores = _stores(tmp_path)

    class FailingDecisionEvidenceStore(ShadowDecisionEvidenceStore):
        def write(self, record):
            raise RuntimeError("evidence conflict token=abc123")

    stores["decision_evidence_store"] = FailingDecisionEvidenceStore(
        tmp_path / "evidence" / "decisions"
    )

    with pytest.raises(RuntimeError, match="evidence conflict"):
        run_shadow_scheduler(
            SHADOW_CONFIG,
            as_of=datetime(2026, 6, 3, 1, 15, tzinfo=UTC),
            evaluator=_evaluation,
            panel_refresher=lambda contract: _panel(date(2026, 6, 2)),
            execution_id="execution-1",
            **stores,
        )

    attempts = stores["attempt_store"].list_for(date(2026, 6, 3))
    assert [attempt["outcome"] for attempt in attempts] == ["failed"]
    assert attempts[0]["error_type"] == "RuntimeError"
    assert "abc123" not in attempts[0]["reason"]
    decision = stores["journal"].read(date(2026, 6, 3))
    assert decision is not None
    assert attempts[0]["decision_fingerprint"] == decision["output_fingerprint"]


def test_scheduler_materializes_label_after_fourteen_completed_bars(
    tmp_path: Path,
) -> None:
    stores = _stores(tmp_path)
    first_runtime = datetime(2026, 6, 3, 1, 15, tzinfo=UTC)
    run_shadow_scheduler(
        SHADOW_CONFIG,
        as_of=first_runtime,
        evaluator=_evaluation,
        panel_refresher=lambda contract: _panel(date(2026, 6, 2)),
        execution_id="execution-1",
        **stores,
    )

    before_maturity = run_shadow_scheduler(
        SHADOW_CONFIG,
        as_of=datetime(2026, 6, 16, 1, 15, tzinfo=UTC),
        evaluator=_evaluation,
        panel_refresher=lambda contract: _panel(date(2026, 6, 15)),
        execution_id="execution-before-maturity",
        **stores,
    )
    assert before_maturity.label_evidence == ()
    assert stores["label_evidence_store"].read(date(2026, 6, 3)) is None

    result = run_shadow_scheduler(
        SHADOW_CONFIG,
        as_of=datetime(2026, 6, 17, 1, 15, tzinfo=UTC),
        evaluator=_evaluation,
        panel_refresher=lambda contract: _panel(date(2026, 6, 16)),
        execution_id="execution-2",
        **stores,
    )

    label = stores["label_evidence_store"].read(date(2026, 6, 3))
    assert label is not None
    assert label["target_end_date"] == "2026-06-16"
    assert len(label["path_adj_closes"]) == 14
    intraday_return = (
        label["effective_adj_close"] / label["entry_adj_open"]
    ) - 1.0
    expected_gross_return = 0.50 * intraday_return
    assert label["turnover"] == pytest.approx(0.50)
    assert label["daily_gross_return"] == pytest.approx(expected_gross_return)
    assert label["strategy_return"] == pytest.approx(
        expected_gross_return - 0.50 * 35.0 / 10_000.0
    )
    assert label["decision_fingerprint"] == stores["journal"].read(
        date(2026, 6, 3)
    )["output_fingerprint"]
    assert any(write.record["effective_date"] == "2026-06-03" for write in result.label_evidence)


def test_scheduler_cli_requires_one_shot_mode() -> None:
    with pytest.raises(SystemExit):
        cli.scheduler_main(["--config", str(SHADOW_CONFIG)])
