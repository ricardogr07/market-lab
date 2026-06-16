from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pytest

from marketlab.reports.phase8_gates import (
    calculate_bull_participation_gate,
    calculate_signal_validity_gate,
)
from marketlab.shadow import (
    ShadowAttemptStore,
    ShadowDecisionEvidenceStore,
    ShadowDecisionJournal,
    ShadowLabelEvidenceStore,
    ShadowReportError,
    build_shadow_status,
    verify_shadow_contract,
    write_final_shadow_report,
    write_monthly_shadow_report,
)

ROOT = Path(__file__).resolve().parents[2]
SHADOW_CONFIG = ROOT / "configs" / "experiment.btc_phase9_shadow_daily.yaml"


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


def _contract_fields() -> dict[str, object]:
    contract = verify_shadow_contract(SHADOW_CONFIG)
    return {
        "candidate_id": contract.candidate_id,
        "behavior_version": contract.behavior_version,
        "config_hash": contract.config_hash,
        "behavior_hash": contract.behavior_hash,
        "code_lock": contract.code_lock,
    }


def _populate(tmp_path: Path):
    stores = _stores(tmp_path)
    contract_fields = _contract_fields()
    fixtures = [
        {
            "effective_date": "2026-06-03",
            "score": 0.20,
            "tier": 0.25,
            "benchmark_return": 0.0,
            "strategy_return": -0.000875,
            "utility": -0.10,
            "target": 0.25,
            "turnover": 0.25,
            "gate_bull": False,
        },
        {
            "effective_date": "2026-06-04",
            "score": 0.80,
            "tier": 1.0,
            "benchmark_return": 0.10,
            "strategy_return": 0.097375,
            "utility": 0.20,
            "target": 1.0,
            "turnover": 0.75,
            "gate_bull": True,
        },
    ]
    for index, fixture in enumerate(fixtures, start=1):
        effective_date = fixture["effective_date"]
        decision = stores["journal"].write(
            {
                "schema_version": 1,
                "effective_date": effective_date,
                "status": "success",
                "selection_source": "strict",
                "fallback_mode": "none",
                "target_allocation": fixture["tier"],
                **contract_fields,
            }
        ).record
        stores["attempt_store"].write(
            {
                "schema_version": 1,
                "attempt_id": f"execution-{index}",
                "execution_id": f"execution-{index}",
                "effective_date": effective_date,
                "scheduled_date": effective_date,
                "outcome": "success",
                **contract_fields,
            }
        )
        evidence = stores["decision_evidence_store"].write(
            {
                "schema_version": 1,
                "effective_date": effective_date,
                "decision_fingerprint": decision["output_fingerprint"],
                "raw_score": fixture["score"],
                "selected_tier": fixture["tier"],
                "selection_source": "strict",
                "fallback_mode": "none",
                "regime_classification": "bull" if fixture["gate_bull"] else "bear",
                "gate_bull": fixture["gate_bull"],
                "diagnostic_fingerprint": f"{index}" * 64,
                **contract_fields,
            }
        ).record
        stores["label_evidence_store"].write(
            {
                "schema_version": 1,
                "effective_date": effective_date,
                "decision_fingerprint": decision["output_fingerprint"],
                "decision_evidence_fingerprint": evidence["output_fingerprint"],
                "diagnostic_fingerprint": evidence["diagnostic_fingerprint"],
                "benchmark_return": fixture["benchmark_return"],
                "daily_benchmark_return": fixture["benchmark_return"],
                "overnight_asset_return": 0.0,
                "intraday_asset_return": fixture["benchmark_return"],
                "daily_gross_return": (
                    fixture["tier"] * fixture["benchmark_return"]
                ),
                "strategy_return": fixture["strategy_return"],
                "realized_utility": fixture["utility"],
                "realized_target_weight": fixture["target"],
                "exposure": fixture["tier"],
                "turnover": fixture["turnover"],
                **contract_fields,
            }
        )
    return stores


def test_phase8_pure_gate_calculators_preserve_complete_conditions() -> None:
    signal = calculate_signal_validity_gate(
        {
            "score_target_weight_correlation": 0.1,
            "score_forward_return_correlation": 0.2,
            "score_realized_utility_correlation": 0.3,
            "predicted_tier_100_fraction": 0.25,
            "any_selected_oos_predicted_tier_100": True,
        }
    )
    bull = calculate_bull_participation_gate(
        {
            "gate_bull_average_long_exposure": 0.50,
            "gate_bull_active_return_sum": 0.01,
            "gate_bull_underexposed_positive_benchmark_return_sum": 0.0,
            "selected_fold_fraction": 0.75,
        }
    )

    assert signal.passed is True
    assert bull.passed is True


def test_status_classifies_successful_dates_from_canonical_records(
    tmp_path: Path,
) -> None:
    stores = _populate(tmp_path)

    status = build_shadow_status(
        SHADOW_CONFIG,
        as_of=date(2026, 6, 4),
        **stores,
    )

    assert status["counts"] == {"successful": 2}
    assert status["integrity_errors"] == []


def test_status_marks_success_without_mature_label_as_label_pending(
    tmp_path: Path,
) -> None:
    stores = _populate(tmp_path)
    stores["label_evidence_store"].path_for(date(2026, 6, 4)).unlink()

    status = build_shadow_status(
        SHADOW_CONFIG,
        as_of=date(2026, 6, 4),
        **stores,
    )

    assert status["counts"] == {"label-pending": 1, "successful": 1}


def test_monthly_report_calculates_exact_cost_scenarios_and_is_deterministic(
    tmp_path: Path,
) -> None:
    stores = _populate(tmp_path)
    output_root = tmp_path / "reports" / "monthly"

    first_json, first_md, report = write_monthly_shadow_report(
        SHADOW_CONFIG,
        as_of=date(2026, 6, 4),
        output_root=output_root,
        **stores,
    )
    original_json = first_json.read_bytes()
    original_markdown = first_md.read_bytes()
    second_json, second_md, second_report = write_monthly_shadow_report(
        SHADOW_CONFIG,
        as_of=date(2026, 6, 4),
        output_root=output_root,
        **stores,
    )

    strategy_35 = (1.0 + (0.25 * 0.0 - 0.25 * 35 / 10_000)) * (
        1.0 + (1.0 * 0.10 - 0.75 * 35 / 10_000)
    ) - 1.0
    benchmark = (1.0 + 0.0) * (1.0 + 0.10) - 1.0
    assert report["active_returns"]["35_bps"]["active_return"] == pytest.approx(
        strategy_35 - benchmark
    )
    assert report["active_returns"]["50_bps"]["active_return"] != report[
        "active_returns"
    ]["35_bps"]["active_return"]
    assert report["provisional"] is True
    assert report["promotion_decision"] is None
    assert report["graduation_checks"]["zero_best_active_fallback"] is True
    assert report["graduation_checks"]["zero_regime_policy_fallback"] is True
    assert second_report == report
    assert second_json.read_bytes() == original_json
    assert second_md.read_bytes() == original_markdown


def test_final_report_rejects_early_generation(tmp_path: Path) -> None:
    stores = _populate(tmp_path)

    with pytest.raises(ShadowReportError, match="2027-06-16"):
        write_final_shadow_report(
            SHADOW_CONFIG,
            as_of=date(2027, 6, 15),
            output_root=tmp_path / "reports" / "final",
            **stores,
        )


def test_reports_fail_closed_on_tampered_evidence(tmp_path: Path) -> None:
    stores = _populate(tmp_path)
    path = stores["label_evidence_store"].path_for(date(2026, 6, 3))
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["benchmark_return"] = 0.99
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ShadowReportError, match="fingerprint"):
        write_monthly_shadow_report(
            SHADOW_CONFIG,
            as_of=date(2026, 6, 4),
            output_root=tmp_path / "reports" / "monthly",
            **stores,
        )


def test_reports_fail_closed_on_tampered_attempt(tmp_path: Path) -> None:
    stores = _populate(tmp_path)
    path = stores["attempt_store"].path_for(date(2026, 6, 3), "execution-1")
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["outcome"] = "failed"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ShadowReportError, match="fingerprint"):
        write_monthly_shadow_report(
            SHADOW_CONFIG,
            as_of=date(2026, 6, 4),
            output_root=tmp_path / "reports" / "monthly",
            **stores,
        )


def test_report_integrity_requires_decision_evidence_for_successful_decisions(
    tmp_path: Path,
) -> None:
    stores = _populate(tmp_path)
    stores["decision_evidence_store"].path_for(date(2026, 6, 4)).unlink()
    stores["label_evidence_store"].path_for(date(2026, 6, 4)).unlink()

    _, _, report = write_monthly_shadow_report(
        SHADOW_CONFIG,
        as_of=date(2026, 6, 4),
        output_root=tmp_path / "reports" / "monthly",
        **stores,
    )

    assert report["integrity"]["passed"] is False
    assert (
        "missing_decision_evidence:2026-06-04"
        in report["integrity"]["cross_link_errors"]
    )
    assert report["graduation_checks"]["fingerprint_integrity"] is False


def test_final_report_remains_informational(tmp_path: Path) -> None:
    stores = _populate(tmp_path)

    _, markdown_path, report = write_final_shadow_report(
        SHADOW_CONFIG,
        as_of=date(2027, 6, 16),
        output_root=tmp_path / "reports" / "final",
        **stores,
    )

    assert report["informational_only"] is True
    assert report["promotion_decision"] is None
    markdown = markdown_path.read_text(encoding="utf-8")
    assert "cannot enable paper execution" in markdown
    assert "call a broker" in markdown
