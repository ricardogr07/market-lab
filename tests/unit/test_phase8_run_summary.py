from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from marketlab.cli import main
from marketlab.reports.phase8_summary import (
    build_phase8_run_summary,
    write_phase8_run_summary,
)


def _write_phase8_artifacts(run_dir: Path) -> None:
    run_dir.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "condition": "overall",
                "passed": False,
                "observed": "False",
                "required": "all rows pass",
            },
            {
                "condition": "selected_walk_forward_fold_fraction",
                "passed": False,
                "observed": "0.50",
                "required": ">= 0.75",
            },
        ]
    ).to_csv(run_dir / "strict_research_gate.csv", index=False)
    pd.DataFrame(
        {
            "fold_id": [1, 2, 3, 4],
            "selection_status": [
                "selected",
                "selected",
                "no_valid_candidate",
                "no_valid_candidate",
            ],
            "selected_allocation_score_transform": [
                "bull_shift_18",
                "identity",
                pd.NA,
                pd.NA,
            ],
            "selected_regime_gate_bull_floor": [1.0, 1.0, pd.NA, pd.NA],
            "validation_score_policy_repair_authorized": [True, False, False, False],
            "validation_guarded_gate_bull_risk_off_override_authorized": [
                True,
                False,
                False,
                False,
            ],
        }
    ).to_csv(run_dir / "ml_strategy_tuning_selections.csv", index=False)
    pd.DataFrame(
        {
            "fold_id": [1, 1, 2],
            "passed_gate": [True, False, False],
            "failure_reasons": [
                "",
                "inactive_candidate;non_positive_required_benchmark_excess",
                "insufficient_predicted_tier_support;negative_score_forward_return_correlation",
            ],
            "validation_score_forward_return_correlation": [0.20, -0.10, -0.20],
            "validation_raw_score_forward_return_correlation": [0.10, -0.20, -0.30],
            "validation_score_policy_repair_authorized": [True, False, False],
            "validation_guarded_gate_bull_risk_off_override_authorized": [
                True,
                False,
                False,
            ],
            "guarded_gate_bull_risk_off_override_denied_reason": [
                "",
                "negative_validation_raw_score_forward_return_correlation",
                "negative_validation_raw_score_forward_return_correlation",
            ],
            "min_selection_validation_cost_benchmark_excess_cumulative_return": [
                0.02,
                -0.03,
                -0.05,
            ],
            "score_policy_repair_denied_reason": [
                "",
                "negative_validation_raw_score_forward_return_correlation",
                "negative_validation_raw_score_forward_return_correlation",
            ],
            "validation_score_target_correlation": [0.30, 0.10, 0.05],
            "validation_gate_bull_average_exposure": [1.0, 0.25, 0.50],
            "validation_gate_bull_underexposed_positive_benchmark_fraction": [
                0.0,
                1.0,
                0.5,
            ],
            "validation_gate_bull_underexposed_positive_benchmark_return_sum": [
                0.0,
                0.03,
                0.02,
            ],
        }
    ).to_csv(run_dir / "ml_strategy_tuning_candidates.csv", index=False)
    pd.DataFrame(
        {
            "scope": ["global", "global", "global"],
            "target_weight": [0.0, 0.25, 0.50],
            "row_fraction": [0.60, 0.20, 0.20],
        }
    ).to_csv(run_dir / "allocation_target_diagnostics.csv", index=False)
    pd.DataFrame(
        {
            "predicted_tier_weight": [0.0, 0.25, 0.25, 0.50, 1.0],
            "score_policy_repair_authorized": [True, True, True, True, True],
            "score_policy_triggered_100": [False, False, False, False, True],
            "guarded_gate_bull_risk_off_override_authorized": [
                True,
                True,
                True,
                True,
                True,
            ],
            "guarded_gate_bull_risk_off_override_triggered": [
                False,
                True,
                False,
                False,
                False,
            ],
        }
    ).to_csv(run_dir / "allocation_probability_diagnostics.csv", index=False)
    pd.DataFrame(
        {
            "strategy": [
                "buy_hold",
                "btc_rebalanced_25",
                "ml_indicator_tuned__long_only__cash",
            ],
            "cumulative_return": [0.10, 0.04, 0.12],
        }
    ).to_csv(run_dir / "strategy_summary.csv", index=False)
    pd.DataFrame(
        {
            "slice_name": ["bull", "bear"],
            "active_return": [0.05, -0.02],
        }
    ).to_csv(run_dir / "regime_slice_diagnostics.csv", index=False)


def test_build_phase8_run_summary_reads_persisted_run_artifacts(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    _write_phase8_artifacts(run_dir)

    summary = build_phase8_run_summary(run_dir)
    by_metric = summary.set_index("metric")

    assert bool(by_metric.loc["strict_gate_overall", "value"]) is False
    assert by_metric.loc["selected_fold_fraction", "value"] == 0.5
    assert by_metric.loc["no_valid_candidate_folds", "value"] == 2
    assert (
        by_metric.loc[
            "candidate_rejection_reason_non_positive_required_benchmark_excess",
            "value",
        ]
        == 1
    )
    assert by_metric.loc["target_tier_25_global_fraction", "value"] == 0.2
    assert by_metric.loc["predicted_tier_25_fraction", "value"] == 0.4
    assert by_metric.loc["selected_score_transform_mode", "value"] == "bull_shift_18"
    assert by_metric.loc["selected_score_transform_identity_folds", "value"] == 1
    assert by_metric.loc["selected_regime_gate_bull_floor_mode", "value"] == 1.0
    assert (
        by_metric.loc[
            "selected_validation_score_policy_repair_authorized_fraction",
            "value",
        ]
        == pytest.approx(0.5)
    )
    assert by_metric.loc["score_policy_triggered_100_fraction", "value"] == pytest.approx(
        0.2
    )
    assert by_metric.loc[
        "selected_validation_guarded_gate_bull_risk_off_override_authorized_fraction",
        "value",
    ] == pytest.approx(0.5)
    assert by_metric.loc[
        "guarded_gate_bull_risk_off_override_triggered_fraction",
        "value",
    ] == pytest.approx(0.2)
    assert by_metric.loc[
        "min_selection_validation_cost_benchmark_excess_cumulative_return_min",
        "value",
    ] == pytest.approx(-0.05)
    assert (
        by_metric.loc[
            "validation_raw_score_forward_return_correlation_min",
            "value",
        ]
        == pytest.approx(-0.30)
    )
    assert (
        by_metric.loc[
            "candidate_rejection_reason_negative_score_forward_return_correlation",
            "value",
        ]
        == 1
    )
    assert (
        by_metric.loc[
            "negative_validation_score_forward_return_correlation_candidates",
            "value",
        ]
        == 2
    )
    assert by_metric.loc["active_return_vs_buy_hold", "value"] == pytest.approx(0.02)
    assert by_metric.loc["bear_active_return", "value"] == pytest.approx(-0.02)


def test_phase8_summary_cli_writes_summary_csv(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    output_path = tmp_path / "summary.csv"
    _write_phase8_artifacts(run_dir)

    assert main(["phase8-summary", "--run-dir", str(run_dir), "--output", str(output_path)]) == 0

    written = pd.read_csv(output_path)
    assert "failed_strict_gate_rows" in set(written["metric"])


def test_write_phase8_run_summary_defaults_to_run_directory(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    _write_phase8_artifacts(run_dir)

    output_path = write_phase8_run_summary(run_dir)

    assert output_path == run_dir / "phase8_run_summary.csv"
    assert output_path.exists()
