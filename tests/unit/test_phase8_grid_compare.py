from __future__ import annotations

from pathlib import Path

import pandas as pd

from marketlab.cli import main
from marketlab.reports.phase8_grid_compare import (
    build_phase8_grid_comparison,
    write_phase8_grid_comparison,
)


def _write_complete_run(run_dir: Path, *, strategy_return: float) -> None:
    run_dir.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "strategy": "buy_hold",
                "cumulative_return": 10.0,
                "sharpe_like": 0.9,
                "max_drawdown": -0.55,
            },
            {
                "strategy": "btc_static_25",
                "cumulative_return": 3.0,
            },
            {
                "strategy": "btc_static_50",
                "cumulative_return": 6.0,
            },
            {
                "strategy": "btc_static_75",
                "cumulative_return": 8.0,
            },
            {
                "strategy": "btc_rebalanced_25",
                "cumulative_return": 1.5,
            },
            {
                "strategy": "btc_rebalanced_50",
                "cumulative_return": 4.0,
            },
            {
                "strategy": "btc_rebalanced_75",
                "cumulative_return": 7.0,
            },
            {
                "strategy": "ml_indicator_tuned__long_only__cash",
                "cumulative_return": strategy_return,
                "sharpe_like": 1.1,
                "max_drawdown": -0.35,
                "avg_gross_exposure": 0.44,
                "avg_turnover": 0.01,
                "up_capture": 0.65,
                "down_capture": 0.50,
            },
        ]
    ).to_csv(run_dir / "strategy_summary.csv", index=False)
    pd.DataFrame(
        [
            {"condition": "overall", "passed": False, "observed": "", "required": ""},
        ]
    ).to_csv(run_dir / "strict_research_gate.csv", index=False)
    pd.DataFrame(
        [
            {
                "fold_id": 1,
                "selection_status": "selected",
            },
        ]
    ).to_csv(run_dir / "ml_strategy_tuning_selections.csv", index=False)
    pd.DataFrame(
        [
            {
                "section": "fold_selection",
                "metric": "selected_fold_fraction",
                "value": 0.73,
                "detail": "",
            },
            {
                "section": "fold_selection",
                "metric": "no_valid_candidate_folds",
                "value": 4,
                "detail": "",
            },
            {
                "section": "target_support",
                "metric": "target_tier_25_global_fraction",
                "value": 0.04,
                "detail": "",
            },
            {
                "section": "target_support",
                "metric": "target_tier_50_global_fraction",
                "value": 0.05,
                "detail": "",
            },
            {
                "section": "predicted_support",
                "metric": "predicted_tier_25_fraction",
                "value": 0.80,
                "detail": "",
            },
            {
                "section": "predicted_support",
                "metric": "predicted_tier_50_fraction",
                "value": 0.10,
                "detail": "",
            },
            {
                "section": "fold_selection",
                "metric": "selected_regime_gate_bull_floor_mode",
                "value": 1.0,
                "detail": "",
            },
            {
                "section": "score_policy_repair",
                "metric": "selected_validation_score_policy_repair_authorized_fraction",
                "value": 0.75,
                "detail": "",
            },
        ]
    ).to_csv(run_dir / "phase8_run_summary.csv", index=False)
    pd.DataFrame(
        [
            {
                "methodology_gate": "deployment_gate",
                "section": "strict_gate",
                "metric": "overall",
                "passed": False,
                "value": "",
                "required": "",
                "diagnostic_only": False,
                "source_artifact": "strict_research_gate.csv",
                "detail": "",
            },
            {
                "methodology_gate": "risk_allocation_gate",
                "section": "summary",
                "metric": "overall",
                "passed": True,
                "value": True,
                "required": "",
                "diagnostic_only": False,
                "source_artifact": "",
                "detail": "",
            },
            {
                "methodology_gate": "selection_coverage_gate",
                "section": "summary",
                "metric": "overall",
                "passed": False,
                "value": False,
                "required": "",
                "diagnostic_only": False,
                "source_artifact": "",
                "detail": "",
            },
            {
                "methodology_gate": "target_support_gate",
                "section": "summary",
                "metric": "overall",
                "passed": False,
                "value": False,
                "required": "",
                "diagnostic_only": False,
                "source_artifact": "",
                "detail": "",
            },
            {
                "methodology_gate": "signal_validity_gate",
                "section": "summary",
                "metric": "overall",
                "passed": False,
                "value": False,
                "required": "",
                "diagnostic_only": False,
                "source_artifact": "",
                "detail": "",
            },
            {
                "methodology_gate": "bull_participation_gate",
                "section": "summary",
                "metric": "overall",
                "passed": False,
                "value": False,
                "required": "",
                "diagnostic_only": False,
                "source_artifact": "",
                "detail": "",
            },
            {
                "methodology_gate": "counterfactual_hypothesis",
                "section": "summary",
                "metric": "counterfactual_pass_available",
                "passed": True,
                "value": True,
                "required": "",
                "diagnostic_only": True,
                "source_artifact": "",
                "detail": "",
            },
        ]
    ).to_csv(run_dir / "phase8_methodology_review.csv", index=False)
    pd.DataFrame(
        [
            {
                "section": "score_deciles",
                "metric": "score_target_weight_correlation",
                "value": 0.10,
                "detail": "",
            },
            {
                "section": "score_deciles",
                "metric": "score_forward_return_correlation",
                "value": -0.02,
                "detail": "",
            },
            {
                "section": "score_deciles",
                "metric": "score_realized_utility_correlation",
                "value": 0.03,
                "detail": "",
            },
            {
                "section": "score_deciles",
                "metric": "predicted_tier_100_fraction",
                "value": 0.20,
                "detail": "",
            },
            {
                "section": "model_family_support",
                "metric": "any_selected_oos_predicted_tier_100",
                "value": True,
                "detail": "",
            },
            {
                "section": "candidate_score_validity",
                "metric": "validation_score_forward_return_correlation_mean",
                "value": -0.10,
                "detail": "",
            },
            {
                "section": "candidate_score_validity",
                "metric": "validation_score_forward_return_correlation_min",
                "value": -0.30,
                "detail": "",
            },
            {
                "section": "candidate_score_validity",
                "metric": "validation_raw_score_forward_return_correlation_min",
                "value": -0.40,
                "detail": "",
            },
            {
                "section": "score_policy_repair",
                "metric": "score_policy_triggered_100_fraction",
                "value": 0.15,
                "detail": "",
            },
            {
                "section": "score_policy_repair",
                "metric": "score_policy_repair_authorized_fraction",
                "value": 0.75,
                "detail": "",
            },
            {
                "section": "candidate_score_validity",
                "metric": "negative_validation_score_forward_return_correlation_candidates",
                "value": 2,
                "detail": "",
            },
            {
                "section": "candidate_score_validity",
                "metric": "validation_gate_bull_underexposed_positive_benchmark_fraction_mean",
                "value": 0.50,
                "detail": "",
            },
        ]
    ).to_csv(run_dir / "phase8_score_diagnostic_summary.csv", index=False)
    pd.DataFrame(
        [
            {
                "section": "runtime_participation",
                "metric": "gate_bull_average_long_exposure",
                "value": 0.55,
                "detail": "",
            },
            {
                "section": "bull_active_return",
                "metric": "gate_bull_active_return_sum",
                "value": -1.0,
                "detail": "",
            },
            {
                "section": "bull_active_return",
                "metric": "gate_bull_underexposed_positive_benchmark_fraction",
                "value": 0.80,
                "detail": "",
            },
            {
                "section": "bull_active_return",
                "metric": "gate_bull_underexposed_positive_benchmark_return_sum",
                "value": 5.0,
                "detail": "",
            },
        ]
    ).to_csv(run_dir / "phase8_bull_participation_summary.csv", index=False)
    pd.DataFrame(
        [
            {
                "scenario": "force_runtime_bull_100",
                "metric": "cumulative_return",
                "value": 20.0,
                "detail": "",
            },
            {
                "scenario": "force_runtime_bull_100",
                "metric": "avg_long_exposure",
                "value": 0.60,
                "detail": "",
            },
            {
                "scenario": "buy_hold_gate_bull_model_elsewhere",
                "metric": "cumulative_return",
                "value": 18.0,
                "detail": "",
            },
            {
                "scenario": "buy_hold_gate_bull_model_elsewhere",
                "metric": "avg_long_exposure",
                "value": 0.58,
                "detail": "",
            },
        ]
    ).to_csv(run_dir / "phase8_bull_counterfactual_summary.csv", index=False)
    pd.DataFrame(
        [
            {
                "scenario": "force_runtime_bull_100",
                "condition": "overall",
                "passed": True,
            },
        ]
    ).to_csv(run_dir / "phase8_bull_counterfactual_gate.csv", index=False)


def test_build_phase8_grid_comparison_summarizes_complete_and_incomplete_runs(
    tmp_path: Path,
) -> None:
    runs_root = tmp_path / "runs"
    first_run = runs_root / "btc_phase8_grid_fixture" / "20260501T000000Z"
    latest_run = runs_root / "btc_phase8_grid_fixture" / "20260502T000000Z"
    incomplete_run = runs_root / "btc_phase8_grid_fixture" / "20260503T000000Z"
    _write_complete_run(first_run, strategy_return=8.0)
    _write_complete_run(latest_run, strategy_return=12.0)
    incomplete_run.mkdir(parents=True)
    (incomplete_run / "partial.txt").write_text("partial", encoding="utf-8")

    comparison = build_phase8_grid_comparison(runs_root=runs_root)

    assert len(comparison) == 3
    first = comparison.loc[comparison["run_id"].eq("20260501T000000Z")].iloc[0]
    latest = comparison.loc[comparison["run_id"].eq("20260502T000000Z")].iloc[0]
    incomplete = comparison.loc[comparison["run_id"].eq("20260503T000000Z")].iloc[0]
    assert first["artifact_status"] == "complete"
    assert first["recommended_artifact_action"] == "keep_for_grid_comparison"
    assert bool(first["latest_complete_for_experiment"]) is False
    assert latest["active_return_vs_buy_hold"] == 2.0
    assert latest["bull_upside_capture_ratio"] == 0.65
    assert latest["gate_bull_underexposed_positive_benchmark_return_sum"] == 5.0
    assert latest["selected_regime_gate_bull_floor_mode"] == 1.0
    assert latest["candidate_validation_score_forward_return_correlation_min"] == -0.30
    assert latest["candidate_validation_raw_score_forward_return_correlation_min"] == -0.40
    assert latest["score_policy_triggered_100_fraction"] == 0.15
    assert latest["score_policy_repair_authorized_fraction"] == 0.75
    assert latest["selected_validation_score_policy_repair_authorized_fraction"] == 0.75
    assert latest["negative_validation_score_forward_return_correlation_candidates"] == 2
    assert latest["recommended_artifact_action"] == "keep_latest_complete"
    assert bool(latest["latest_complete_for_experiment"]) is True
    assert incomplete["artifact_status"] == "incomplete"
    assert incomplete["recommended_artifact_action"] == "archive_or_prune_after_manifest"


def test_write_phase8_grid_comparison_and_cli_write_output(tmp_path: Path) -> None:
    run_dir = tmp_path / "runs" / "btc_phase8_grid_fixture" / "20260501T000000Z"
    _write_complete_run(run_dir, strategy_return=12.0)
    output_path = tmp_path / "comparison.csv"

    written_path = write_phase8_grid_comparison(
        run_dirs=[run_dir],
        output_path=output_path,
    )

    assert written_path == output_path
    assert output_path.exists()

    cli_output_path = tmp_path / "cli-comparison.csv"
    assert (
        main(
            [
                "phase8-grid-compare",
                "--run-dir",
                str(run_dir),
                "--output",
                str(cli_output_path),
            ]
        )
        == 0
    )
    assert cli_output_path.exists()
