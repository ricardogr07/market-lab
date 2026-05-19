from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from marketlab.cli import main
from marketlab.reports.phase8_score_diagnostic import (
    build_phase8_score_diagnostic,
    write_phase8_score_diagnostic,
)


def _write_score_artifacts(run_dir: Path) -> None:
    run_dir.mkdir(parents=True)
    dates = pd.date_range("2020-01-01", periods=10, freq="D")
    pd.DataFrame(
        {
            "model_name": ["logistic_l1"] * 5 + ["random_forest"] * 5,
            "fold_id": [1] * 5 + [2] * 5,
            "signal_date": dates - pd.Timedelta(days=1),
            "effective_date": dates,
            "symbol": "BTC-USD",
            "runtime_regime": ["bull", "bull", "risk_off", "bear", "bull"] * 2,
            "target_weight": [1.0, 1.0, 0.0, 0.25, 0.50, 1.0, 0.0, 0.0, 0.50, 1.0],
            "predicted_tier_weight": [0.50, 0.50, 0.25, 0.25, 0.50, 1.0, 0.25, 0.50, 0.50, 1.0],
            "score": [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.00],
            "prob_tier_100": [0.05, 0.06, 0.07, 0.08, 0.09, 0.30, 0.31, 0.32, 0.33, 0.34],
            "forward_return": [-0.04, -0.02, -0.01, 0.0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06],
            "realized_utility": [-0.03, -0.01, -0.005, 0.0, 0.01, 0.03, 0.04, 0.05, 0.06, 0.07],
        }
    ).to_csv(run_dir / "allocation_probability_diagnostics.csv", index=False)
    pd.DataFrame(
        [
            {
                "fold_id": 1,
                "model_name": "logistic_l1",
                "utility_profile": "partial_p25",
                "rolling_train_bars": 730,
                "min_holding_period_bars": 18,
                "hysteresis_margin": 0.02,
                "regime_policy": "bull100_sideways25",
                "threshold": 0.0,
                "validation_predicted_25_fraction": 0.20,
                "validation_predicted_50_fraction": 0.80,
                "validation_predicted_100_fraction": 0.0,
                "min_validation_predicted_target_fraction": 0.20,
                "min_benchmark_excess_cumulative_return": -0.01,
                "passed_gate": False,
            },
            {
                "fold_id": 2,
                "model_name": "random_forest",
                "utility_profile": "partial_p25",
                "rolling_train_bars": 730,
                "min_holding_period_bars": 18,
                "hysteresis_margin": 0.02,
                "regime_policy": "bull100_sideways25",
                "threshold": 0.0,
                "validation_predicted_25_fraction": 0.10,
                "validation_predicted_50_fraction": 0.70,
                "validation_predicted_100_fraction": 0.20,
                "min_validation_predicted_target_fraction": 0.10,
                "min_benchmark_excess_cumulative_return": 0.03,
                "passed_gate": True,
            },
        ]
    ).to_csv(run_dir / "ml_strategy_tuning_candidates.csv", index=False)
    pd.DataFrame(
        {
            "fold_id": [1],
            "selection_status": ["selected"],
            "selected_model_name": ["logistic_l1"],
            "selected_utility_profile": ["partial_p25"],
            "selected_rolling_train_bars": [730],
            "selected_min_holding_period_bars": [18],
            "selected_hysteresis_margin": [0.02],
            "selected_regime_policy": ["bull100_sideways25"],
            "selected_threshold": [0.0],
            "validation_predicted_25_fraction": [0.20],
            "validation_predicted_50_fraction": [0.80],
        }
    ).to_csv(run_dir / "ml_strategy_tuning_selections.csv", index=False)


def test_build_phase8_score_diagnostic_reports_deciles_confusion_and_support(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    _write_score_artifacts(run_dir)

    detail, summary = build_phase8_score_diagnostic(run_dir)
    by_metric = summary.set_index("metric")

    assert by_metric.loc["score_forward_return_correlation", "value"] > 0.9
    assert by_metric.loc["predicted_tier_100_fraction", "value"] == pytest.approx(0.20)
    assert bool(by_metric.loc["any_selected_oos_predicted_tier_100", "value"]) is True
    assert bool(
        by_metric.loc["candidate_validation_predicted_100_available", "value"]
    ) is True

    deciles = detail.loc[
        detail["section"].eq("score_deciles") & detail["metric"].eq("score_mean")
    ]
    assert deciles["group"].astype(int).tolist() == list(range(1, 11))

    confusion = detail.loc[
        detail["section"].eq("target_prediction_confusion")
        & detail["group"].astype(str).eq("runtime_bull")
        & detail["subgroup"].astype(str).eq("100_to_50")
    ]
    assert not confusion.empty

    selected_support = detail.loc[
        detail["section"].eq("candidate_score_support")
        & detail["group"].astype(str).eq("selected")
        & detail["metric"].eq("validation_predicted_50_fraction_mean")
    ].iloc[0]
    assert selected_support["value"] == pytest.approx(0.80)


def test_build_phase8_score_diagnostic_handles_missing_artifacts(tmp_path: Path) -> None:
    detail, summary = build_phase8_score_diagnostic(tmp_path / "missing-run")
    by_metric = summary.set_index("metric")

    assert detail.empty
    assert bool(by_metric.loc["allocation_probability_present", "value"]) is False
    assert bool(by_metric.loc["candidate_artifact_present", "value"]) is False
    assert bool(by_metric.loc["stability_present", "value"]) is False


def test_phase8_score_diagnostic_cli_writes_outputs(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    output_dir = tmp_path / "score"
    _write_score_artifacts(run_dir)

    assert (
        main(
            [
                "phase8-score-diagnostic",
                "--run-dir",
                str(run_dir),
                "--output-dir",
                str(output_dir),
            ]
        )
        == 0
    )

    assert (output_dir / "phase8_score_diagnostic.csv").exists()
    assert (output_dir / "phase8_score_diagnostic_summary.csv").exists()


def test_write_phase8_score_diagnostic_defaults_to_run_directory(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    _write_score_artifacts(run_dir)

    detail_path, summary_path = write_phase8_score_diagnostic(run_dir)

    assert detail_path == run_dir / "phase8_score_diagnostic.csv"
    assert summary_path == run_dir / "phase8_score_diagnostic_summary.csv"
    assert detail_path.exists()
    assert summary_path.exists()
