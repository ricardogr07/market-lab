from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from marketlab.cli import main
from marketlab.reports.phase8_bull_participation import (
    build_phase8_bull_participation,
    write_phase8_bull_participation,
)


def _write_bull_participation_artifacts(run_dir: Path) -> None:
    run_dir.mkdir(parents=True)
    dates = pd.date_range("2020-01-01", periods=4, freq="D")
    pd.DataFrame(
        {
            "model_name": "logistic_l1",
            "fold_id": [1, 1, 2, 2],
            "signal_date": dates - pd.Timedelta(days=1),
            "effective_date": dates,
            "symbol": "BTC-USD",
            "runtime_regime": ["bull", "risk_off", "bull", "risk_off"],
            "crypto_regime_risk_off": [0, 1, 0, 1],
            "crypto_regime_trend_state": [1, 1, 1, -1],
            "gate_bull": [True, True, False, True],
            "target": [3, 3, 0, 2],
            "target_weight": [1.0, 1.0, 0.0, 0.50],
            "predicted_target": [2, 2, 1, 2],
            "predicted_weight": [0.50, 0.50, 0.25, 0.50],
            "predicted_tier_weight": [0.50, 0.50, 0.25, 0.50],
            "score": [0.44, 0.46, 0.35, 0.41],
            "prob_tier_0": [0.10, 0.10, 0.40, 0.15],
            "prob_tier_25": [0.20, 0.20, 0.30, 0.15],
            "prob_tier_50": [0.50, 0.45, 0.20, 0.50],
            "prob_tier_100": [0.20, 0.25, 0.10, 0.20],
        }
    ).to_csv(run_dir / "allocation_probability_diagnostics.csv", index=False)
    pd.DataFrame(
        {
            "date": dates,
            "strategy": "ml_indicator_tuned__long_only__cash",
            "long_exposure": [1.0, 0.25, 0.50, 0.25],
            "gross_exposure": [1.0, 0.25, 0.50, 0.25],
            "cash_weight": [0.0, 0.75, 0.50, 0.75],
        }
    ).to_csv(run_dir / "daily_exposure.csv", index=False)
    pd.DataFrame(
        {
            "date": dates,
            "strategy": "ml_indicator_tuned__long_only__cash",
            "benchmark_strategy": "buy_hold",
            "strategy_net_return": [0.04, 0.02, 0.01, -0.02],
            "benchmark_net_return": [0.05, 0.04, -0.01, 0.02],
            "excess_return": [-0.01, -0.02, 0.03, -0.04],
        }
    ).to_csv(run_dir / "benchmark_relative.csv", index=False)
    pd.DataFrame(
        {
            "fold_id": [1, 2, 3],
            "selection_status": ["selected", "selected", "no_valid_candidate"],
            "selection_source": ["best_active_fallback", "strict", "none"],
            "selected_regime_policy": [
                "bull100_sideways25",
                "bull100_sideways25",
                pd.NA,
            ],
            "selected_regime_bull_floor": [1.0, 1.0, pd.NA],
            "selected_regime_risk_off_cap": [0.25, 0.25, pd.NA],
        }
    ).to_csv(run_dir / "ml_strategy_tuning_selections.csv", index=False)


def test_build_phase8_bull_participation_summarizes_prediction_compression_and_bull_exposure(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    _write_bull_participation_artifacts(run_dir)

    detail, summary = build_phase8_bull_participation(run_dir)
    by_metric = summary.set_index("metric")

    assert by_metric.loc["target_tier_100_fraction", "value"] == pytest.approx(0.50)
    assert by_metric.loc["predicted_tier_100_fraction", "value"] == pytest.approx(0.0)
    assert by_metric.loc["gate_bull_runtime_risk_off_fraction", "value"] == pytest.approx(
        2 / 3
    )
    assert by_metric.loc["gate_bull_average_long_exposure", "value"] == pytest.approx(
        0.50
    )
    assert by_metric.loc[
        "gate_bull_underexposed_positive_benchmark_fraction",
        "value",
    ] == pytest.approx(2 / 3)
    assert by_metric.loc["selection_source_best_active_fallback_folds", "value"] == 1

    confusion = detail.loc[
        detail["metric"].eq("target_vs_predicted_tier_fraction")
        & detail["group"].astype(str).eq("100")
        & detail["subgroup"].astype(str).eq("50")
    ].iloc[0]
    assert confusion["value"] == pytest.approx(1.0)


def test_build_phase8_bull_participation_handles_missing_artifacts(tmp_path: Path) -> None:
    detail, summary = build_phase8_bull_participation(tmp_path / "missing-run")
    by_metric = summary.set_index("metric")

    assert detail.empty
    assert bool(by_metric.loc["allocation_probability_present", "value"]) is False
    assert bool(by_metric.loc["daily_exposure_present", "value"]) is False
    assert bool(by_metric.loc["benchmark_relative_present", "value"]) is False
    assert bool(by_metric.loc["selection_artifact_present", "value"]) is False


def test_phase8_bull_participation_cli_writes_outputs(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    output_dir = tmp_path / "bull"
    _write_bull_participation_artifacts(run_dir)

    assert (
        main(
            [
                "phase8-bull-participation",
                "--run-dir",
                str(run_dir),
                "--output-dir",
                str(output_dir),
            ]
        )
        == 0
    )

    assert (output_dir / "phase8_bull_participation.csv").exists()
    assert (output_dir / "phase8_bull_participation_summary.csv").exists()


def test_write_phase8_bull_participation_defaults_to_run_directory(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    _write_bull_participation_artifacts(run_dir)

    detail_path, summary_path = write_phase8_bull_participation(run_dir)

    assert detail_path == run_dir / "phase8_bull_participation.csv"
    assert summary_path == run_dir / "phase8_bull_participation_summary.csv"
    assert detail_path.exists()
    assert summary_path.exists()
