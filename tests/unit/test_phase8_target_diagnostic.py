from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from marketlab.cli import main
from marketlab.reports.phase8_target_diagnostic import (
    build_phase8_target_diagnostic,
    write_phase8_target_diagnostic,
)


def _write_target_artifacts(run_dir: Path) -> None:
    run_dir.mkdir(parents=True)
    dates = pd.date_range("2020-01-01", periods=8, freq="D")
    pd.DataFrame(
        {
            "model_name": ["logistic_l1"] * 8,
            "fold_id": [1] * 4 + [2] * 4,
            "signal_date": dates - pd.Timedelta(days=1),
            "effective_date": dates,
            "symbol": "BTC-USD",
            "runtime_regime": [
                "bull",
                "bull",
                "risk_off",
                "bear",
                "sideways",
                "bull",
                "bull",
                "risk_off",
            ],
            "target_weight": [1.0, 0.50, 0.25, 0.0, 0.25, 1.0, 0.50, 0.0],
            "predicted_tier_weight": [1.0, 0.50, 0.25, 0.0, 0.25, 0.50, 0.50, 0.0],
            "score": [0.90, 0.70, 0.30, 0.10, 0.40, 0.80, 0.65, 0.05],
            "forward_return": [0.08, 0.05, -0.04, -0.08, 0.02, 0.04, 0.03, -0.03],
            "forward_drawdown": [-0.02, -0.12, -0.18, -0.22, -0.03, -0.04, -0.01, -0.11],
            "realized_utility": [0.08, 0.01, -0.01, 0.0, 0.01, 0.02, 0.01, 0.0],
        }
    ).to_csv(run_dir / "allocation_probability_diagnostics.csv", index=False)


def _write_gate_config(tmp_path: Path) -> Path:
    config_dir = tmp_path / "configs"
    data_dir = tmp_path / "data"
    config_dir.mkdir()
    data_dir.mkdir()
    dates = pd.date_range("2019-12-28", periods=12, freq="D")
    prices = [100, 101, 102, 103, 104, 105, 106, 104, 103, 107, 108, 109]
    pd.DataFrame(
        {
            "symbol": ["BTC-USD"] * len(dates),
            "timestamp": dates,
            "open": prices,
            "high": prices,
            "low": prices,
            "close": prices,
            "volume": [1000] * len(dates),
            "adj_close": prices,
            "adj_factor": [1.0] * len(dates),
            "adj_open": prices,
            "adj_high": prices,
            "adj_low": prices,
        }
    ).to_csv(data_dir / "panel.csv", index=False)
    config_path = config_dir / "experiment.btc_phase8_test.yaml"
    config_path.write_text(
        """
experiment_name: btc_phase8_test
data:
  symbols: ["BTC-USD"]
  interval: "1d"
  cache_dir: "data"
  prepared_panel_filename: "panel.csv"
features:
  crypto_regime_features_enabled: true
  crypto_regime_trend_windows: [2]
  crypto_regime_volatility_window: 2
  crypto_regime_percentile_window: 2
  crypto_regime_drawdown_window: 2
  crypto_regime_volume_window: 2
evaluation:
  strict_research_gate:
    recent_window_months: 1
paper:
  enabled: false
""",
        encoding="utf-8",
    )
    return config_path


def test_build_phase8_target_diagnostic_reports_target_score_pivot_metrics(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    _write_target_artifacts(run_dir)

    detail, summary = build_phase8_target_diagnostic(run_dir)
    by_metric = summary.set_index("metric")

    assert by_metric.loc["bull_continuation_rows", "value"] == 4
    assert by_metric.loc["bull_continuation_full_target_fraction", "value"] == pytest.approx(
        0.50
    )
    assert by_metric.loc[
        "bull_continuation_full_prediction_fraction",
        "value",
    ] == pytest.approx(0.25)
    assert by_metric.loc[
        "positive_return_rows_assigned_below_100_fraction",
        "value",
    ] == pytest.approx(3 / 5)
    assert by_metric.loc[
        "drawdown_defense_rows_assigned_below_100_fraction",
        "value",
    ] == pytest.approx(1.0)
    assert by_metric.loc["score_target_weight_correlation", "value"] > 0.9

    runtime_rows = detail.loc[
        detail["section"].eq("runtime_regime")
        & detail["metric"].eq("full_target_fraction")
        & detail["group"].eq("bull")
    ]
    assert runtime_rows.iloc[0]["value"] == pytest.approx(0.50)

    matrix = detail.loc[
        detail["section"].eq("target_prediction_matrix")
        & detail["group"].astype(str).eq("100")
        & detail["subgroup"].astype(str).eq("50")
    ]
    assert not matrix.empty


def test_build_phase8_target_diagnostic_handles_missing_artifacts(tmp_path: Path) -> None:
    detail, summary = build_phase8_target_diagnostic(tmp_path / "missing-run")
    by_metric = summary.set_index("metric")

    assert detail.empty
    assert bool(by_metric.loc["allocation_probability_present", "value"]) is False


def test_build_phase8_target_diagnostic_joins_config_gate_bull_labels(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    _write_target_artifacts(run_dir)
    config_path = _write_gate_config(tmp_path)

    detail, summary = build_phase8_target_diagnostic(run_dir, config_path=config_path)
    by_metric = summary.set_index("metric")

    assert by_metric.loc["runtime_regime_source", "value"] == "config"
    assert by_metric.loc["gate_bull_rows", "value"] > 0
    assert not detail.loc[detail["section"].eq("gate_bull_label")].empty


def test_phase8_target_diagnostic_cli_writes_outputs(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    output_dir = tmp_path / "target"
    _write_target_artifacts(run_dir)

    assert (
        main(
            [
                "phase8-target-diagnostic",
                "--run-dir",
                str(run_dir),
                "--output-dir",
                str(output_dir),
            ]
        )
        == 0
    )

    assert (output_dir / "phase8_target_diagnostic.csv").exists()
    assert (output_dir / "phase8_target_diagnostic_summary.csv").exists()


def test_write_phase8_target_diagnostic_defaults_to_run_directory(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    _write_target_artifacts(run_dir)

    detail_path, summary_path = write_phase8_target_diagnostic(run_dir)

    assert detail_path == run_dir / "phase8_target_diagnostic.csv"
    assert summary_path == run_dir / "phase8_target_diagnostic_summary.csv"
    assert detail_path.exists()
    assert summary_path.exists()
