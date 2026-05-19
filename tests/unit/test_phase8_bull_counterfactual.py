from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from marketlab.cli import main
from marketlab.reports.phase8_bull_counterfactual import (
    build_phase8_bull_counterfactual,
    write_phase8_bull_counterfactual,
)


def _write_counterfactual_config(path: Path, cache_dir: Path) -> None:
    path.write_text(
        f"""
experiment_name: phase8_counterfactual_test
data:
  symbols: ["BTC-USD"]
  start_date: "2020-01-01"
  end_date: "2020-01-04"
  interval: "1d"
  cache_dir: "{cache_dir.as_posix()}"
  prepared_panel_filename: "missing.csv"
portfolio:
  costs:
    bps_per_trade: 35
evaluation:
  benchmark_strategy: "buy_hold"
  periods_per_year: 365
  cost_sensitivity_bps: [35, 50]
  ml_strategy_tuning:
    enabled: true
    max_annualized_turnover: 365
  strict_research_gate:
    enabled: true
    strategy_name: "ml_indicator_tuned__long_only__cash"
    benchmark_strategy: "buy_hold"
    required_benchmark_strategies: ["buy_hold", "btc_static_25"]
    cost_gate_bps: 35
    acceptable_cost_bps: 50
    min_positive_regime_slices: 1
    min_average_exposure: 0.20
    max_average_exposure: 0.85
""".strip(),
        encoding="utf-8",
    )


def _write_counterfactual_artifacts(run_dir: Path, *, include_regime: bool = True) -> None:
    run_dir.mkdir(parents=True)
    dates = pd.date_range("2020-01-01", periods=4, freq="D")
    if include_regime:
        pd.DataFrame(
            {
                "model_name": "logistic_l1",
                "fold_id": [1, 1, 1, 1],
                "signal_date": dates - pd.Timedelta(days=1),
                "effective_date": dates,
                "symbol": "BTC-USD",
                "runtime_regime": ["bull", "risk_off", "bear", "sideways"],
                "crypto_regime_risk_off": [0, 1, 0, 0],
                "crypto_regime_trend_state": [1, 1, -1, 0],
                "gate_bull": [True, True, False, False],
            }
        ).to_csv(run_dir / "allocation_probability_diagnostics.csv", index=False)
    pd.DataFrame(
        {
            "date": dates,
            "strategy": "ml_indicator_tuned__long_only__cash",
            "long_exposure": [0.25, 0.25, 0.25, 0.25],
        }
    ).to_csv(run_dir / "daily_exposure.csv", index=False)
    pd.DataFrame(
        {
            "date": dates,
            "strategy": "ml_indicator_tuned__long_only__cash",
            "benchmark_strategy": "buy_hold",
            "strategy_net_return": [0.025, 0.025, -0.025, 0.025],
            "benchmark_net_return": [0.10, 0.10, -0.10, 0.10],
            "excess_return": [-0.075, -0.075, 0.075, -0.075],
            "strategy_equity": [1.025, 1.050625, 1.024359375, 1.049968359375],
            "benchmark_equity": [1.10, 1.21, 1.089, 1.1979],
            "relative_equity": [0.931818, 0.868285, 0.940642, 0.876512],
        }
    ).to_csv(run_dir / "benchmark_relative.csv", index=False)
    pd.DataFrame(
        {
            "strategy": ["buy_hold", "btc_static_25"],
            "cumulative_return": [0.1979, 0.0495],
            "sharpe_like": [1.0, 0.8],
            "max_drawdown": [-0.10, -0.03],
        }
    ).to_csv(run_dir / "strategy_summary.csv", index=False)
    pd.DataFrame(
        {
            "strategy": ["buy_hold", "btc_static_25", "buy_hold", "btc_static_25"],
            "bps_per_trade": [35, 35, 50, 50],
            "cumulative_return": [0.1979, 0.0495, 0.1979, 0.0495],
        }
    ).to_csv(run_dir / "cost_sensitivity.csv", index=False)


def test_build_phase8_bull_counterfactual_mutates_expected_day_sets(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    config_path = tmp_path / "config.yaml"
    _write_counterfactual_config(config_path, tmp_path / "cache")
    _write_counterfactual_artifacts(run_dir)

    detail, summary, gate = build_phase8_bull_counterfactual(
        run_dir,
        config_path=config_path,
    )
    mutated = summary.loc[summary["metric"].eq("mutated_days")].set_index("scenario")

    assert mutated.loc["actual_runtime", "value"] == 0
    assert mutated.loc["force_runtime_bull_100", "value"] == 1
    assert mutated.loc["gate_bull_overrides_risk_off", "value"] == 1
    assert mutated.loc["buy_hold_gate_bull_model_elsewhere", "value"] == 2

    cumulative = summary.loc[summary["metric"].eq("cumulative_return")].set_index("scenario")
    assert cumulative.loc["force_runtime_bull_100", "value"] > cumulative.loc[
        "actual_runtime",
        "value",
    ]
    assert cumulative.loc["buy_hold_gate_bull_model_elsewhere", "value"] > cumulative.loc[
        "gate_bull_overrides_risk_off",
        "value",
    ]

    gate_rows = gate.loc[
        gate["scenario"].eq("buy_hold_gate_bull_model_elsewhere")
        & gate["condition"].eq("net_cumulative_return_beats_btc_static_25")
    ]
    assert not gate_rows.empty
    assert bool(gate["diagnostic_only"].all()) is True

    slice_rows = detail.loc[
        detail["scenario"].eq("buy_hold_gate_bull_model_elsewhere")
        & detail["section"].eq("regime_slices")
        & detail["group"].eq("gate_bull")
    ]
    assert not slice_rows.empty


def test_build_phase8_bull_counterfactual_requires_regime_labels(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    config_path = tmp_path / "config.yaml"
    _write_counterfactual_config(config_path, tmp_path / "cache")
    _write_counterfactual_artifacts(run_dir, include_regime=False)

    with pytest.raises(ValueError, match="runtime/gate regime"):
        build_phase8_bull_counterfactual(run_dir, config_path=config_path)


def test_phase8_bull_counterfactual_cli_writes_outputs(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    output_dir = tmp_path / "counterfactual"
    config_path = tmp_path / "config.yaml"
    _write_counterfactual_config(config_path, tmp_path / "cache")
    _write_counterfactual_artifacts(run_dir)

    assert (
        main(
            [
                "phase8-bull-counterfactual",
                "--run-dir",
                str(run_dir),
                "--config",
                str(config_path),
                "--output-dir",
                str(output_dir),
            ]
        )
        == 0
    )

    assert (output_dir / "phase8_bull_counterfactual.csv").exists()
    assert (output_dir / "phase8_bull_counterfactual_summary.csv").exists()
    assert (output_dir / "phase8_bull_counterfactual_gate.csv").exists()


def test_write_phase8_bull_counterfactual_defaults_to_run_directory(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    config_path = tmp_path / "config.yaml"
    _write_counterfactual_config(config_path, tmp_path / "cache")
    _write_counterfactual_artifacts(run_dir)

    detail_path, summary_path, gate_path = write_phase8_bull_counterfactual(
        run_dir,
        config_path=config_path,
    )

    assert detail_path == run_dir / "phase8_bull_counterfactual.csv"
    assert summary_path == run_dir / "phase8_bull_counterfactual_summary.csv"
    assert gate_path == run_dir / "phase8_bull_counterfactual_gate.csv"
    assert detail_path.exists()
    assert summary_path.exists()
    assert gate_path.exists()
