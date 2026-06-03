from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from marketlab.cli import main
from marketlab.reports.phase8_regime_policy_sweep import (
    build_phase8_regime_policy_sweep,
    write_phase8_regime_policy_sweep,
)


def _write_sweep_config(path: Path, cache_dir: Path) -> None:
    path.write_text(
        f"""
experiment_name: phase8_regime_policy_sweep_test
data:
  symbols: ["BTC-USD"]
  interval: "1d"
  cache_dir: "{cache_dir.as_posix()}"
  prepared_panel_filename: "missing.csv"
portfolio:
  costs:
    bps_per_trade: 35
evaluation:
  benchmark_strategy: "buy_hold"
  periods_per_year: 365
  ml_strategy_tuning:
    enabled: true
    max_annualized_turnover: 365
  strict_research_gate:
    enabled: true
    strategy_name: "ml_indicator_tuned__long_only__cash"
    benchmark_strategy: "buy_hold"
    required_benchmark_strategies: ["buy_hold", "btc_static_25"]
    min_average_exposure: 0.20
    max_average_exposure: 0.85
""".strip(),
        encoding="utf-8",
    )


def _write_sweep_artifacts(run_dir: Path) -> None:
    run_dir.mkdir(parents=True)
    dates = pd.date_range("2020-01-01", periods=5, freq="D")
    pd.DataFrame(
        {
            "model_name": "logistic_l1",
            "fold_id": [1] * len(dates),
            "signal_date": dates - pd.Timedelta(days=1),
            "effective_date": dates,
            "symbol": "BTC-USD",
            "runtime_regime": ["bull", "risk_off", "sideways", "bear", "bull"],
            "crypto_regime_risk_off": [0, 1, 0, 0, 0],
            "crypto_regime_trend_state": [1, 1, 0, -1, 1],
            "gate_bull": [True, True, False, False, True],
        }
    ).to_csv(run_dir / "allocation_probability_diagnostics.csv", index=False)
    pd.DataFrame(
        {
            "date": dates,
            "strategy": "ml_indicator_tuned__long_only__cash",
            "long_exposure": [0.25, 0.25, 0.25, 0.25, 0.25],
        }
    ).to_csv(run_dir / "daily_exposure.csv", index=False)
    pd.DataFrame(
        {
            "date": dates,
            "strategy": "ml_indicator_tuned__long_only__cash",
            "benchmark_strategy": "buy_hold",
            "strategy_net_return": [0.025, 0.025, 0.0, -0.025, 0.025],
            "benchmark_net_return": [0.10, 0.10, 0.0, -0.10, 0.10],
            "excess_return": [-0.075, -0.075, 0.0, 0.075, -0.075],
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


def test_build_phase8_regime_policy_sweep_ranks_policy_metrics(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    config_path = tmp_path / "config.yaml"
    _write_sweep_config(config_path, tmp_path / "cache")
    _write_sweep_artifacts(run_dir)

    detail, summary = build_phase8_regime_policy_sweep(
        run_dir,
        config_path=config_path,
    )

    assert not detail.empty
    assert len(summary) == 24
    assert summary.iloc[0]["active_return_vs_buy_hold"] >= summary.iloc[-1][
        "active_return_vs_buy_hold"
    ]
    gate_override_rows = summary.loc[summary["gate_bull_override"].astype(bool)]
    assert not gate_override_rows.empty
    assert gate_override_rows["gate_bull_underexposed_positive_benchmark_fraction"].min() == (
        pytest.approx(0.0)
    )
    assert {
        "policy_metrics",
        "benchmark_deltas",
        "gate_bull",
        "runtime_regime",
    }.issubset(set(detail["section"]))


def test_phase8_regime_policy_sweep_cli_writes_outputs(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    output_dir = tmp_path / "sweep"
    config_path = tmp_path / "config.yaml"
    _write_sweep_config(config_path, tmp_path / "cache")
    _write_sweep_artifacts(run_dir)

    assert (
        main(
            [
                "phase8-regime-policy-sweep",
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

    assert (output_dir / "phase8_regime_policy_sweep.csv").exists()
    assert (output_dir / "phase8_regime_policy_sweep_summary.csv").exists()


def test_write_phase8_regime_policy_sweep_defaults_to_run_directory(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    config_path = tmp_path / "config.yaml"
    _write_sweep_config(config_path, tmp_path / "cache")
    _write_sweep_artifacts(run_dir)

    detail_path, summary_path = write_phase8_regime_policy_sweep(
        run_dir,
        config_path=config_path,
    )

    assert detail_path == run_dir / "phase8_regime_policy_sweep.csv"
    assert summary_path == run_dir / "phase8_regime_policy_sweep_summary.csv"
    assert detail_path.exists()
    assert summary_path.exists()
