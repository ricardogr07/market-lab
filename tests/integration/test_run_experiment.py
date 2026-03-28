from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
from tests.integration import _cli_harness

from marketlab.data.panel import save_panel_csv

EXPECTED_METRICS_COLUMNS = _cli_harness.EXPECTED_METRICS_COLUMNS
MODEL_SUMMARY_COLUMNS = _cli_harness.MODEL_SUMMARY_COLUMNS
FOLD_SUMMARY_COLUMNS = _cli_harness.FOLD_SUMMARY_COLUMNS
assert_command_ok = _cli_harness.assert_command_ok
build_synthetic_panel = _cli_harness.build_synthetic_panel
latest_run_dir = _cli_harness.latest_run_dir
load_fixture_panel = _cli_harness.load_fixture_panel
write_yaml_config = _cli_harness.write_yaml_config
run_marketlab_cli = getattr(
    _cli_harness,
    "run_marketlab_cli",
    _cli_harness.run_launcher_command,
)
stdout_path = getattr(
    _cli_harness,
    "stdout_path",
    _cli_harness.printed_path,
)

PERFORMANCE_COLUMNS = [
    "date",
    "strategy",
    "gross_return",
    "net_return",
    "turnover",
    "equity",
]


def _write_run_experiment_config(tmp_path: Path) -> Path:
    cache_dir = tmp_path / "cache"
    save_panel_csv(
        build_synthetic_panel(
            (
                ("AAA", 100.0, 0.45),
                ("BBB", 130.0, 0.40),
                ("CCC", 160.0, 0.35),
                ("DDD", 190.0, 0.30),
            ),
            start_date="2020-01-01",
            end_date="2024-12-31",
        ),
        cache_dir / "panel.csv",
    )

    return write_yaml_config(
        tmp_path / "integration.yaml",
        {
            "experiment_name": "integration_fixture",
            "data": {
                "symbols": ["AAA", "BBB", "CCC", "DDD"],
                "start_date": "2020-01-01",
                "end_date": "2024-12-31",
                "interval": "1d",
                "cache_dir": str(cache_dir),
                "prepared_panel_filename": "panel.csv",
            },
            "features": {
                "return_windows": [5, 10],
                "ma_windows": [5, 10],
                "vol_windows": [5],
                "momentum_window": 10,
            },
            "target": {
                "horizon_days": 5,
                "type": "direction",
            },
            "portfolio": {
                "ranking": {
                    "long_n": 2,
                    "short_n": 2,
                    "rebalance_frequency": "W-FRI",
                    "weighting": "equal",
                },
                "costs": {"bps_per_trade": 10},
            },
            "baselines": {
                "buy_hold": True,
                "sma": {"enabled": True, "fast_window": 5, "slow_window": 10},
            },
            "models": [
                {"name": "logistic_regression"},
                {"name": "random_forest"},
            ],
            "evaluation": {
                "walk_forward": {
                    "train_years": 1,
                    "test_months": 2,
                    "step_months": 2,
                }
            },
            "artifacts": {
                "output_dir": str(tmp_path / "runs"),
                "save_predictions": True,
                "save_metrics_csv": True,
                "save_report_md": True,
                "save_plots": True,
            },
        },
    )


def _write_backtest_config(tmp_path: Path) -> Path:
    cache_dir = tmp_path / "cache"
    save_panel_csv(load_fixture_panel(), cache_dir / "panel.csv")

    return write_yaml_config(
        tmp_path / "backtest.yaml",
        {
            "experiment_name": "integration_backtest_fixture",
            "data": {
                "symbols": ["VOO", "QQQ", "SMH", "XLV", "IEMG"],
                "start_date": "2024-01-01",
                "end_date": "2024-01-31",
                "interval": "1d",
                "cache_dir": str(cache_dir),
                "prepared_panel_filename": "panel.csv",
            },
            "features": {
                "return_windows": [2, 3],
                "ma_windows": [2, 3],
                "vol_windows": [2],
                "momentum_window": 2,
            },
            "portfolio": {
                "costs": {"bps_per_trade": 10},
            },
            "baselines": {
                "buy_hold": True,
                "sma": {"enabled": True, "fast_window": 2, "slow_window": 3},
            },
            "artifacts": {
                "output_dir": str(tmp_path / "runs"),
                "save_predictions": False,
                "save_metrics_csv": True,
                "save_report_md": True,
                "save_plots": True,
            },
        },
    )


def test_run_experiment_produces_baseline_and_ml_artifacts(tmp_path: Path) -> None:
    config_path = _write_run_experiment_config(tmp_path)

    result = run_marketlab_cli("run-experiment", config_path)
    assert_command_ok(result)

    run_root = tmp_path / "runs" / "integration_fixture"
    run_dir = latest_run_dir(run_root)

    assert stdout_path(result) == run_dir.resolve()
    assert {path.name for path in run_dir.iterdir()} == {
        "metrics.csv",
        "performance.csv",
        "report.md",
        "cumulative_returns.png",
        "drawdown.png",
        "model_summary.csv",
        "fold_summary.csv",
        "models",
    }

    metrics = pd.read_csv(run_dir / "metrics.csv")
    performance = pd.read_csv(run_dir / "performance.csv", parse_dates=["date"])
    model_summary = pd.read_csv(run_dir / "model_summary.csv")
    fold_summary = pd.read_csv(run_dir / "fold_summary.csv")
    report_text = (run_dir / "report.md").read_text(encoding="utf-8")

    expected_strategies = {
        "buy_hold",
        "sma",
        "ml_logistic_regression",
        "ml_random_forest",
    }
    assert list(metrics.columns) == EXPECTED_METRICS_COLUMNS
    assert list(performance.columns) == PERFORMANCE_COLUMNS
    assert list(model_summary.columns) == MODEL_SUMMARY_COLUMNS
    assert list(fold_summary.columns) == FOLD_SUMMARY_COLUMNS
    assert set(metrics["strategy"]) == expected_strategies
    assert set(performance["strategy"]) == expected_strategies
    assert set(model_summary["model_name"]) == {"logistic_regression", "random_forest"}
    assert not fold_summary.empty
    assert (run_dir / "models").is_dir()

    date_sequences = {
        strategy: tuple(frame["date"].tolist())
        for strategy, frame in performance.groupby("strategy", sort=False)
    }
    first_sequence = next(iter(date_sequences.values()))
    assert all(sequence == first_sequence for sequence in date_sequences.values())

    for _, strategy_frame in performance.groupby("strategy", sort=False):
        expected_equity = (1.0 + strategy_frame["net_return"]).cumprod()
        assert strategy_frame["equity"].tolist() == pytest.approx(expected_equity.tolist())

    assert "## Strategy Metrics" in report_text
    assert "## Model Summary" in report_text
    assert "## Fold Summary" in report_text
    assert "## Headline Outcomes" in report_text
    assert "Phase 2 baseline plus ML experiment" in report_text
    assert "ml_logistic_regression" in report_text


def test_backtest_remains_baseline_only(tmp_path: Path) -> None:
    config_path = _write_backtest_config(tmp_path)

    result = run_marketlab_cli("backtest", config_path)
    assert_command_ok(result)

    run_root = tmp_path / "runs" / "integration_backtest_fixture"
    run_dir = latest_run_dir(run_root)

    assert stdout_path(result) == run_dir.resolve()
    assert {path.name for path in run_dir.iterdir()} == {
        "metrics.csv",
        "performance.csv",
        "report.md",
        "cumulative_returns.png",
        "drawdown.png",
    }

    metrics = pd.read_csv(run_dir / "metrics.csv")
    performance = pd.read_csv(run_dir / "performance.csv")

    assert list(metrics.columns) == EXPECTED_METRICS_COLUMNS
    assert list(performance.columns) == PERFORMANCE_COLUMNS
    assert set(metrics["strategy"]) == {"buy_hold", "sma"}
    assert set(performance["strategy"]) == {"buy_hold", "sma"}
    assert not (run_dir / "model_summary.csv").exists()
    assert not (run_dir / "fold_summary.csv").exists()
