from __future__ import annotations

from pathlib import Path

from tests.integration import _cli_harness

from marketlab.data.panel import save_panel_csv

assert_command_ok = _cli_harness.assert_command_ok
build_synthetic_panel = _cli_harness.build_synthetic_panel
load_fixture_panel = _cli_harness.load_fixture_panel
run_marketlab_cli = getattr(
    _cli_harness,
    "run_marketlab_cli",
    _cli_harness.run_launcher_command,
)
write_yaml_config = _cli_harness.write_yaml_config

ERROR_MESSAGE = "baselines.optimized is configured, but optimized baseline methods are not implemented yet."


def _write_backtest_config(tmp_path: Path) -> Path:
    cache_dir = tmp_path / "cache"
    save_panel_csv(load_fixture_panel(), cache_dir / "panel.csv")
    return write_yaml_config(
        tmp_path / "backtest.yaml",
        {
            "experiment_name": "optimized_scaffold_backtest",
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
                "optimized": {
                    "enabled": True,
                    "method": "mean_variance",
                },
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


def _write_run_experiment_config(tmp_path: Path) -> Path:
    cache_dir = tmp_path / "cache"
    save_panel_csv(
        build_synthetic_panel(
            (("AAA", 100.0, 0.45), ("BBB", 130.0, 0.40), ("CCC", 160.0, 0.35), ("DDD", 190.0, 0.30)),
            start_date="2020-01-01",
            end_date="2024-12-31",
        ),
        cache_dir / "panel.csv",
    )
    return write_yaml_config(
        tmp_path / "integration.yaml",
        {
            "experiment_name": "optimized_scaffold_experiment",
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
                    "mode": "long_short",
                    "min_score_threshold": 0.0,
                    "cash_when_underfilled": False,
                },
                "costs": {"bps_per_trade": 10},
            },
            "baselines": {
                "buy_hold": True,
                "sma": {"enabled": True, "fast_window": 5, "slow_window": 10},
                "optimized": {
                    "enabled": True,
                    "method": "mean_variance",
                },
            },
            "models": [{"name": "logistic_regression"}],
            "evaluation": {
                "walk_forward": {
                    "train_years": 1,
                    "test_months": 2,
                    "step_months": 2,
                },
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


def test_backtest_rejects_unimplemented_optimized_baseline(tmp_path: Path) -> None:
    config_path = _write_backtest_config(tmp_path)

    result = run_marketlab_cli("backtest", config_path)

    assert result.returncode != 0
    combined_output = f"{result.stdout}\n{result.stderr}"
    assert ERROR_MESSAGE in combined_output
    assert not (tmp_path / "runs" / "optimized_scaffold_backtest").exists()



def test_run_experiment_rejects_unimplemented_optimized_baseline(tmp_path: Path) -> None:
    config_path = _write_run_experiment_config(tmp_path)

    result = run_marketlab_cli("run-experiment", config_path)

    assert result.returncode != 0
    combined_output = f"{result.stdout}\n{result.stderr}"
    assert ERROR_MESSAGE in combined_output
    assert not (tmp_path / "runs" / "optimized_scaffold_experiment").exists()
