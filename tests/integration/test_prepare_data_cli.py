from __future__ import annotations

from pathlib import Path

from tests.integration._cli_harness import (
    DEFAULT_FIXTURE_SYMBOLS,
    assert_command_ok,
    printed_path,
    run_launcher_command,
    write_raw_symbol_cache,
    write_yaml_config,
)

from marketlab.data.panel import PANEL_COLUMNS, load_panel_csv


def _write_prepare_data_config(tmp_path: Path) -> tuple[Path, Path]:
    cache_dir = tmp_path / "cache"
    write_raw_symbol_cache(cache_dir, symbols=DEFAULT_FIXTURE_SYMBOLS)

    config_path = write_yaml_config(
        tmp_path / "prepare-data.yaml",
        {
            "experiment_name": "integration_prepare_data_fixture",
            "data": {
                "symbols": DEFAULT_FIXTURE_SYMBOLS,
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
            "target": {
                "horizon_days": 5,
                "type": "direction",
            },
            "portfolio": {
                "ranking": {
                    "rebalance_frequency": "W-FRI",
                    "long_n": 2,
                    "short_n": 2,
                    "weighting": "equal",
                },
                "costs": {"bps_per_trade": 10},
            },
            "baselines": {
                "buy_hold": True,
                "sma": {"enabled": True, "fast_window": 2, "slow_window": 3},
            },
            "models": [],
            "evaluation": {
                "walk_forward": {
                    "train_years": 1,
                    "test_months": 2,
                    "step_months": 2,
                }
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
    return config_path, cache_dir / "panel.csv"


def test_prepare_data_cli_builds_and_reuses_cached_panel(tmp_path: Path) -> None:
    config_path, prepared_panel_path = _write_prepare_data_config(tmp_path)

    first_result = run_launcher_command("prepare-data", config_path)
    assert_command_ok(first_result)

    printed_panel_path = printed_path(first_result)
    assert printed_panel_path == prepared_panel_path.resolve()
    assert prepared_panel_path.exists()

    panel = load_panel_csv(prepared_panel_path)
    assert panel.columns.tolist() == PANEL_COLUMNS
    assert set(panel["symbol"]) == set(DEFAULT_FIXTURE_SYMBOLS)

    first_mtime = prepared_panel_path.stat().st_mtime_ns

    second_result = run_launcher_command("prepare-data", config_path)
    assert_command_ok(second_result)

    assert printed_path(second_result) == prepared_panel_path.resolve()
    assert prepared_panel_path.stat().st_mtime_ns == first_mtime
