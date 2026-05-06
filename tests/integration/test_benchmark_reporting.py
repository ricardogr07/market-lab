from __future__ import annotations

from pathlib import Path

import pandas as pd
from tests.integration import _cli_harness
from tests.integration.test_run_experiment import (
    _write_backtest_config,
    _write_run_experiment_config,
    assert_command_ok,
    latest_run_dir,
    run_marketlab_cli,
    stdout_path,
    write_yaml_config,
)

from marketlab.data.panel import save_panel_csv

BENCHMARK_RELATIVE_COLUMNS = _cli_harness.BENCHMARK_RELATIVE_COLUMNS
STRATEGY_SUMMARY_COLUMNS = _cli_harness.STRATEGY_SUMMARY_COLUMNS



def test_backtest_writes_benchmark_relative_artifacts(tmp_path: Path) -> None:
    config_path = _write_backtest_config(
        tmp_path,
        evaluation={"benchmark_strategy": "buy_hold"},
    )

    result = run_marketlab_cli("backtest", config_path)
    assert_command_ok(result)

    run_root = tmp_path / "runs" / "integration_backtest_fixture"
    run_dir = latest_run_dir(run_root)
    benchmark_relative_path = run_dir / "benchmark_relative.csv"
    strategy_summary_path = run_dir / "strategy_summary.csv"
    report_path = run_dir / "report.md"

    assert stdout_path(result) == run_dir.resolve()
    assert benchmark_relative_path.exists()

    benchmark_relative = pd.read_csv(benchmark_relative_path)
    strategy_summary = pd.read_csv(strategy_summary_path)
    report_text = report_path.read_text(encoding="utf-8")

    assert list(benchmark_relative.columns) == BENCHMARK_RELATIVE_COLUMNS
    assert list(strategy_summary.columns) == STRATEGY_SUMMARY_COLUMNS
    assert set(benchmark_relative["strategy"]) == {"buy_hold", "sma"}
    assert benchmark_relative["benchmark_strategy"].eq("buy_hold").all()
    assert strategy_summary["benchmark_strategy"].eq("buy_hold").all()
    assert "## Benchmark-Relative Summary" in report_text
    assert "benchmark_relative.csv" in report_text



def test_run_experiment_writes_benchmark_relative_artifacts(tmp_path: Path) -> None:
    config_path = _write_run_experiment_config(
        tmp_path,
        models=[{"name": "logistic_regression"}],
        evaluation={"benchmark_strategy": "buy_hold"},
    )

    result = run_marketlab_cli("run-experiment", config_path)
    assert_command_ok(result)

    run_root = tmp_path / "runs" / "integration_fixture"
    run_dir = latest_run_dir(run_root)
    benchmark_relative_path = run_dir / "benchmark_relative.csv"
    strategy_summary_path = run_dir / "strategy_summary.csv"
    report_path = run_dir / "report.md"

    assert benchmark_relative_path.exists()

    benchmark_relative = pd.read_csv(benchmark_relative_path)
    strategy_summary = pd.read_csv(strategy_summary_path)
    report_text = report_path.read_text(encoding="utf-8")

    assert list(benchmark_relative.columns) == BENCHMARK_RELATIVE_COLUMNS
    assert list(strategy_summary.columns) == STRATEGY_SUMMARY_COLUMNS
    assert set(strategy_summary["strategy"]) == {
        "buy_hold",
        "sma",
        "ml_logistic_regression",
    }
    assert set(benchmark_relative["strategy"]) == {
        "buy_hold",
        "sma",
        "ml_logistic_regression",
    }
    assert benchmark_relative["benchmark_strategy"].eq("buy_hold").all()
    assert strategy_summary["benchmark_strategy"].eq("buy_hold").all()
    assert "## Benchmark-Relative Summary" in report_text



def test_backtest_rejects_unknown_benchmark_strategy(tmp_path: Path) -> None:
    config_path = _write_backtest_config(
        tmp_path,
        evaluation={"benchmark_strategy": "missing"},
    )

    result = run_marketlab_cli("backtest", config_path)

    assert result.returncode != 0
    combined_output = f"{result.stdout}\n{result.stderr}"
    assert "evaluation.benchmark_strategy='missing'" in combined_output
    assert "Available strategies: buy_hold, sma" in combined_output


def test_crypto_hourly_indicator_backtest_writes_buy_hold_comparison(
    tmp_path: Path,
) -> None:
    cache_dir = tmp_path / "cache"
    fixture_panel = pd.read_csv(
        Path("tests/fixtures/crypto_hourly_panel.csv"),
        parse_dates=["timestamp"],
    )
    save_panel_csv(fixture_panel, cache_dir / "panel.csv")
    config_path = write_yaml_config(
        tmp_path / "crypto_hourly.yaml",
        {
            "experiment_name": "crypto_hourly_fixture",
            "data": {
                "symbols": ["BTC-USD"],
                "start_date": "2024-01-01",
                "end_date": "2024-01-02",
                "interval": "1h",
                "cache_dir": str(cache_dir),
                "prepared_panel_filename": "panel.csv",
            },
            "features": {
                "return_windows": [1, 2],
                "ma_windows": [2, 3],
                "vol_windows": [2],
                "momentum_window": 2,
            },
            "portfolio": {
                "ranking": {
                    "long_n": 1,
                    "short_n": 1,
                    "rebalance_frequency": "bar",
                    "mode": "long_only",
                    "cash_when_underfilled": True,
                },
                "costs": {"bps_per_trade": 20},
            },
            "baselines": {
                "buy_hold": True,
                "sma": {"enabled": False},
                "indicator_stack": {
                    "enabled": True,
                    "ema_fast_window": 2,
                    "ema_slow_window": 4,
                    "rsi_window": 3,
                    "rsi_min": 40.0,
                    "rsi_max": 100.0,
                    "macd_fast_window": 2,
                    "macd_slow_window": 5,
                    "macd_signal_window": 2,
                    "bollinger_window": 3,
                    "bollinger_std": 0.5,
                    "bollinger_mode": "breakout",
                    "volume_window": 2,
                    "volume_multiplier": 1.0,
                    "use_vwap": True,
                    "vwap_window": 3,
                    "min_confirmations": 3,
                },
            },
            "models": [],
            "evaluation": {
                "benchmark_strategy": "buy_hold",
                "periods_per_year": 8760,
                "cost_sensitivity_bps": [5.0, 20.0, 40.0],
            },
            "artifacts": {
                "output_dir": str(tmp_path / "runs"),
                "save_predictions": False,
                "save_metrics_csv": True,
                "save_report_md": True,
                "save_plots": False,
            },
        },
    )

    result = run_marketlab_cli("backtest", config_path)
    assert_command_ok(result)

    run_root = tmp_path / "runs" / "crypto_hourly_fixture"
    run_dir = latest_run_dir(run_root)
    benchmark_relative = pd.read_csv(run_dir / "benchmark_relative.csv")
    strategy_summary = pd.read_csv(run_dir / "strategy_summary.csv")
    report_text = (run_dir / "report.md").read_text(encoding="utf-8")

    assert stdout_path(result) == run_dir.resolve()
    assert set(strategy_summary["strategy"]) == {"buy_hold", "indicator_stack"}
    assert set(benchmark_relative["strategy"]) == {"buy_hold", "indicator_stack"}
    assert benchmark_relative["benchmark_strategy"].eq("buy_hold").all()
    assert strategy_summary["benchmark_strategy"].eq("buy_hold").all()
    assert "## Trend-Signal Acceptance Gate" in report_text
    assert "`indicator_stack` gate:" in report_text


def test_crypto_15m_visual_signal_backtest_writes_focused_artifacts(
    tmp_path: Path,
) -> None:
    cache_dir = tmp_path / "cache"
    fixture_panel = pd.read_csv(
        Path("tests/fixtures/crypto_15m_panel.csv"),
        parse_dates=["timestamp"],
    )
    save_panel_csv(fixture_panel, cache_dir / "panel.csv")
    config_path = write_yaml_config(
        tmp_path / "crypto_15m.yaml",
        {
            "experiment_name": "crypto_15m_visual_fixture",
            "data": {
                "symbols": ["BTC-USD"],
                "start_date": "2024-01-01",
                "end_date": "2024-01-01",
                "interval": "15m",
                "cache_dir": str(cache_dir),
                "prepared_panel_filename": "panel.csv",
            },
            "features": {
                "return_windows": [1, 2],
                "ma_windows": [2, 3],
                "vol_windows": [2],
                "momentum_window": 2,
            },
            "portfolio": {
                "ranking": {
                    "long_n": 1,
                    "short_n": 1,
                    "rebalance_frequency": "bar",
                    "mode": "long_only",
                    "cash_when_underfilled": True,
                },
                "costs": {"bps_per_trade": 20},
            },
            "baselines": {
                "buy_hold": True,
                "sma": {"enabled": False},
                "indicator_stack": {
                    "enabled": True,
                    "ema_fast_window": 2,
                    "ema_slow_window": 4,
                    "rsi_window": 3,
                    "rsi_min": 40.0,
                    "rsi_max": 100.0,
                    "macd_fast_window": 2,
                    "macd_slow_window": 5,
                    "macd_signal_window": 2,
                    "bollinger_window": 3,
                    "bollinger_std": 0.5,
                    "bollinger_mode": "breakout",
                    "volume_window": 2,
                    "volume_multiplier": 1.0,
                    "use_vwap": True,
                    "vwap_window": 3,
                    "min_confirmations": 3,
                },
            },
            "models": [],
            "evaluation": {
                "benchmark_strategy": "buy_hold",
                "cost_sensitivity_bps": [5.0, 20.0, 40.0],
                "focus_start": "2024-01-01 01:00:00",
                "focus_end": "2024-01-01 03:00:00",
                "visualize_signals": True,
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

    result = run_marketlab_cli("backtest", config_path)
    assert_command_ok(result)

    run_root = tmp_path / "runs" / "crypto_15m_visual_fixture"
    run_dir = latest_run_dir(run_root)
    diagnostics_path = run_dir / "indicator_diagnostics.csv"
    benchmark_relative_path = run_dir / "benchmark_relative.csv"
    report_path = run_dir / "report.md"
    plot_paths = [
        run_dir / "signal_price_overlay.png",
        run_dir / "signal_confirmations.png",
        run_dir / "signal_performance_focus.png",
    ]

    diagnostics = pd.read_csv(diagnostics_path, parse_dates=["timestamp", "effective_date"])
    report_text = report_path.read_text(encoding="utf-8")

    assert benchmark_relative_path.exists()
    assert len(diagnostics) == 23
    assert diagnostics["timestamp"].min() == pd.Timestamp("2024-01-01 00:00:00")
    assert diagnostics["effective_date"].min() == pd.Timestamp("2024-01-01 00:15:00")
    assert "target_weight" in diagnostics.columns
    for plot_path in plot_paths:
        assert plot_path.exists()
        assert plot_path.stat().st_size > 0
        assert plot_path.name in report_text
    assert "## Signal Inspection" in report_text
    assert "indicator_diagnostics.csv" in report_text
    assert "benchmark_relative.csv" in report_text


def test_crypto_15m_chart_pattern_backtest_writes_pattern_artifacts(
    tmp_path: Path,
) -> None:
    cache_dir = tmp_path / "cache"
    fixture_panel = pd.read_csv(
        Path("tests/fixtures/crypto_15m_panel.csv"),
        parse_dates=["timestamp"],
    )
    save_panel_csv(fixture_panel, cache_dir / "panel.csv")
    config_path = write_yaml_config(
        tmp_path / "crypto_patterns.yaml",
        {
            "experiment_name": "crypto_15m_patterns_fixture",
            "data": {
                "symbols": ["BTC-USD"],
                "start_date": "2024-01-01",
                "end_date": "2024-01-01",
                "interval": "15m",
                "cache_dir": str(cache_dir),
                "prepared_panel_filename": "panel.csv",
            },
            "features": {
                "return_windows": [1, 2],
                "ma_windows": [2, 3],
                "vol_windows": [2],
                "momentum_window": 2,
            },
            "portfolio": {
                "ranking": {
                    "long_n": 1,
                    "short_n": 1,
                    "rebalance_frequency": "bar",
                    "mode": "long_only",
                    "cash_when_underfilled": True,
                },
                "costs": {"bps_per_trade": 20},
            },
            "baselines": {
                "buy_hold": True,
                "sma": {"enabled": False},
                "chart_patterns": {
                    "enabled": True,
                    "lookback_bars": 8,
                    "level_tolerance_pct": 0.03,
                    "breakout_pct": 0.001,
                    "rectangle_max_range_pct": 0.08,
                    "flag_pole_bars": 3,
                    "flag_consolidation_bars": 3,
                    "flag_min_pole_return": 0.01,
                    "flag_max_retrace_pct": 0.03,
                    "volume_window": 2,
                    "volume_multiplier": 1.0,
                    "min_bullish_patterns": 1,
                },
            },
            "models": [],
            "evaluation": {
                "benchmark_strategy": "buy_hold",
                "focus_start": "2024-01-01 00:00:00",
                "focus_end": "2024-01-01 05:45:00",
                "visualize_signals": True,
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

    result = run_marketlab_cli("backtest", config_path)
    assert_command_ok(result)

    run_root = tmp_path / "runs" / "crypto_15m_patterns_fixture"
    run_dir = latest_run_dir(run_root)
    diagnostics_path = run_dir / "pattern_diagnostics.csv"
    report_path = run_dir / "report.md"
    plot_paths = [
        run_dir / "pattern_price_overlay.png",
        run_dir / "pattern_detections.png",
        run_dir / "pattern_detection_windows.png",
        run_dir / "pattern_performance_focus.png",
    ]

    diagnostics = pd.read_csv(diagnostics_path, parse_dates=["timestamp", "effective_date"])
    report_text = report_path.read_text(encoding="utf-8")

    assert set(pd.read_csv(run_dir / "strategy_summary.csv")["strategy"]) == {
        "buy_hold",
        "chart_patterns",
    }
    assert len(diagnostics) == 23
    assert "bull_flag_breakout" in diagnostics.columns
    assert "target_weight" in diagnostics.columns
    for plot_path in plot_paths:
        assert plot_path.exists()
        assert plot_path.stat().st_size > 0
        assert plot_path.name in report_text
    assert "pattern_diagnostics.csv" in report_text
    assert "## Signal Inspection" in report_text


def test_crypto_pattern_meta_label_backtest_writes_overlay_artifacts(
    tmp_path: Path,
) -> None:
    cache_dir = tmp_path / "cache"
    fixture_panel = pd.read_csv(
        Path("tests/fixtures/crypto_15m_panel.csv"),
        parse_dates=["timestamp"],
    )
    save_panel_csv(fixture_panel, cache_dir / "panel.csv")
    config_path = write_yaml_config(
        tmp_path / "crypto_pattern_meta.yaml",
        {
            "experiment_name": "crypto_pattern_meta_fixture",
            "data": {
                "symbols": ["BTC-USD"],
                "start_date": "2024-01-01",
                "end_date": "2024-01-01",
                "interval": "15m",
                "cache_dir": str(cache_dir),
                "prepared_panel_filename": "panel.csv",
            },
            "features": {
                "return_windows": [1, 2],
                "ma_windows": [2, 3],
                "vol_windows": [2],
                "momentum_window": 2,
            },
            "portfolio": {
                "ranking": {
                    "long_n": 1,
                    "short_n": 1,
                    "rebalance_frequency": "bar",
                    "mode": "long_only",
                    "cash_when_underfilled": True,
                },
                "costs": {"bps_per_trade": 10},
            },
            "baselines": {
                "buy_hold": True,
                "sma": {"enabled": False},
                "chart_patterns": {
                    "enabled": True,
                    "lookback_bars": 8,
                    "level_tolerance_pct": 0.02,
                    "breakout_pct": 0.001,
                    "rectangle_max_range_pct": 0.08,
                    "flag_pole_bars": 3,
                    "flag_consolidation_bars": 3,
                    "flag_min_pole_return": 0.01,
                    "flag_max_retrace_pct": 0.03,
                    "volume_window": 2,
                    "volume_multiplier": 1.0,
                    "min_bullish_patterns": 1,
                },
                "pattern_exit_overlay": {
                    "enabled": True,
                    "min_bearish_patterns": 1,
                    "min_bullish_reentry_patterns": 1,
                    "trend_ema_window": 3,
                    "reentry_clear_bars": 1,
                    "require_price_below_trend_for_exit": False,
                    "bearish_confirmation_window_bars": 1,
                    "min_cash_bars": 0,
                    "exit_cooldown_bars": 0,
                },
                "pattern_meta_label": {
                    "enabled": True,
                    "label_horizon_bars": 2,
                    "exit_probability_threshold": 0.55,
                    "exit_probability_threshold_grid": [0.45, 0.55],
                    "tuning_mode": "nested_walk_forward",
                    "min_oos_exit_count": 0,
                    "max_average_exposure_for_active": 1.0,
                    "models": ["logistic_l1"],
                },
                "pattern_partial_exposure_overlay": {
                    "enabled": True,
                    "partial_weight": 0.5,
                    "partial_exit_probability_threshold_grid": [0.45],
                    "full_exit_probability_threshold_grid": [0.55],
                },
            },
            "models": [],
            "evaluation": {
                "benchmark_strategy": "buy_hold",
                "periods_per_year": 35040,
                "walk_forward": {
                    "train_years": 1,
                    "test_months": 1,
                    "step_months": 1,
                    "min_train_rows": 1,
                    "min_test_rows": 1,
                },
            },
            "artifacts": {
                "output_dir": str(tmp_path / "runs"),
                "save_predictions": False,
                "save_metrics_csv": True,
                "save_report_md": True,
                "save_plots": False,
            },
        },
    )

    result = run_marketlab_cli("backtest", config_path)
    assert_command_ok(result)

    run_dir = latest_run_dir(tmp_path / "runs" / "crypto_pattern_meta_fixture")
    strategy_summary = pd.read_csv(run_dir / "strategy_summary.csv")
    threshold_sweep = pd.read_csv(run_dir / "pattern_meta_threshold_sweep.csv")
    report_text = (run_dir / "report.md").read_text(encoding="utf-8")

    assert {
        "buy_hold",
        "chart_patterns",
        "pattern_exit_overlay",
        "pattern_meta_label_exit_overlay",
        "pattern_partial_exposure_overlay",
    }.issubset(set(strategy_summary["strategy"]))
    for filename in [
        "pattern_diagnostics.csv",
        "pattern_exit_overlay_diagnostics.csv",
        "pattern_meta_labels.csv",
        "pattern_meta_predictions.csv",
        "pattern_meta_fold_diagnostics.csv",
        "pattern_meta_threshold_sweep.csv",
        "pattern_meta_tuning_candidates.csv",
        "pattern_meta_tuning_selections.csv",
        "pattern_partial_exposure_diagnostics.csv",
        "pattern_partial_threshold_sweep.csv",
        "benchmark_relative.csv",
    ]:
        assert (run_dir / filename).exists()
    assert "Pattern exit overlay diagnostics" in report_text
    assert "Pattern meta predictions" in report_text
    assert "## Pattern Meta Threshold Sweep" in report_text
    assert "## Pattern Meta Tuning" in report_text
    assert "Pattern partial exposure diagnostics" in report_text
    assert {"exit_count", "cash_bar_count", "average_exposure"}.issubset(
        set(threshold_sweep.columns)
    )
    assert "`pattern_exit_overlay` gate:" in report_text
    assert "`pattern_meta_label_exit_overlay` gate:" in report_text
