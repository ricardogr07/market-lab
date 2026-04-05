from __future__ import annotations

import pandas as pd
from tests.integration.test_run_experiment import (
    _write_backtest_config,
    _write_run_experiment_config,
    assert_command_ok,
    latest_run_dir,
    run_marketlab_cli,
)

from marketlab.reports.analytics import DAILY_EXPOSURE_COLUMNS, GROUP_EXPOSURE_COLUMNS


def test_backtest_writes_daily_exposure_artifact(tmp_path) -> None:
    config_path = _write_backtest_config(tmp_path)

    result = run_marketlab_cli("backtest", config_path)
    assert_command_ok(result)

    run_dir = latest_run_dir(tmp_path / "runs" / "integration_backtest_fixture")
    daily_exposure = pd.read_csv(run_dir / "daily_exposure.csv", parse_dates=["date"])
    strategy_summary = pd.read_csv(run_dir / "strategy_summary.csv")
    report_text = (run_dir / "report.md").read_text(encoding="utf-8")

    assert list(daily_exposure.columns) == DAILY_EXPOSURE_COLUMNS
    assert set(daily_exposure["strategy"]) == {"buy_hold", "sma"}
    assert strategy_summary["avg_gross_exposure"].notna().all()
    assert strategy_summary["avg_engine_cash_weight"].notna().all()
    assert strategy_summary["max_group_weight"].isna().all()
    assert "## Exposure Summary" in report_text


def test_run_experiment_writes_group_exposure_for_grouped_runs(tmp_path) -> None:
    config_path = _write_run_experiment_config(
        tmp_path,
        symbol_groups={
            "AAA": "growth",
            "BBB": "growth",
            "CCC": "defensive",
            "DDD": "defensive",
        },
        allocation={
            "enabled": True,
            "mode": "group_weights",
            "group_weights": {"growth": 0.75, "defensive": 0.25},
        },
        models=[{"name": "logistic_regression"}],
    )

    result = run_marketlab_cli("run-experiment", config_path)
    assert_command_ok(result)

    run_dir = latest_run_dir(tmp_path / "runs" / "integration_fixture")
    daily_exposure = pd.read_csv(run_dir / "daily_exposure.csv", parse_dates=["date"])
    group_exposure = pd.read_csv(run_dir / "group_exposure.csv", parse_dates=["date"])
    strategy_summary = pd.read_csv(run_dir / "strategy_summary.csv")
    report_text = (run_dir / "report.md").read_text(encoding="utf-8")

    assert list(daily_exposure.columns) == DAILY_EXPOSURE_COLUMNS
    assert list(group_exposure.columns) == GROUP_EXPOSURE_COLUMNS
    assert set(group_exposure["group_name"]) == {"growth", "defensive"}
    assert set(group_exposure["strategy"]) == set(strategy_summary["strategy"])
    assert strategy_summary["max_group_weight"].notna().all()
    assert "## Exposure Summary" in report_text
    assert "group_exposure.csv" in report_text
