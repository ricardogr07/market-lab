from pathlib import Path

import pandas as pd
import pytest
from tests.integration import _cli_harness
from tests.integration.test_run_experiment import (
    _write_backtest_config,
    _write_run_experiment_config,
    assert_command_ok,
    latest_run_dir,
    run_marketlab_cli,
)

COST_SENSITIVITY_COLUMNS = _cli_harness.COST_SENSITIVITY_COLUMNS


def test_backtest_writes_cost_sensitivity_artifact(tmp_path: Path) -> None:
    config_path = _write_backtest_config(tmp_path)

    result = run_marketlab_cli("backtest", config_path)
    assert_command_ok(result)

    run_dir = latest_run_dir(tmp_path / "runs" / "integration_backtest_fixture")
    cost_sensitivity = pd.read_csv(run_dir / "cost_sensitivity.csv")
    strategy_summary = pd.read_csv(run_dir / "strategy_summary.csv")
    report_text = (run_dir / "report.md").read_text(encoding="utf-8")

    assert list(cost_sensitivity.columns) == COST_SENSITIVITY_COLUMNS
    assert set(cost_sensitivity["strategy"]) == {"buy_hold", "sma"}
    assert set(cost_sensitivity["bps_per_trade"]) == {0.0, 10.0}
    assert "## Cost Sensitivity" in report_text

    base_rows = cost_sensitivity.loc[cost_sensitivity["bps_per_trade"] == 10.0]
    merged = base_rows.merge(
        strategy_summary.loc[:, ["strategy", "cumulative_return", "cost_drag"]],
        on="strategy",
        suffixes=("_scenario", "_summary"),
        how="inner",
        validate="one_to_one",
    )
    assert merged["cumulative_return_scenario"].tolist() == pytest.approx(
        merged["cumulative_return_summary"].tolist()
    )
    assert merged["cost_drag_scenario"].tolist() == pytest.approx(
        merged["cost_drag_summary"].tolist()
    )


def test_run_experiment_writes_cost_sensitivity_grid(tmp_path: Path) -> None:
    config_path = _write_run_experiment_config(
        tmp_path,
        models=[{"name": "logistic_regression"}],
        evaluation={"cost_sensitivity_bps": [5.0, 25.0]},
    )

    result = run_marketlab_cli("run-experiment", config_path)
    assert_command_ok(result)

    run_dir = latest_run_dir(tmp_path / "runs" / "integration_fixture")
    cost_sensitivity = pd.read_csv(run_dir / "cost_sensitivity.csv")
    strategy_summary = pd.read_csv(run_dir / "strategy_summary.csv")
    report_text = (run_dir / "report.md").read_text(encoding="utf-8")

    assert list(cost_sensitivity.columns) == COST_SENSITIVITY_COLUMNS
    assert set(cost_sensitivity["strategy"]) == {
        "buy_hold",
        "sma",
        "ml_logistic_regression",
    }
    assert set(cost_sensitivity["bps_per_trade"]) == {0.0, 5.0, 10.0, 25.0}
    assert "## Cost Sensitivity" in report_text

    for _, strategy_rows in cost_sensitivity.groupby("strategy", sort=False):
        ordered = strategy_rows.sort_values("bps_per_trade").reset_index(drop=True)
        assert ordered["total_cost_return"].is_monotonic_increasing
        assert ordered["cost_drag"].is_monotonic_increasing
        assert ordered.loc[0, "cumulative_return"] >= ordered.loc[len(ordered) - 1, "cumulative_return"]

    base_rows = cost_sensitivity.loc[cost_sensitivity["bps_per_trade"] == 10.0]
    merged = base_rows.merge(
        strategy_summary.loc[:, ["strategy", "cumulative_return", "cost_drag"]],
        on="strategy",
        suffixes=("_scenario", "_summary"),
        how="inner",
        validate="one_to_one",
    )
    assert merged["cumulative_return_scenario"].tolist() == pytest.approx(
        merged["cumulative_return_summary"].tolist()
    )
    assert merged["cost_drag_scenario"].tolist() == pytest.approx(
        merged["cost_drag_summary"].tolist()
    )
