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
)

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
