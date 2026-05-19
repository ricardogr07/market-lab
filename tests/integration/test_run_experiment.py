from __future__ import annotations

import math
from pathlib import Path

import pandas as pd
import pytest
from tests.integration import _cli_harness

from marketlab.config import load_config
from marketlab.data.panel import save_panel_csv
from marketlab.evaluation import build_walk_forward_folds

EXPECTED_METRICS_COLUMNS = _cli_harness.EXPECTED_METRICS_COLUMNS
MODEL_SUMMARY_COLUMNS = _cli_harness.MODEL_SUMMARY_COLUMNS
FOLD_SUMMARY_COLUMNS = _cli_harness.FOLD_SUMMARY_COLUMNS
FOLD_DIAGNOSTICS_COLUMNS = _cli_harness.FOLD_DIAGNOSTICS_COLUMNS
RANKING_DIAGNOSTICS_COLUMNS = _cli_harness.RANKING_DIAGNOSTICS_COLUMNS
CALIBRATION_DIAGNOSTICS_COLUMNS = _cli_harness.CALIBRATION_DIAGNOSTICS_COLUMNS
SCORE_HISTOGRAM_COLUMNS = _cli_harness.SCORE_HISTOGRAM_COLUMNS
THRESHOLD_DIAGNOSTICS_COLUMNS = _cli_harness.THRESHOLD_DIAGNOSTICS_COLUMNS
STRATEGY_SUMMARY_COLUMNS = _cli_harness.STRATEGY_SUMMARY_COLUMNS
MONTHLY_RETURNS_COLUMNS = _cli_harness.MONTHLY_RETURNS_COLUMNS
TURNOVER_COSTS_COLUMNS = _cli_harness.TURNOVER_COSTS_COLUMNS
COST_SENSITIVITY_COLUMNS = _cli_harness.COST_SENSITIVITY_COLUMNS
FACTOR_DIAGNOSTICS_COLUMNS = _cli_harness.FACTOR_DIAGNOSTICS_COLUMNS
COVARIANCE_DIAGNOSTICS_COLUMNS = _cli_harness.COVARIANCE_DIAGNOSTICS_COLUMNS
assert_command_ok = _cli_harness.assert_command_ok
build_synthetic_panel = _cli_harness.build_synthetic_panel
latest_run_dir = _cli_harness.latest_run_dir
load_fixture_panel = _cli_harness.load_fixture_panel
write_factor_model_csv = _cli_harness.write_factor_model_csv
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

DEFAULT_MODEL_NAMES = {
    "extra_trees",
    "gradient_boosting",
    "hist_gradient_boosting",
    "logistic_l1",
    "logistic_regression",
    "random_forest",
}


def _write_factor_model(
    tmp_path: Path,
    dates: pd.Series | pd.Index | list[pd.Timestamp],
) -> Path:
    ordered_dates = pd.Index(pd.to_datetime(dates)).drop_duplicates().sort_values()
    frame = pd.DataFrame(
        {
            "date": ordered_dates,
            "MKT": [0.001 + (0.0002 * index) for index in range(len(ordered_dates))],
            "VALUE": [0.0005 * ((-1) ** index) for index in range(len(ordered_dates))],
        }
    )
    return write_factor_model_csv(tmp_path / "inputs" / "factor_returns.csv", frame)


def _write_run_experiment_config(
    tmp_path: Path,
    *,
    walk_forward: dict[str, int | float] | None = None,
    ranking: dict[str, object] | None = None,
    risk: dict[str, object] | None = None,
    symbol_specs: tuple[tuple[str, float, float], ...] | None = None,
    symbol_groups: dict[str, str] | None = None,
    allocation: dict[str, object] | None = None,
    partial_allocation_benchmarks: dict[str, object] | None = None,
    rebalanced_partial_allocation_benchmarks: dict[str, object] | None = None,
    indicator_stack: dict[str, object] | None = None,
    optimized: dict[str, object] | None = None,
    models: list[dict[str, str]] | None = None,
    features: dict[str, object] | None = None,
    target: dict[str, object] | None = None,
    evaluation: dict[str, object] | None = None,
) -> Path:
    cache_dir = tmp_path / "cache"
    resolved_symbol_specs = symbol_specs or (
        ("AAA", 100.0, 0.45),
        ("BBB", 130.0, 0.40),
        ("CCC", 160.0, 0.35),
        ("DDD", 190.0, 0.30),
    )
    resolved_symbol_groups = symbol_groups or {}
    resolved_models = models or [
        {"name": "logistic_regression"},
        {"name": "logistic_l1"},
        {"name": "random_forest"},
        {"name": "extra_trees"},
        {"name": "gradient_boosting"},
        {"name": "hist_gradient_boosting"},
    ]
    save_panel_csv(
        build_synthetic_panel(
            resolved_symbol_specs,
            start_date="2020-01-01",
            end_date="2024-12-31",
        ),
        cache_dir / "panel.csv",
    )

    walk_forward_payload: dict[str, int | float] = {
        "train_years": 1,
        "test_months": 2,
        "step_months": 2,
    }
    if walk_forward is not None:
        walk_forward_payload.update(walk_forward)

    ranking_payload: dict[str, object] = {
        "long_n": 2,
        "short_n": 2,
        "rebalance_frequency": "W-FRI",
        "weighting": "equal",
        "mode": "long_short",
        "min_score_threshold": 0.0,
        "cash_when_underfilled": False,
    }
    if ranking is not None:
        ranking_payload.update(ranking)

    baselines_payload: dict[str, object] = {
        "buy_hold": True,
        "sma": {"enabled": True, "fast_window": 5, "slow_window": 10},
    }
    if allocation is not None:
        baselines_payload["allocation"] = allocation
    if partial_allocation_benchmarks is not None:
        baselines_payload["partial_allocation_benchmarks"] = partial_allocation_benchmarks
    if rebalanced_partial_allocation_benchmarks is not None:
        baselines_payload["rebalanced_partial_allocation_benchmarks"] = (
            rebalanced_partial_allocation_benchmarks
        )
    if indicator_stack is not None:
        baselines_payload["indicator_stack"] = indicator_stack
    if optimized is not None:
        baselines_payload["optimized"] = optimized

    return write_yaml_config(
        tmp_path / "integration.yaml",
        {
            "experiment_name": "integration_fixture",
            "data": {
                "symbols": [symbol for symbol, _, _ in resolved_symbol_specs],
                "start_date": "2020-01-01",
                "end_date": "2024-12-31",
                "interval": "1d",
                "cache_dir": str(cache_dir),
                "prepared_panel_filename": "panel.csv",
                "symbol_groups": resolved_symbol_groups,
            },
            "features": features or {
                "return_windows": [5, 10],
                "ma_windows": [5, 10],
                "vol_windows": [5],
                "momentum_window": 10,
            },
            "target": target or {
                "horizon_days": 5,
                "type": "direction",
            },
            "portfolio": {
                "ranking": ranking_payload,
                "risk": risk or {},
                "costs": {"bps_per_trade": 10},
            },
            "baselines": baselines_payload,
            "models": resolved_models,
            "evaluation": {
                "walk_forward": walk_forward_payload,
                **(evaluation or {}),
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


def _write_backtest_config(
    tmp_path: Path,
    *,
    symbol_groups: dict[str, str] | None = None,
    allocation: dict[str, object] | None = None,
    optimized: dict[str, object] | None = None,
    evaluation: dict[str, object] | None = None,
) -> Path:
    cache_dir = tmp_path / "cache"
    save_panel_csv(load_fixture_panel(), cache_dir / "panel.csv")
    resolved_symbol_groups = symbol_groups or {}
    baselines_payload: dict[str, object] = {
        "buy_hold": True,
        "sma": {"enabled": True, "fast_window": 2, "slow_window": 3},
    }
    if allocation is not None:
        baselines_payload["allocation"] = allocation
    if optimized is not None:
        baselines_payload["optimized"] = optimized

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
                "symbol_groups": resolved_symbol_groups,
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
            "baselines": baselines_payload,
            "evaluation": evaluation or {},
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
        "strategy_summary.csv",
        "monthly_returns.csv",
        "turnover_costs.csv",
        "cost_sensitivity.csv",
        "daily_exposure.csv",
        "report.md",
        "cumulative_returns.png",
        "drawdown.png",
        "turnover.png",
        "calibration_curves.png",
        "score_histograms.png",
        "threshold_sweeps.png",
        "fold_diagnostics.csv",
        "ranking_diagnostics.csv",
        "calibration_diagnostics.csv",
        "score_histograms.csv",
        "threshold_diagnostics.csv",
        "model_summary.csv",
        "fold_summary.csv",
        "models",
    }
    metrics = pd.read_csv(run_dir / "metrics.csv")
    performance = pd.read_csv(run_dir / "performance.csv", parse_dates=["date"])
    strategy_summary = pd.read_csv(run_dir / "strategy_summary.csv", parse_dates=["start_date", "end_date"])
    monthly_returns = pd.read_csv(run_dir / "monthly_returns.csv")
    turnover_costs = pd.read_csv(run_dir / "turnover_costs.csv", parse_dates=["date"])
    cost_sensitivity = pd.read_csv(run_dir / "cost_sensitivity.csv")
    fold_diagnostics = pd.read_csv(run_dir / "fold_diagnostics.csv")
    ranking_diagnostics = pd.read_csv(run_dir / "ranking_diagnostics.csv")
    calibration_diagnostics = pd.read_csv(run_dir / "calibration_diagnostics.csv")
    score_histograms = pd.read_csv(run_dir / "score_histograms.csv")
    threshold_diagnostics = pd.read_csv(run_dir / "threshold_diagnostics.csv")
    model_summary = pd.read_csv(run_dir / "model_summary.csv")
    fold_summary = pd.read_csv(run_dir / "fold_summary.csv")
    report_text = (run_dir / "report.md").read_text(encoding="utf-8")

    expected_strategies = {
        "buy_hold",
        "sma",
        "ml_logistic_regression",
        "ml_logistic_l1",
        "ml_random_forest",
        "ml_extra_trees",
        "ml_gradient_boosting",
        "ml_hist_gradient_boosting",
    }
    assert list(metrics.columns) == EXPECTED_METRICS_COLUMNS
    assert list(performance.columns) == PERFORMANCE_COLUMNS
    assert list(strategy_summary.columns) == STRATEGY_SUMMARY_COLUMNS
    assert list(monthly_returns.columns) == MONTHLY_RETURNS_COLUMNS
    assert list(turnover_costs.columns) == TURNOVER_COSTS_COLUMNS
    assert list(cost_sensitivity.columns) == COST_SENSITIVITY_COLUMNS
    assert list(fold_diagnostics.columns) == FOLD_DIAGNOSTICS_COLUMNS
    assert list(ranking_diagnostics.columns) == RANKING_DIAGNOSTICS_COLUMNS
    assert list(calibration_diagnostics.columns) == CALIBRATION_DIAGNOSTICS_COLUMNS
    assert list(score_histograms.columns) == SCORE_HISTOGRAM_COLUMNS
    assert list(threshold_diagnostics.columns) == THRESHOLD_DIAGNOSTICS_COLUMNS
    assert list(model_summary.columns) == MODEL_SUMMARY_COLUMNS
    assert list(fold_summary.columns) == FOLD_SUMMARY_COLUMNS
    assert set(metrics["strategy"]) == expected_strategies
    assert set(performance["strategy"]) == expected_strategies
    assert set(strategy_summary["strategy"]) == expected_strategies
    assert set(monthly_returns["strategy"]) == expected_strategies
    assert set(turnover_costs["strategy"]) == expected_strategies
    assert set(cost_sensitivity["strategy"]) == expected_strategies
    assert set(model_summary["model_name"]) == DEFAULT_MODEL_NAMES
    assert set(ranking_diagnostics["model_name"]) == DEFAULT_MODEL_NAMES
    assert set(calibration_diagnostics["model_name"]) == DEFAULT_MODEL_NAMES
    assert set(score_histograms["model_name"]) == DEFAULT_MODEL_NAMES
    assert set(threshold_diagnostics["model_name"]) == DEFAULT_MODEL_NAMES
    assert not fold_diagnostics.empty
    assert not ranking_diagnostics.empty
    assert not calibration_diagnostics.empty
    assert not score_histograms.empty
    assert not threshold_diagnostics.empty
    assert not fold_summary.empty
    assert (run_dir / "models").is_dir()
    assert set(fold_diagnostics["status"]).issubset({"used", "skipped"})
    assert "used" in set(fold_diagnostics["status"])
    assert set(ranking_diagnostics["bucket_status"]).issubset({"used", "underfilled"})
    assert set(threshold_diagnostics["threshold_status"]).issubset({"used", "empty"})

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
    assert "## Strategy Summary" in report_text
    assert "## Monthly Net Returns" in report_text
    assert "## Turnover And Costs" in report_text
    assert "## Cost Sensitivity" in report_text
    assert "## Walk-Forward Diagnostics" in report_text
    assert "## Model Summary" in report_text
    assert "## Fold Summary" in report_text
    assert "## Headline Outcomes" in report_text
    assert "## Calibration And Threshold Diagnostics" in report_text
    assert "Phase 2 baseline plus ML experiment" in report_text
    assert "ml_logistic_regression" in report_text
    assert "ml_logistic_l1" in report_text
    assert "ml_extra_trees" in report_text
    assert "ml_gradient_boosting" in report_text
    assert "ml_hist_gradient_boosting" in report_text
    assert "- Used candidates:" in report_text
    assert "- Skipped candidates:" in report_text
    assert "- Best model by mean ROC AUC:" in report_text
    assert "- Best model by mean top-bottom spread:" in report_text
    assert "![Calibration Curves](calibration_curves.png)" in report_text
    assert "![Score Histograms](score_histograms.png)" in report_text
    assert "![Threshold Sweeps](threshold_sweeps.png)" in report_text


def test_run_experiment_writes_diagnostics_before_failing_on_zero_usable_folds(tmp_path: Path) -> None:
    config_path = _write_run_experiment_config(
        tmp_path,
        walk_forward={"min_train_rows": 5000},
    )

    result = run_marketlab_cli("run-experiment", config_path)

    assert result.returncode != 0
    combined_output = f"{result.stdout}\n{result.stderr}"
    run_root = tmp_path / "runs" / "integration_fixture"
    run_dir = latest_run_dir(run_root)
    diagnostics_path = (run_dir / "fold_diagnostics.csv").resolve()
    fold_diagnostics = pd.read_csv(diagnostics_path)

    assert "No walk-forward folds are available for run-experiment." in combined_output
    assert str(diagnostics_path) in combined_output
    assert diagnostics_path.exists()
    assert list(fold_diagnostics.columns) == FOLD_DIAGNOSTICS_COLUMNS
    assert (fold_diagnostics["status"] == "skipped").all()
    assert fold_diagnostics["fold_id"].isna().all()


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
        "strategy_summary.csv",
        "monthly_returns.csv",
        "turnover_costs.csv",
        "cost_sensitivity.csv",
        "daily_exposure.csv",
        "report.md",
        "cumulative_returns.png",
        "drawdown.png",
        "turnover.png",
    }
    metrics = pd.read_csv(run_dir / "metrics.csv")
    performance = pd.read_csv(run_dir / "performance.csv")
    strategy_summary = pd.read_csv(run_dir / "strategy_summary.csv")
    monthly_returns = pd.read_csv(run_dir / "monthly_returns.csv")
    turnover_costs = pd.read_csv(run_dir / "turnover_costs.csv")
    cost_sensitivity = pd.read_csv(run_dir / "cost_sensitivity.csv")
    report_text = (run_dir / "report.md").read_text(encoding="utf-8")

    assert list(metrics.columns) == EXPECTED_METRICS_COLUMNS
    assert list(performance.columns) == PERFORMANCE_COLUMNS
    assert list(strategy_summary.columns) == STRATEGY_SUMMARY_COLUMNS
    assert list(monthly_returns.columns) == MONTHLY_RETURNS_COLUMNS
    assert list(turnover_costs.columns) == TURNOVER_COSTS_COLUMNS
    assert list(cost_sensitivity.columns) == COST_SENSITIVITY_COLUMNS
    assert set(metrics["strategy"]) == {"buy_hold", "sma"}
    assert set(performance["strategy"]) == {"buy_hold", "sma"}
    assert set(strategy_summary["strategy"]) == {"buy_hold", "sma"}
    assert set(monthly_returns["strategy"]) == {"buy_hold", "sma"}
    assert set(turnover_costs["strategy"]) == {"buy_hold", "sma"}
    assert set(cost_sensitivity["strategy"]) == {"buy_hold", "sma"}
    assert "## Strategy Summary" in report_text
    assert "## Monthly Net Returns" in report_text
    assert "## Turnover And Costs" in report_text
    assert "## Cost Sensitivity" in report_text
    assert not (run_dir / "group_exposure.csv").exists()
    assert not (run_dir / 'fold_diagnostics.csv').exists()
    assert not (run_dir / "ranking_diagnostics.csv").exists()
    assert not (run_dir / "calibration_diagnostics.csv").exists()
    assert not (run_dir / "score_histograms.csv").exists()
    assert not (run_dir / "threshold_diagnostics.csv").exists()
    assert not (run_dir / "model_summary.csv").exists()
    assert not (run_dir / "fold_summary.csv").exists()
    assert not (run_dir / "calibration_curves.png").exists()
    assert not (run_dir / "score_histograms.png").exists()
    assert not (run_dir / "threshold_sweeps.png").exists()
    assert not (run_dir / "factor_diagnostics.csv").exists()
    assert not (run_dir / "covariance_diagnostics.csv").exists()


def test_backtest_supports_config_defined_allocation_baseline(tmp_path: Path) -> None:
    config_path = _write_backtest_config(
        tmp_path,
        allocation={
            "enabled": True,
            "mode": "equal",
        },
    )

    result = run_marketlab_cli("backtest", config_path)
    assert_command_ok(result)

    run_root = tmp_path / "runs" / "integration_backtest_fixture"
    run_dir = latest_run_dir(run_root)
    metrics = pd.read_csv(run_dir / "metrics.csv")
    strategy_summary = pd.read_csv(run_dir / "strategy_summary.csv")
    report_text = (run_dir / "report.md").read_text(encoding="utf-8")

    assert set(metrics["strategy"]) == {"buy_hold", "sma", "allocation_equal"}
    assert set(strategy_summary["strategy"]) == {"buy_hold", "sma", "allocation_equal"}

    allocation_row = strategy_summary.loc[
        strategy_summary["strategy"] == "allocation_equal"
    ].iloc[0]
    buy_hold_row = strategy_summary.loc[
        strategy_summary["strategy"] == "buy_hold"
    ].iloc[0]
    assert allocation_row["total_turnover"] > 0.0
    assert allocation_row["cumulative_return"] != pytest.approx(
        buy_hold_row["cumulative_return"]
    )
    assert "allocation_equal" in report_text


def test_backtest_supports_factor_and_covariance_diagnostics_for_optimized_baseline(
    tmp_path: Path,
) -> None:
    factor_model_path = _write_factor_model(
        tmp_path,
        pd.bdate_range("2024-01-01", "2024-01-31"),
    )
    config_path = _write_backtest_config(
        tmp_path,
        optimized={
            "enabled": True,
            "method": "mean_variance",
            "lookback_days": 3,
            "target_gross_exposure": 0.7,
            "risk_aversion": 1.0,
        },
        evaluation={"factor_model_path": str(factor_model_path)},
    )

    result = run_marketlab_cli("backtest", config_path)
    assert_command_ok(result)

    run_root = tmp_path / "runs" / "integration_backtest_fixture"
    run_dir = latest_run_dir(run_root)
    factor_diagnostics = pd.read_csv(
        run_dir / "factor_diagnostics.csv",
        parse_dates=["start_date", "end_date"],
    )
    covariance_diagnostics = pd.read_csv(
        run_dir / "covariance_diagnostics.csv",
        parse_dates=["signal_date", "effective_date"],
    )
    report_text = (run_dir / "report.md").read_text(encoding="utf-8")

    assert list(factor_diagnostics.columns) == FACTOR_DIAGNOSTICS_COLUMNS
    assert list(covariance_diagnostics.columns) == COVARIANCE_DIAGNOSTICS_COLUMNS
    assert set(factor_diagnostics["strategy"]) == {"buy_hold", "sma", "mean_variance"}
    assert set(covariance_diagnostics["strategy"]) == {"mean_variance"}
    assert "## Factor Attribution Diagnostics" in report_text
    assert "## Covariance Diagnostics" in report_text
    assert "factor_diagnostics.csv" in report_text
    assert "covariance_diagnostics.csv" in report_text



def test_run_experiment_supports_mean_variance_baseline(tmp_path: Path) -> None:
    config_path = _write_run_experiment_config(
        tmp_path,
        models=[{"name": "logistic_regression"}],
        optimized={
            "enabled": True,
            "method": "mean_variance",
            "lookback_days": 5,
            "target_gross_exposure": 0.7,
            "risk_aversion": 1.0,
        },
    )

    result = run_marketlab_cli("run-experiment", config_path)
    assert_command_ok(result)

    run_root = tmp_path / "runs" / "integration_fixture"
    run_dir = latest_run_dir(run_root)
    metrics = pd.read_csv(run_dir / "metrics.csv")
    performance = pd.read_csv(run_dir / "performance.csv")
    strategy_summary = pd.read_csv(run_dir / "strategy_summary.csv")
    report_text = (run_dir / "report.md").read_text(encoding="utf-8")

    expected_strategies = {"buy_hold", "sma", "mean_variance", "ml_logistic_regression"}
    assert set(metrics["strategy"]) == expected_strategies
    assert set(performance["strategy"]) == expected_strategies
    assert set(strategy_summary["strategy"]) == expected_strategies
    mean_variance_row = strategy_summary.loc[
        strategy_summary["strategy"] == "mean_variance"
    ].iloc[0]
    assert mean_variance_row["avg_gross_exposure"] <= 0.7 + 1e-6
    assert "mean_variance" in report_text
    assert "Covariance Diagnostics" in report_text
    assert "covariance_diagnostics.csv" in report_text
    assert (run_dir / "covariance_diagnostics.csv").exists()
    assert not (run_dir / "factor_diagnostics.csv").exists()


def test_run_experiment_supports_risk_parity_baseline(tmp_path: Path) -> None:
    config_path = _write_run_experiment_config(
        tmp_path,
        models=[{"name": "logistic_regression"}],
        optimized={
            "enabled": True,
            "method": "risk_parity",
            "lookback_days": 5,
            "target_gross_exposure": 0.7,
        },
    )

    result = run_marketlab_cli("run-experiment", config_path)
    assert_command_ok(result)

    run_root = tmp_path / "runs" / "integration_fixture"
    run_dir = latest_run_dir(run_root)
    metrics = pd.read_csv(run_dir / "metrics.csv")
    performance = pd.read_csv(run_dir / "performance.csv")
    strategy_summary = pd.read_csv(run_dir / "strategy_summary.csv")
    report_text = (run_dir / "report.md").read_text(encoding="utf-8")

    expected_strategies = {"buy_hold", "sma", "risk_parity", "ml_logistic_regression"}
    assert set(metrics["strategy"]) == expected_strategies
    assert set(performance["strategy"]) == expected_strategies
    assert set(strategy_summary["strategy"]) == expected_strategies
    risk_parity_row = strategy_summary.loc[
        strategy_summary["strategy"] == "risk_parity"
    ].iloc[0]
    assert risk_parity_row["avg_gross_exposure"] <= 0.71
    assert "risk_parity" in report_text
    assert "Covariance Diagnostics" in report_text
    assert "covariance_diagnostics.csv" in report_text
    assert (run_dir / "covariance_diagnostics.csv").exists()
    assert not (run_dir / "factor_diagnostics.csv").exists()


def test_run_experiment_adds_factor_diagnostics_for_all_strategies_and_covariance_for_optimized_only(
    tmp_path: Path,
) -> None:
    factor_model_path = _write_factor_model(
        tmp_path,
        pd.bdate_range("2020-01-01", "2024-12-31"),
    )
    config_path = _write_run_experiment_config(
        tmp_path,
        models=[{"name": "logistic_regression"}],
        optimized={
            "enabled": True,
            "method": "mean_variance",
            "lookback_days": 5,
            "rebalance_frequency": "W-MON",
            "target_gross_exposure": 0.7,
            "risk_aversion": 1.0,
        },
        evaluation={"factor_model_path": str(factor_model_path)},
    )

    result = run_marketlab_cli("run-experiment", config_path)
    assert_command_ok(result)

    run_root = tmp_path / "runs" / "integration_fixture"
    run_dir = latest_run_dir(run_root)
    performance = pd.read_csv(run_dir / "performance.csv", parse_dates=["date"])
    factor_diagnostics = pd.read_csv(
        run_dir / "factor_diagnostics.csv",
        parse_dates=["start_date", "end_date"],
    )
    covariance_diagnostics = pd.read_csv(
        run_dir / "covariance_diagnostics.csv",
        parse_dates=["signal_date", "effective_date"],
    )
    report_text = (run_dir / "report.md").read_text(encoding="utf-8")

    expected_strategies = {"buy_hold", "sma", "mean_variance", "ml_logistic_regression"}
    performance_dates = pd.Index(performance["date"]).sort_values()
    covariance_effective_dates = pd.Index(
        pd.to_datetime(covariance_diagnostics["effective_date"]).drop_duplicates()
    ).sort_values()
    pre_oos_dates = covariance_effective_dates[covariance_effective_dates < performance_dates.min()]
    in_window_dates = covariance_effective_dates[covariance_effective_dates >= performance_dates.min()]

    assert list(factor_diagnostics.columns) == FACTOR_DIAGNOSTICS_COLUMNS
    assert list(covariance_diagnostics.columns) == COVARIANCE_DIAGNOSTICS_COLUMNS
    assert set(factor_diagnostics["strategy"]) == expected_strategies
    assert set(covariance_diagnostics["strategy"]) == {"mean_variance"}
    assert len(pre_oos_dates) == 1
    assert (covariance_diagnostics["effective_date"] == pre_oos_dates[0]).sum() == 16
    assert len(in_window_dates) > 0
    assert in_window_dates.max() <= performance_dates.max()
    assert set(in_window_dates) <= set(performance_dates)
    assert "## Factor Attribution Diagnostics" in report_text
    assert "## Covariance Diagnostics" in report_text
    assert "factor_diagnostics.csv" in report_text
    assert "covariance_diagnostics.csv" in report_text


def test_run_experiment_supports_group_weight_allocation_baseline(tmp_path: Path) -> None:
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
    )

    result = run_marketlab_cli("run-experiment", config_path)
    assert_command_ok(result)

    run_root = tmp_path / "runs" / "integration_fixture"
    run_dir = latest_run_dir(run_root)
    metrics = pd.read_csv(run_dir / "metrics.csv")
    performance = pd.read_csv(run_dir / "performance.csv")
    strategy_summary = pd.read_csv(run_dir / "strategy_summary.csv")
    report_text = (run_dir / "report.md").read_text(encoding="utf-8")

    assert "allocation_group_weights" in set(metrics["strategy"])
    assert "allocation_group_weights" in set(performance["strategy"])
    assert "allocation_group_weights" in set(strategy_summary["strategy"])
    allocation_row = strategy_summary.loc[
        strategy_summary["strategy"] == "allocation_group_weights"
    ].iloc[0]
    assert allocation_row["total_turnover"] > 0.0
    assert "allocation_group_weights" in report_text


def test_run_experiment_supports_long_only_strategy_variants(tmp_path: Path) -> None:
    config_path = _write_run_experiment_config(
        tmp_path,
        ranking={"mode": "long_only"},
    )

    result = run_marketlab_cli("run-experiment", config_path)
    assert_command_ok(result)

    run_root = tmp_path / "runs" / "integration_fixture"
    run_dir = latest_run_dir(run_root)
    metrics = pd.read_csv(run_dir / "metrics.csv")
    performance = pd.read_csv(run_dir / "performance.csv")
    strategy_summary = pd.read_csv(run_dir / "strategy_summary.csv")
    report_text = (run_dir / "report.md").read_text(encoding="utf-8")

    expected_strategies = {
        "buy_hold",
        "sma",
        "ml_logistic_regression__long_only",
        "ml_logistic_l1__long_only",
        "ml_random_forest__long_only",
        "ml_extra_trees__long_only",
        "ml_gradient_boosting__long_only",
        "ml_hist_gradient_boosting__long_only",
    }
    assert set(metrics["strategy"]) == expected_strategies
    assert set(performance["strategy"]) == expected_strategies
    assert set(strategy_summary["strategy"]) == expected_strategies
    assert "ml_logistic_regression__long_only" in report_text
    assert "ml_logistic_l1__long_only" in report_text


def test_run_experiment_supports_single_symbol_long_only_timing_runs(tmp_path: Path) -> None:
    config_path = _write_run_experiment_config(
        tmp_path,
        models=[{"name": "logistic_regression"}],
        ranking={
            "mode": "long_only",
            "long_n": 1,
            "short_n": 1,
        },
        walk_forward={
            "train_years": 3,
            "test_months": 3,
            "step_months": 3,
            "min_train_rows": 100,
            "min_test_rows": 10,
            "min_train_positive_rate": 0.05,
            "min_test_positive_rate": 0.05,
            "embargo_periods": 1,
        },
        symbol_specs=(("VOO", 100.0, 0.45),),
    )

    result = run_marketlab_cli("run-experiment", config_path)
    assert_command_ok(result)

    run_root = tmp_path / "runs" / "integration_fixture"
    run_dir = latest_run_dir(run_root)
    metrics = pd.read_csv(run_dir / "metrics.csv")
    ranking_diagnostics = pd.read_csv(run_dir / "ranking_diagnostics.csv")
    model_summary = pd.read_csv(run_dir / "model_summary.csv")
    fold_summary = pd.read_csv(run_dir / "fold_summary.csv")
    report_text = (run_dir / "report.md").read_text(encoding="utf-8")

    assert set(metrics["strategy"]) == {
        "buy_hold",
        "sma",
        "ml_logistic_regression__long_only",
    }
    assert set(ranking_diagnostics["evaluation_mode"]) == {"long_only"}
    assert ranking_diagnostics["bucket_status"].eq("used").all()
    assert ranking_diagnostics["top_bucket_size"].eq(1).all()
    assert ranking_diagnostics["bottom_bucket_size"].eq(0).all()
    assert ranking_diagnostics["top_bucket_return"].notna().all()
    assert ranking_diagnostics["top_bottom_spread"].isna().all()
    assert model_summary["mean_top_bucket_return"].notna().all()
    assert model_summary["mean_top_bucket_signal_count"].gt(0).all()
    assert model_summary["mean_top_bottom_spread"].isna().all()
    assert fold_summary["best_model_by_top_bucket_return"].eq("logistic_regression").all()
    assert fold_summary["best_top_bucket_return"].notna().all()
    assert fold_summary["best_model_by_top_bottom_spread"].isna().all()
    assert fold_summary["best_top_bottom_spread"].isna().all()
    assert "ml_logistic_regression__long_only" in report_text
    assert "- Best model by mean top-bucket return:" in report_text


def test_run_experiment_supports_daily_one_day_single_symbol_timing_runs(
    tmp_path: Path,
) -> None:
    config_path = _write_run_experiment_config(
        tmp_path,
        models=[{"name": "logistic_regression"}],
        ranking={
            "mode": "long_only",
            "long_n": 1,
            "short_n": 1,
            "rebalance_frequency": "D",
            "min_score_threshold": 0.55,
            "cash_when_underfilled": True,
        },
        walk_forward={
            "train_years": 3,
            "test_months": 1,
            "step_months": 1,
            "min_train_rows": 200,
            "min_test_rows": 15,
            "min_train_positive_rate": 0.05,
            "min_test_positive_rate": 0.05,
            "embargo_periods": 1,
        },
        symbol_specs=(("VOO", 100.0, 0.45),),
    )
    config_text = config_path.read_text(encoding="utf-8").replace(
        'horizon_days: 5',
        'horizon_days: 1',
    )
    config_path.write_text(config_text, encoding="utf-8")

    result = run_marketlab_cli("run-experiment", config_path)
    assert_command_ok(result)

    run_root = tmp_path / "runs" / "integration_fixture"
    run_dir = latest_run_dir(run_root)
    metrics = pd.read_csv(run_dir / "metrics.csv")
    ranking_diagnostics = pd.read_csv(run_dir / "ranking_diagnostics.csv")
    report_text = (run_dir / "report.md").read_text(encoding="utf-8")

    assert set(metrics["strategy"]) == {
        "buy_hold",
        "sma",
        "ml_logistic_regression__long_only__thr0p55__cash",
    }
    assert set(ranking_diagnostics["evaluation_mode"]) == {"long_only"}
    assert ranking_diagnostics["top_bucket_size"].eq(1).all()
    assert "ml_logistic_regression__long_only__thr0p55__cash" in report_text




def test_run_experiment_writes_ml_strategy_threshold_sweep(
    tmp_path: Path,
) -> None:
    config_path = _write_run_experiment_config(
        tmp_path,
        models=[{"name": "logistic_regression"}],
        symbol_specs=(("BTC-USD", 100.0, 0.45),),
        features={
            "return_windows": [3, 6],
            "ma_windows": [3, 6],
            "vol_windows": [3],
            "momentum_window": 3,
            "crypto_time_series_enabled": True,
            "crypto_return_windows": [1, 3],
            "crypto_vol_windows": [3],
            "crypto_ma_windows": [3],
            "crypto_rsi_window": 3,
            "crypto_macd_fast_window": 3,
            "crypto_macd_slow_window": 6,
            "crypto_macd_signal_window": 3,
            "crypto_bollinger_window": 3,
            "crypto_volume_window": 3,
        },
        target={"horizon_days": 1, "type": "direction"},
        ranking={
            "mode": "long_only",
            "long_n": 1,
            "short_n": 1,
            "rebalance_frequency": "bar",
            "min_score_threshold": 0.55,
            "cash_when_underfilled": True,
        },
        walk_forward={
            "train_years": 1,
            "test_months": 2,
            "step_months": 2,
            "min_train_rows": 100,
            "min_test_rows": 10,
            "min_train_positive_rate": 0.05,
            "min_test_positive_rate": 0.05,
            "embargo_periods": 1,
        },
        evaluation={
            "benchmark_strategy": "buy_hold",
            "ml_strategy_threshold_sweep": {
                "enabled": True,
                "thresholds": [0.50, 0.55],
                "min_exposure_changes": 1,
                "max_average_exposure_for_active": 1.0,
            },
        },
    )

    result = run_marketlab_cli("run-experiment", config_path)
    assert_command_ok(result)

    run_root = tmp_path / "runs" / "integration_fixture"
    run_dir = latest_run_dir(run_root)
    sweep = pd.read_csv(run_dir / "ml_strategy_threshold_sweep.csv")
    report_text = (run_dir / "report.md").read_text(encoding="utf-8")

    assert not sweep.empty
    assert set(sweep["threshold"]) == {0.50, 0.55}
    assert {
        "model_name",
        "threshold",
        "buy_hold_cumulative_return",
        "excess_cumulative_return",
        "exposure_changes",
        "average_exposure",
        "passed_gate",
    }.issubset(sweep.columns)
    assert sweep["model_name"].eq("logistic_regression").all()
    assert "## ML Strategy Threshold Sweep" in report_text
    assert "ML strategy threshold sweep" in report_text


def test_run_experiment_writes_indicator_ml_strategy_tuning(
    tmp_path: Path,
) -> None:
    config_path = _write_run_experiment_config(
        tmp_path,
        models=[{"name": "logistic_regression"}],
        symbol_specs=(("BTC-USD", 100.0, 0.45),),
        features={
            "return_windows": [3, 6],
            "ma_windows": [3, 6],
            "vol_windows": [3],
            "momentum_window": 3,
            "indicator_stack_ml_features_enabled": True,
            "crypto_time_series_enabled": True,
            "crypto_return_windows": [1, 3],
            "crypto_vol_windows": [3],
            "crypto_ma_windows": [3],
            "crypto_rsi_window": 3,
            "crypto_macd_fast_window": 3,
            "crypto_macd_slow_window": 6,
            "crypto_macd_signal_window": 3,
            "crypto_bollinger_window": 3,
            "crypto_volume_window": 3,
        },
        target={"horizon_days": 1, "type": "direction"},
        ranking={
            "mode": "long_only",
            "long_n": 1,
            "short_n": 1,
            "rebalance_frequency": "bar",
            "min_score_threshold": 0.55,
            "cash_when_underfilled": True,
        },
        indicator_stack={
            "enabled": True,
            "ema_fast_window": 3,
            "ema_slow_window": 6,
            "rsi_window": 3,
            "macd_fast_window": 3,
            "macd_slow_window": 6,
            "macd_signal_window": 3,
            "bollinger_window": 3,
            "volume_window": 3,
            "min_confirmations": 2,
        },
        partial_allocation_benchmarks={
            "enabled": True,
            "weights": [0.25, 0.50, 0.75],
        },
        rebalanced_partial_allocation_benchmarks={
            "enabled": True,
            "weights": [0.25, 0.50, 0.75],
        },
        walk_forward={
            "train_years": 1,
            "test_months": 2,
            "step_months": 2,
            "min_train_rows": 100,
            "min_test_rows": 10,
            "min_train_positive_rate": 0.05,
            "min_test_positive_rate": 0.05,
            "embargo_periods": 1,
        },
        evaluation={
            "benchmark_strategy": "buy_hold",
            "ml_strategy_tuning": {
                "enabled": True,
                "thresholds": [0.50, 0.55],
                "validation_months": 2,
                "min_validation_rows": 10,
                "min_exposure_changes": 1,
                "max_average_exposure_for_active": 1.0,
            },
        },
    )

    result = run_marketlab_cli("run-experiment", config_path)
    assert_command_ok(result)

    run_root = tmp_path / "runs" / "integration_fixture"
    run_dir = latest_run_dir(run_root)
    metrics = pd.read_csv(run_dir / "metrics.csv")
    performance = pd.read_csv(run_dir / "performance.csv")
    strategy_summary = pd.read_csv(run_dir / "strategy_summary.csv")
    cost_sensitivity = pd.read_csv(run_dir / "cost_sensitivity.csv")
    candidates = pd.read_csv(run_dir / "ml_strategy_tuning_candidates.csv")
    selections = pd.read_csv(run_dir / "ml_strategy_tuning_selections.csv")
    folds = pd.read_csv(run_dir / "fold_diagnostics.csv")
    report_text = (run_dir / "report.md").read_text(encoding="utf-8")

    assert "ml_indicator_tuned__long_only__cash" in set(strategy_summary["strategy"])
    assert {"btc_static_25", "btc_static_50", "btc_static_75"}.issubset(
        set(metrics["strategy"])
    )
    assert {"btc_rebalanced_25", "btc_rebalanced_50", "btc_rebalanced_75"}.issubset(
        set(metrics["strategy"])
    )
    assert {"btc_static_25", "btc_static_50", "btc_static_75"}.issubset(
        set(performance["strategy"])
    )
    assert {"btc_rebalanced_25", "btc_rebalanced_50", "btc_rebalanced_75"}.issubset(
        set(performance["strategy"])
    )
    assert {"btc_static_25", "btc_static_50", "btc_static_75"}.issubset(
        set(strategy_summary["strategy"])
    )
    assert {"btc_rebalanced_25", "btc_rebalanced_50", "btc_rebalanced_75"}.issubset(
        set(strategy_summary["strategy"])
    )
    assert {"btc_static_25", "btc_static_50", "btc_static_75"}.issubset(
        set(cost_sensitivity["strategy"])
    )
    assert {"btc_rebalanced_25", "btc_rebalanced_50", "btc_rebalanced_75"}.issubset(
        set(cost_sensitivity["strategy"])
    )
    assert not candidates.empty
    assert not selections.empty
    assert {
        "fold_id",
        "model_name",
        "threshold",
        "excess_cumulative_return",
        "sharpe_like_delta",
        "drawdown_delta",
        "active_candidate",
        "passed_gate",
    }.issubset(candidates.columns)
    assert selections["selected_strategy"].eq("ml_indicator_tuned__long_only__cash").all()
    selected_rows = selections.loc[selections["selection_status"] == "selected"]
    assert selected_rows["passed_gate"].astype(bool).all()

    used_folds = folds.loc[folds["status"] == "used", ["fold_id", "test_start"]].copy()
    joined = selections.merge(used_folds, on="fold_id", how="inner")
    assert pd.to_datetime(joined["validation_end"]).lt(pd.to_datetime(joined["test_start"])).all()

    passed = candidates.loc[candidates["passed_gate"].astype(bool)]
    if not passed.empty:
        assert passed["excess_cumulative_return"].gt(0.0).all()
        assert (
            passed["sharpe_like_delta"].gt(0.0) | passed["drawdown_delta"].ge(0.0)
        ).all()
    assert "## ML Strategy Tuning" in report_text
    assert "ML strategy tuning candidates" in report_text
    assert "btc_static_25" in report_text
    assert "btc_rebalanced_25" in report_text


def test_run_experiment_writes_allocation_utility_outputs(
    tmp_path: Path,
) -> None:
    config_path = _write_run_experiment_config(
        tmp_path,
        models=[{"name": "logistic_regression"}],
        symbol_specs=(("BTC-USD", 100.0, 1.00),),
        features={
            "return_windows": [3, 5, 7],
            "ma_windows": [3, 5, 7],
            "vol_windows": [3, 5],
            "momentum_window": 3,
            "crypto_time_series_enabled": True,
            "crypto_return_windows": [1, 3, 5],
            "crypto_vol_windows": [3, 5],
            "crypto_ma_windows": [3, 5],
            "crypto_rsi_window": 3,
            "crypto_macd_fast_window": 3,
            "crypto_macd_slow_window": 6,
            "crypto_macd_signal_window": 3,
            "crypto_bollinger_window": 3,
            "crypto_volume_window": 3,
            "crypto_regime_features_enabled": True,
            "crypto_regime_trend_windows": [3, 6, 12],
            "crypto_regime_volatility_window": 3,
            "crypto_regime_percentile_window": 12,
            "crypto_regime_drawdown_window": 12,
            "crypto_regime_volume_window": 3,
        },
        target={"horizon_days": 3, "type": "allocation_utility"},
        ranking={
            "mode": "long_only",
            "long_n": 1,
            "short_n": 1,
            "rebalance_frequency": "bar",
            "min_score_threshold": 0.55,
            "cash_when_underfilled": True,
        },
        partial_allocation_benchmarks={
            "enabled": True,
            "weights": [0.25, 0.50, 0.75],
        },
        rebalanced_partial_allocation_benchmarks={
            "enabled": True,
            "weights": [0.25, 0.50, 0.75],
        },
        walk_forward={
            "train_years": 1,
            "test_months": 2,
            "step_months": 2,
            "min_train_rows": 100,
            "min_test_rows": 10,
            "min_train_positive_rate": 0.05,
            "min_test_positive_rate": 0.05,
            "embargo_periods": 1,
        },
        evaluation={
            "benchmark_strategy": "buy_hold",
            "cost_sensitivity_bps": [35, 50],
            "ml_strategy_tuning": {
                "enabled": True,
                "allocation_mode": "direct_tiered",
                "thresholds": [0.50],
                "tier_thresholds": [0.25, 0.50, 0.75],
                "validation_months": 2,
                "min_validation_rows": 10,
                "min_exposure_changes": 0,
                "min_average_exposure_for_active": 0.0,
                "max_average_exposure_for_active": 1.0,
                "rolling_train_bars_grid": [120],
                "min_holding_period_bars_grid": [0],
                "hysteresis_margin_grid": [0.0],
                "regime_participation_policies": [
                    {
                        "name": "model_only",
                        "bull_floor": 0.0,
                        "sideways_floor": 0.0,
                        "bear_floor": 0.0,
                        "risk_off_cap": 0.25,
                    },
                    {
                        "name": "bull100_sideways25",
                        "bull_floor": 1.0,
                        "sideways_floor": 0.25,
                        "bear_floor": 0.0,
                        "risk_off_cap": 0.25,
                    },
                ],
                "max_annualized_turnover": 365.0,
                "objective": "net_return_and_risk_vs_required_benchmarks",
                "selection_benchmark_strategies": [
                    "buy_hold",
                    "btc_rebalanced_25",
                    "btc_rebalanced_50",
                    "btc_rebalanced_75",
                ],
                "allocation_utility_profiles": [
                    {
                        "name": "fixture_profile",
                        "drawdown_penalty": 0.50,
                        "volatility_penalty": 0.25,
                        "risk_penalty_power": 2.0,
                    }
                ],
                "allocation_class_weighting": "balanced_partial_boost",
                "allocation_partial_class_weight_multiplier": 2.0,
                "allocation_probability_calibration": "sigmoid",
                "allocation_calibration_cv": 3,
            },
            "strict_research_gate": {
                "enabled": True,
                "required_benchmark_strategies": [
                    "buy_hold",
                    "btc_static_25",
                    "btc_static_50",
                    "btc_static_75",
                    "btc_rebalanced_25",
                    "btc_rebalanced_50",
                    "btc_rebalanced_75",
                ],
                "required_partial_target_weights": [0.25, 0.50],
                "min_partial_target_fraction": 0.05,
                "min_partial_target_fold_fraction": 0.60,
                "required_predicted_target_weights": [0.25, 0.50],
                "min_predicted_target_fraction": 0.0,
                "min_predicted_target_fold_fraction": 0.0,
            },
        },
    )

    result = run_marketlab_cli("run-experiment", config_path)
    assert_command_ok(result)

    run_root = tmp_path / "runs" / "integration_fixture"
    run_dir = latest_run_dir(run_root)
    strategy_summary = pd.read_csv(run_dir / "strategy_summary.csv")
    target_diagnostics = pd.read_csv(run_dir / "allocation_target_diagnostics.csv")
    probability_diagnostics = pd.read_csv(run_dir / "allocation_probability_diagnostics.csv")
    feature_importance = pd.read_csv(run_dir / "feature_importance.csv")
    tuning_candidates = pd.read_csv(run_dir / "ml_strategy_tuning_candidates.csv")
    tuning_selections = pd.read_csv(run_dir / "ml_strategy_tuning_selections.csv")
    phase8_summary = pd.read_csv(run_dir / "phase8_run_summary.csv")
    strict_gate = pd.read_csv(run_dir / "strict_research_gate.csv")
    report_text = (run_dir / "report.md").read_text(encoding="utf-8")

    assert "ml_indicator_tuned__long_only__cash" in set(strategy_summary["strategy"])
    assert {"btc_rebalanced_25", "btc_rebalanced_50", "btc_rebalanced_75"}.issubset(
        set(strategy_summary["strategy"])
    )
    assert {"scope", "target_weight", "row_count", "avg_forward_return"}.issubset(
        target_diagnostics.columns
    )
    assert {
        "selection_benchmark_strategies",
        "min_benchmark_excess_cumulative_return",
        "utility_profile",
        "allocation_class_weighting",
        "allocation_score_policy",
        "allocation_score_policy_prob100_threshold",
        "calibration_status",
        "regime_policy",
        "regime_bull_floor",
        "regime_sideways_floor",
        "regime_bear_floor",
        "regime_risk_off_cap",
        "failure_reasons",
        "validation_predicted_25_fraction",
        "validation_predicted_50_fraction",
        "validation_predicted_100_fraction",
        "validation_score_policy_triggered_100_fraction",
    }.issubset(tuning_candidates.columns)
    assert {
        "selection_policy",
        "selection_source",
        "allocation_score_policy",
        "allocation_score_policy_prob100_threshold",
        "selected_regime_policy",
        "selected_regime_bull_floor",
        "selected_regime_sideways_floor",
        "selected_regime_bear_floor",
        "selected_regime_risk_off_cap",
    }.issubset(tuning_selections.columns)
    assert "selected_fold_fraction" in set(phase8_summary["metric"])
    assert {
        "partial_target_25_global_fraction",
        "partial_target_50_fold_fraction",
        "predicted_target_25_global_fraction",
        "predicted_target_50_fold_fraction",
    }.issubset(set(strict_gate["condition"]))
    assert {
        "score",
        "runtime_regime",
        "crypto_regime_risk_off",
        "crypto_regime_trend_state",
        "predicted_tier_weight",
        "allocation_score_policy",
        "allocation_score_policy_prob100_threshold",
        "raw_expected_allocation_score",
        "final_allocation_score",
        "score_policy_triggered_100",
        "prob_tier_0",
        "prob_tier_25",
        "prob_tier_50",
        "prob_tier_100",
        "fold_predicted_25_fraction",
        "fold_predicted_50_fraction",
        "fold_predicted_100_fraction",
        "fold_score_policy_triggered_100_fraction",
    }.issubset(probability_diagnostics.columns)
    assert {"feature", "importance_type", "importance"}.issubset(feature_importance.columns)
    assert "## Allocation Utility Diagnostics" in report_text
    assert "## Phase 8 Run Summary" in report_text
    assert "Target-support gate requires" in report_text
    assert "Predicted-support gate requires" in report_text
    assert "Allocation score policy" in report_text
    assert "Allocation target diagnostics" in report_text
    assert "selection_policy" in report_text


def test_run_experiment_supports_diagnostic_best_active_fallback_selection(
    tmp_path: Path,
) -> None:
    config_path = _write_run_experiment_config(
        tmp_path,
        models=[{"name": "logistic_regression"}],
        symbol_specs=(("BTC-USD", 100.0, 1.00),),
        features={
            "return_windows": [3, 5, 7],
            "ma_windows": [3, 5, 7],
            "vol_windows": [3, 5],
            "momentum_window": 3,
            "crypto_time_series_enabled": True,
            "crypto_return_windows": [1, 3, 5],
            "crypto_vol_windows": [3, 5],
            "crypto_ma_windows": [3, 5],
            "crypto_rsi_window": 3,
            "crypto_macd_fast_window": 3,
            "crypto_macd_slow_window": 6,
            "crypto_macd_signal_window": 3,
            "crypto_bollinger_window": 3,
            "crypto_volume_window": 3,
            "crypto_regime_features_enabled": True,
            "crypto_regime_trend_windows": [3, 6, 12],
            "crypto_regime_volatility_window": 3,
            "crypto_regime_percentile_window": 12,
            "crypto_regime_drawdown_window": 12,
            "crypto_regime_volume_window": 3,
        },
        target={"horizon_days": 3, "type": "allocation_utility"},
        ranking={
            "mode": "long_only",
            "long_n": 1,
            "short_n": 1,
            "rebalance_frequency": "bar",
            "min_score_threshold": 0.55,
            "cash_when_underfilled": True,
        },
        partial_allocation_benchmarks={
            "enabled": True,
            "weights": [0.25, 0.50, 0.75],
        },
        rebalanced_partial_allocation_benchmarks={
            "enabled": True,
            "weights": [0.25, 0.50, 0.75],
        },
        walk_forward={
            "train_years": 1,
            "test_months": 2,
            "step_months": 2,
            "min_train_rows": 100,
            "min_test_rows": 10,
            "min_train_positive_rate": 0.0,
            "min_test_positive_rate": 0.0,
            "embargo_periods": 1,
        },
        evaluation={
            "benchmark_strategy": "buy_hold",
            "cost_sensitivity_bps": [35, 50],
            "ml_strategy_tuning": {
                "enabled": True,
                "allocation_mode": "direct_tiered",
                "selection_policy": "best_active_fallback",
                "thresholds": [0.50],
                "tier_thresholds": [0.25, 0.50, 0.75],
                "validation_months": 2,
                "min_validation_rows": 10,
                "min_exposure_changes": 0,
                "min_average_exposure_for_active": 0.0,
                "max_average_exposure_for_active": 1.0,
                "rolling_train_bars_grid": [120],
                "min_holding_period_bars_grid": [0],
                "hysteresis_margin_grid": [0.0],
                "regime_participation_policies": [
                    {
                        "name": "bull100_sideways25",
                        "bull_floor": 1.0,
                        "sideways_floor": 0.25,
                        "bear_floor": 0.0,
                        "risk_off_cap": 0.25,
                    }
                ],
                "max_annualized_turnover": 365.0,
                "objective": "net_return_and_risk_vs_required_benchmarks",
                "selection_benchmark_strategies": ["btc_static_75"],
                "allocation_utility_profiles": [
                    {
                        "name": "fixture_profile",
                        "drawdown_penalty": 0.50,
                        "volatility_penalty": 0.25,
                        "risk_penalty_power": 2.0,
                    }
                ],
                "allocation_class_weighting": "balanced_partial_boost",
                "allocation_partial_class_weight_multiplier": 2.0,
            },
            "strict_research_gate": {
                "enabled": True,
                "required_benchmark_strategies": ["buy_hold"],
                "required_partial_target_weights": [0.25, 0.50],
                "min_partial_target_fraction": 0.0,
                "min_partial_target_fold_fraction": 0.0,
                "required_predicted_target_weights": [],
                "min_predicted_target_fraction": 0.0,
                "min_predicted_target_fold_fraction": 0.0,
            },
        },
    )
    dates = pd.bdate_range("2020-01-01", "2024-12-31")
    close = pd.Series(
        [100.0 + (index * 0.05) + (2.0 * math.sin(index / 3.0)) for index in range(len(dates))]
    )
    open_price = close.shift(1).fillna(close.iloc[0])
    high = pd.concat([open_price, close], axis=1).max(axis=1) + 0.2
    low = pd.concat([open_price, close], axis=1).min(axis=1) - 0.2
    save_panel_csv(
        pd.DataFrame(
            {
                "symbol": "BTC-USD",
                "timestamp": dates,
                "open": open_price,
                "high": high,
                "low": low,
                "close": close,
                "volume": [1_000_000 + index for index in range(len(dates))],
                "adj_close": close,
                "adj_factor": 1.0,
                "adj_open": open_price,
                "adj_high": high,
                "adj_low": low,
            }
        ),
        tmp_path / "cache" / "panel.csv",
    )
    result = run_marketlab_cli("run-experiment", config_path)
    assert_command_ok(result)

    run_dir = latest_run_dir(tmp_path / "runs" / "integration_fixture")
    selections = pd.read_csv(run_dir / "ml_strategy_tuning_selections.csv")
    phase8_summary = pd.read_csv(run_dir / "phase8_run_summary.csv")
    strategy_summary = pd.read_csv(run_dir / "strategy_summary.csv")

    fallback_rows = selections.loc[
        selections["selection_source"].eq("best_active_fallback")
    ]
    assert not fallback_rows.empty
    assert fallback_rows["selection_status"].eq("selected").all()
    assert not fallback_rows["passed_gate"].astype(bool).any()
    assert selections["selection_policy"].eq("best_active_fallback").all()
    assert (
        strategy_summary.loc[
            strategy_summary["strategy"].eq("ml_indicator_tuned__long_only__cash"),
            "avg_long_exposure",
        ].iloc[0]
        > 0.0
    )
    selected_fraction = phase8_summary.loc[
        phase8_summary["metric"].eq("selected_fold_fraction"),
        "value",
    ].iloc[0]
    assert float(selected_fraction) > 0.0


def test_run_experiment_supports_no_candidate_regime_fallback_selection(
    tmp_path: Path,
) -> None:
    strict_benchmarks = [
        "buy_hold",
        "btc_static_25",
        "btc_static_50",
        "btc_static_75",
        "btc_rebalanced_25",
        "btc_rebalanced_50",
        "btc_rebalanced_75",
    ]
    config_path = _write_run_experiment_config(
        tmp_path,
        models=[{"name": "logistic_regression"}],
        symbol_specs=(("BTC-USD", 100.0, 1.00),),
        features={
            "return_windows": [3, 5, 7],
            "ma_windows": [3, 5, 7],
            "vol_windows": [3, 5],
            "momentum_window": 3,
            "crypto_time_series_enabled": True,
            "crypto_return_windows": [1, 3, 5],
            "crypto_vol_windows": [3, 5],
            "crypto_ma_windows": [3, 5],
            "crypto_rsi_window": 3,
            "crypto_macd_fast_window": 3,
            "crypto_macd_slow_window": 6,
            "crypto_macd_signal_window": 3,
            "crypto_bollinger_window": 3,
            "crypto_volume_window": 3,
            "crypto_regime_features_enabled": True,
            "crypto_regime_trend_windows": [3, 6, 12],
            "crypto_regime_volatility_window": 3,
            "crypto_regime_percentile_window": 12,
            "crypto_regime_drawdown_window": 12,
            "crypto_regime_volume_window": 3,
        },
        target={"horizon_days": 3, "type": "allocation_utility"},
        ranking={
            "mode": "long_only",
            "long_n": 1,
            "short_n": 1,
            "rebalance_frequency": "bar",
            "min_score_threshold": 0.55,
            "cash_when_underfilled": True,
        },
        partial_allocation_benchmarks={
            "enabled": True,
            "weights": [0.25, 0.50, 0.75],
        },
        rebalanced_partial_allocation_benchmarks={
            "enabled": True,
            "weights": [0.25, 0.50, 0.75],
        },
        walk_forward={
            "train_years": 1,
            "test_months": 2,
            "step_months": 2,
            "min_train_rows": 100,
            "min_test_rows": 10,
            "min_train_positive_rate": 0.0,
            "min_test_positive_rate": 0.0,
            "embargo_periods": 1,
        },
        evaluation={
            "benchmark_strategy": "buy_hold",
            "cost_sensitivity_bps": [35, 50],
            "ml_strategy_tuning": {
                "enabled": True,
                "allocation_mode": "direct_tiered",
                "selection_policy": "best_active_fallback",
                "no_candidate_fallback_regime_policy": "bull100_sideways25",
                "thresholds": [0.50],
                "tier_thresholds": [0.25, 0.50, 0.75],
                "validation_months": 2,
                "min_validation_rows": 10,
                "min_exposure_changes": 9999,
                "min_average_exposure_for_active": 0.0,
                "max_average_exposure_for_active": 1.0,
                "rolling_train_bars_grid": [120],
                "min_holding_period_bars_grid": [0],
                "hysteresis_margin_grid": [0.0],
                "regime_participation_policies": [
                    {
                        "name": "bull100_sideways25",
                        "bull_floor": 1.0,
                        "sideways_floor": 0.25,
                        "bear_floor": 0.0,
                        "risk_off_cap": 0.25,
                    }
                ],
                "max_annualized_turnover": 365.0,
                "objective": "net_return_and_risk_vs_required_benchmarks",
                "selection_benchmark_strategies": strict_benchmarks,
                "allocation_utility_profiles": [
                    {
                        "name": "fixture_profile",
                        "drawdown_penalty": 0.50,
                        "volatility_penalty": 0.25,
                        "risk_penalty_power": 2.0,
                    }
                ],
                "allocation_score_policy": "expected_allocation",
                "allocation_class_weighting": "balanced_partial_boost",
                "allocation_partial_class_weight_multiplier": 2.0,
            },
            "strict_research_gate": {
                "enabled": True,
                "required_benchmark_strategies": strict_benchmarks,
                "required_partial_target_weights": [0.25, 0.50],
                "min_partial_target_fraction": 0.0,
                "min_partial_target_fold_fraction": 0.0,
                "required_predicted_target_weights": [],
                "min_predicted_target_fraction": 0.0,
                "min_predicted_target_fold_fraction": 0.0,
            },
        },
    )

    result = run_marketlab_cli("run-experiment", config_path)
    assert_command_ok(result)

    run_dir = latest_run_dir(tmp_path / "runs" / "integration_fixture")
    selections = pd.read_csv(run_dir / "ml_strategy_tuning_selections.csv")
    candidates = pd.read_csv(run_dir / "ml_strategy_tuning_candidates.csv")
    strategy_summary = pd.read_csv(run_dir / "strategy_summary.csv")
    strict_gate = pd.read_csv(run_dir / "strict_research_gate.csv")

    fallback_rows = selections.loc[
        selections["selection_source"].eq("deterministic_regime_fallback")
    ]
    assert not fallback_rows.empty
    assert fallback_rows["selection_status"].eq("selected").all()
    assert not fallback_rows["passed_gate"].astype(bool).any()
    assert fallback_rows["selected_model_name"].isna().all()
    assert fallback_rows["selected_regime_policy"].eq("bull100_sideways25").all()
    assert fallback_rows["selected_regime_bull_floor"].eq(1.0).all()
    assert fallback_rows["selected_regime_sideways_floor"].eq(0.25).all()
    assert fallback_rows["selected_regime_risk_off_cap"].eq(0.25).all()
    assert not candidates["passed_gate"].astype(bool).any()
    assert (run_dir / "strict_research_gate.csv").exists()
    assert (
        strict_gate.loc[
            strict_gate["condition"].eq("selected_walk_forward_fold_fraction"),
            "passed",
        ]
        .astype(bool)
        .all()
    )
    assert (
        strategy_summary.loc[
            strategy_summary["strategy"].eq("ml_indicator_tuned__long_only__cash"),
            "avg_long_exposure",
        ].iloc[0]
        > 0.0
    )


def test_run_experiment_writes_regime_state_outputs(tmp_path: Path) -> None:
    config_path = _write_run_experiment_config(
        tmp_path,
        models=[{"name": "logistic_regression"}],
        symbol_specs=(("BTC-USD", 100.0, 1.00),),
        features={
            "return_windows": [3, 5, 7],
            "ma_windows": [3, 5, 7],
            "vol_windows": [3, 5],
            "momentum_window": 3,
            "crypto_time_series_enabled": True,
            "crypto_return_windows": [1, 3, 5],
            "crypto_vol_windows": [3, 5],
            "crypto_ma_windows": [3, 5],
            "crypto_rsi_window": 3,
            "crypto_macd_fast_window": 3,
            "crypto_macd_slow_window": 6,
            "crypto_macd_signal_window": 3,
            "crypto_bollinger_window": 3,
            "crypto_volume_window": 3,
            "crypto_regime_features_enabled": True,
            "crypto_regime_trend_windows": [3, 6, 12],
            "crypto_regime_volatility_window": 3,
            "crypto_regime_percentile_window": 12,
            "crypto_regime_drawdown_window": 12,
            "crypto_regime_volume_window": 3,
        },
        target={"horizon_days": 3, "type": "regime_state"},
        ranking={
            "mode": "long_only",
            "long_n": 1,
            "short_n": 1,
            "rebalance_frequency": "bar",
            "min_score_threshold": 0.55,
            "cash_when_underfilled": True,
        },
        partial_allocation_benchmarks={
            "enabled": True,
            "weights": [0.25, 0.50, 0.75],
        },
        rebalanced_partial_allocation_benchmarks={
            "enabled": True,
            "weights": [0.25, 0.50, 0.75],
        },
        walk_forward={
            "train_years": 1,
            "test_months": 2,
            "step_months": 2,
            "min_train_rows": 100,
            "min_test_rows": 10,
            "min_train_positive_rate": 0.05,
            "min_test_positive_rate": 0.05,
            "embargo_periods": 1,
        },
        evaluation={
            "benchmark_strategy": "buy_hold",
            "cost_sensitivity_bps": [35, 50],
            "ml_strategy_tuning": {
                "enabled": True,
                "allocation_mode": "direct_tiered",
                "thresholds": [0.50],
                "tier_thresholds": [0.25, 0.50, 0.75],
                "validation_months": 2,
                "min_validation_rows": 10,
                "min_exposure_changes": 0,
                "min_average_exposure_for_active": 0.0,
                "max_average_exposure_for_active": 1.0,
                "rolling_train_bars_grid": [120],
                "min_holding_period_bars_grid": [0],
                "hysteresis_margin_grid": [0.0],
                "regime_participation_policies": [
                    {
                        "name": "model_only",
                        "bull_floor": 0.0,
                        "sideways_floor": 0.0,
                        "bear_floor": 0.0,
                        "risk_off_cap": 0.25,
                    },
                    {
                        "name": "bull100_sideways25",
                        "bull_floor": 1.0,
                        "sideways_floor": 0.25,
                        "bear_floor": 0.0,
                        "risk_off_cap": 0.25,
                    },
                ],
                "max_annualized_turnover": 365.0,
                "objective": "net_return_and_risk_vs_required_benchmarks",
                "selection_benchmark_strategies": [
                    "buy_hold",
                    "btc_rebalanced_25",
                    "btc_rebalanced_50",
                    "btc_rebalanced_75",
                ],
                "allocation_utility_profiles": [
                    {
                        "name": "fixture_profile",
                        "drawdown_penalty": 0.50,
                        "volatility_penalty": 0.25,
                        "risk_penalty_power": 2.0,
                    }
                ],
                "allocation_class_weighting": "balanced_partial_boost",
                "allocation_partial_class_weight_multiplier": 2.0,
                "allocation_probability_calibration": "sigmoid",
                "allocation_calibration_cv": 3,
            },
            "strict_research_gate": {
                "enabled": True,
                "required_benchmark_strategies": [
                    "buy_hold",
                    "btc_static_25",
                    "btc_static_50",
                    "btc_static_75",
                    "btc_rebalanced_25",
                    "btc_rebalanced_50",
                    "btc_rebalanced_75",
                ],
                "required_partial_target_weights": [0.25, 0.50],
                "min_partial_target_fraction": 0.05,
                "min_partial_target_fold_fraction": 0.60,
                "required_predicted_target_weights": [0.25, 0.50],
                "min_predicted_target_fraction": 0.0,
                "min_predicted_target_fold_fraction": 0.0,
            },
        },
    )

    result = run_marketlab_cli("run-experiment", config_path)
    assert_command_ok(result)

    run_root = tmp_path / "runs" / "integration_fixture"
    run_dir = latest_run_dir(run_root)
    target_diagnostics = pd.read_csv(run_dir / "allocation_target_diagnostics.csv")
    probability_diagnostics = pd.read_csv(run_dir / "allocation_probability_diagnostics.csv")
    tuning_candidates = pd.read_csv(run_dir / "ml_strategy_tuning_candidates.csv")
    phase8_summary = pd.read_csv(run_dir / "phase8_run_summary.csv")
    report_text = (run_dir / "report.md").read_text(encoding="utf-8")

    assert set(target_diagnostics["target"].dropna().astype(int)).issubset({0, 1, 2})
    assert {"predicted_tier_weight", "prob_tier_0", "prob_tier_50"}.issubset(
        probability_diagnostics.columns
    )
    assert {
        "utility_profile",
        "calibration_status",
        "regime_policy",
        "allocation_score_policy",
    }.issubset(
        tuning_candidates.columns
    )
    assert "candidate_rejections" in set(phase8_summary["section"])
    assert "`regime_state` maps utility tiers" in report_text


def test_btc_long_history_config_produces_walk_forward_folds() -> None:
    config = load_config("configs/experiment.btc_phase8_regime_allocation_1d_long_history.yaml")
    dates = pd.date_range(config.data.start_date, config.data.end_date, freq="D")
    modeling_dataset = pd.DataFrame(
        {
            "signal_date": dates,
            "target_end_date": dates,
            "target": [index % 2 for index in range(len(dates))],
        }
    )

    folds = build_walk_forward_folds(
        modeling_dataset=modeling_dataset,
        walk_forward=config.evaluation.walk_forward,
        frequency=config.data.interval,
    )

    assert len(folds) >= 10
    assert all(fold.train_rows >= config.evaluation.walk_forward.min_train_rows for fold in folds)
    assert all(fold.test_rows >= config.evaluation.walk_forward.min_test_rows for fold in folds)


def test_run_experiment_supports_capped_long_short_strategy_variants(tmp_path: Path) -> None:
    config_path = _write_run_experiment_config(
        tmp_path,
        models=[{"name": "logistic_regression"}],
        symbol_groups={
            "AAA": "growth",
            "BBB": "growth",
            "CCC": "defensive",
            "DDD": "defensive",
        },
        risk={
            "max_position_weight": 0.20,
            "max_group_weight": 0.30,
            "max_long_exposure": 0.40,
            "max_short_exposure": 0.40,
        },
    )

    result = run_marketlab_cli("run-experiment", config_path)
    assert_command_ok(result)

    run_root = tmp_path / "runs" / "integration_fixture"
    run_dir = latest_run_dir(run_root)
    metrics = pd.read_csv(run_dir / "metrics.csv")
    performance = pd.read_csv(run_dir / "performance.csv")
    strategy_summary = pd.read_csv(run_dir / "strategy_summary.csv")
    report_text = (run_dir / "report.md").read_text(encoding="utf-8")

    expected_strategy = (
        "ml_logistic_regression__poscap0p20__groupcap0p30__longcap0p40__shortcap0p40"
    )
    assert set(metrics["strategy"]) == {"buy_hold", "sma", expected_strategy}
    assert set(performance["strategy"]) == {"buy_hold", "sma", expected_strategy}
    assert set(strategy_summary["strategy"]) == {"buy_hold", "sma", expected_strategy}
    assert expected_strategy in report_text


def test_run_experiment_supports_capped_long_only_strategy_variants(tmp_path: Path) -> None:
    config_path = _write_run_experiment_config(
        tmp_path,
        models=[{"name": "logistic_regression"}],
        ranking={
            "mode": "long_only",
        },
        risk={
            "max_long_exposure": 0.60,
        },
    )

    result = run_marketlab_cli("run-experiment", config_path)
    assert_command_ok(result)

    run_root = tmp_path / "runs" / "integration_fixture"
    run_dir = latest_run_dir(run_root)
    metrics = pd.read_csv(run_dir / "metrics.csv")
    performance = pd.read_csv(run_dir / "performance.csv")
    strategy_summary = pd.read_csv(run_dir / "strategy_summary.csv")
    report_text = (run_dir / "report.md").read_text(encoding="utf-8")

    expected_strategy = "ml_logistic_regression__long_only__longcap0p60"
    assert set(metrics["strategy"]) == {"buy_hold", "sma", expected_strategy}
    assert set(performance["strategy"]) == {"buy_hold", "sma", expected_strategy}
    assert set(strategy_summary["strategy"]) == {"buy_hold", "sma", expected_strategy}
    assert expected_strategy in report_text

def test_run_experiment_supports_gated_cash_strategy_variants(tmp_path: Path) -> None:
    config_path = _write_run_experiment_config(
        tmp_path,
        ranking={
            "mode": "long_short",
            "min_score_threshold": 0.99,
            "cash_when_underfilled": True,
        },
    )

    result = run_marketlab_cli("run-experiment", config_path)
    assert_command_ok(result)

    run_root = tmp_path / "runs" / "integration_fixture"
    run_dir = latest_run_dir(run_root)
    metrics = pd.read_csv(run_dir / "metrics.csv")
    performance = pd.read_csv(run_dir / "performance.csv")
    strategy_summary = pd.read_csv(run_dir / "strategy_summary.csv")
    report_text = (run_dir / "report.md").read_text(encoding="utf-8")

    expected_strategies = {
        "buy_hold",
        "sma",
        "ml_logistic_regression__long_short__thr0p99__cash",
        "ml_logistic_l1__long_short__thr0p99__cash",
        "ml_random_forest__long_short__thr0p99__cash",
        "ml_extra_trees__long_short__thr0p99__cash",
        "ml_gradient_boosting__long_short__thr0p99__cash",
        "ml_hist_gradient_boosting__long_short__thr0p99__cash",
    }
    assert set(metrics["strategy"]) == expected_strategies
    assert set(performance["strategy"]) == expected_strategies
    assert set(strategy_summary["strategy"]) == expected_strategies
    assert "ml_logistic_regression__long_short__thr0p99__cash" in report_text
    assert "ml_logistic_l1__long_short__thr0p99__cash" in report_text






