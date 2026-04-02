from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
from tests.integration import _cli_harness

from marketlab.data.panel import save_panel_csv

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

DEFAULT_MODEL_NAMES = {
    "extra_trees",
    "gradient_boosting",
    "hist_gradient_boosting",
    "logistic_l1",
    "logistic_regression",
    "random_forest",
}


def _write_run_experiment_config(
    tmp_path: Path,
    *,
    walk_forward: dict[str, int | float] | None = None,
    ranking: dict[str, object] | None = None,
) -> Path:
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
                "ranking": ranking_payload,
                "costs": {"bps_per_trade": 10},
            },
            "baselines": {
                "buy_hold": True,
                "sma": {"enabled": True, "fast_window": 5, "slow_window": 10},
            },
            "models": [
                {"name": "logistic_regression"},
                {"name": "logistic_l1"},
                {"name": "random_forest"},
                {"name": "extra_trees"},
                {"name": "gradient_boosting"},
                {"name": "hist_gradient_boosting"},
            ],
            "evaluation": {
                "walk_forward": walk_forward_payload,
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
        "strategy_summary.csv",
        "monthly_returns.csv",
        "turnover_costs.csv",
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
    report_text = (run_dir / "report.md").read_text(encoding="utf-8")

    assert list(metrics.columns) == EXPECTED_METRICS_COLUMNS
    assert list(performance.columns) == PERFORMANCE_COLUMNS
    assert list(strategy_summary.columns) == STRATEGY_SUMMARY_COLUMNS
    assert list(monthly_returns.columns) == MONTHLY_RETURNS_COLUMNS
    assert list(turnover_costs.columns) == TURNOVER_COSTS_COLUMNS
    assert set(metrics["strategy"]) == {"buy_hold", "sma"}
    assert set(performance["strategy"]) == {"buy_hold", "sma"}
    assert set(strategy_summary["strategy"]) == {"buy_hold", "sma"}
    assert set(monthly_returns["strategy"]) == {"buy_hold", "sma"}
    assert set(turnover_costs["strategy"]) == {"buy_hold", "sma"}
    assert "## Strategy Summary" in report_text
    assert "## Monthly Net Returns" in report_text
    assert "## Turnover And Costs" in report_text
    assert not (run_dir / "fold_diagnostics.csv").exists()
    assert not (run_dir / "ranking_diagnostics.csv").exists()
    assert not (run_dir / "calibration_diagnostics.csv").exists()
    assert not (run_dir / "score_histograms.csv").exists()
    assert not (run_dir / "threshold_diagnostics.csv").exists()
    assert not (run_dir / "model_summary.csv").exists()
    assert not (run_dir / "fold_summary.csv").exists()
    assert not (run_dir / "calibration_curves.png").exists()
    assert not (run_dir / "score_histograms.png").exists()
    assert not (run_dir / "threshold_sweeps.png").exists()


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
