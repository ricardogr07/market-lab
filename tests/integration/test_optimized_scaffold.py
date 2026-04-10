from __future__ import annotations

from pathlib import Path

import pandas as pd
from tests.integration import _cli_harness

from marketlab.data.panel import save_panel_csv

assert_command_ok = _cli_harness.assert_command_ok
build_synthetic_panel = _cli_harness.build_synthetic_panel
latest_run_dir = _cli_harness.latest_run_dir
load_fixture_panel = _cli_harness.load_fixture_panel
run_marketlab_cli = getattr(
    _cli_harness,
    "run_marketlab_cli",
    _cli_harness.run_launcher_command,
)
write_yaml_config = _cli_harness.write_yaml_config


def _black_litterman_optimized(symbols: list[str]) -> dict[str, object]:
    equilibrium = {symbol: round(1.0 / len(symbols), 6) for symbol in symbols}
    first, second, third = symbols[:3]
    last = symbols[-1]
    return {
        "enabled": True,
        "method": "black_litterman",
        "lookback_days": 3,
        "target_gross_exposure": 0.6,
        "risk_aversion": 1.0,
        "equilibrium_weights": equilibrium,
        "tau": 0.05,
        "views": [
            {
                "name": "growth_over_defensive",
                "weights": {first: 1.0, second: 1.0, third: -1.0},
                "view_return": 0.0025,
            },
            {
                "name": "core_over_tail",
                "weights": {first: 1.0, last: -1.0},
                "view_return": 0.001,
            },
        ],
    }


def _write_backtest_config(
    tmp_path: Path,
    *,
    optimized: dict[str, object] | None = None,
    buy_hold: bool = True,
    sma_enabled: bool = True,
) -> Path:
    cache_dir = tmp_path / "cache"
    save_panel_csv(load_fixture_panel(), cache_dir / "panel.csv")
    baselines_payload: dict[str, object] = {
        "buy_hold": buy_hold,
        "sma": {"enabled": sma_enabled, "fast_window": 2, "slow_window": 3},
    }
    if optimized is not None:
        baselines_payload["optimized"] = optimized
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
            "baselines": baselines_payload,
            "artifacts": {
                "output_dir": str(tmp_path / "runs"),
                "save_predictions": False,
                "save_metrics_csv": True,
                "save_report_md": True,
                "save_plots": True,
            },
        },
    )


def _write_run_experiment_config(
    tmp_path: Path,
    *,
    optimized: dict[str, object] | None = None,
) -> Path:
    cache_dir = tmp_path / "cache"
    save_panel_csv(
        build_synthetic_panel(
            (("AAA", 100.0, 0.45), ("BBB", 130.0, 0.40), ("CCC", 160.0, 0.35), ("DDD", 190.0, 0.30)),
            start_date="2020-01-01",
            end_date="2024-12-31",
        ),
        cache_dir / "panel.csv",
    )
    baselines_payload: dict[str, object] = {
        "buy_hold": True,
        "sma": {"enabled": True, "fast_window": 5, "slow_window": 10},
    }
    if optimized is not None:
        baselines_payload["optimized"] = optimized
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
            "baselines": baselines_payload,
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


def test_backtest_supports_mean_variance_optimized_baseline(tmp_path: Path) -> None:
    config_path = _write_backtest_config(
        tmp_path,
        optimized={
            "enabled": True,
            "method": "mean_variance",
            "lookback_days": 3,
            "target_gross_exposure": 0.6,
            "risk_aversion": 1.0,
        },
    )

    result = run_marketlab_cli("backtest", config_path)
    assert_command_ok(result)

    run_root = tmp_path / "runs" / "optimized_scaffold_backtest"
    run_dir = latest_run_dir(run_root)
    metrics = pd.read_csv(run_dir / "metrics.csv")
    strategy_summary = pd.read_csv(run_dir / "strategy_summary.csv")
    report_text = (run_dir / "report.md").read_text(encoding="utf-8")

    assert set(metrics["strategy"]) == {"buy_hold", "sma", "mean_variance"}
    assert set(strategy_summary["strategy"]) == {"buy_hold", "sma", "mean_variance"}
    mean_variance_row = strategy_summary.loc[
        strategy_summary["strategy"] == "mean_variance"
    ].iloc[0]
    assert mean_variance_row["avg_gross_exposure"] <= 0.6 + 1e-6
    assert "mean_variance" in report_text
    assert "Covariance Diagnostics" in report_text
    assert "covariance_diagnostics.csv" in report_text
    assert (run_dir / "covariance_diagnostics.csv").exists()
    assert not (run_dir / "black_litterman_assumptions.csv").exists()



def test_backtest_keeps_mean_variance_as_cash_when_no_windows_exist(tmp_path: Path) -> None:
    config_path = _write_backtest_config(
        tmp_path,
        optimized={
            "enabled": True,
            "method": "mean_variance",
            "lookback_days": 10_000,
            "target_gross_exposure": 0.6,
            "risk_aversion": 1.0,
        },
        buy_hold=False,
        sma_enabled=False,
    )

    result = run_marketlab_cli("backtest", config_path)
    assert_command_ok(result)

    run_root = tmp_path / "runs" / "optimized_scaffold_backtest"
    run_dir = latest_run_dir(run_root)
    metrics = pd.read_csv(run_dir / "metrics.csv")
    strategy_summary = pd.read_csv(run_dir / "strategy_summary.csv")

    assert set(metrics["strategy"]) == {"mean_variance"}
    assert set(strategy_summary["strategy"]) == {"mean_variance"}
    mean_variance_row = strategy_summary.loc[
        strategy_summary["strategy"] == "mean_variance"
    ].iloc[0]
    assert mean_variance_row["avg_gross_exposure"] == 0.0
    assert mean_variance_row["avg_cash_weight"] == 1.0
    assert not (run_dir / "covariance_diagnostics.csv").exists()


def test_backtest_supports_risk_parity_optimized_baseline(tmp_path: Path) -> None:
    config_path = _write_backtest_config(
        tmp_path,
        optimized={
            "enabled": True,
            "method": "risk_parity",
            "lookback_days": 3,
            "target_gross_exposure": 0.6,
        },
    )

    result = run_marketlab_cli("backtest", config_path)
    assert_command_ok(result)

    run_root = tmp_path / "runs" / "optimized_scaffold_backtest"
    run_dir = latest_run_dir(run_root)
    metrics = pd.read_csv(run_dir / "metrics.csv")
    strategy_summary = pd.read_csv(run_dir / "strategy_summary.csv")
    report_text = (run_dir / "report.md").read_text(encoding="utf-8")

    assert set(metrics["strategy"]) == {"buy_hold", "sma", "risk_parity"}
    assert set(strategy_summary["strategy"]) == {"buy_hold", "sma", "risk_parity"}
    risk_parity_row = strategy_summary.loc[
        strategy_summary["strategy"] == "risk_parity"
    ].iloc[0]
    assert risk_parity_row["avg_gross_exposure"] <= 0.6 + 1e-6
    assert "risk_parity" in report_text
    assert "Covariance Diagnostics" in report_text
    assert "covariance_diagnostics.csv" in report_text
    assert (run_dir / "covariance_diagnostics.csv").exists()


def test_backtest_keeps_risk_parity_as_cash_when_no_windows_exist(tmp_path: Path) -> None:
    config_path = _write_backtest_config(
        tmp_path,
        optimized={
            "enabled": True,
            "method": "risk_parity",
            "lookback_days": 10_000,
            "target_gross_exposure": 0.6,
        },
        buy_hold=False,
        sma_enabled=False,
    )

    result = run_marketlab_cli("backtest", config_path)
    assert_command_ok(result)

    run_root = tmp_path / "runs" / "optimized_scaffold_backtest"
    run_dir = latest_run_dir(run_root)
    metrics = pd.read_csv(run_dir / "metrics.csv")
    strategy_summary = pd.read_csv(run_dir / "strategy_summary.csv")

    assert set(metrics["strategy"]) == {"risk_parity"}
    assert set(strategy_summary["strategy"]) == {"risk_parity"}
    risk_parity_row = strategy_summary.loc[
        strategy_summary["strategy"] == "risk_parity"
    ].iloc[0]
    assert risk_parity_row["avg_gross_exposure"] == 0.0
    assert risk_parity_row["avg_cash_weight"] == 1.0
    assert not (run_dir / "covariance_diagnostics.csv").exists()


def test_backtest_supports_black_litterman_optimized_baseline(tmp_path: Path) -> None:
    config_path = _write_run_experiment_config(
        tmp_path,
        optimized=_black_litterman_optimized(["AAA", "BBB", "CCC", "DDD"]),
    )

    result = run_marketlab_cli("backtest", config_path)
    assert_command_ok(result)

    run_root = tmp_path / "runs" / "optimized_scaffold_experiment"
    run_dir = latest_run_dir(run_root)
    metrics = pd.read_csv(run_dir / "metrics.csv")
    strategy_summary = pd.read_csv(run_dir / "strategy_summary.csv")
    assumptions = pd.read_csv(run_dir / "black_litterman_assumptions.csv")
    report_text = (run_dir / "report.md").read_text(encoding="utf-8")

    assert "black_litterman" in set(metrics["strategy"])
    assert "black_litterman" in set(strategy_summary["strategy"])
    assert not assumptions.empty
    assert set(assumptions.columns) >= {
        "signal_date",
        "effective_date",
        "symbol",
        "equilibrium_weight",
        "implied_prior_return",
        "posterior_expected_return",
        "tau",
    }
    assert assumptions["tau"].eq(0.05).all()
    assert (run_dir / "covariance_diagnostics.csv").exists()
    assert "Black-Litterman Assumptions" in report_text
    assert "Covariance Diagnostics" in report_text
    assert "growth_over_defensive" in report_text
    assert "black_litterman_assumptions.csv" in report_text
    assert "covariance_diagnostics.csv" in report_text


def test_backtest_black_litterman_cash_fallback_skips_assumptions_artifact(tmp_path: Path) -> None:
    optimized = _black_litterman_optimized(["VOO", "QQQ", "SMH", "XLV", "IEMG"])
    optimized["lookback_days"] = 10_000
    config_path = _write_backtest_config(
        tmp_path,
        optimized=optimized,
        buy_hold=False,
        sma_enabled=False,
    )

    result = run_marketlab_cli("backtest", config_path)
    assert_command_ok(result)

    run_root = tmp_path / "runs" / "optimized_scaffold_backtest"
    run_dir = latest_run_dir(run_root)
    metrics = pd.read_csv(run_dir / "metrics.csv")
    strategy_summary = pd.read_csv(run_dir / "strategy_summary.csv")
    report_text = (run_dir / "report.md").read_text(encoding="utf-8")

    assert set(metrics["strategy"]) == {"black_litterman"}
    assert set(strategy_summary["strategy"]) == {"black_litterman"}
    black_litterman_row = strategy_summary.loc[
        strategy_summary["strategy"] == "black_litterman"
    ].iloc[0]
    assert black_litterman_row["avg_gross_exposure"] == 0.0
    assert black_litterman_row["avg_cash_weight"] == 1.0
    assert not (run_dir / "black_litterman_assumptions.csv").exists()
    assert not (run_dir / "covariance_diagnostics.csv").exists()
    assert "Black-Litterman Assumptions" not in report_text
    assert "Covariance Diagnostics" not in report_text


def test_run_experiment_supports_black_litterman_optimized_baseline(tmp_path: Path) -> None:
    config_path = _write_run_experiment_config(
        tmp_path,
        optimized=_black_litterman_optimized(["AAA", "BBB", "CCC", "DDD"]),
    )

    result = run_marketlab_cli("run-experiment", config_path)

    assert_command_ok(result)

    run_root = tmp_path / "runs" / "optimized_scaffold_experiment"
    run_dir = latest_run_dir(run_root)
    metrics = pd.read_csv(run_dir / "metrics.csv")
    performance = pd.read_csv(run_dir / "performance.csv", parse_dates=["date"])
    strategy_summary = pd.read_csv(run_dir / "strategy_summary.csv")
    assumptions = pd.read_csv(
        run_dir / "black_litterman_assumptions.csv",
        parse_dates=["signal_date", "effective_date"],
    )
    report_text = (run_dir / "report.md").read_text(encoding="utf-8")

    assert "black_litterman" in set(metrics["strategy"])
    assert "black_litterman" in set(strategy_summary["strategy"])
    assert not assumptions.empty
    assert assumptions["tau"].eq(0.05).all()
    assert assumptions["effective_date"].min() >= performance["date"].min()
    assert assumptions["effective_date"].max() <= performance["date"].max()
    assert set(assumptions["effective_date"]) <= set(performance["date"])
    covariance = pd.read_csv(
        run_dir / "covariance_diagnostics.csv",
        parse_dates=["signal_date", "effective_date"],
    )
    performance_dates = pd.Index(performance["date"]).sort_values()
    covariance_effective_dates = pd.Index(
        pd.to_datetime(covariance["effective_date"]).drop_duplicates()
    ).sort_values()
    pre_oos_dates = covariance_effective_dates[covariance_effective_dates < performance_dates.min()]
    in_window_dates = covariance_effective_dates[covariance_effective_dates >= performance_dates.min()]

    assert not covariance.empty
    assert set(covariance["strategy"]) == {"black_litterman"}
    assert len(pre_oos_dates) == 1
    assert (covariance["effective_date"] == pre_oos_dates[0]).sum() == 16
    assert len(in_window_dates) > 0
    assert in_window_dates.max() <= performance_dates.max()
    assert set(in_window_dates) <= set(performance_dates)
    assert "Black-Litterman Assumptions" in report_text
    assert "Covariance Diagnostics" in report_text
    assert "core_over_tail" in report_text
    assert "black_litterman_assumptions.csv" in report_text
    assert "covariance_diagnostics.csv" in report_text
