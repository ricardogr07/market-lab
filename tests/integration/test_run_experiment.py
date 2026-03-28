from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
import yaml

from marketlab.cli import main
from marketlab.data.panel import save_panel_csv


def _build_synthetic_panel() -> pd.DataFrame:
    trading_dates = pd.bdate_range("2020-01-01", "2024-12-31")
    rows: list[dict[str, object]] = []

    for symbol_index, (symbol, base_price, amplitude) in enumerate(
        (
            ("AAA", 100.0, 0.45),
            ("BBB", 130.0, 0.40),
            ("CCC", 160.0, 0.35),
            ("DDD", 190.0, 0.30),
        )
    ):
        close_price = base_price
        for row_index, timestamp in enumerate(trading_dates):
            week_ordinal = timestamp.to_period("W-FRI").ordinal + symbol_index
            direction = 1.0 if week_ordinal % 2 == 0 else -1.0
            open_price = close_price
            close_price = max(5.0, open_price + (amplitude * direction))
            high_price = max(open_price, close_price) + 0.2
            low_price = min(open_price, close_price) - 0.2

            rows.append(
                {
                    "symbol": symbol,
                    "timestamp": timestamp,
                    "open": round(open_price, 4),
                    "high": round(high_price, 4),
                    "low": round(low_price, 4),
                    "close": round(close_price, 4),
                    "volume": 1_000_000 + (symbol_index * 10_000) + row_index,
                    "adj_close": round(close_price, 4),
                    "adj_factor": 1.0,
                    "adj_open": round(open_price, 4),
                    "adj_high": round(high_price, 4),
                    "adj_low": round(low_price, 4),
                }
            )

    return pd.DataFrame(rows).sort_values(["symbol", "timestamp"]).reset_index(drop=True)


def _write_run_experiment_config(tmp_path: Path) -> Path:
    cache_dir = tmp_path / "cache"
    runs_dir = tmp_path / "runs"
    save_panel_csv(_build_synthetic_panel(), cache_dir / "panel.csv")

    config = {
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
            "output_dir": str(runs_dir),
            "save_predictions": True,
            "save_metrics_csv": True,
            "save_report_md": True,
            "save_plots": True,
        },
    }

    config_path = tmp_path / "integration.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return config_path


def _write_backtest_config(tmp_path: Path) -> Path:
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir(parents=True)
    fixture_panel = Path(__file__).resolve().parents[1] / "fixtures" / "market_panel.csv"
    (cache_dir / "panel.csv").write_text(fixture_panel.read_text(encoding="utf-8"), encoding="utf-8")

    config = {
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
    }

    config_path = tmp_path / "backtest.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return config_path


def test_run_experiment_produces_baseline_and_ml_artifacts(tmp_path: Path) -> None:
    config_path = _write_run_experiment_config(tmp_path)

    result = main(["run-experiment", "--config", str(config_path)])

    assert result == 0

    run_root = tmp_path / "runs" / "integration_fixture"
    run_dirs = list(run_root.iterdir())
    assert len(run_dirs) == 1
    run_dir = run_dirs[0]

    assert (run_dir / "metrics.csv").exists()
    assert (run_dir / "performance.csv").exists()
    assert (run_dir / "report.md").exists()
    assert (run_dir / "cumulative_returns.png").exists()
    assert (run_dir / "drawdown.png").exists()
    assert (run_dir / "model_summary.csv").exists()
    assert (run_dir / "fold_summary.csv").exists()

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
    assert set(metrics["strategy"]) == expected_strategies
    assert set(performance["strategy"]) == expected_strategies
    assert set(model_summary["model_name"]) == {"logistic_regression", "random_forest"}
    assert not fold_summary.empty

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

    result = main(["backtest", "--config", str(config_path)])

    assert result == 0

    run_root = tmp_path / "runs" / "integration_backtest_fixture"
    run_dirs = list(run_root.iterdir())
    assert len(run_dirs) == 1
    run_dir = run_dirs[0]

    metrics = pd.read_csv(run_dir / "metrics.csv")
    performance = pd.read_csv(run_dir / "performance.csv")

    assert set(metrics["strategy"]) == {"buy_hold", "sma"}
    assert set(performance["strategy"]) == {"buy_hold", "sma"}
    assert not (run_dir / "model_summary.csv").exists()
    assert not (run_dir / "fold_summary.csv").exists()
