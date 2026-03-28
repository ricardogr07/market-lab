from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
import yaml

from marketlab.cli import main
from marketlab.data.panel import save_panel_csv


def _build_synthetic_panel() -> pd.DataFrame:
    trading_dates = pd.bdate_range("2020-01-01", "2022-12-30")
    rows: list[dict[str, object]] = []

    for symbol_index, (symbol, base_price, amplitude) in enumerate(
        (
            ("AAA", 100.0, 0.45),
            ("BBB", 130.0, 0.40),
            ("CCC", 160.0, 0.35),
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


def _write_config(tmp_path: Path, *, models: list[dict[str, str]]) -> Path:
    cache_dir = tmp_path / "cache"
    runs_dir = tmp_path / "runs"
    panel_path = cache_dir / "panel.csv"

    save_panel_csv(_build_synthetic_panel(), panel_path)

    config = {
        "experiment_name": "integration_train_models",
        "data": {
            "symbols": ["AAA", "BBB", "CCC"],
            "start_date": "2020-01-01",
            "end_date": "2022-12-30",
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
                "rebalance_frequency": "W-FRI",
            }
        },
        "models": models,
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
            "save_report_md": False,
            "save_plots": False,
        },
    }

    config_path = tmp_path / "train_models.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return config_path


def test_train_models_writes_fold_metrics_manifest_predictions_and_summaries(tmp_path: Path) -> None:
    config_path = _write_config(
        tmp_path,
        models=[
            {"name": "logistic_regression"},
            {"name": "random_forest"},
        ],
    )

    result = main(["train-models", "--config", str(config_path)])

    assert result == 0

    run_root = tmp_path / "runs" / "integration_train_models"
    run_dirs = list(run_root.iterdir())
    assert len(run_dirs) == 1
    run_dir = run_dirs[0]

    folds_path = run_dir / "folds.csv"
    manifest_path = run_dir / "model_manifest.csv"
    metrics_path = run_dir / "model_metrics.csv"
    predictions_path = run_dir / "predictions.csv"
    model_summary_path = run_dir / "model_summary.csv"
    fold_summary_path = run_dir / "fold_summary.csv"

    assert folds_path.exists()
    assert manifest_path.exists()
    assert metrics_path.exists()
    assert predictions_path.exists()
    assert model_summary_path.exists()
    assert fold_summary_path.exists()

    folds = pd.read_csv(folds_path)
    manifest = pd.read_csv(manifest_path)
    metrics = pd.read_csv(metrics_path)
    predictions = pd.read_csv(predictions_path)
    model_summary = pd.read_csv(model_summary_path)
    fold_summary = pd.read_csv(fold_summary_path)

    assert not folds.empty
    assert not model_summary.empty
    assert not fold_summary.empty
    assert set(manifest["model_name"]) == {"logistic_regression", "random_forest"}
    assert set(metrics["model_name"]) == {"logistic_regression", "random_forest"}
    assert set(predictions["model_name"]) == {"logistic_regression", "random_forest"}
    assert set(model_summary["model_name"]) == {"logistic_regression", "random_forest"}
    assert set(fold_summary["fold_id"]) == set(folds["fold_id"])
    assert predictions["score"].between(0.0, 1.0).all()
    assert predictions["predicted_target"].isin([0, 1]).all()
    assert predictions.groupby("model_name")["fold_id"].nunique().gt(0).all()

    for relative_model_path in manifest["model_path"]:
        assert (run_dir / relative_model_path).exists()


def test_train_models_surfaces_unsupported_model_name(tmp_path: Path) -> None:
    config_path = _write_config(
        tmp_path,
        models=[{"name": "not_real"}],
    )

    with pytest.raises(ValueError, match="Unsupported model 'not_real'"):
        main(["train-models", "--config", str(config_path)])
