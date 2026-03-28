from __future__ import annotations

from pathlib import Path

import pandas as pd
from tests.integration import _cli_harness

from marketlab.data.panel import save_panel_csv

assert_command_ok = _cli_harness.assert_command_ok
build_synthetic_panel = _cli_harness.build_synthetic_panel
latest_run_dir = _cli_harness.latest_run_dir
write_yaml_config = _cli_harness.write_yaml_config
MODEL_SUMMARY_COLUMNS = _cli_harness.MODEL_SUMMARY_COLUMNS
FOLD_SUMMARY_COLUMNS = _cli_harness.FOLD_SUMMARY_COLUMNS
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

MODEL_METRICS_COLUMNS = [
    "model_name",
    "fold_id",
    "train_start",
    "train_end",
    "label_cutoff",
    "test_start",
    "test_end",
    "train_rows",
    "test_rows",
    "accuracy",
    "target_rate",
    "prediction_rate",
    "roc_auc",
    "log_loss",
]
PREDICTIONS_COLUMNS = [
    "model_name",
    "fold_id",
    "symbol",
    "signal_date",
    "effective_date",
    "target_end_date",
    "forward_return",
    "target",
    "score",
    "predicted_target",
]


def _write_config(tmp_path: Path, *, models: list[dict[str, str]]) -> Path:
    cache_dir = tmp_path / "cache"
    save_panel_csv(
        build_synthetic_panel(
            (
                ("AAA", 100.0, 0.45),
                ("BBB", 130.0, 0.40),
                ("CCC", 160.0, 0.35),
            ),
            start_date="2020-01-01",
            end_date="2022-12-30",
        ),
        cache_dir / "panel.csv",
    )

    return write_yaml_config(
        tmp_path / "train_models.yaml",
        {
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
                "output_dir": str(tmp_path / "runs"),
                "save_predictions": True,
                "save_metrics_csv": True,
                "save_report_md": False,
                "save_plots": False,
            },
        },
    )


def test_train_models_writes_fold_metrics_manifest_predictions_and_summaries(tmp_path: Path) -> None:
    config_path = _write_config(
        tmp_path,
        models=[
            {"name": "logistic_regression"},
            {"name": "random_forest"},
        ],
    )

    result = run_marketlab_cli("train-models", config_path)
    assert_command_ok(result)

    run_root = tmp_path / "runs" / "integration_train_models"
    run_dir = latest_run_dir(run_root)

    assert stdout_path(result) == run_dir.resolve()
    assert (run_dir / "models").is_dir()
    assert {path.name for path in run_dir.iterdir()} == {
        "folds.csv",
        "model_manifest.csv",
        "model_metrics.csv",
        "predictions.csv",
        "model_summary.csv",
        "fold_summary.csv",
        "models",
    }

    folds = pd.read_csv(run_dir / "folds.csv")
    manifest = pd.read_csv(run_dir / "model_manifest.csv")
    metrics = pd.read_csv(run_dir / "model_metrics.csv")
    predictions = pd.read_csv(run_dir / "predictions.csv")
    model_summary = pd.read_csv(run_dir / "model_summary.csv")
    fold_summary = pd.read_csv(run_dir / "fold_summary.csv")

    assert list(metrics.columns) == MODEL_METRICS_COLUMNS
    assert list(predictions.columns) == PREDICTIONS_COLUMNS
    assert list(model_summary.columns) == MODEL_SUMMARY_COLUMNS
    assert list(fold_summary.columns) == FOLD_SUMMARY_COLUMNS
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

    result = run_marketlab_cli("train-models", config_path)

    assert result.returncode != 0
    combined_output = f"{result.stdout}\n{result.stderr}"
    assert "Unsupported model 'not_real'" in combined_output
