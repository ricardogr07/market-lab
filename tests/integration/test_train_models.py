from __future__ import annotations

from pathlib import Path

import pandas as pd
from tests.integration import _cli_harness

from marketlab.data.panel import save_panel_csv

assert_command_ok = _cli_harness.assert_command_ok
build_synthetic_panel = _cli_harness.build_synthetic_panel
latest_run_dir = _cli_harness.latest_run_dir
write_yaml_config = _cli_harness.write_yaml_config
MODEL_METRICS_COLUMNS = _cli_harness.MODEL_METRICS_COLUMNS
MODEL_SUMMARY_COLUMNS = _cli_harness.MODEL_SUMMARY_COLUMNS
FOLD_SUMMARY_COLUMNS = _cli_harness.FOLD_SUMMARY_COLUMNS
FOLD_DIAGNOSTICS_COLUMNS = _cli_harness.FOLD_DIAGNOSTICS_COLUMNS
RANKING_DIAGNOSTICS_COLUMNS = _cli_harness.RANKING_DIAGNOSTICS_COLUMNS
CALIBRATION_DIAGNOSTICS_COLUMNS = _cli_harness.CALIBRATION_DIAGNOSTICS_COLUMNS
SCORE_HISTOGRAM_COLUMNS = _cli_harness.SCORE_HISTOGRAM_COLUMNS
THRESHOLD_DIAGNOSTICS_COLUMNS = _cli_harness.THRESHOLD_DIAGNOSTICS_COLUMNS
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

DEFAULT_MODEL_NAMES = {
    "extra_trees",
    "gradient_boosting",
    "hist_gradient_boosting",
    "logistic_l1",
    "logistic_regression",
    "random_forest",
}


def _write_config(
    tmp_path: Path,
    *,
    models: list[dict[str, str]],
    walk_forward: dict[str, int | float] | None = None,
    ranking: dict[str, object] | None = None,
    symbol_specs: tuple[tuple[str, float, float], ...] | None = None,
) -> Path:
    cache_dir = tmp_path / "cache"
    resolved_symbol_specs = symbol_specs or (
        ("AAA", 100.0, 0.45),
        ("BBB", 130.0, 0.40),
        ("CCC", 160.0, 0.35),
    )
    save_panel_csv(
        build_synthetic_panel(
            resolved_symbol_specs,
            start_date="2020-01-01",
            end_date="2022-12-30",
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
        tmp_path / "train_models.yaml",
        {
            "experiment_name": "integration_train_models",
            "data": {
                "symbols": [symbol for symbol, _, _ in resolved_symbol_specs],
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
                "ranking": ranking_payload,
            },
            "models": models,
            "evaluation": {
                "walk_forward": walk_forward_payload,
            },
            "artifacts": {
                "output_dir": str(tmp_path / "runs"),
                "save_predictions": True,
                "save_metrics_csv": True,
                "save_report_md": False,
                "save_plots": True,
            },
        },
    )


def test_train_models_writes_fold_metrics_manifest_predictions_and_summaries(tmp_path: Path) -> None:
    config_path = _write_config(
        tmp_path,
        models=[
            {"name": "logistic_regression"},
            {"name": "logistic_l1"},
            {"name": "random_forest"},
            {"name": "extra_trees"},
            {"name": "gradient_boosting"},
            {"name": "hist_gradient_boosting"},
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
        "fold_diagnostics.csv",
        "ranking_diagnostics.csv",
        "calibration_diagnostics.csv",
        "score_histograms.csv",
        "threshold_diagnostics.csv",
        "model_manifest.csv",
        "model_metrics.csv",
        "predictions.csv",
        "model_summary.csv",
        "fold_summary.csv",
        "calibration_curves.png",
        "score_histograms.png",
        "threshold_sweeps.png",
        "models",
    }

    folds = pd.read_csv(run_dir / "folds.csv")
    fold_diagnostics = pd.read_csv(run_dir / "fold_diagnostics.csv")
    ranking_diagnostics = pd.read_csv(run_dir / "ranking_diagnostics.csv")
    calibration_diagnostics = pd.read_csv(run_dir / "calibration_diagnostics.csv")
    score_histograms = pd.read_csv(run_dir / "score_histograms.csv")
    threshold_diagnostics = pd.read_csv(run_dir / "threshold_diagnostics.csv")
    manifest = pd.read_csv(run_dir / "model_manifest.csv")
    metrics = pd.read_csv(run_dir / "model_metrics.csv")
    predictions = pd.read_csv(run_dir / "predictions.csv")
    model_summary = pd.read_csv(run_dir / "model_summary.csv")
    fold_summary = pd.read_csv(run_dir / "fold_summary.csv")

    assert list(fold_diagnostics.columns) == FOLD_DIAGNOSTICS_COLUMNS
    assert list(ranking_diagnostics.columns) == RANKING_DIAGNOSTICS_COLUMNS
    assert list(calibration_diagnostics.columns) == CALIBRATION_DIAGNOSTICS_COLUMNS
    assert list(score_histograms.columns) == SCORE_HISTOGRAM_COLUMNS
    assert list(threshold_diagnostics.columns) == THRESHOLD_DIAGNOSTICS_COLUMNS
    assert list(metrics.columns) == MODEL_METRICS_COLUMNS
    assert list(predictions.columns) == PREDICTIONS_COLUMNS
    assert list(model_summary.columns) == MODEL_SUMMARY_COLUMNS
    assert list(fold_summary.columns) == FOLD_SUMMARY_COLUMNS
    assert not folds.empty
    assert not fold_diagnostics.empty
    assert not ranking_diagnostics.empty
    assert not calibration_diagnostics.empty
    assert not score_histograms.empty
    assert not threshold_diagnostics.empty
    assert not model_summary.empty
    assert not fold_summary.empty
    assert set(fold_diagnostics["status"]).issubset({"used", "skipped"})
    assert "used" in set(fold_diagnostics["status"])
    assert set(ranking_diagnostics["bucket_status"]).issubset({"used", "underfilled"})
    assert set(threshold_diagnostics["threshold_status"]).issubset({"used", "empty"})
    assert set(manifest["model_name"]) == DEFAULT_MODEL_NAMES
    assert set(metrics["model_name"]) == DEFAULT_MODEL_NAMES
    assert set(predictions["model_name"]) == DEFAULT_MODEL_NAMES
    assert set(ranking_diagnostics["model_name"]) == DEFAULT_MODEL_NAMES
    assert set(calibration_diagnostics["model_name"]) == DEFAULT_MODEL_NAMES
    assert set(score_histograms["model_name"]) == DEFAULT_MODEL_NAMES
    assert set(threshold_diagnostics["model_name"]) == DEFAULT_MODEL_NAMES
    assert set(model_summary["model_name"]) == DEFAULT_MODEL_NAMES
    assert set(fold_summary["fold_id"]) == set(folds["fold_id"])
    assert set(folds["fold_id"]) == set(
        fold_diagnostics.loc[fold_diagnostics["status"] == "used", "fold_id"].dropna().astype(int)
    )
    assert predictions["score"].between(0.0, 1.0).all()
    assert predictions["predicted_target"].isin([0, 1]).all()
    assert predictions.groupby("model_name")["fold_id"].nunique().gt(0).all()
    assert metrics["spread_signal_count"].ge(0).all()
    assert metrics["ece"].ge(0.0).all()
    assert metrics["max_calibration_gap"].ge(0.0).all()

    metrics_index = metrics.set_index(["model_name", "fold_id"])
    used_signal_counts = (
        ranking_diagnostics.loc[ranking_diagnostics["bucket_status"] == "used"]
        .groupby(["model_name", "fold_id"])["signal_date"]
        .nunique()
    )
    for key, count in used_signal_counts.items():
        assert int(metrics_index.loc[key, "spread_signal_count"]) == int(count)

    for relative_model_path in manifest["model_path"]:
        assert (run_dir / relative_model_path).exists()
    for plot_name in ["calibration_curves.png", "score_histograms.png", "threshold_sweeps.png"]:
        assert (run_dir / plot_name).exists()


def test_train_models_writes_diagnostics_before_failing_on_zero_usable_folds(tmp_path: Path) -> None:
    config_path = _write_config(
        tmp_path,
        models=[{"name": "logistic_regression"}],
        walk_forward={"min_train_rows": 1000},
    )

    result = run_marketlab_cli("train-models", config_path)

    assert result.returncode != 0
    combined_output = f"{result.stdout}\n{result.stderr}"
    run_root = tmp_path / "runs" / "integration_train_models"
    run_dir = latest_run_dir(run_root)
    diagnostics_path = (run_dir / "fold_diagnostics.csv").resolve()
    fold_diagnostics = pd.read_csv(diagnostics_path)

    assert "No walk-forward folds are available for train-models." in combined_output
    assert str(diagnostics_path) in combined_output
    assert diagnostics_path.exists()
    assert list(fold_diagnostics.columns) == FOLD_DIAGNOSTICS_COLUMNS
    assert (fold_diagnostics["status"] == "skipped").all()
    assert fold_diagnostics["fold_id"].isna().all()


def test_train_models_surfaces_unsupported_model_name(tmp_path: Path) -> None:
    config_path = _write_config(
        tmp_path,
        models=[{"name": "not_real"}],
    )

    result = run_marketlab_cli("train-models", config_path)

    assert result.returncode != 0
    combined_output = f"{result.stdout}\n{result.stderr}"
    assert "Unsupported model 'not_real'" in combined_output


def test_train_models_keeps_existing_artifact_contract_with_strategy_mode_fields(tmp_path: Path) -> None:
    config_path = _write_config(
        tmp_path,
        models=[{"name": "logistic_regression"}],
        ranking={
            "mode": "long_only",
            "min_score_threshold": 0.8,
            "cash_when_underfilled": True,
        },
    )

    result = run_marketlab_cli("train-models", config_path)
    assert_command_ok(result)

    run_root = tmp_path / "runs" / "integration_train_models"
    run_dir = latest_run_dir(run_root)
    metrics = pd.read_csv(run_dir / "model_metrics.csv")
    ranking_diagnostics = pd.read_csv(run_dir / "ranking_diagnostics.csv")
    model_summary = pd.read_csv(run_dir / "model_summary.csv")

    assert list(metrics.columns) == MODEL_METRICS_COLUMNS
    assert list(model_summary.columns) == MODEL_SUMMARY_COLUMNS
    assert set(ranking_diagnostics["model_name"]) == {"logistic_regression"}
    assert set(ranking_diagnostics["evaluation_mode"]) == {"long_only"}
    assert metrics["top_bucket_signal_count"].gt(0).all()
    assert metrics["spread_signal_count"].eq(0).all()


def test_train_models_supports_single_symbol_long_only_evaluation(tmp_path: Path) -> None:
    config_path = _write_config(
        tmp_path,
        models=[{"name": "logistic_regression"}],
        ranking={
            "mode": "long_only",
            "long_n": 1,
            "short_n": 1,
        },
        symbol_specs=(("VOO", 100.0, 0.45),),
    )

    result = run_marketlab_cli("train-models", config_path)
    assert_command_ok(result)

    run_root = tmp_path / "runs" / "integration_train_models"
    run_dir = latest_run_dir(run_root)
    metrics = pd.read_csv(run_dir / "model_metrics.csv")
    ranking_diagnostics = pd.read_csv(run_dir / "ranking_diagnostics.csv")
    model_summary = pd.read_csv(run_dir / "model_summary.csv")
    fold_summary = pd.read_csv(run_dir / "fold_summary.csv")

    assert set(ranking_diagnostics["evaluation_mode"]) == {"long_only"}
    assert ranking_diagnostics["bucket_status"].eq("used").all()
    assert ranking_diagnostics["top_bucket_size"].eq(1).all()
    assert ranking_diagnostics["bottom_bucket_size"].eq(0).all()
    assert ranking_diagnostics["top_bucket_return"].notna().all()
    assert ranking_diagnostics["bottom_bucket_return"].isna().all()
    assert ranking_diagnostics["top_bottom_spread"].isna().all()
    assert metrics["top_bucket_signal_count"].gt(0).all()
    assert metrics["top_bucket_return"].notna().all()
    assert metrics["top_bucket_hit_rate"].between(0.0, 1.0).all()
    assert metrics["spread_signal_count"].eq(0).all()
    assert metrics["top_bottom_spread"].isna().all()
    assert model_summary["mean_top_bucket_return"].notna().all()
    assert model_summary["worst_top_bucket_return"].notna().all()
    assert model_summary["mean_top_bucket_signal_count"].gt(0).all()
    assert model_summary["mean_top_bottom_spread"].isna().all()
    assert fold_summary["best_model_by_top_bucket_return"].eq("logistic_regression").all()
    assert fold_summary["best_top_bucket_return"].notna().all()
    assert fold_summary["best_model_by_top_bottom_spread"].isna().all()
    assert fold_summary["best_top_bottom_spread"].isna().all()
