from __future__ import annotations

import pandas as pd

MODEL_SUMMARY_COLUMNS = [
    "model_name",
    "estimator_label",
    "fold_count",
    "first_test_start",
    "last_test_end",
    "mean_accuracy",
    "mean_balanced_accuracy",
    "mean_precision",
    "mean_recall",
    "mean_f1",
    "mean_roc_auc",
    "mean_log_loss",
    "mean_brier_score",
    "mean_ece",
    "mean_max_calibration_gap",
    "mean_target_rate",
    "mean_prediction_rate",
    "mean_rank_corr",
    "mean_top_bucket_return",
    "mean_bottom_bucket_return",
    "mean_top_bottom_spread",
    "mean_spread_hit_rate",
    "worst_top_bottom_spread",
    "mean_train_rows",
    "mean_test_rows",
]

FOLD_SUMMARY_COLUMNS = [
    "fold_id",
    "label_cutoff",
    "test_start",
    "test_end",
    "train_rows",
    "test_rows",
    "models_evaluated",
    "mean_accuracy",
    "mean_balanced_accuracy",
    "mean_precision",
    "mean_recall",
    "mean_f1",
    "mean_roc_auc",
    "mean_log_loss",
    "mean_brier_score",
    "mean_ece",
    "mean_max_calibration_gap",
    "mean_rank_corr",
    "mean_top_bucket_return",
    "mean_bottom_bucket_return",
    "mean_top_bottom_spread",
    "mean_spread_hit_rate",
    "worst_top_bottom_spread",
    "best_model_by_roc_auc",
    "best_roc_auc",
    "best_model_by_top_bottom_spread",
    "best_top_bottom_spread",
]


def _require_columns(frame: pd.DataFrame, required: set[str], label: str) -> None:
    missing = required - set(frame.columns)
    if missing:
        joined = ", ".join(sorted(missing))
        raise ValueError(f"{label} is missing required columns: {joined}")


def build_model_summary(
    model_metrics: pd.DataFrame,
    model_manifest: pd.DataFrame,
) -> pd.DataFrame:
    _require_columns(
        model_metrics,
        {
            "model_name",
            "fold_id",
            "accuracy",
            "balanced_accuracy",
            "precision",
            "recall",
            "f1",
            "roc_auc",
            "log_loss",
            "brier_score",
            "ece",
            "max_calibration_gap",
            "target_rate",
            "prediction_rate",
            "rank_corr",
            "top_bucket_return",
            "bottom_bucket_return",
            "top_bottom_spread",
            "spread_hit_rate",
            "worst_top_bottom_spread",
            "train_rows",
            "test_rows",
        },
        "Model metrics",
    )
    _require_columns(
        model_manifest,
        {"model_name", "fold_id", "estimator_label", "test_start", "test_end"},
        "Model manifest",
    )

    if model_metrics.empty or model_manifest.empty:
        return pd.DataFrame(columns=MODEL_SUMMARY_COLUMNS)

    metric_rows = model_metrics.loc[
        :,
        [
            "model_name",
            "fold_id",
            "accuracy",
            "balanced_accuracy",
            "precision",
            "recall",
            "f1",
            "roc_auc",
            "log_loss",
            "brier_score",
            "ece",
            "max_calibration_gap",
            "target_rate",
            "prediction_rate",
            "rank_corr",
            "top_bucket_return",
            "bottom_bucket_return",
            "top_bottom_spread",
            "spread_hit_rate",
            "worst_top_bottom_spread",
            "train_rows",
            "test_rows",
        ],
    ].copy()
    manifest_rows = model_manifest.loc[
        :,
        ["model_name", "fold_id", "estimator_label", "test_start", "test_end"],
    ].copy()

    merged = metric_rows.merge(
        manifest_rows.loc[:, ["model_name", "fold_id", "estimator_label"]],
        on=["model_name", "fold_id"],
        how="inner",
        validate="one_to_one",
    )
    if merged.empty:
        return pd.DataFrame(columns=MODEL_SUMMARY_COLUMNS)

    metrics_summary = (
        merged.groupby(["model_name", "estimator_label"], as_index=False)
        .agg(
            fold_count=("fold_id", "nunique"),
            mean_accuracy=("accuracy", "mean"),
            mean_balanced_accuracy=("balanced_accuracy", "mean"),
            mean_precision=("precision", "mean"),
            mean_recall=("recall", "mean"),
            mean_f1=("f1", "mean"),
            mean_roc_auc=("roc_auc", "mean"),
            mean_log_loss=("log_loss", "mean"),
            mean_brier_score=("brier_score", "mean"),
            mean_ece=("ece", "mean"),
            mean_max_calibration_gap=("max_calibration_gap", "mean"),
            mean_target_rate=("target_rate", "mean"),
            mean_prediction_rate=("prediction_rate", "mean"),
            mean_rank_corr=("rank_corr", "mean"),
            mean_top_bucket_return=("top_bucket_return", "mean"),
            mean_bottom_bucket_return=("bottom_bucket_return", "mean"),
            mean_top_bottom_spread=("top_bottom_spread", "mean"),
            mean_spread_hit_rate=("spread_hit_rate", "mean"),
            worst_top_bottom_spread=("worst_top_bottom_spread", "min"),
            mean_train_rows=("train_rows", "mean"),
            mean_test_rows=("test_rows", "mean"),
        )
        .sort_values(["model_name"])
        .reset_index(drop=True)
    )
    date_summary = (
        manifest_rows.groupby(["model_name", "estimator_label"], as_index=False)
        .agg(
            first_test_start=("test_start", "min"),
            last_test_end=("test_end", "max"),
        )
        .sort_values(["model_name"])
        .reset_index(drop=True)
    )
    summary = metrics_summary.merge(
        date_summary,
        on=["model_name", "estimator_label"],
        how="inner",
        validate="one_to_one",
    )
    return summary.loc[:, MODEL_SUMMARY_COLUMNS]


def build_fold_summary(
    model_metrics: pd.DataFrame,
    model_manifest: pd.DataFrame,
) -> pd.DataFrame:
    _require_columns(
        model_metrics,
        {
            "model_name",
            "fold_id",
            "accuracy",
            "balanced_accuracy",
            "precision",
            "recall",
            "f1",
            "roc_auc",
            "log_loss",
            "brier_score",
            "ece",
            "max_calibration_gap",
            "rank_corr",
            "top_bucket_return",
            "bottom_bucket_return",
            "top_bottom_spread",
            "spread_hit_rate",
            "worst_top_bottom_spread",
        },
        "Model metrics",
    )
    _require_columns(
        model_manifest,
        {
            "model_name",
            "fold_id",
            "label_cutoff",
            "test_start",
            "test_end",
            "train_rows",
            "test_rows",
        },
        "Model manifest",
    )

    if model_metrics.empty or model_manifest.empty:
        return pd.DataFrame(columns=FOLD_SUMMARY_COLUMNS)

    metric_rows = model_metrics.loc[
        :,
        [
            "model_name",
            "fold_id",
            "accuracy",
            "balanced_accuracy",
            "precision",
            "recall",
            "f1",
            "roc_auc",
            "log_loss",
            "brier_score",
            "ece",
            "max_calibration_gap",
            "rank_corr",
            "top_bucket_return",
            "bottom_bucket_return",
            "top_bottom_spread",
            "spread_hit_rate",
            "worst_top_bottom_spread",
        ],
    ].copy()
    manifest_rows = model_manifest.loc[
        :,
        ["model_name", "fold_id", "label_cutoff", "test_start", "test_end", "train_rows", "test_rows"],
    ].copy()
    merged = metric_rows.merge(
        manifest_rows,
        on=["model_name", "fold_id"],
        how="inner",
        validate="one_to_one",
    )
    if merged.empty:
        return pd.DataFrame(columns=FOLD_SUMMARY_COLUMNS)

    base = (
        merged.groupby("fold_id", as_index=False)
        .agg(
            label_cutoff=("label_cutoff", "first"),
            test_start=("test_start", "first"),
            test_end=("test_end", "first"),
            train_rows=("train_rows", "first"),
            test_rows=("test_rows", "first"),
            models_evaluated=("model_name", "nunique"),
            mean_accuracy=("accuracy", "mean"),
            mean_balanced_accuracy=("balanced_accuracy", "mean"),
            mean_precision=("precision", "mean"),
            mean_recall=("recall", "mean"),
            mean_f1=("f1", "mean"),
            mean_roc_auc=("roc_auc", "mean"),
            mean_log_loss=("log_loss", "mean"),
            mean_brier_score=("brier_score", "mean"),
            mean_ece=("ece", "mean"),
            mean_max_calibration_gap=("max_calibration_gap", "mean"),
            mean_rank_corr=("rank_corr", "mean"),
            mean_top_bucket_return=("top_bucket_return", "mean"),
            mean_bottom_bucket_return=("bottom_bucket_return", "mean"),
            mean_top_bottom_spread=("top_bottom_spread", "mean"),
            mean_spread_hit_rate=("spread_hit_rate", "mean"),
            worst_top_bottom_spread=("worst_top_bottom_spread", "min"),
        )
        .sort_values("fold_id")
        .reset_index(drop=True)
    )

    best_rows: list[dict[str, object]] = []
    for fold_id, fold_rows in merged.groupby("fold_id", sort=True):
        ranked_roc = fold_rows.dropna(subset=["roc_auc"]).sort_values(
            ["roc_auc", "model_name"],
            ascending=[False, True],
        )
        if ranked_roc.empty:
            best_model_by_roc_auc = ""
            best_roc_auc = float("nan")
        else:
            best_roc = ranked_roc.iloc[0]
            best_model_by_roc_auc = str(best_roc["model_name"])
            best_roc_auc = float(best_roc["roc_auc"])

        ranked_spread = fold_rows.dropna(subset=["top_bottom_spread"]).sort_values(
            ["top_bottom_spread", "model_name"],
            ascending=[False, True],
        )
        if ranked_spread.empty:
            best_model_by_top_bottom_spread = ""
            best_top_bottom_spread = float("nan")
        else:
            best_spread = ranked_spread.iloc[0]
            best_model_by_top_bottom_spread = str(best_spread["model_name"])
            best_top_bottom_spread = float(best_spread["top_bottom_spread"])

        best_rows.append(
            {
                "fold_id": fold_id,
                "best_model_by_roc_auc": best_model_by_roc_auc,
                "best_roc_auc": best_roc_auc,
                "best_model_by_top_bottom_spread": best_model_by_top_bottom_spread,
                "best_top_bottom_spread": best_top_bottom_spread,
            }
        )

    best_frame = pd.DataFrame(best_rows)
    summary = base.merge(best_frame, on="fold_id", how="left", validate="one_to_one")
    return summary.loc[:, FOLD_SUMMARY_COLUMNS]
