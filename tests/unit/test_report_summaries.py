from __future__ import annotations

import math

import pandas as pd
import pytest

from marketlab.reports.summary import (
    FOLD_SUMMARY_COLUMNS,
    MODEL_SUMMARY_COLUMNS,
    build_fold_summary,
    build_model_summary,
)


@pytest.fixture()
def model_manifest() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "model_name": [
                "gradient_boosting",
                "logistic_regression",
                "gradient_boosting",
                "logistic_regression",
            ],
            "estimator_label": [
                "GradientBoostingClassifier",
                "LogisticRegression",
                "GradientBoostingClassifier",
                "LogisticRegression",
            ],
            "fold_id": [1, 1, 2, 2],
            "label_cutoff": pd.to_datetime([
                "2023-01-06",
                "2023-01-06",
                "2023-03-03",
                "2023-03-03",
            ]),
            "test_start": pd.to_datetime([
                "2023-01-09",
                "2023-01-09",
                "2023-03-06",
                "2023-03-06",
            ]),
            "test_end": pd.to_datetime([
                "2023-02-24",
                "2023-02-24",
                "2023-04-28",
                "2023-04-28",
            ]),
            "train_rows": [120, 120, 130, 130],
            "test_rows": [20, 20, 22, 22],
        }
    )


@pytest.fixture()
def model_metrics() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "model_name": [
                "gradient_boosting",
                "logistic_regression",
                "gradient_boosting",
                "logistic_regression",
            ],
            "fold_id": [1, 1, 2, 2],
            "accuracy": [0.60, 0.55, 0.68, 0.66],
            "balanced_accuracy": [0.61, 0.56, 0.69, 0.67],
            "precision": [0.62, 0.57, 0.70, 0.68],
            "recall": [0.63, 0.58, 0.71, 0.69],
            "f1": [0.64, 0.59, 0.72, 0.70],
            "roc_auc": [0.71, 0.67, float("nan"), float("nan")],
            "log_loss": [0.59, 0.63, float("nan"), float("nan")],
            "brier_score": [0.22, 0.24, 0.19, 0.20],
            "target_rate": [0.50, 0.50, 0.45, 0.45],
            "prediction_rate": [0.52, 0.49, 0.44, 0.46],
            "rank_corr": [0.15, 0.10, 0.05, 0.08],
            "top_bucket_return": [0.020, 0.015, 0.010, 0.005],
            "bottom_bucket_return": [-0.015, -0.005, -0.010, -0.008],
            "top_bottom_spread": [0.035, 0.020, 0.020, 0.013],
            "spread_hit_rate": [0.60, 0.55, 0.50, 0.45],
            "worst_top_bottom_spread": [-0.020, -0.030, -0.040, -0.050],
            "spread_signal_count": [8, 8, 9, 9],
            "train_rows": [120, 120, 130, 130],
            "test_rows": [20, 20, 22, 22],
        }
    )


def test_build_model_summary_aggregates_one_row_per_model(
    model_metrics: pd.DataFrame,
    model_manifest: pd.DataFrame,
) -> None:
    summary = build_model_summary(model_metrics, model_manifest)

    assert list(summary.columns) == MODEL_SUMMARY_COLUMNS
    assert summary["model_name"].tolist() == ["gradient_boosting", "logistic_regression"]

    gb_row = summary.loc[summary["model_name"] == "gradient_boosting"].iloc[0]
    assert gb_row["estimator_label"] == "GradientBoostingClassifier"
    assert gb_row["fold_count"] == 2
    assert gb_row["first_test_start"] == pd.Timestamp("2023-01-09")
    assert gb_row["last_test_end"] == pd.Timestamp("2023-04-28")
    assert gb_row["mean_accuracy"] == pytest.approx(0.64)
    assert gb_row["mean_balanced_accuracy"] == pytest.approx(0.65)
    assert gb_row["mean_precision"] == pytest.approx(0.66)
    assert gb_row["mean_recall"] == pytest.approx(0.67)
    assert gb_row["mean_f1"] == pytest.approx(0.68)
    assert gb_row["mean_roc_auc"] == pytest.approx(0.71)
    assert gb_row["mean_log_loss"] == pytest.approx(0.59)
    assert gb_row["mean_brier_score"] == pytest.approx(0.205)
    assert gb_row["mean_rank_corr"] == pytest.approx(0.10)
    assert gb_row["mean_top_bottom_spread"] == pytest.approx(0.0275)
    assert gb_row["worst_top_bottom_spread"] == pytest.approx(-0.04)
    assert gb_row["mean_train_rows"] == pytest.approx(125.0)
    assert gb_row["mean_test_rows"] == pytest.approx(21.0)


def test_build_fold_summary_aggregates_one_row_per_fold_and_selects_best_models(
    model_metrics: pd.DataFrame,
    model_manifest: pd.DataFrame,
) -> None:
    summary = build_fold_summary(model_metrics, model_manifest)

    assert list(summary.columns) == FOLD_SUMMARY_COLUMNS
    assert summary["fold_id"].tolist() == [1, 2]

    first_fold = summary.loc[summary["fold_id"] == 1].iloc[0]
    assert first_fold["models_evaluated"] == 2
    assert first_fold["mean_accuracy"] == pytest.approx(0.575)
    assert first_fold["mean_balanced_accuracy"] == pytest.approx(0.585)
    assert first_fold["mean_precision"] == pytest.approx(0.595)
    assert first_fold["mean_recall"] == pytest.approx(0.605)
    assert first_fold["mean_f1"] == pytest.approx(0.615)
    assert first_fold["mean_roc_auc"] == pytest.approx(0.69)
    assert first_fold["mean_log_loss"] == pytest.approx(0.61)
    assert first_fold["mean_brier_score"] == pytest.approx(0.23)
    assert first_fold["mean_rank_corr"] == pytest.approx(0.125)
    assert first_fold["mean_top_bottom_spread"] == pytest.approx(0.0275)
    assert first_fold["mean_spread_hit_rate"] == pytest.approx(0.575)
    assert first_fold["worst_top_bottom_spread"] == pytest.approx(-0.03)
    assert first_fold["best_model_by_roc_auc"] == "gradient_boosting"
    assert first_fold["best_roc_auc"] == pytest.approx(0.71)
    assert first_fold["best_model_by_top_bottom_spread"] == "gradient_boosting"
    assert first_fold["best_top_bottom_spread"] == pytest.approx(0.035)


def test_build_fold_summary_keeps_nan_roc_metrics_for_single_class_folds_but_still_ranks_by_spread(
    model_metrics: pd.DataFrame,
    model_manifest: pd.DataFrame,
) -> None:
    summary = build_fold_summary(model_metrics, model_manifest)

    second_fold = summary.loc[summary["fold_id"] == 2].iloc[0]
    assert math.isnan(second_fold["mean_roc_auc"])
    assert math.isnan(second_fold["mean_log_loss"])
    assert second_fold["best_model_by_roc_auc"] == ""
    assert math.isnan(second_fold["best_roc_auc"])
    assert second_fold["best_model_by_top_bottom_spread"] == "gradient_boosting"
    assert second_fold["best_top_bottom_spread"] == pytest.approx(0.02)
