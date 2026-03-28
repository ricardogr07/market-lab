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
            "roc_auc": [0.71, 0.67, float("nan"), float("nan")],
            "log_loss": [0.59, 0.63, float("nan"), float("nan")],
            "target_rate": [0.50, 0.50, 0.45, 0.45],
            "prediction_rate": [0.52, 0.49, 0.44, 0.46],
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
    assert gb_row["mean_roc_auc"] == pytest.approx(0.71)
    assert gb_row["mean_log_loss"] == pytest.approx(0.59)
    assert gb_row["mean_train_rows"] == pytest.approx(125.0)
    assert gb_row["mean_test_rows"] == pytest.approx(21.0)


def test_build_fold_summary_aggregates_one_row_per_fold_and_selects_best_model(
    model_metrics: pd.DataFrame,
    model_manifest: pd.DataFrame,
) -> None:
    summary = build_fold_summary(model_metrics, model_manifest)

    assert list(summary.columns) == FOLD_SUMMARY_COLUMNS
    assert summary["fold_id"].tolist() == [1, 2]

    first_fold = summary.loc[summary["fold_id"] == 1].iloc[0]
    assert first_fold["models_evaluated"] == 2
    assert first_fold["mean_accuracy"] == pytest.approx(0.575)
    assert first_fold["mean_roc_auc"] == pytest.approx(0.69)
    assert first_fold["mean_log_loss"] == pytest.approx(0.61)
    assert first_fold["best_model_by_roc_auc"] == "gradient_boosting"
    assert first_fold["best_roc_auc"] == pytest.approx(0.71)


def test_build_fold_summary_keeps_nan_metrics_for_single_class_folds(
    model_metrics: pd.DataFrame,
    model_manifest: pd.DataFrame,
) -> None:
    summary = build_fold_summary(model_metrics, model_manifest)

    second_fold = summary.loc[summary["fold_id"] == 2].iloc[0]
    assert math.isnan(second_fold["mean_roc_auc"])
    assert math.isnan(second_fold["mean_log_loss"])
    assert second_fold["best_model_by_roc_auc"] == ""
    assert math.isnan(second_fold["best_roc_auc"])
