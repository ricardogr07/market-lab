from __future__ import annotations

import math

import pandas as pd
import pytest

from marketlab.models.evaluation import (
    CALIBRATION_DIAGNOSTICS_COLUMNS,
    SCORE_HISTOGRAM_COLUMNS,
    THRESHOLD_DIAGNOSTICS_COLUMNS,
    build_calibration_diagnostics,
    build_score_histograms,
    build_threshold_diagnostics,
    summarize_calibration_diagnostics,
)


def test_build_calibration_diagnostics_and_summary_from_fixed_bins() -> None:
    diagnostics = build_calibration_diagnostics(
        model_name="logistic_regression",
        fold_id=1,
        predictions=pd.DataFrame(
            {
                "score": [0.10, 0.20, 0.80, 0.90],
                "target": [0, 0, 1, 1],
                "forward_return": [-0.01, 0.00, 0.02, 0.03],
            }
        ),
    )

    assert list(diagnostics.columns) == CALIBRATION_DIAGNOSTICS_COLUMNS
    assert len(diagnostics) == 10
    occupied = diagnostics.loc[diagnostics["sample_count"] > 0].reset_index(drop=True)
    assert occupied["bin_id"].tolist() == [2, 3, 9, 10]

    summary = summarize_calibration_diagnostics(diagnostics)
    assert summary["ece"] == pytest.approx(0.15)
    assert summary["max_calibration_gap"] == pytest.approx(0.2)


def test_build_calibration_diagnostics_handles_single_class_constant_scores() -> None:
    diagnostics = build_calibration_diagnostics(
        model_name="random_forest",
        fold_id=2,
        predictions=pd.DataFrame(
            {
                "score": [0.5, 0.5, 0.5],
                "target": [1, 1, 1],
                "forward_return": [0.01, -0.02, 0.03],
            }
        ),
    )

    occupied = diagnostics.loc[diagnostics["sample_count"] > 0]
    assert occupied["bin_id"].tolist() == [6]
    assert occupied.iloc[0]["observed_positive_rate"] == pytest.approx(1.0)
    summary = summarize_calibration_diagnostics(diagnostics)
    assert summary["ece"] == pytest.approx(0.5)
    assert summary["max_calibration_gap"] == pytest.approx(0.5)


def test_build_score_histograms_emits_all_bins_for_both_classes() -> None:
    histograms = build_score_histograms(
        model_name="gradient_boosting",
        fold_id=3,
        predictions=pd.DataFrame(
            {
                "score": [0.15, 0.35, 0.65, 0.85],
                "target": [0, 1, 1, 1],
                "forward_return": [0.01, 0.02, -0.01, 0.03],
            }
        ),
    )

    assert list(histograms.columns) == SCORE_HISTOGRAM_COLUMNS
    assert len(histograms) == 20
    per_target = histograms.groupby("target")["fraction_within_target"].sum()
    assert per_target.loc[0] == pytest.approx(1.0)
    assert per_target.loc[1] == pytest.approx(1.0)


def test_build_threshold_diagnostics_handles_empty_thresholds_and_return_columns() -> None:
    diagnostics = build_threshold_diagnostics(
        model_name="logistic_regression",
        fold_id=4,
        predictions=pd.DataFrame(
            {
                "score": [0.10, 0.40, 0.90],
                "target": [0, 1, 1],
                "forward_return": [0.01, -0.02, 0.03],
            }
        ),
    )

    assert list(diagnostics.columns) == THRESHOLD_DIAGNOSTICS_COLUMNS
    used_row = diagnostics.loc[diagnostics["threshold"] == 0.5].iloc[0]
    assert used_row["threshold_status"] == "used"
    assert used_row["predicted_positive_count"] == 1
    assert used_row["precision"] == pytest.approx(1.0)
    assert used_row["recall"] == pytest.approx(0.5)
    assert used_row["f1"] == pytest.approx(2.0 / 3.0)
    assert used_row["avg_forward_return_predicted_positive"] == pytest.approx(0.03)
    assert used_row["negative_forward_return_rate_predicted_positive"] == pytest.approx(0.0)
    assert used_row["worst_forward_return_predicted_positive"] == pytest.approx(0.03)

    empty_row = diagnostics.loc[diagnostics["threshold"] == 0.95].iloc[0]
    assert empty_row["threshold_status"] == "empty"
    assert empty_row["predicted_positive_count"] == 0
    assert math.isnan(float(empty_row["avg_forward_return_predicted_positive"]))
    assert math.isnan(float(empty_row["negative_forward_return_rate_predicted_positive"]))
    assert math.isnan(float(empty_row["worst_forward_return_predicted_positive"]))
