from __future__ import annotations

import math

import pandas as pd
import pytest

from marketlab.models.evaluation import (
    BUCKET_STATUS_UNDERFILLED,
    BUCKET_STATUS_USED,
    build_ranking_diagnostics,
    classification_metrics,
    summarize_ranking_diagnostics,
)


def test_classification_metrics_are_deterministic_for_single_class_folds() -> None:
    metrics = classification_metrics(
        truth=pd.Series([1, 1, 1, 1], dtype=int),
        predicted=pd.Series([1, 1, 0, 0], dtype=int),
        scores=pd.Series([0.9, 0.8, 0.4, 0.3], dtype=float),
    )

    assert metrics["accuracy"] == pytest.approx(0.5)
    assert metrics["balanced_accuracy"] == pytest.approx(0.5)
    assert metrics["precision"] == pytest.approx(1.0)
    assert metrics["recall"] == pytest.approx(0.5)
    assert metrics["f1"] == pytest.approx(2.0 / 3.0)
    assert metrics["brier_score"] == pytest.approx(0.225)
    assert math.isnan(metrics["roc_auc"])
    assert math.isnan(metrics["log_loss"])


def test_build_ranking_diagnostics_and_summary_exclude_underfilled_rows_from_spread_aggregates() -> None:
    diagnostics = build_ranking_diagnostics(
        model_name="logistic_regression",
        fold_id=3,
        predictions=pd.DataFrame(
            {
                "symbol": ["AAA", "BBB", "CCC", "DDD", "AAA", "BBB", "CCC"],
                "signal_date": pd.to_datetime(
                    [
                        "2024-01-05",
                        "2024-01-05",
                        "2024-01-05",
                        "2024-01-05",
                        "2024-01-12",
                        "2024-01-12",
                        "2024-01-12",
                    ]
                ),
                "effective_date": pd.to_datetime(
                    [
                        "2024-01-08",
                        "2024-01-08",
                        "2024-01-08",
                        "2024-01-08",
                        "2024-01-16",
                        "2024-01-16",
                        "2024-01-16",
                    ]
                ),
                "forward_return": [0.04, 0.01, -0.02, -0.03, 0.02, -0.01, 0.00],
                "score": [0.9, 0.7, 0.2, 0.1, 0.8, 0.4, 0.2],
            }
        ),
        long_n=2,
        short_n=2,
    )

    assert diagnostics["bucket_status"].tolist() == [BUCKET_STATUS_USED, BUCKET_STATUS_UNDERFILLED]

    used_row = diagnostics.iloc[0]
    assert used_row["top_bucket_size"] == 2
    assert used_row["bottom_bucket_size"] == 2
    assert used_row["top_bucket_return"] == pytest.approx(0.025)
    assert used_row["bottom_bucket_return"] == pytest.approx(-0.025)
    assert used_row["top_bottom_spread"] == pytest.approx(0.05)

    summary = summarize_ranking_diagnostics(diagnostics)
    assert summary["spread_signal_count"] == 1
    assert summary["top_bucket_return"] == pytest.approx(0.025)
    assert summary["bottom_bucket_return"] == pytest.approx(-0.025)
    assert summary["top_bottom_spread"] == pytest.approx(0.05)
    assert summary["spread_hit_rate"] == pytest.approx(1.0)
    assert summary["worst_top_bottom_spread"] == pytest.approx(0.05)
    assert summary["rank_corr"] == pytest.approx(diagnostics["rank_corr"].mean())


def test_build_ranking_diagnostics_keeps_nan_rank_corr_for_constant_inputs() -> None:
    diagnostics = build_ranking_diagnostics(
        model_name="random_forest",
        fold_id=2,
        predictions=pd.DataFrame(
            {
                "symbol": ["AAA", "BBB", "CCC", "DDD"],
                "signal_date": pd.to_datetime(["2024-01-05"] * 4),
                "effective_date": pd.to_datetime(["2024-01-08"] * 4),
                "forward_return": [0.01, 0.02, 0.03, 0.04],
                "score": [0.5, 0.5, 0.5, 0.5],
            }
        ),
        long_n=2,
        short_n=2,
    )

    assert len(diagnostics) == 1
    assert math.isnan(float(diagnostics.loc[0, "rank_corr"]))
