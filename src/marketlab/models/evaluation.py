from __future__ import annotations

import math

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score

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
    "balanced_accuracy",
    "precision",
    "recall",
    "f1",
    "brier_score",
    "target_rate",
    "prediction_rate",
    "roc_auc",
    "log_loss",
    "rank_corr",
    "top_bucket_return",
    "bottom_bucket_return",
    "top_bottom_spread",
    "spread_hit_rate",
    "worst_top_bottom_spread",
    "spread_signal_count",
]

RANKING_DIAGNOSTICS_COLUMNS = [
    "model_name",
    "fold_id",
    "signal_date",
    "effective_date",
    "symbol_count",
    "top_bucket_size",
    "bottom_bucket_size",
    "bucket_status",
    "rank_corr",
    "top_bucket_return",
    "bottom_bucket_return",
    "top_bottom_spread",
]

BUCKET_STATUS_USED = "used"
BUCKET_STATUS_UNDERFILLED = "underfilled"


def _binary_recall(
    truth: pd.Series,
    predicted: pd.Series,
    *,
    label: int,
) -> float:
    truth_mask = truth.eq(label)
    if not truth_mask.any():
        return 0.0
    true_positives = truth_mask & predicted.eq(label)
    return float(true_positives.sum() / truth_mask.sum())


def classification_metrics(
    truth: pd.Series,
    predicted: pd.Series,
    scores: pd.Series,
) -> dict[str, float]:
    truth_int = truth.astype(int).reset_index(drop=True)
    predicted_int = predicted.astype(int).reset_index(drop=True)
    score_float = scores.astype(float).reset_index(drop=True)
    clipped_scores = np.clip(score_float.to_numpy(dtype=float), 1e-9, 1 - 1e-9)

    positive_truth = truth_int.eq(1)
    positive_pred = predicted_int.eq(1)
    true_positive_count = int((positive_truth & positive_pred).sum())
    positive_predicted_count = int(positive_pred.sum())
    positive_truth_count = int(positive_truth.sum())

    precision = (
        float(true_positive_count / positive_predicted_count)
        if positive_predicted_count > 0
        else 0.0
    )
    recall = (
        float(true_positive_count / positive_truth_count)
        if positive_truth_count > 0
        else 0.0
    )
    f1 = (
        float(2.0 * precision * recall / (precision + recall))
        if (precision + recall) > 0.0
        else 0.0
    )

    recalls = []
    for label in [0, 1]:
        if truth_int.eq(label).any():
            recalls.append(_binary_recall(truth_int, predicted_int, label=label))
    balanced_accuracy = float(sum(recalls) / len(recalls)) if recalls else float("nan")

    metrics: dict[str, float] = {
        "accuracy": float(accuracy_score(truth_int, predicted_int)),
        "balanced_accuracy": balanced_accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "brier_score": float(
            np.mean(
                np.square(
                    score_float.to_numpy(dtype=float) - truth_int.to_numpy(dtype=float)
                )
            )
        ),
        "target_rate": float(truth_int.mean()),
        "prediction_rate": float(predicted_int.mean()),
    }

    if truth_int.nunique() > 1:
        metrics["roc_auc"] = float(roc_auc_score(truth_int, score_float))
        metrics["log_loss"] = float(log_loss(truth_int, clipped_scores, labels=[0, 1]))
    else:
        metrics["roc_auc"] = float("nan")
        metrics["log_loss"] = float("nan")

    return metrics


def _spearman_rank_corr(
    scores: pd.Series,
    forward_returns: pd.Series,
) -> float:
    ranked_scores = scores.rank(method="average")
    ranked_returns = forward_returns.rank(method="average")
    if ranked_scores.nunique(dropna=True) < 2 or ranked_returns.nunique(dropna=True) < 2:
        return float("nan")
    return float(ranked_scores.corr(ranked_returns))


def _bucket_frame(
    signal_rows: pd.DataFrame,
    *,
    long_n: int,
    short_n: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    long_bucket = signal_rows.sort_values(
        ["score", "symbol"],
        ascending=[False, True],
    ).head(long_n)
    short_bucket = signal_rows.loc[~signal_rows["symbol"].isin(long_bucket["symbol"])].sort_values(
        ["score", "symbol"],
        ascending=[True, True],
    ).head(short_n)
    return long_bucket, short_bucket


def build_ranking_diagnostics(
    model_name: str,
    fold_id: int,
    predictions: pd.DataFrame,
    *,
    long_n: int,
    short_n: int,
) -> pd.DataFrame:
    required = {
        "symbol",
        "signal_date",
        "effective_date",
        "forward_return",
        "score",
    }
    missing = required - set(predictions.columns)
    if missing:
        joined = ", ".join(sorted(missing))
        raise ValueError(f"Prediction frame is missing required columns: {joined}")

    if predictions.empty:
        return pd.DataFrame(columns=RANKING_DIAGNOSTICS_COLUMNS)

    working = predictions.loc[
        :,
        ["symbol", "signal_date", "effective_date", "forward_return", "score"],
    ].copy()
    working["signal_date"] = pd.to_datetime(working["signal_date"], errors="coerce")
    working["effective_date"] = pd.to_datetime(working["effective_date"], errors="coerce")
    diagnostics_rows: list[dict[str, object]] = []

    for (signal_date, effective_date), signal_rows in working.groupby(
        ["signal_date", "effective_date"],
        sort=True,
    ):
        long_bucket, short_bucket = _bucket_frame(
            signal_rows,
            long_n=long_n,
            short_n=short_n,
        )
        top_bucket_return = (
            float(long_bucket["forward_return"].mean())
            if not long_bucket.empty
            else float("nan")
        )
        bottom_bucket_return = (
            float(short_bucket["forward_return"].mean())
            if not short_bucket.empty
            else float("nan")
        )
        top_bottom_spread = (
            float(top_bucket_return - bottom_bucket_return)
            if not math.isnan(top_bucket_return) and not math.isnan(bottom_bucket_return)
            else float("nan")
        )
        bucket_status = (
            BUCKET_STATUS_USED
            if len(long_bucket) == long_n and len(short_bucket) == short_n
            else BUCKET_STATUS_UNDERFILLED
        )
        diagnostics_rows.append(
            {
                "model_name": model_name,
                "fold_id": fold_id,
                "signal_date": pd.Timestamp(signal_date),
                "effective_date": pd.Timestamp(effective_date),
                "symbol_count": int(signal_rows["symbol"].nunique()),
                "top_bucket_size": int(len(long_bucket)),
                "bottom_bucket_size": int(len(short_bucket)),
                "bucket_status": bucket_status,
                "rank_corr": _spearman_rank_corr(
                    signal_rows["score"],
                    signal_rows["forward_return"],
                ),
                "top_bucket_return": top_bucket_return,
                "bottom_bucket_return": bottom_bucket_return,
                "top_bottom_spread": top_bottom_spread,
            }
        )

    diagnostics = pd.DataFrame(diagnostics_rows)
    return diagnostics.loc[:, RANKING_DIAGNOSTICS_COLUMNS]


def summarize_ranking_diagnostics(
    ranking_diagnostics: pd.DataFrame,
) -> dict[str, float]:
    if ranking_diagnostics.empty:
        return {
            "rank_corr": float("nan"),
            "top_bucket_return": float("nan"),
            "bottom_bucket_return": float("nan"),
            "top_bottom_spread": float("nan"),
            "spread_hit_rate": float("nan"),
            "worst_top_bottom_spread": float("nan"),
            "spread_signal_count": 0,
        }

    used_rows = ranking_diagnostics.loc[
        ranking_diagnostics["bucket_status"] == BUCKET_STATUS_USED
    ].copy()
    spread_rows = used_rows.dropna(subset=["top_bottom_spread"]).copy()

    return {
        "rank_corr": float(ranking_diagnostics["rank_corr"].mean()),
        "top_bucket_return": float(spread_rows["top_bucket_return"].mean()),
        "bottom_bucket_return": float(spread_rows["bottom_bucket_return"].mean()),
        "top_bottom_spread": float(spread_rows["top_bottom_spread"].mean()),
        "spread_hit_rate": float((spread_rows["top_bottom_spread"] > 0.0).mean())
        if not spread_rows.empty
        else float("nan"),
        "worst_top_bottom_spread": float(spread_rows["top_bottom_spread"].min())
        if not spread_rows.empty
        else float("nan"),
        "spread_signal_count": int(len(spread_rows)),
    }
