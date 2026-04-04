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
    "ece",
    "max_calibration_gap",
    "target_rate",
    "prediction_rate",
    "roc_auc",
    "log_loss",
    "rank_corr",
    "top_bucket_return",
    "top_bucket_hit_rate",
    "bottom_bucket_return",
    "top_bottom_spread",
    "spread_hit_rate",
    "worst_top_bucket_return",
    "worst_top_bottom_spread",
    "top_bucket_signal_count",
    "spread_signal_count",
]

RANKING_DIAGNOSTICS_COLUMNS = [
    "model_name",
    "fold_id",
    "evaluation_mode",
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

CALIBRATION_DIAGNOSTICS_COLUMNS = [
    "model_name",
    "fold_id",
    "bin_id",
    "bin_left",
    "bin_right",
    "sample_count",
    "sample_fraction",
    "mean_score",
    "observed_positive_rate",
    "calibration_gap",
    "absolute_calibration_gap",
    "avg_forward_return",
    "negative_forward_return_rate",
]

SCORE_HISTOGRAM_COLUMNS = [
    "model_name",
    "fold_id",
    "target",
    "bin_id",
    "bin_left",
    "bin_right",
    "sample_count",
    "fraction_within_target",
]

THRESHOLD_DIAGNOSTICS_COLUMNS = [
    "model_name",
    "fold_id",
    "threshold",
    "threshold_status",
    "predicted_positive_count",
    "predicted_positive_rate",
    "precision",
    "recall",
    "f1",
    "balanced_accuracy",
    "avg_forward_return_predicted_positive",
    "negative_forward_return_rate_predicted_positive",
    "worst_forward_return_predicted_positive",
]

BUCKET_STATUS_USED = "used"
BUCKET_STATUS_UNDERFILLED = "underfilled"
THRESHOLD_STATUS_USED = "used"
THRESHOLD_STATUS_EMPTY = "empty"
VALID_EVALUATION_MODES = {"long_only", "long_short"}

SCORE_BIN_COUNT = 10
THRESHOLD_GRID = [step / 100 for step in range(5, 100, 5)]


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
    mode: str,
    long_n: int,
    short_n: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    long_bucket = signal_rows.sort_values(
        ["score", "symbol"],
        ascending=[False, True],
    ).head(long_n)
    if mode == "long_only":
        short_bucket = signal_rows.iloc[0:0].copy()
    else:
        short_bucket = signal_rows.loc[
            ~signal_rows["symbol"].isin(long_bucket["symbol"])
        ].sort_values(
            ["score", "symbol"],
            ascending=[True, True],
        ).head(short_n)
    return long_bucket, short_bucket


def _score_bin_bounds(
    bin_id: int,
    *,
    bin_count: int,
) -> tuple[float, float]:
    return float((bin_id - 1) / bin_count), float(bin_id / bin_count)


def _score_bin_ids(
    scores: pd.Series,
    *,
    bin_count: int,
) -> pd.Series:
    clipped_scores = np.clip(scores.astype(float).to_numpy(dtype=float), 0.0, 1.0)
    bin_indexes = np.minimum((clipped_scores * bin_count).astype(int), bin_count - 1)
    return pd.Series(bin_indexes + 1, index=scores.index, dtype=int)


def _prediction_frame_for_score_bins(
    predictions: pd.DataFrame,
) -> pd.DataFrame:
    required = {"score", "target", "forward_return"}
    missing = required - set(predictions.columns)
    if missing:
        joined = ", ".join(sorted(missing))
        raise ValueError(f"Prediction frame is missing required columns: {joined}")

    working = predictions.loc[:, ["score", "target", "forward_return"]].copy()
    working["score"] = working["score"].astype(float).clip(0.0, 1.0)
    working["target"] = working["target"].astype(int)
    working["forward_return"] = working["forward_return"].astype(float)
    working["bin_id"] = _score_bin_ids(working["score"], bin_count=SCORE_BIN_COUNT)
    return working


def build_calibration_diagnostics(
    model_name: str,
    fold_id: int,
    predictions: pd.DataFrame,
) -> pd.DataFrame:
    if predictions.empty:
        return pd.DataFrame(columns=CALIBRATION_DIAGNOSTICS_COLUMNS)

    working = _prediction_frame_for_score_bins(predictions)
    total_count = int(len(working))
    diagnostics_rows: list[dict[str, object]] = []

    for bin_id in range(1, SCORE_BIN_COUNT + 1):
        bin_left, bin_right = _score_bin_bounds(bin_id, bin_count=SCORE_BIN_COUNT)
        bin_rows = working.loc[working["bin_id"] == bin_id]
        sample_count = int(len(bin_rows))
        sample_fraction = float(sample_count / total_count) if total_count > 0 else 0.0
        if sample_count > 0:
            mean_score = float(bin_rows["score"].mean())
            observed_positive_rate = float(bin_rows["target"].mean())
            avg_forward_return = float(bin_rows["forward_return"].mean())
            negative_forward_return_rate = float((bin_rows["forward_return"] < 0.0).mean())
            calibration_gap = float(observed_positive_rate - mean_score)
            absolute_calibration_gap = abs(calibration_gap)
        else:
            mean_score = float("nan")
            observed_positive_rate = float("nan")
            avg_forward_return = float("nan")
            negative_forward_return_rate = float("nan")
            calibration_gap = float("nan")
            absolute_calibration_gap = float("nan")

        diagnostics_rows.append(
            {
                "model_name": model_name,
                "fold_id": fold_id,
                "bin_id": bin_id,
                "bin_left": bin_left,
                "bin_right": bin_right,
                "sample_count": sample_count,
                "sample_fraction": sample_fraction,
                "mean_score": mean_score,
                "observed_positive_rate": observed_positive_rate,
                "calibration_gap": calibration_gap,
                "absolute_calibration_gap": absolute_calibration_gap,
                "avg_forward_return": avg_forward_return,
                "negative_forward_return_rate": negative_forward_return_rate,
            }
        )

    diagnostics = pd.DataFrame(diagnostics_rows)
    return diagnostics.loc[:, CALIBRATION_DIAGNOSTICS_COLUMNS]


def summarize_calibration_diagnostics(
    calibration_diagnostics: pd.DataFrame,
) -> dict[str, float]:
    if calibration_diagnostics.empty:
        return {"ece": float("nan"), "max_calibration_gap": float("nan")}

    occupied = calibration_diagnostics.loc[calibration_diagnostics["sample_count"] > 0].copy()
    if occupied.empty:
        return {"ece": float("nan"), "max_calibration_gap": float("nan")}

    return {
        "ece": float((occupied["sample_fraction"] * occupied["absolute_calibration_gap"]).sum()),
        "max_calibration_gap": float(occupied["absolute_calibration_gap"].max()),
    }


def build_score_histograms(
    model_name: str,
    fold_id: int,
    predictions: pd.DataFrame,
) -> pd.DataFrame:
    if predictions.empty:
        return pd.DataFrame(columns=SCORE_HISTOGRAM_COLUMNS)

    working = _prediction_frame_for_score_bins(predictions)
    histogram_rows: list[dict[str, object]] = []

    for target in [0, 1]:
        target_rows = working.loc[working["target"] == target]
        target_count = int(len(target_rows))
        for bin_id in range(1, SCORE_BIN_COUNT + 1):
            bin_left, bin_right = _score_bin_bounds(bin_id, bin_count=SCORE_BIN_COUNT)
            sample_count = int((target_rows["bin_id"] == bin_id).sum())
            histogram_rows.append(
                {
                    "model_name": model_name,
                    "fold_id": fold_id,
                    "target": target,
                    "bin_id": bin_id,
                    "bin_left": bin_left,
                    "bin_right": bin_right,
                    "sample_count": sample_count,
                    "fraction_within_target": float(sample_count / target_count) if target_count > 0 else 0.0,
                }
            )

    histograms = pd.DataFrame(histogram_rows)
    return histograms.loc[:, SCORE_HISTOGRAM_COLUMNS]


def build_threshold_diagnostics(
    model_name: str,
    fold_id: int,
    predictions: pd.DataFrame,
) -> pd.DataFrame:
    required = {"score", "target", "forward_return"}
    missing = required - set(predictions.columns)
    if missing:
        joined = ", ".join(sorted(missing))
        raise ValueError(f"Prediction frame is missing required columns: {joined}")

    if predictions.empty:
        return pd.DataFrame(columns=THRESHOLD_DIAGNOSTICS_COLUMNS)

    working = predictions.loc[:, ["score", "target", "forward_return"]].copy()
    working["score"] = working["score"].astype(float).clip(0.0, 1.0)
    working["target"] = working["target"].astype(int)
    working["forward_return"] = working["forward_return"].astype(float)
    truth = working["target"]
    total_count = int(len(working))
    diagnostics_rows: list[dict[str, object]] = []

    for threshold in THRESHOLD_GRID:
        predicted_positive = working["score"] >= threshold
        predicted_positive_count = int(predicted_positive.sum())
        threshold_metrics = classification_metrics(
            truth=truth,
            predicted=predicted_positive.astype(int),
            scores=working["score"],
        )

        if predicted_positive_count > 0:
            positive_rows = working.loc[predicted_positive]
            avg_forward_return = float(positive_rows["forward_return"].mean())
            negative_forward_return_rate = float((positive_rows["forward_return"] < 0.0).mean())
            worst_forward_return = float(positive_rows["forward_return"].min())
            threshold_status = THRESHOLD_STATUS_USED
        else:
            avg_forward_return = float("nan")
            negative_forward_return_rate = float("nan")
            worst_forward_return = float("nan")
            threshold_status = THRESHOLD_STATUS_EMPTY

        diagnostics_rows.append(
            {
                "model_name": model_name,
                "fold_id": fold_id,
                "threshold": float(threshold),
                "threshold_status": threshold_status,
                "predicted_positive_count": predicted_positive_count,
                "predicted_positive_rate": float(predicted_positive_count / total_count) if total_count > 0 else 0.0,
                "precision": threshold_metrics["precision"],
                "recall": threshold_metrics["recall"],
                "f1": threshold_metrics["f1"],
                "balanced_accuracy": threshold_metrics["balanced_accuracy"],
                "avg_forward_return_predicted_positive": avg_forward_return,
                "negative_forward_return_rate_predicted_positive": negative_forward_return_rate,
                "worst_forward_return_predicted_positive": worst_forward_return,
            }
        )

    diagnostics = pd.DataFrame(diagnostics_rows)
    return diagnostics.loc[:, THRESHOLD_DIAGNOSTICS_COLUMNS]


def build_ranking_diagnostics(
    model_name: str,
    fold_id: int,
    predictions: pd.DataFrame,
    *,
    long_n: int,
    short_n: int,
    mode: str = "long_short",
) -> pd.DataFrame:
    if mode not in VALID_EVALUATION_MODES:
        supported = ", ".join(sorted(VALID_EVALUATION_MODES))
        raise ValueError(
            f"Unsupported ranking evaluation mode '{mode}'. Expected one of: {supported}"
        )

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
            mode=mode,
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
        long_bucket_complete = len(long_bucket) == long_n
        short_bucket_complete = mode == "long_only" or len(short_bucket) == short_n
        bucket_status = (
            BUCKET_STATUS_USED
            if long_bucket_complete and short_bucket_complete
            else BUCKET_STATUS_UNDERFILLED
        )
        diagnostics_rows.append(
            {
                "model_name": model_name,
                "fold_id": fold_id,
                "evaluation_mode": mode,
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
            "top_bucket_hit_rate": float("nan"),
            "bottom_bucket_return": float("nan"),
            "top_bottom_spread": float("nan"),
            "spread_hit_rate": float("nan"),
            "worst_top_bucket_return": float("nan"),
            "worst_top_bottom_spread": float("nan"),
            "top_bucket_signal_count": 0,
            "spread_signal_count": 0,
        }

    used_rows = ranking_diagnostics.loc[
        ranking_diagnostics["bucket_status"] == BUCKET_STATUS_USED
    ].copy()
    top_rows = used_rows.dropna(subset=["top_bucket_return"]).copy()
    spread_rows = used_rows.dropna(subset=["top_bottom_spread"]).copy()

    return {
        "rank_corr": float(ranking_diagnostics["rank_corr"].mean()),
        "top_bucket_return": float(top_rows["top_bucket_return"].mean()),
        "top_bucket_hit_rate": float((top_rows["top_bucket_return"] > 0.0).mean())
        if not top_rows.empty
        else float("nan"),
        "bottom_bucket_return": float(spread_rows["bottom_bucket_return"].mean()),
        "top_bottom_spread": float(spread_rows["top_bottom_spread"].mean()),
        "spread_hit_rate": float((spread_rows["top_bottom_spread"] > 0.0).mean())
        if not spread_rows.empty
        else float("nan"),
        "worst_top_bucket_return": float(top_rows["top_bucket_return"].min())
        if not top_rows.empty
        else float("nan"),
        "worst_top_bottom_spread": float(spread_rows["top_bottom_spread"].min())
        if not spread_rows.empty
        else float("nan"),
        "top_bucket_signal_count": int(len(top_rows)),
        "spread_signal_count": int(len(spread_rows)),
    }
