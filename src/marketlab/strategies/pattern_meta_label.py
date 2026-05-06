from __future__ import annotations

import pandas as pd

from marketlab.config import WalkForwardConfig
from marketlab.evaluation import build_walk_forward_folds, slice_fold_rows
from marketlab.evaluation.walk_forward import build_walk_forward_diagnostics
from marketlab.models.registry import build_model_estimator, predict_direction_scores
from marketlab.strategies.chart_patterns import PATTERN_COLUMNS
from marketlab.strategies.pattern_exit_overlay import (
    DIAGNOSTIC_COLUMNS as OVERLAY_DIAGNOSTIC_COLUMNS,
)

METADATA_COLUMNS = {
    "symbol",
    "signal_date",
    "effective_date",
    "target_end_date",
    "forward_return",
    "target",
}
LABEL_COLUMNS = [
    "symbol",
    "signal_date",
    "effective_date",
    "target_end_date",
    "forward_return",
    "turnover_cost",
    "target",
    "close",
    "trend_ema",
    "support_level",
    "resistance_level",
    "distance_to_support_pct",
    "distance_to_resistance_pct",
    "volume_average",
    "volume_confirmed",
    *PATTERN_COLUMNS,
    "bullish_pattern_count",
    "bearish_pattern_count",
]
PREDICTION_COLUMNS = [
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
FOLD_DIAGNOSTIC_COLUMNS = [
    "model_name",
    "fold_id",
    "status",
    "skip_reasons",
    "label_cutoff",
    "test_start",
    "test_end",
    "train_rows",
    "test_rows",
    "train_positive_rate",
    "test_positive_rate",
]
META_OVERLAY_COLUMNS = [
    *OVERLAY_DIAGNOSTIC_COLUMNS,
    "meta_exit_score",
    "meta_exit_predicted",
    "meta_exit_model_count",
]


def _feature_columns(labels: pd.DataFrame) -> list[str]:
    return [
        column
        for column in labels.columns
        if column not in METADATA_COLUMNS and column != "turnover_cost"
    ]


def _panel_price_frame(panel: pd.DataFrame) -> pd.DataFrame:
    return (
        panel.sort_values(["symbol", "timestamp"])
        .loc[:, ["timestamp", "symbol", "adj_open", "adj_close"]]
        .reset_index(drop=True)
    )


def _candidate_rows(overlay_diagnostics: pd.DataFrame) -> pd.DataFrame:
    if overlay_diagnostics.empty:
        return pd.DataFrame(columns=overlay_diagnostics.columns)
    candidates = overlay_diagnostics.loc[
        overlay_diagnostics["bearish_exit_candidate"].fillna(False).astype(bool)
    ].copy()
    candidates["signal_date"] = pd.to_datetime(candidates["timestamp"])
    candidates["effective_date"] = pd.to_datetime(candidates["effective_date"])
    return candidates.sort_values(["symbol", "signal_date"]).reset_index(drop=True)


def build_labels(
    panel: pd.DataFrame,
    overlay_diagnostics: pd.DataFrame,
    *,
    label_horizon_bars: int,
    cost_bps: float,
) -> pd.DataFrame:
    prices = _panel_price_frame(panel)
    candidates = _candidate_rows(overlay_diagnostics)
    cost = float(cost_bps) / 10_000.0
    rows: list[dict[str, object]] = []

    for _, symbol_prices in prices.groupby("symbol", sort=False):
        symbol = str(symbol_prices["symbol"].iloc[0])
        price_rows = symbol_prices.reset_index(drop=True)
        timestamp_to_index = {
            pd.Timestamp(timestamp): index
            for index, timestamp in enumerate(pd.to_datetime(price_rows["timestamp"]))
        }
        symbol_candidates = candidates.loc[candidates["symbol"] == symbol]
        for candidate in symbol_candidates.itertuples(index=False):
            effective_date = pd.Timestamp(candidate.effective_date)
            start_index = timestamp_to_index.get(effective_date)
            if start_index is None:
                continue
            target_index = start_index + label_horizon_bars - 1
            if target_index >= len(price_rows):
                continue

            entry_open = float(price_rows.loc[start_index, "adj_open"])
            exit_close = float(price_rows.loc[target_index, "adj_close"])
            if entry_open <= 0.0:
                continue
            forward_return = (exit_close / entry_open) - 1.0
            target_end_date = pd.Timestamp(price_rows.loc[target_index, "timestamp"])
            support = candidate.support_level
            resistance = candidate.resistance_level
            close = float(candidate.close)
            support_distance = (
                (close / float(support)) - 1.0 if pd.notna(support) and float(support) else 0.0
            )
            resistance_distance = (
                (float(resistance) / close) - 1.0
                if pd.notna(resistance) and close
                else 0.0
            )
            rows.append(
                {
                    "symbol": symbol,
                    "signal_date": pd.Timestamp(candidate.signal_date),
                    "effective_date": effective_date,
                    "target_end_date": target_end_date,
                    "forward_return": forward_return,
                    "turnover_cost": cost,
                    "target": int(forward_return < -cost),
                    "close": close,
                    "trend_ema": candidate.trend_ema,
                    "support_level": support,
                    "resistance_level": resistance,
                    "distance_to_support_pct": support_distance,
                    "distance_to_resistance_pct": resistance_distance,
                    "volume_average": candidate.volume_average,
                    "volume_confirmed": int(bool(candidate.volume_confirmed)),
                    **{column: int(bool(getattr(candidate, column))) for column in PATTERN_COLUMNS},
                    "bullish_pattern_count": candidate.bullish_pattern_count,
                    "bearish_pattern_count": candidate.bearish_pattern_count,
                }
            )

    labels = pd.DataFrame(rows, columns=LABEL_COLUMNS)
    if labels.empty:
        return labels
    for column in _feature_columns(labels):
        labels[column] = pd.to_numeric(labels[column], errors="coerce").fillna(0.0)
    return labels.sort_values(["signal_date", "symbol"]).reset_index(drop=True)


def _fold_frequency(rebalance_frequency: str) -> str:
    return "D" if rebalance_frequency == "bar" else rebalance_frequency


def predict_exit_candidates(
    labels: pd.DataFrame,
    *,
    walk_forward: WalkForwardConfig,
    rebalance_frequency: str,
    model_names: list[str],
    threshold: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if labels.empty:
        return (
            pd.DataFrame(columns=PREDICTION_COLUMNS),
            pd.DataFrame(columns=FOLD_DIAGNOSTIC_COLUMNS),
        )

    frequency = _fold_frequency(rebalance_frequency)
    folds = build_walk_forward_folds(labels, walk_forward, frequency=frequency)
    base_diagnostics = build_walk_forward_diagnostics(labels, walk_forward, frequency=frequency)
    feature_columns = _feature_columns(labels)
    predictions: list[pd.DataFrame] = []
    diagnostics: list[dict[str, object]] = []

    if not folds and not base_diagnostics.empty:
        for model_name in model_names:
            for row in base_diagnostics.itertuples(index=False):
                diagnostics.append(
                    {
                        "model_name": model_name,
                        "fold_id": row.fold_id,
                        "status": row.status,
                        "skip_reasons": row.skip_reasons,
                        "label_cutoff": row.label_cutoff,
                        "test_start": row.test_start,
                        "test_end": row.test_end,
                        "train_rows": row.train_rows,
                        "test_rows": row.test_rows,
                        "train_positive_rate": row.train_positive_rate,
                        "test_positive_rate": row.test_positive_rate,
                    }
                )

    for model_name in model_names:
        for fold in folds:
            train_rows, test_rows = slice_fold_rows(labels, fold)
            train_positive_rate = float(train_rows["target"].mean()) if not train_rows.empty else float("nan")
            test_positive_rate = float(test_rows["target"].mean()) if not test_rows.empty else float("nan")
            skip_reasons: list[str] = []
            if train_rows.empty:
                skip_reasons.append("empty_train")
            if test_rows.empty:
                skip_reasons.append("empty_test")
            if train_rows["target"].nunique(dropna=True) < 2:
                skip_reasons.append("single_train_class")
            status = "skipped" if skip_reasons else "used"
            diagnostics.append(
                {
                    "model_name": model_name,
                    "fold_id": fold.fold_id,
                    "status": status,
                    "skip_reasons": ";".join(skip_reasons),
                    "label_cutoff": fold.label_cutoff,
                    "test_start": fold.test_start,
                    "test_end": fold.test_end,
                    "train_rows": len(train_rows),
                    "test_rows": len(test_rows),
                    "train_positive_rate": train_positive_rate,
                    "test_positive_rate": test_positive_rate,
                }
            )
            if skip_reasons:
                continue

            _, estimator = build_model_estimator(model_name, target_type="direction")
            estimator.fit(train_rows.loc[:, feature_columns], train_rows["target"].astype(int))
            scores = predict_direction_scores(estimator, test_rows.loc[:, feature_columns])
            fold_predictions = test_rows.loc[
                :,
                [
                    "symbol",
                    "signal_date",
                    "effective_date",
                    "target_end_date",
                    "forward_return",
                    "target",
                ],
            ].copy()
            fold_predictions.insert(0, "fold_id", fold.fold_id)
            fold_predictions.insert(0, "model_name", model_name)
            fold_predictions["score"] = scores.to_numpy(dtype=float)
            fold_predictions["predicted_target"] = (
                fold_predictions["score"].ge(threshold).astype(int)
            )
            predictions.append(fold_predictions)

    prediction_frame = (
        pd.concat(predictions, ignore_index=True)
        if predictions
        else pd.DataFrame(columns=PREDICTION_COLUMNS)
    )
    diagnostic_frame = pd.DataFrame(diagnostics, columns=FOLD_DIAGNOSTIC_COLUMNS)
    return prediction_frame, diagnostic_frame


def _combined_predictions(predictions: pd.DataFrame, *, threshold: float) -> pd.DataFrame:
    if predictions.empty:
        return pd.DataFrame(
            columns=[
                "symbol",
                "signal_date",
                "effective_date",
                "meta_exit_score",
                "meta_exit_predicted",
                "meta_exit_model_count",
            ]
        )
    grouped = (
        predictions.groupby(["symbol", "signal_date", "effective_date"], as_index=False)
        .agg(
            meta_exit_score=("score", "mean"),
            meta_exit_model_count=("model_name", "nunique"),
        )
        .sort_values(["symbol", "signal_date"])
        .reset_index(drop=True)
    )
    grouped["meta_exit_predicted"] = grouped["meta_exit_score"].ge(threshold)
    return grouped


def generate_meta_overlay_diagnostics(
    overlay_diagnostics: pd.DataFrame,
    predictions: pd.DataFrame,
    *,
    threshold: float,
    reentry_clear_bars: int,
    require_price_below_trend_for_exit: bool = False,
    bearish_confirmation_window_bars: int = 1,
    min_cash_bars: int = 0,
    exit_cooldown_bars: int = 0,
    reentry_requires_price_above_trend: bool = False,
    strategy_name: str = "pattern_meta_label_exit_overlay",
) -> pd.DataFrame:
    if overlay_diagnostics.empty:
        return pd.DataFrame(columns=META_OVERLAY_COLUMNS)

    working = overlay_diagnostics.copy()
    working["timestamp"] = pd.to_datetime(working["timestamp"])
    working["effective_date"] = pd.to_datetime(working["effective_date"])
    prediction_keys = _combined_predictions(predictions, threshold=threshold)
    if not prediction_keys.empty:
        prediction_keys["signal_date"] = pd.to_datetime(prediction_keys["signal_date"])
        prediction_keys["effective_date"] = pd.to_datetime(prediction_keys["effective_date"])
    working = working.merge(
        prediction_keys,
        left_on=["symbol", "timestamp", "effective_date"],
        right_on=["symbol", "signal_date", "effective_date"],
        how="left",
    )
    working["meta_exit_score"] = pd.to_numeric(
        working["meta_exit_score"],
        errors="coerce",
    )
    working["meta_exit_predicted"] = working["meta_exit_predicted"].where(
        working["meta_exit_predicted"].notna(),
        False,
    ).astype(bool)
    working["meta_exit_model_count"] = (
        pd.to_numeric(working["meta_exit_model_count"], errors="coerce").fillna(0).astype(int)
    )

    rows: list[dict[str, object]] = []
    for _, symbol_rows in working.sort_values(["symbol", "timestamp"]).groupby(
        "symbol",
        sort=False,
    ):
        current_weight = 1.0
        clear_count = 0
        bearish_confirmation_count = 0
        bars_since_exit: int | None = None
        bars_since_reentry: int | None = 0
        cash_bars = 0
        for row in symbol_rows.itertuples(index=False):
            bearish_clear = bool(row.bearish_clear)
            clear_count = clear_count + 1 if bearish_clear else 0
            raw_bearish_exit = bool(row.bearish_exit_candidate)
            bearish_confirmation_count = (
                bearish_confirmation_count + 1 if raw_bearish_exit else 0
            )
            bullish_reentry = bool(row.bullish_reentry_candidate)
            trend_reentry = pd.notna(row.trend_ema) and float(row.close) >= float(row.trend_ema)
            below_trend = pd.notna(row.trend_ema) and float(row.close) < float(row.trend_ema)
            exit_blocked_by_trend = (
                raw_bearish_exit and require_price_below_trend_for_exit and not below_trend
            )
            exit_blocked_by_confirmation = (
                raw_bearish_exit
                and bearish_confirmation_count < bearish_confirmation_window_bars
            )
            exit_blocked_by_cooldown = (
                raw_bearish_exit
                and bars_since_exit is not None
                and bars_since_exit <= exit_cooldown_bars
            )
            bearish_exit = (
                raw_bearish_exit
                and bool(row.meta_exit_predicted)
                and not exit_blocked_by_trend
                and not exit_blocked_by_confirmation
                and not exit_blocked_by_cooldown
            )
            reentry_blocked_by_min_cash = (
                current_weight == 0.0
                and cash_bars < min_cash_bars
                and (bullish_reentry or trend_reentry)
            )
            reentry_signal = (
                clear_count >= reentry_clear_bars
                and (bullish_reentry or trend_reentry)
                and not reentry_blocked_by_min_cash
            )
            if reentry_requires_price_above_trend:
                reentry_signal = reentry_signal and trend_reentry

            exit_reason = ""
            reentry_reason = ""
            if current_weight > 0.0 and bearish_exit:
                current_weight = 0.0
                bars_since_exit = 0
                bars_since_reentry = None
                cash_bars = 0
                exit_reason = "meta_bearish_pattern"
            elif current_weight == 0.0 and reentry_signal:
                current_weight = 1.0
                bars_since_reentry = 0
                cash_bars = 0
                reentry_reason = "bullish_or_trend_reentry"

            output = {
                column: getattr(row, column)
                for column in OVERLAY_DIAGNOSTIC_COLUMNS
                if hasattr(row, column)
            }
            output["strategy"] = strategy_name
            output["bearish_confirmation_count"] = bearish_confirmation_count
            output["exit_blocked_by_trend"] = exit_blocked_by_trend
            output["exit_blocked_by_confirmation"] = exit_blocked_by_confirmation
            output["exit_blocked_by_cooldown"] = exit_blocked_by_cooldown
            output["reentry_blocked_by_min_cash"] = reentry_blocked_by_min_cash
            output["bars_since_exit"] = pd.NA if bars_since_exit is None else bars_since_exit
            output["bars_since_reentry"] = (
                pd.NA if bars_since_reentry is None else bars_since_reentry
            )
            output["exit_reason"] = exit_reason
            output["reentry_reason"] = reentry_reason
            output["reentry_clear_count"] = clear_count
            output["target_weight"] = current_weight
            output["meta_exit_score"] = row.meta_exit_score
            output["meta_exit_predicted"] = bool(row.meta_exit_predicted)
            output["meta_exit_model_count"] = int(row.meta_exit_model_count)
            rows.append(output)

            if current_weight == 0.0:
                cash_bars += 1
                if bars_since_exit is not None:
                    bars_since_exit += 1
            else:
                if bars_since_reentry is not None:
                    bars_since_reentry += 1
                if bars_since_exit is not None:
                    bars_since_exit += 1

    return pd.DataFrame(rows, columns=META_OVERLAY_COLUMNS)


def generate_weights(diagnostics: pd.DataFrame) -> pd.DataFrame:
    if diagnostics.empty:
        return pd.DataFrame(columns=["strategy", "effective_date", "symbol", "weight"])
    return diagnostics.loc[
        :,
        ["strategy", "effective_date", "symbol", "target_weight"],
    ].rename(columns={"target_weight": "weight"})
