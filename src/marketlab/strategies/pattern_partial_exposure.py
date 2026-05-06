from __future__ import annotations

import pandas as pd

from marketlab.strategies.pattern_exit_overlay import (
    DIAGNOSTIC_COLUMNS as OVERLAY_DIAGNOSTIC_COLUMNS,
)
from marketlab.strategies.pattern_meta_label import _combined_predictions

PARTIAL_DIAGNOSTIC_COLUMNS = [
    *OVERLAY_DIAGNOSTIC_COLUMNS,
    "meta_exit_score",
    "meta_exit_predicted",
    "meta_exit_model_count",
    "partial_exit_predicted",
    "full_exit_predicted",
    "risk_state",
]


def generate_diagnostics(
    overlay_diagnostics: pd.DataFrame,
    predictions: pd.DataFrame,
    *,
    partial_threshold: float,
    full_threshold: float,
    partial_weight: float,
    reentry_clear_bars: int,
    require_price_below_trend_for_exit: bool = False,
    bearish_confirmation_window_bars: int = 1,
    min_cash_bars: int = 0,
    exit_cooldown_bars: int = 0,
    reentry_requires_price_above_trend: bool = False,
    strategy_name: str = "pattern_partial_exposure_overlay",
) -> pd.DataFrame:
    if overlay_diagnostics.empty:
        return pd.DataFrame(columns=PARTIAL_DIAGNOSTIC_COLUMNS)

    working = overlay_diagnostics.copy()
    working["timestamp"] = pd.to_datetime(working["timestamp"])
    working["effective_date"] = pd.to_datetime(working["effective_date"])
    prediction_keys = _combined_predictions(predictions, threshold=partial_threshold)
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
            score = row.meta_exit_score
            partial_predicted = pd.notna(score) and float(score) >= partial_threshold
            full_predicted = pd.notna(score) and float(score) >= full_threshold
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
            risk_exit = (
                raw_bearish_exit
                and partial_predicted
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
            if risk_exit:
                target_weight = 0.0 if full_predicted else partial_weight
                if target_weight < current_weight:
                    current_weight = target_weight
                    bars_since_exit = 0
                    bars_since_reentry = None
                    if current_weight == 0.0:
                        cash_bars = 0
                    exit_reason = (
                        "meta_full_exit" if current_weight == 0.0 else "meta_partial_exit"
                    )
            elif reentry_signal and current_weight < 1.0:
                current_weight = min(1.0, current_weight + partial_weight)
                bars_since_reentry = 0
                if current_weight > 0.0:
                    cash_bars = 0
                reentry_reason = "incremental_reentry"

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
            output["meta_exit_score"] = score
            output["meta_exit_predicted"] = bool(full_predicted)
            output["meta_exit_model_count"] = int(row.meta_exit_model_count)
            output["partial_exit_predicted"] = bool(partial_predicted)
            output["full_exit_predicted"] = bool(full_predicted)
            output["risk_state"] = (
                "cash"
                if current_weight == 0.0
                else "partial"
                if current_weight < 1.0
                else "long"
            )
            rows.append(output)

            if current_weight == 0.0:
                cash_bars += 1
            if current_weight < 1.0:
                if bars_since_exit is not None:
                    bars_since_exit += 1
            else:
                if bars_since_reentry is not None:
                    bars_since_reentry += 1
                if bars_since_exit is not None:
                    bars_since_exit += 1

    return pd.DataFrame(rows, columns=PARTIAL_DIAGNOSTIC_COLUMNS)


def generate_weights(diagnostics: pd.DataFrame) -> pd.DataFrame:
    if diagnostics.empty:
        return pd.DataFrame(columns=["strategy", "effective_date", "symbol", "weight"])
    return diagnostics.loc[
        :,
        ["strategy", "effective_date", "symbol", "target_weight"],
    ].rename(columns={"target_weight": "weight"})
