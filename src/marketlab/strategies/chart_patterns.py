from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

from marketlab.rebalance import next_effective_dates, rebalance_signal_dates
from marketlab.strategies.pattern_catalog import PATTERN_CATALOG

BULLISH_PATTERN_COLUMNS = [
    "ascending_triangle_breakout",
    "symmetrical_triangle_breakout",
    "bullish_rectangle_breakout",
    "inverse_head_and_shoulders_breakout",
    "double_bottom_breakout",
    "triple_bottom_breakout",
    "falling_wedge_breakout",
    "bull_flag_breakout",
    "pennant_breakout",
    "cup_and_handle_breakout",
    "ascending_channel_continuation",
    "megaphone_breakout",
]
BEARISH_PATTERN_COLUMNS = [
    "descending_triangle_breakdown",
    "bearish_rectangle_breakdown",
    "head_and_shoulders_breakdown",
    "double_top_breakdown",
    "triple_top_breakdown",
    "rising_wedge_breakdown",
    "bear_flag_breakdown",
    "descending_channel_breakdown",
    "diamond_breakdown",
]
PATTERN_COLUMNS = [*BULLISH_PATTERN_COLUMNS, *BEARISH_PATTERN_COLUMNS]
DIAGNOSTIC_COLUMNS = [
    "strategy",
    "timestamp",
    "effective_date",
    "symbol",
    "close",
    "support_level",
    "resistance_level",
    "volume_average",
    "volume_confirmed",
    *PATTERN_COLUMNS,
    "bullish_pattern_count",
    "bearish_pattern_count",
    "target_weight",
]
ARCHETYPE_THRESHOLD = 0.995
ARCHETYPE_TEMPLATES = {
    column: np.asarray(spec.closes, dtype=float)
    for spec in PATTERN_CATALOG
    for column in spec.implemented_columns
}
def _slope_array(values: np.ndarray) -> float:
    if len(values) < 2:
        return 0.0
    x = np.arange(len(values), dtype=float)
    x_mean = float(x.mean())
    y_mean = float(values.mean())
    denominator = float(np.sum((x - x_mean) ** 2))
    if denominator == 0.0:
        return 0.0
    return float(np.sum((x - x_mean) * (values - y_mean)) / denominator)


def _within_tolerance(first: float, second: float, tolerance_pct: float) -> bool:
    denominator = max(abs(first), abs(second), 1e-12)
    return abs(first - second) / denominator <= tolerance_pct


def _normalized_shape(values: np.ndarray) -> np.ndarray:
    span = float(values.max() - values.min())
    if span == 0.0:
        return np.zeros_like(values, dtype=float)
    return (values - float(values.mean())) / span


NORMALIZED_ARCHETYPE_TEMPLATES = {
    column: _normalized_shape(template) for column, template in ARCHETYPE_TEMPLATES.items()
}


def _detect_archetype_matrix(
    closes: np.ndarray,
    volume_confirmed: np.ndarray,
) -> dict[str, np.ndarray]:
    detections = {column: np.zeros(len(closes), dtype=bool) for column in PATTERN_COLUMNS}
    templates_by_length: dict[int, list[tuple[str, np.ndarray]]] = {}
    for column, template in NORMALIZED_ARCHETYPE_TEMPLATES.items():
        templates_by_length.setdefault(len(template), []).append((column, template))

    for window_length, templates in templates_by_length.items():
        if len(closes) < window_length:
            continue
        windows = sliding_window_view(closes, window_length)
        spans = windows.max(axis=1) - windows.min(axis=1)
        normalized = np.zeros_like(windows, dtype=float)
        non_flat = spans != 0.0
        normalized[non_flat] = (
            windows[non_flat] - windows[non_flat].mean(axis=1, keepdims=True)
        ) / spans[non_flat, None]
        target_index = np.arange(window_length - 1, len(closes))
        confirmed = volume_confirmed[target_index]
        for column, template in templates:
            distances = np.sqrt(np.mean((normalized - template) ** 2, axis=1))
            matched = (1.0 - distances >= ARCHETYPE_THRESHOLD) & confirmed
            detections[column][target_index] = matched
    return detections


def _split_extreme_pair_array(
    values: np.ndarray,
    *,
    mode: str,
) -> tuple[float, float, int, int]:
    midpoint = len(values) // 2
    first_half = values[:midpoint]
    second_half = values[midpoint:]
    if len(first_half) == 0 or len(second_half) == 0:
        return np.nan, np.nan, -1, -1

    first_pos = int(np.argmin(first_half) if mode == "min" else np.argmax(first_half))
    second_local_pos = int(
        np.argmin(second_half) if mode == "min" else np.argmax(second_half)
    )
    second_pos = midpoint + second_local_pos
    return (
        float(values[first_pos]),
        float(values[second_pos]),
        first_pos,
        second_pos,
    )


def _detect_for_symbol(
    symbol_frame: pd.DataFrame,
    *,
    lookback_bars: int,
    triangle_slope_min: float,
    level_tolerance_pct: float,
    breakout_pct: float,
    rectangle_max_range_pct: float,
    flag_pole_bars: int,
    flag_consolidation_bars: int,
    flag_min_pole_return: float,
    flag_max_retrace_pct: float,
    volume_window: int,
    volume_multiplier: float,
) -> pd.DataFrame:
    frame = symbol_frame.sort_values("timestamp").reset_index(drop=True).copy()
    frame["volume_average"] = frame["volume"].rolling(
        volume_window,
        min_periods=volume_window,
    ).mean()
    timestamps = frame["timestamp"].to_numpy()
    symbol = str(frame["symbol"].iloc[0]) if not frame.empty else ""
    closes = frame["adj_close"].to_numpy(dtype=float)
    highs = frame["adj_high"].to_numpy(dtype=float)
    lows = frame["adj_low"].to_numpy(dtype=float)
    volumes = frame["volume"].to_numpy(dtype=float)
    volume_averages = frame["volume_average"].to_numpy(dtype=float)
    volume_confirmed = np.isfinite(volume_averages) & (
        volumes >= volume_averages * volume_multiplier
    )
    pattern_arrays = _detect_archetype_matrix(closes, volume_confirmed)
    support_levels = np.full(len(frame), np.nan, dtype=float)
    resistance_levels = np.full(len(frame), np.nan, dtype=float)

    for index in range(len(frame)):
        prior_start = max(0, index - lookback_bars)
        prior_highs = highs[prior_start:index]
        prior_lows = lows[prior_start:index]
        current_close = float(closes[index])
        is_volume_confirmed = bool(volume_confirmed[index])

        if len(prior_highs) >= max(lookback_bars // 2, 4):
            resistance = float(np.max(prior_highs))
            support = float(np.min(prior_lows))
            support_levels[index] = support
            resistance_levels[index] = resistance
            range_pct = (resistance - support) / max(abs(current_close), 1e-12)
            high_slope = _slope_array(prior_highs)
            low_slope = _slope_array(prior_lows)

            breaks_resistance = bool(current_close > resistance * (1.0 + breakout_pct))
            breaks_support = bool(current_close < support * (1.0 - breakout_pct))
            flat_resistance = (
                abs(high_slope) / max(abs(current_close), 1e-12)
                <= level_tolerance_pct / lookback_bars
            )
            flat_support = (
                abs(low_slope) / max(abs(current_close), 1e-12)
                <= level_tolerance_pct / lookback_bars
            )

            pattern_arrays["ascending_triangle_breakout"][index] = (
                pattern_arrays["ascending_triangle_breakout"][index]
                or (
                is_volume_confirmed
                and breaks_resistance
                and flat_resistance
                and low_slope > triangle_slope_min
                )
            )
            pattern_arrays["descending_triangle_breakdown"][index] = (
                pattern_arrays["descending_triangle_breakdown"][index]
                or (
                is_volume_confirmed
                and breaks_support
                and flat_support
                and high_slope < -triangle_slope_min
                )
            )
            pattern_arrays["bullish_rectangle_breakout"][index] = (
                pattern_arrays["bullish_rectangle_breakout"][index]
                or (
                    is_volume_confirmed
                    and breaks_resistance
                    and range_pct <= rectangle_max_range_pct
                )
            )
            pattern_arrays["bearish_rectangle_breakdown"][index] = (
                pattern_arrays["bearish_rectangle_breakdown"][index]
                or (
                    is_volume_confirmed
                    and breaks_support
                    and range_pct <= rectangle_max_range_pct
                )
            )

            low_a, low_b, low_a_pos, low_b_pos = _split_extreme_pair_array(
                prior_lows,
                mode="min",
            )
            if low_a_pos >= 0 and low_b_pos > low_a_pos:
                neckline = float(np.max(prior_highs[low_a_pos : low_b_pos + 1]))
                pattern_arrays["double_bottom_breakout"][index] = (
                    pattern_arrays["double_bottom_breakout"][index]
                    or (
                        is_volume_confirmed
                        and _within_tolerance(low_a, low_b, level_tolerance_pct)
                        and current_close > neckline * (1.0 + breakout_pct)
                    )
                )

            high_a, high_b, high_a_pos, high_b_pos = _split_extreme_pair_array(
                prior_highs,
                mode="max",
            )
            if high_a_pos >= 0 and high_b_pos > high_a_pos:
                neckline = float(np.min(prior_lows[high_a_pos : high_b_pos + 1]))
                pattern_arrays["double_top_breakdown"][index] = (
                    pattern_arrays["double_top_breakdown"][index]
                    or (
                        is_volume_confirmed
                        and _within_tolerance(high_a, high_b, level_tolerance_pct)
                        and current_close < neckline * (1.0 - breakout_pct)
                    )
                )

        flag_span = flag_pole_bars + flag_consolidation_bars
        if index >= flag_span:
            pole_start = float(closes[index - flag_span])
            pole_end = float(closes[index - flag_consolidation_bars])
            consolidation_highs = highs[index - flag_consolidation_bars : index]
            consolidation_lows = lows[index - flag_consolidation_bars : index]
            consolidation_closes = closes[index - flag_consolidation_bars : index]
            pole_return = (pole_end / pole_start) - 1.0
            consolidation_high = float(np.max(consolidation_highs))
            consolidation_low = float(np.min(consolidation_lows))
            consolidation_slope = _slope_array(consolidation_closes)
            pullback = (pole_end - consolidation_low) / max(abs(pole_end), 1e-12)
            bounce = (consolidation_high - pole_end) / max(abs(pole_end), 1e-12)
            pattern_arrays["bull_flag_breakout"][index] = (
                pattern_arrays["bull_flag_breakout"][index]
                or (
                    is_volume_confirmed
                    and pole_return >= flag_min_pole_return
                    and pullback <= flag_max_retrace_pct
                    and consolidation_slope <= 0.0
                    and current_close > consolidation_high * (1.0 + breakout_pct)
                )
            )
            pattern_arrays["bear_flag_breakdown"][index] = (
                pattern_arrays["bear_flag_breakdown"][index]
                or (
                    is_volume_confirmed
                    and pole_return <= -flag_min_pole_return
                    and bounce <= flag_max_retrace_pct
                    and consolidation_slope >= 0.0
                    and current_close < consolidation_low * (1.0 - breakout_pct)
                )
            )

    result = pd.DataFrame(
        {
            "timestamp": timestamps,
            "symbol": symbol,
            "close": closes,
            "support_level": support_levels,
            "resistance_level": resistance_levels,
            "volume_average": volume_averages,
            "volume_confirmed": volume_confirmed,
        }
    )
    for column in PATTERN_COLUMNS:
        result[column] = pattern_arrays[column]
    result["bullish_pattern_count"] = result[BULLISH_PATTERN_COLUMNS].sum(axis=1)
    result["bearish_pattern_count"] = result[BEARISH_PATTERN_COLUMNS].sum(axis=1)
    return result


def build_pattern_frame(
    panel: pd.DataFrame,
    *,
    lookback_bars: int,
    triangle_slope_min: float,
    level_tolerance_pct: float,
    breakout_pct: float,
    rectangle_max_range_pct: float,
    flag_pole_bars: int,
    flag_consolidation_bars: int,
    flag_min_pole_return: float,
    flag_max_retrace_pct: float,
    volume_window: int,
    volume_multiplier: float,
) -> pd.DataFrame:
    working = panel.sort_values(["symbol", "timestamp"]).copy()
    frames = [
        _detect_for_symbol(
            symbol_frame,
            lookback_bars=lookback_bars,
            triangle_slope_min=triangle_slope_min,
            level_tolerance_pct=level_tolerance_pct,
            breakout_pct=breakout_pct,
            rectangle_max_range_pct=rectangle_max_range_pct,
            flag_pole_bars=flag_pole_bars,
            flag_consolidation_bars=flag_consolidation_bars,
            flag_min_pole_return=flag_min_pole_return,
            flag_max_retrace_pct=flag_max_retrace_pct,
            volume_window=volume_window,
            volume_multiplier=volume_multiplier,
        )
        for _, symbol_frame in working.groupby("symbol", sort=False)
    ]
    return pd.concat(frames, ignore_index=True).sort_values(["symbol", "timestamp"]).reset_index(drop=True)


def generate_diagnostics(
    panel: pd.DataFrame,
    *,
    frequency: str,
    lookback_bars: int,
    triangle_slope_min: float,
    level_tolerance_pct: float,
    breakout_pct: float,
    rectangle_max_range_pct: float,
    flag_pole_bars: int,
    flag_consolidation_bars: int,
    flag_min_pole_return: float,
    flag_max_retrace_pct: float,
    volume_window: int,
    volume_multiplier: float,
    min_bullish_patterns: int,
    strategy_name: str = "chart_patterns",
) -> pd.DataFrame:
    pattern_frame = build_pattern_frame(
        panel,
        lookback_bars=lookback_bars,
        triangle_slope_min=triangle_slope_min,
        level_tolerance_pct=level_tolerance_pct,
        breakout_pct=breakout_pct,
        rectangle_max_range_pct=rectangle_max_range_pct,
        flag_pole_bars=flag_pole_bars,
        flag_consolidation_bars=flag_consolidation_bars,
        flag_min_pole_return=flag_min_pole_return,
        flag_max_retrace_pct=flag_max_retrace_pct,
        volume_window=volume_window,
        volume_multiplier=volume_multiplier,
    )
    effective_dates = next_effective_dates(
        pattern_frame,
        rebalance_signal_dates(pattern_frame, frequency),
    )
    if effective_dates.empty:
        return pd.DataFrame(columns=DIAGNOSTIC_COLUMNS)

    effective_frame = (
        effective_dates.rename_axis("timestamp")
        .reset_index()
        .rename(columns={0: "effective_date"})
    )
    if "effective_date" not in effective_frame.columns:
        effective_frame = effective_frame.rename(columns={effective_dates.name: "effective_date"})
    diagnostics = pattern_frame.merge(effective_frame, on="timestamp", how="inner")
    diagnostics["strategy"] = strategy_name
    selected = (
        diagnostics["bullish_pattern_count"].ge(min_bullish_patterns)
        & diagnostics["bearish_pattern_count"].eq(0)
    )
    selected_count = selected.groupby(diagnostics["timestamp"]).transform("sum")
    diagnostics["target_weight"] = 0.0
    selected_with_capacity = selected & selected_count.gt(0)
    diagnostics.loc[selected_with_capacity, "target_weight"] = (
        1.0 / selected_count.loc[selected_with_capacity]
    )
    diagnostics["volume_confirmed"] = diagnostics["volume_confirmed"].astype(bool)
    for column in PATTERN_COLUMNS:
        diagnostics[column] = diagnostics[column].astype(bool)

    return diagnostics.loc[:, DIAGNOSTIC_COLUMNS].sort_values(
        ["timestamp", "symbol"],
    ).reset_index(drop=True)


def generate_weights(
    panel: pd.DataFrame,
    *,
    frequency: str,
    lookback_bars: int,
    triangle_slope_min: float,
    level_tolerance_pct: float,
    breakout_pct: float,
    rectangle_max_range_pct: float,
    flag_pole_bars: int,
    flag_consolidation_bars: int,
    flag_min_pole_return: float,
    flag_max_retrace_pct: float,
    volume_window: int,
    volume_multiplier: float,
    min_bullish_patterns: int,
    strategy_name: str = "chart_patterns",
) -> pd.DataFrame:
    diagnostics = generate_diagnostics(
        panel,
        frequency=frequency,
        lookback_bars=lookback_bars,
        triangle_slope_min=triangle_slope_min,
        level_tolerance_pct=level_tolerance_pct,
        breakout_pct=breakout_pct,
        rectangle_max_range_pct=rectangle_max_range_pct,
        flag_pole_bars=flag_pole_bars,
        flag_consolidation_bars=flag_consolidation_bars,
        flag_min_pole_return=flag_min_pole_return,
        flag_max_retrace_pct=flag_max_retrace_pct,
        volume_window=volume_window,
        volume_multiplier=volume_multiplier,
        min_bullish_patterns=min_bullish_patterns,
        strategy_name=strategy_name,
    )
    if diagnostics.empty:
        return pd.DataFrame(columns=["strategy", "effective_date", "symbol", "weight"])
    return diagnostics.loc[
        :,
        ["strategy", "effective_date", "symbol", "target_weight"],
    ].rename(columns={"target_weight": "weight"})
