from __future__ import annotations

import pandas as pd

from marketlab.strategies.chart_patterns import (
    BEARISH_PATTERN_COLUMNS,
    BULLISH_PATTERN_COLUMNS,
    PATTERN_COLUMNS,
)

DIAGNOSTIC_COLUMNS = [
    "strategy",
    "timestamp",
    "effective_date",
    "symbol",
    "close",
    "trend_ema",
    "support_level",
    "resistance_level",
    "volume_average",
    "volume_confirmed",
    *PATTERN_COLUMNS,
    "bullish_pattern_count",
    "bearish_pattern_count",
    "bearish_exit_candidate",
    "bullish_reentry_candidate",
    "bearish_clear",
    "bearish_confirmation_count",
    "exit_blocked_by_trend",
    "exit_blocked_by_confirmation",
    "exit_blocked_by_cooldown",
    "reentry_blocked_by_min_cash",
    "bars_since_exit",
    "bars_since_reentry",
    "exit_reason",
    "reentry_reason",
    "reentry_clear_count",
    "target_weight",
]


def _trend_frame(panel: pd.DataFrame, *, trend_ema_window: int) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for _, symbol_frame in panel.sort_values(["symbol", "timestamp"]).groupby(
        "symbol",
        sort=False,
    ):
        working = symbol_frame.loc[:, ["timestamp", "symbol", "adj_close"]].copy()
        working["trend_ema"] = working["adj_close"].ewm(
            span=trend_ema_window,
            adjust=False,
        ).mean()
        rows.append(working.loc[:, ["timestamp", "symbol", "trend_ema"]])
    if not rows:
        return pd.DataFrame(columns=["timestamp", "symbol", "trend_ema"])
    return pd.concat(rows, ignore_index=True)


def _prepare_overlay_input(
    panel: pd.DataFrame,
    pattern_diagnostics: pd.DataFrame,
    *,
    min_bearish_patterns: int,
    min_bullish_reentry_patterns: int,
    trend_ema_window: int,
) -> pd.DataFrame:
    if pattern_diagnostics.empty:
        return pd.DataFrame(columns=DIAGNOSTIC_COLUMNS)

    diagnostics = pattern_diagnostics.copy()
    diagnostics["timestamp"] = pd.to_datetime(diagnostics["timestamp"])
    diagnostics["effective_date"] = pd.to_datetime(diagnostics["effective_date"])
    diagnostics = diagnostics.merge(
        _trend_frame(panel, trend_ema_window=trend_ema_window),
        on=["timestamp", "symbol"],
        how="left",
    )
    for column in PATTERN_COLUMNS:
        diagnostics[column] = diagnostics[column].fillna(False).astype(bool)
    diagnostics["bullish_pattern_count"] = diagnostics[BULLISH_PATTERN_COLUMNS].sum(axis=1)
    diagnostics["bearish_pattern_count"] = diagnostics[BEARISH_PATTERN_COLUMNS].sum(axis=1)
    diagnostics["bearish_exit_candidate"] = (
        diagnostics["bearish_pattern_count"].ge(min_bearish_patterns)
        & diagnostics["bullish_pattern_count"].eq(0)
    )
    diagnostics["bullish_reentry_candidate"] = (
        diagnostics["bullish_pattern_count"].ge(min_bullish_reentry_patterns)
        & diagnostics["close"].ge(diagnostics["trend_ema"])
    )
    diagnostics["bearish_clear"] = diagnostics["bearish_pattern_count"].eq(0)
    return diagnostics.sort_values(["symbol", "timestamp"]).reset_index(drop=True)


def generate_diagnostics(
    panel: pd.DataFrame,
    pattern_diagnostics: pd.DataFrame,
    *,
    min_bearish_patterns: int,
    min_bullish_reentry_patterns: int,
    trend_ema_window: int,
    reentry_clear_bars: int,
    require_price_below_trend_for_exit: bool = False,
    bearish_confirmation_window_bars: int = 1,
    min_cash_bars: int = 0,
    exit_cooldown_bars: int = 0,
    reentry_requires_price_above_trend: bool = False,
    strategy_name: str = "pattern_exit_overlay",
) -> pd.DataFrame:
    diagnostics = _prepare_overlay_input(
        panel,
        pattern_diagnostics,
        min_bearish_patterns=min_bearish_patterns,
        min_bullish_reentry_patterns=min_bullish_reentry_patterns,
        trend_ema_window=trend_ema_window,
    )
    if diagnostics.empty:
        return pd.DataFrame(columns=DIAGNOSTIC_COLUMNS)

    rows: list[dict[str, object]] = []
    for _, symbol_rows in diagnostics.groupby("symbol", sort=False):
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
                exit_reason = "bearish_pattern"
            elif current_weight == 0.0 and reentry_signal:
                current_weight = 1.0
                bars_since_reentry = 0
                cash_bars = 0
                reentry_reason = "bullish_or_trend_reentry"

            output = row._asdict()
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

    return pd.DataFrame(rows, columns=DIAGNOSTIC_COLUMNS)


def generate_weights(
    diagnostics: pd.DataFrame,
) -> pd.DataFrame:
    if diagnostics.empty:
        return pd.DataFrame(columns=["strategy", "effective_date", "symbol", "weight"])
    return diagnostics.loc[
        :,
        ["strategy", "effective_date", "symbol", "target_weight"],
    ].rename(columns={"target_weight": "weight"})
