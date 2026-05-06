from __future__ import annotations

import pandas as pd

from marketlab.rebalance import next_effective_dates, rebalance_signal_dates

DIAGNOSTIC_COLUMNS = [
    "strategy",
    "timestamp",
    "effective_date",
    "symbol",
    "close",
    "ema_fast",
    "ema_slow",
    "rsi",
    "macd",
    "macd_signal",
    "bollinger_mid",
    "bollinger_upper",
    "bollinger_lower",
    "volume_average",
    "vwap",
    "ema_confirmed",
    "rsi_confirmed",
    "macd_confirmed",
    "bollinger_confirmed",
    "volume_confirmed",
    "vwap_confirmed",
    "confirmation_count",
    "target_weight",
]


def _rsi(close: pd.Series, window: int) -> pd.Series:
    delta = close.diff()
    gains = delta.clip(lower=0.0)
    losses = -delta.clip(upper=0.0)
    avg_gain = gains.rolling(window, min_periods=window).mean()
    avg_loss = losses.rolling(window, min_periods=window).mean()
    relative_strength = avg_gain / avg_loss.mask(avg_loss == 0.0)
    rsi = 100.0 - (100.0 / (1.0 + relative_strength))
    rsi = rsi.mask(avg_loss == 0.0, 100.0)
    return rsi.where(avg_gain.notna() & avg_loss.notna())


def _rolling_vwap(frame: pd.DataFrame, window: int) -> pd.Series:
    typical_price = (frame["adj_high"] + frame["adj_low"] + frame["adj_close"]) / 3.0
    price_volume = typical_price * frame["volume"]
    rolling_volume = frame["volume"].rolling(window, min_periods=window).sum()
    return price_volume.rolling(window, min_periods=window).sum() / rolling_volume


def build_indicator_frame(
    panel: pd.DataFrame,
    *,
    ema_fast_window: int,
    ema_slow_window: int,
    rsi_window: int,
    rsi_min: float,
    rsi_max: float,
    macd_fast_window: int,
    macd_slow_window: int,
    macd_signal_window: int,
    bollinger_window: int,
    bollinger_std: float,
    bollinger_mode: str,
    volume_window: int,
    volume_multiplier: float,
    vwap_window: int,
    use_vwap: bool,
) -> pd.DataFrame:
    working = panel.sort_values(["symbol", "timestamp"]).copy()
    frames: list[pd.DataFrame] = []

    for _, symbol_frame in working.groupby("symbol", sort=False):
        frame = symbol_frame.copy()
        close = frame["adj_close"]
        frame["ema_fast"] = close.ewm(span=ema_fast_window, adjust=False).mean()
        frame["ema_slow"] = close.ewm(span=ema_slow_window, adjust=False).mean()
        frame["ema_confirmed"] = frame["ema_fast"].gt(frame["ema_slow"])

        frame["rsi"] = _rsi(close, rsi_window)
        frame["rsi_confirmed"] = frame["rsi"].between(rsi_min, rsi_max, inclusive="both")

        macd_fast = close.ewm(span=macd_fast_window, adjust=False).mean()
        macd_slow = close.ewm(span=macd_slow_window, adjust=False).mean()
        frame["macd"] = macd_fast - macd_slow
        frame["macd_signal"] = frame["macd"].ewm(span=macd_signal_window, adjust=False).mean()
        frame["macd_confirmed"] = frame["macd"].gt(frame["macd_signal"])

        rolling_mean = close.rolling(bollinger_window, min_periods=bollinger_window).mean()
        rolling_std = close.rolling(bollinger_window, min_periods=bollinger_window).std(ddof=0)
        frame["bollinger_mid"] = rolling_mean
        frame["bollinger_upper"] = rolling_mean + (bollinger_std * rolling_std)
        frame["bollinger_lower"] = rolling_mean - (bollinger_std * rolling_std)
        if bollinger_mode == "breakout":
            frame["bollinger_confirmed"] = close.gt(frame["bollinger_upper"])
        else:
            frame["bollinger_confirmed"] = close.le(frame["bollinger_lower"])

        frame["volume_average"] = frame["volume"].rolling(
            volume_window,
            min_periods=volume_window,
        ).mean()
        frame["volume_std"] = frame["volume"].rolling(
            volume_window,
            min_periods=volume_window,
        ).std(ddof=0)
        frame["volume_confirmed"] = frame["volume"].ge(
            frame["volume_average"] * volume_multiplier
        )

        if use_vwap:
            frame["vwap"] = _rolling_vwap(frame, vwap_window)
            frame["vwap_confirmed"] = close.ge(frame["vwap"])
        else:
            frame["vwap"] = pd.NA
            frame["vwap_confirmed"] = False

        confirmation_columns = [
            "ema_confirmed",
            "rsi_confirmed",
            "macd_confirmed",
            "bollinger_confirmed",
            "volume_confirmed",
        ]
        if use_vwap:
            confirmation_columns.append("vwap_confirmed")
        frame["confirmation_count"] = frame[confirmation_columns].fillna(False).sum(axis=1)
        frames.append(frame)

    return pd.concat(frames, ignore_index=True).sort_values(["symbol", "timestamp"]).reset_index(drop=True)


def generate_weights(
    panel: pd.DataFrame,
    *,
    frequency: str,
    ema_fast_window: int,
    ema_slow_window: int,
    rsi_window: int,
    rsi_min: float,
    rsi_max: float,
    macd_fast_window: int,
    macd_slow_window: int,
    macd_signal_window: int,
    bollinger_window: int,
    bollinger_std: float,
    bollinger_mode: str,
    volume_window: int,
    volume_multiplier: float,
    vwap_window: int,
    use_vwap: bool,
    min_confirmations: int,
    strategy_name: str = "indicator_stack",
) -> pd.DataFrame:
    diagnostics = generate_diagnostics(
        panel,
        frequency=frequency,
        ema_fast_window=ema_fast_window,
        ema_slow_window=ema_slow_window,
        rsi_window=rsi_window,
        rsi_min=rsi_min,
        rsi_max=rsi_max,
        macd_fast_window=macd_fast_window,
        macd_slow_window=macd_slow_window,
        macd_signal_window=macd_signal_window,
        bollinger_window=bollinger_window,
        bollinger_std=bollinger_std,
        bollinger_mode=bollinger_mode,
        volume_window=volume_window,
        volume_multiplier=volume_multiplier,
        vwap_window=vwap_window,
        use_vwap=use_vwap,
        min_confirmations=min_confirmations,
        strategy_name=strategy_name,
    )
    if diagnostics.empty:
        return pd.DataFrame(columns=["strategy", "effective_date", "symbol", "weight"])

    return diagnostics.loc[
        :,
        ["strategy", "effective_date", "symbol", "target_weight"],
    ].rename(columns={"target_weight": "weight"})


def generate_diagnostics(
    panel: pd.DataFrame,
    *,
    frequency: str,
    ema_fast_window: int,
    ema_slow_window: int,
    rsi_window: int,
    rsi_min: float,
    rsi_max: float,
    macd_fast_window: int,
    macd_slow_window: int,
    macd_signal_window: int,
    bollinger_window: int,
    bollinger_std: float,
    bollinger_mode: str,
    volume_window: int,
    volume_multiplier: float,
    vwap_window: int,
    use_vwap: bool,
    min_confirmations: int,
    strategy_name: str = "indicator_stack",
) -> pd.DataFrame:
    indicator_frame = build_indicator_frame(
        panel,
        ema_fast_window=ema_fast_window,
        ema_slow_window=ema_slow_window,
        rsi_window=rsi_window,
        rsi_min=rsi_min,
        rsi_max=rsi_max,
        macd_fast_window=macd_fast_window,
        macd_slow_window=macd_slow_window,
        macd_signal_window=macd_signal_window,
        bollinger_window=bollinger_window,
        bollinger_std=bollinger_std,
        bollinger_mode=bollinger_mode,
        volume_window=volume_window,
        volume_multiplier=volume_multiplier,
        vwap_window=vwap_window,
        use_vwap=use_vwap,
    )
    symbols = sorted(indicator_frame["symbol"].unique())
    effective_dates = next_effective_dates(
        indicator_frame,
        rebalance_signal_dates(indicator_frame, frequency),
    )
    rows: list[dict[str, object]] = []

    for signal_date, effective_date in effective_dates.items():
        signal_slice = indicator_frame.loc[indicator_frame["timestamp"] == signal_date].copy()
        signal_slice = signal_slice.set_index("symbol").reindex(symbols)
        selected = signal_slice["confirmation_count"].ge(min_confirmations).fillna(False)
        selected_count = int(selected.sum())
        weight = 1.0 / selected_count if selected_count else 0.0
        for symbol in symbols:
            row = signal_slice.loc[symbol]
            rows.append(
                {
                    "strategy": strategy_name,
                    "timestamp": signal_date,
                    "effective_date": effective_date,
                    "symbol": symbol,
                    "close": row["adj_close"],
                    "ema_fast": row["ema_fast"],
                    "ema_slow": row["ema_slow"],
                    "rsi": row["rsi"],
                    "macd": row["macd"],
                    "macd_signal": row["macd_signal"],
                    "bollinger_mid": row["bollinger_mid"],
                    "bollinger_upper": row["bollinger_upper"],
                    "bollinger_lower": row["bollinger_lower"],
                    "volume_average": row["volume_average"],
                    "vwap": row["vwap"],
                    "ema_confirmed": bool(row["ema_confirmed"]),
                    "rsi_confirmed": bool(row["rsi_confirmed"]),
                    "macd_confirmed": bool(row["macd_confirmed"]),
                    "bollinger_confirmed": bool(row["bollinger_confirmed"]),
                    "volume_confirmed": bool(row["volume_confirmed"]),
                    "vwap_confirmed": bool(row["vwap_confirmed"]),
                    "confirmation_count": row["confirmation_count"],
                    "target_weight": weight if bool(selected.loc[symbol]) else 0.0,
                }
            )

    return pd.DataFrame(rows, columns=DIAGNOSTIC_COLUMNS)
