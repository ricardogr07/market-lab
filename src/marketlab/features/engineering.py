from __future__ import annotations

import math

import pandas as pd


def _add_crypto_time_series_features(
    featured: pd.DataFrame,
    *,
    return_windows: list[int],
    vol_windows: list[int],
    ma_windows: list[int],
    rsi_window: int,
    macd_fast_window: int,
    macd_slow_window: int,
    macd_signal_window: int,
    bollinger_window: int,
    bollinger_std: float,
    volume_window: int,
    time_features: bool,
) -> pd.DataFrame:
    grouped = featured.groupby("symbol", group_keys=False)
    grouped_close = grouped["adj_close"]
    returns_1 = grouped_close.pct_change()

    for window in sorted(set(return_windows)):
        featured[f"crypto_return_{window}"] = grouped_close.transform(
            lambda series, window=window: series.pct_change(window)
        )

    for window in sorted(set(vol_windows)):
        featured[f"crypto_vol_{window}"] = returns_1.groupby(featured["symbol"]).transform(
            lambda series, window=window: series.rolling(window, min_periods=window).std(ddof=0)
        )

    for window in sorted(set(ma_windows)):
        ma = grouped_close.transform(
            lambda series, window=window: series.rolling(window, min_periods=window).mean()
        )
        featured[f"crypto_ma_{window}"] = ma
        featured[f"crypto_price_to_ma_{window}"] = featured["adj_close"] / ma
        featured[f"crypto_ma_slope_{window}"] = ma.groupby(featured["symbol"]).pct_change(
            window,
            fill_method=None,
        )

    close_delta = grouped_close.diff()
    gains = close_delta.clip(lower=0.0)
    losses = -close_delta.clip(upper=0.0)
    avg_gain = gains.groupby(featured["symbol"]).transform(
        lambda series: series.rolling(rsi_window, min_periods=rsi_window).mean()
    )
    avg_loss = losses.groupby(featured["symbol"]).transform(
        lambda series: series.rolling(rsi_window, min_periods=rsi_window).mean()
    )
    relative_strength = avg_gain / avg_loss.replace(0.0, pd.NA)
    featured[f"crypto_rsi_{rsi_window}"] = 100.0 - (100.0 / (1.0 + relative_strength))

    macd_fast = grouped_close.transform(
        lambda series: series.ewm(span=macd_fast_window, adjust=False).mean()
    )
    macd_slow = grouped_close.transform(
        lambda series: series.ewm(span=macd_slow_window, adjust=False).mean()
    )
    macd = macd_fast - macd_slow
    macd_signal = macd.groupby(featured["symbol"]).transform(
        lambda series: series.ewm(span=macd_signal_window, adjust=False).mean()
    )
    featured["crypto_macd"] = macd
    featured["crypto_macd_signal"] = macd_signal
    featured["crypto_macd_hist"] = macd - macd_signal

    bollinger_mean = grouped_close.transform(
        lambda series: series.rolling(bollinger_window, min_periods=bollinger_window).mean()
    )
    bollinger_std_series = grouped_close.transform(
        lambda series: series.rolling(bollinger_window, min_periods=bollinger_window).std(ddof=0)
    )
    featured[f"crypto_bollinger_z_{bollinger_window}"] = (
        (featured["adj_close"] - bollinger_mean)
        / (bollinger_std_series.replace(0.0, pd.NA) * bollinger_std)
    )

    grouped_volume = grouped["volume"]
    volume_mean = grouped_volume.transform(
        lambda series: series.rolling(volume_window, min_periods=volume_window).mean()
    )
    volume_std = grouped_volume.transform(
        lambda series: series.rolling(volume_window, min_periods=volume_window).std(ddof=0)
    )
    featured[f"crypto_volume_z_{volume_window}"] = (
        (featured["volume"] - volume_mean) / volume_std.replace(0.0, pd.NA)
    )

    if time_features:
        timestamps = pd.to_datetime(featured["timestamp"])
        hour_angle = 2.0 * math.pi * (timestamps.dt.hour / 24.0)
        day_angle = 2.0 * math.pi * (timestamps.dt.dayofweek / 7.0)
        featured["crypto_hour_sin"] = hour_angle.map(math.sin)
        featured["crypto_hour_cos"] = hour_angle.map(math.cos)
        featured["crypto_dayofweek_sin"] = day_angle.map(math.sin)
        featured["crypto_dayofweek_cos"] = day_angle.map(math.cos)

    return featured


def add_feature_set(
    panel: pd.DataFrame,
    return_windows: list[int],
    ma_windows: list[int],
    vol_windows: list[int],
    momentum_window: int,
    indicator_stack_ml_features_enabled: bool = False,
    crypto_time_series_enabled: bool = False,
    crypto_return_windows: list[int] | None = None,
    crypto_vol_windows: list[int] | None = None,
    crypto_ma_windows: list[int] | None = None,
    crypto_rsi_window: int = 14,
    crypto_macd_fast_window: int = 12,
    crypto_macd_slow_window: int = 26,
    crypto_macd_signal_window: int = 9,
    crypto_bollinger_window: int = 20,
    crypto_bollinger_std: float = 2.0,
    crypto_volume_window: int = 24,
    crypto_time_features: bool = True,
) -> pd.DataFrame:
    featured = panel.sort_values(["symbol", "timestamp"]).copy()
    grouped_close = featured.groupby("symbol")["adj_close"]
    daily_returns = grouped_close.pct_change()

    for window in sorted(set(return_windows)):
        featured[f"return_{window}"] = grouped_close.transform(
            lambda series, window=window: series.pct_change(window)
        )

    ordered_ma_windows = sorted(set(ma_windows))
    for window in ordered_ma_windows:
        featured[f"ma_{window}"] = grouped_close.transform(
            lambda series, window=window: series.rolling(window, min_periods=window).mean()
        )
        featured[f"price_to_ma_{window}"] = featured["adj_close"] / featured[f"ma_{window}"]

    for fast, slow in zip(ordered_ma_windows, ordered_ma_windows[1:]):
        featured[f"ma_{fast}_minus_ma_{slow}"] = (
            featured[f"ma_{fast}"] - featured[f"ma_{slow}"]
        )

    for window in sorted(set(vol_windows)):
        featured[f"vol_{window}"] = daily_returns.groupby(featured["symbol"]).transform(
            lambda series, window=window: series.rolling(window, min_periods=window).std(ddof=0)
        )

    featured["momentum"] = grouped_close.transform(
        lambda series: series.diff(momentum_window)
    )
    if crypto_time_series_enabled:
        featured = _add_crypto_time_series_features(
            featured,
            return_windows=crypto_return_windows or [],
            vol_windows=crypto_vol_windows or [],
            ma_windows=crypto_ma_windows or [],
            rsi_window=crypto_rsi_window,
            macd_fast_window=crypto_macd_fast_window,
            macd_slow_window=crypto_macd_slow_window,
            macd_signal_window=crypto_macd_signal_window,
            bollinger_window=crypto_bollinger_window,
            bollinger_std=crypto_bollinger_std,
            volume_window=crypto_volume_window,
            time_features=crypto_time_features,
        )
    return featured
