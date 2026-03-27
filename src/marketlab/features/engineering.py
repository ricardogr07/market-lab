from __future__ import annotations

import pandas as pd


def add_feature_set(
    panel: pd.DataFrame,
    return_windows: list[int],
    ma_windows: list[int],
    vol_windows: list[int],
    momentum_window: int,
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
    return featured
