from __future__ import annotations

import pandas as pd


def _weekly_signal_dates(panel: pd.DataFrame) -> list[pd.Timestamp]:
    calendar = pd.DataFrame({"timestamp": sorted(panel["timestamp"].drop_duplicates())})
    calendar["rebalance_period"] = calendar["timestamp"].dt.to_period("W-FRI")
    return calendar.groupby("rebalance_period")["timestamp"].max().sort_values().tolist()


def generate_weights(
    panel: pd.DataFrame,
    fast_window: int,
    slow_window: int,
    strategy_name: str = "sma",
) -> pd.DataFrame:
    if fast_window >= slow_window:
        raise ValueError("SMA strategy requires fast_window < slow_window.")

    working = panel.sort_values(["symbol", "timestamp"]).copy()
    grouped = working.groupby("symbol")["adj_close"]
    working["fast_ma"] = grouped.transform(
        lambda series: series.rolling(fast_window, min_periods=fast_window).mean()
    )
    working["slow_ma"] = grouped.transform(
        lambda series: series.rolling(slow_window, min_periods=slow_window).mean()
    )

    symbols = sorted(working["symbol"].unique())
    unique_dates = pd.Index(sorted(working["timestamp"].drop_duplicates()))
    rows: list[dict[str, object]] = []

    for signal_date in _weekly_signal_dates(working):
        next_dates = unique_dates[unique_dates > signal_date]
        if next_dates.empty:
            continue
        effective_date = next_dates.min()
        signal_slice = (
            working.loc[working["timestamp"] == signal_date, ["symbol", "fast_ma", "slow_ma"]]
            .set_index("symbol")
            .reindex(symbols)
        )
        positive = signal_slice["fast_ma"].gt(signal_slice["slow_ma"]).fillna(False)
        count = int(positive.sum())
        weight = 1.0 / count if count else 0.0

        for symbol in symbols:
            rows.append(
                {
                    "strategy": strategy_name,
                    "effective_date": effective_date,
                    "symbol": symbol,
                    "weight": weight if bool(positive.loc[symbol]) else 0.0,
                }
            )

    return pd.DataFrame(rows)
