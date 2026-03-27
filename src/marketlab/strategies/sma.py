from __future__ import annotations

import pandas as pd

from marketlab.rebalance import next_effective_dates, weekly_signal_dates


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
    rows: list[dict[str, object]] = []

    effective_dates = next_effective_dates(working, weekly_signal_dates(working))
    for signal_date, effective_date in effective_dates.items():
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
