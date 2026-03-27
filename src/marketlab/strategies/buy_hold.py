from __future__ import annotations

import pandas as pd


def generate_weights(panel: pd.DataFrame, strategy_name: str = "buy_hold") -> pd.DataFrame:
    symbols = sorted(panel["symbol"].unique())
    first_date = panel["timestamp"].min()
    weight = 1.0 / len(symbols)
    return pd.DataFrame(
        {
            "strategy": strategy_name,
            "effective_date": [first_date] * len(symbols),
            "symbol": symbols,
            "weight": [weight] * len(symbols),
        }
    )
