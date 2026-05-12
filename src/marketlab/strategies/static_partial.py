from __future__ import annotations

import math

import pandas as pd

WEIGHTS_COLUMNS = ["strategy", "effective_date", "symbol", "weight"]


def strategy_name_for_weight(target_weight: float) -> str:
    percentage = float(target_weight) * 100.0
    if math.isclose(percentage, round(percentage)):
        suffix = str(int(round(percentage)))
    else:
        suffix = f"{percentage:g}".replace(".", "p")
    return f"btc_static_{suffix}"


def generate_weights(
    panel: pd.DataFrame,
    *,
    target_weight: float,
    strategy_name: str | None = None,
) -> pd.DataFrame:
    if panel.empty:
        return pd.DataFrame(columns=WEIGHTS_COLUMNS)
    if not math.isfinite(target_weight) or target_weight <= 0.0 or target_weight >= 1.0:
        raise ValueError("Static partial allocation target_weight must be between 0.0 and 1.0.")

    symbols = sorted(panel["symbol"].drop_duplicates().tolist())
    if len(symbols) != 1:
        raise ValueError("Static partial allocation is scoped to one BTC symbol.")

    return pd.DataFrame(
        {
            "strategy": [strategy_name or strategy_name_for_weight(target_weight)],
            "effective_date": [pd.Timestamp(panel["timestamp"].min())],
            "symbol": [symbols[0]],
            "weight": [float(target_weight)],
        },
        columns=WEIGHTS_COLUMNS,
    )
