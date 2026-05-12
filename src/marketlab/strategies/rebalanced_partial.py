from __future__ import annotations

import math

import pandas as pd

from marketlab.rebalance import signal_effective_dates
from marketlab.strategies.ranking import WEIGHTS_COLUMNS


def strategy_name_for_weight(target_weight: float) -> str:
    percentage = float(target_weight) * 100.0
    if math.isclose(percentage, round(percentage)):
        suffix = str(int(round(percentage)))
    else:
        suffix = f"{percentage:g}".replace(".", "p")
    return f"btc_rebalanced_{suffix}"


def generate_weights(
    panel: pd.DataFrame,
    *,
    target_weight: float,
    frequency: str = "W-FRI",
    strategy_name: str | None = None,
) -> pd.DataFrame:
    if panel.empty:
        return pd.DataFrame(columns=WEIGHTS_COLUMNS)
    if not math.isfinite(target_weight) or target_weight <= 0.0 or target_weight >= 1.0:
        raise ValueError(
            "Rebalanced partial allocation target_weight must be between 0.0 and 1.0."
        )

    symbols = sorted(panel["symbol"].drop_duplicates().tolist())
    if len(symbols) != 1:
        raise ValueError("Rebalanced partial allocation is scoped to one BTC symbol.")

    effective_dates = pd.Index(pd.to_datetime(signal_effective_dates(panel, frequency).dropna()))
    if effective_dates.empty:
        return pd.DataFrame(columns=WEIGHTS_COLUMNS)

    resolved_strategy_name = strategy_name or strategy_name_for_weight(target_weight)
    return pd.DataFrame(
        {
            "strategy": [resolved_strategy_name] * len(effective_dates),
            "effective_date": effective_dates,
            "symbol": [symbols[0]] * len(effective_dates),
            "weight": [float(target_weight)] * len(effective_dates),
        },
        columns=WEIGHTS_COLUMNS,
    ).sort_values(["effective_date", "symbol"]).reset_index(drop=True)
