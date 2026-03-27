from __future__ import annotations

import pandas as pd


def run_backtest(
    panel: pd.DataFrame,
    weights: pd.DataFrame,
    cost_bps: float,
) -> pd.DataFrame:
    required_weight_columns = {"strategy", "effective_date", "symbol", "weight"}
    missing = required_weight_columns - set(weights.columns)
    if missing:
        joined = ", ".join(sorted(missing))
        raise ValueError(f"Weights frame is missing required columns: {joined}")

    strategy_names = weights["strategy"].drop_duplicates().tolist()
    if len(strategy_names) != 1:
        raise ValueError("Backtest engine expects weights for exactly one strategy.")
    strategy_name = strategy_names[0]

    working_panel = panel.sort_values(["timestamp", "symbol"]).copy()
    unique_dates = pd.Index(sorted(working_panel["timestamp"].drop_duplicates()))
    symbols = sorted(working_panel["symbol"].unique())

    adj_close = working_panel.pivot(index="timestamp", columns="symbol", values="adj_close")
    adj_open = working_panel.pivot(index="timestamp", columns="symbol", values="adj_open")

    overnight_returns = (adj_open / adj_close.shift(1)) - 1.0
    intraday_returns = (adj_close / adj_open) - 1.0

    weights_pivot = (
        weights.copy()
        .assign(effective_date=lambda frame: pd.to_datetime(frame["effective_date"]))
        .pivot(index="effective_date", columns="symbol", values="weight")
        .reindex(columns=symbols, fill_value=0.0)
    )
    weights_pivot = weights_pivot.reindex(unique_dates).ffill().fillna(0.0)

    post_open_weights = weights_pivot
    pre_open_weights = post_open_weights.shift(1).fillna(0.0)
    turnover = (post_open_weights - pre_open_weights).abs().sum(axis=1)

    gross_returns = (
        (pre_open_weights * overnight_returns.fillna(0.0)).sum(axis=1)
        + (post_open_weights * intraday_returns.fillna(0.0)).sum(axis=1)
    )
    costs = turnover * (cost_bps / 10_000.0)
    net_returns = gross_returns - costs
    equity = (1.0 + net_returns).cumprod()

    return pd.DataFrame(
        {
            "date": unique_dates,
            "strategy": strategy_name,
            "gross_return": gross_returns.values,
            "net_return": net_returns.values,
            "turnover": turnover.values,
            "equity": equity.values,
        }
    )
