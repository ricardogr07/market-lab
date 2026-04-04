from __future__ import annotations

import pandas as pd

WEIGHT_EPSILON = 1e-12


def _advance_weights(
    asset_weights: pd.Series,
    cash_weight: float,
    asset_returns: pd.Series,
) -> tuple[pd.Series, float, float]:
    evolved_asset_weights = asset_weights * (1.0 + asset_returns.fillna(0.0))
    portfolio_value = float(evolved_asset_weights.sum() + cash_weight)
    if abs(portfolio_value) <= WEIGHT_EPSILON:
        raise ValueError("Portfolio value collapsed to zero; weights are undefined.")

    next_asset_weights = evolved_asset_weights / portfolio_value
    next_cash_weight = cash_weight / portfolio_value
    period_return = portfolio_value - 1.0
    return next_asset_weights, next_cash_weight, period_return


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

    weights_schedule = (
        weights.copy()
        .assign(effective_date=lambda frame: pd.to_datetime(frame["effective_date"]))
        .pivot(index="effective_date", columns="symbol", values="weight")
        .reindex(columns=symbols, fill_value=0.0)
        .sort_index()
    )

    close_weights = pd.Series(0.0, index=symbols, dtype=float)
    close_cash_weight = 1.0
    equity = 1.0
    rows: list[dict[str, object]] = []

    for date in unique_dates:
        date_timestamp = pd.Timestamp(date)
        overnight_row = overnight_returns.loc[date_timestamp].reindex(symbols).fillna(0.0)
        pre_open_weights, cash_open_weight, overnight_return = _advance_weights(
            close_weights,
            close_cash_weight,
            overnight_row,
        )

        if date_timestamp in weights_schedule.index:
            post_open_weights = (
                weights_schedule.loc[date_timestamp]
                .reindex(symbols)
                .fillna(0.0)
                .astype(float)
            )
            turnover = float((post_open_weights - pre_open_weights).abs().sum())
            post_open_cash_weight = 1.0 - float(post_open_weights.sum())
        else:
            post_open_weights = pre_open_weights
            post_open_cash_weight = cash_open_weight
            turnover = 0.0

        intraday_row = intraday_returns.loc[date_timestamp].reindex(symbols).fillna(0.0)
        close_weights, close_cash_weight, intraday_return = _advance_weights(
            post_open_weights,
            post_open_cash_weight,
            intraday_row,
        )

        gross_return = ((1.0 + overnight_return) * (1.0 + intraday_return)) - 1.0
        costs = turnover * (cost_bps / 10_000.0)
        net_return = gross_return - costs
        equity *= 1.0 + net_return

        rows.append(
            {
                "date": date_timestamp,
                "strategy": strategy_name,
                "gross_return": gross_return,
                "net_return": net_return,
                "turnover": turnover,
                "equity": equity,
            }
        )

    return pd.DataFrame(rows)
