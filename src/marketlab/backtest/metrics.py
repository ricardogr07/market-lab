from __future__ import annotations

import math

import pandas as pd


def compute_strategy_metrics(performance: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, float | str]] = []
    for strategy, frame in performance.groupby("strategy", sort=False):
        ordered = frame.sort_values("date").reset_index(drop=True)
        returns = ordered["net_return"]
        equity = ordered["equity"]
        periods = len(ordered)

        cumulative_return = float(equity.iloc[-1] - 1.0) if periods else 0.0
        annualized_return = (
            float((equity.iloc[-1] ** (252.0 / periods)) - 1.0) if periods else 0.0
        )
        annualized_volatility = float(returns.std(ddof=0) * math.sqrt(252.0))
        sharpe_like = (
            annualized_return / annualized_volatility if annualized_volatility else 0.0
        )
        drawdown = (equity / equity.cummax()) - 1.0
        max_drawdown = float(drawdown.min()) if not drawdown.empty else 0.0
        hit_rate = float((returns > 0.0).mean()) if periods else 0.0
        avg_turnover = float(ordered["turnover"].mean()) if periods else 0.0
        total_turnover = float(ordered["turnover"].sum()) if periods else 0.0

        rows.append(
            {
                "strategy": strategy,
                "cumulative_return": cumulative_return,
                "annualized_return": annualized_return,
                "annualized_volatility": annualized_volatility,
                "sharpe_like": sharpe_like,
                "max_drawdown": max_drawdown,
                "hit_rate": hit_rate,
                "avg_turnover": avg_turnover,
                "total_turnover": total_turnover,
            }
        )

    return pd.DataFrame(rows)
