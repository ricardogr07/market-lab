from __future__ import annotations

import pandas as pd

from marketlab.backtest.metrics import compute_strategy_metrics

STRATEGY_SUMMARY_COLUMNS = [
    "strategy",
    "start_date",
    "end_date",
    "trading_days",
    "final_equity",
    "gross_final_equity",
    "gross_cumulative_return",
    "cumulative_return",
    "cost_drag",
    "annualized_return",
    "annualized_volatility",
    "sharpe_like",
    "max_drawdown",
    "hit_rate",
    "avg_turnover",
    "total_turnover",
    "avg_cost_return",
    "total_cost_return",
]

MONTHLY_RETURNS_COLUMNS = [
    "strategy",
    "month",
    "gross_return",
    "net_return",
]

TURNOVER_COSTS_COLUMNS = [
    "date",
    "strategy",
    "turnover",
    "gross_return",
    "net_return",
    "cost_return",
]


def _require_columns(frame: pd.DataFrame, required: set[str], label: str) -> None:
    missing = required - set(frame.columns)
    if missing:
        joined = ", ".join(sorted(missing))
        raise ValueError(f"{label} is missing required columns: {joined}")


def _normalized_performance(performance: pd.DataFrame) -> pd.DataFrame:
    _require_columns(
        performance,
        {"date", "strategy", "gross_return", "net_return", "turnover", "equity"},
        "Performance frame",
    )
    if performance.empty:
        return performance.copy()

    return (
        performance.copy()
        .assign(date=lambda frame: pd.to_datetime(frame["date"]))
        .sort_values(["strategy", "date"])
        .reset_index(drop=True)
    )


def build_turnover_costs(performance: pd.DataFrame) -> pd.DataFrame:
    working = _normalized_performance(performance)
    if working.empty:
        return pd.DataFrame(columns=TURNOVER_COSTS_COLUMNS)

    working["cost_return"] = working["gross_return"] - working["net_return"]
    return working.loc[:, TURNOVER_COSTS_COLUMNS]


def build_monthly_returns(performance: pd.DataFrame) -> pd.DataFrame:
    working = _normalized_performance(performance)
    if working.empty:
        return pd.DataFrame(columns=MONTHLY_RETURNS_COLUMNS)

    monthly = (
        working.assign(month=lambda frame: frame["date"].dt.to_period("M").astype(str))
        .groupby(["strategy", "month"], as_index=False)
        .agg(
            gross_return=("gross_return", lambda values: float((1.0 + values).prod() - 1.0)),
            net_return=("net_return", lambda values: float((1.0 + values).prod() - 1.0)),
        )
        .sort_values(["strategy", "month"])
        .reset_index(drop=True)
    )
    return monthly.loc[:, MONTHLY_RETURNS_COLUMNS]


def build_strategy_summary(performance: pd.DataFrame) -> pd.DataFrame:
    working = _normalized_performance(performance)
    if working.empty:
        return pd.DataFrame(columns=STRATEGY_SUMMARY_COLUMNS)

    metrics = compute_strategy_metrics(working)
    cost_frame = build_turnover_costs(working)

    date_summary = (
        working.groupby("strategy", as_index=False)
        .agg(
            start_date=("date", "min"),
            end_date=("date", "max"),
            trading_days=("date", "size"),
            final_equity=("equity", "last"),
        )
        .sort_values("strategy")
        .reset_index(drop=True)
    )
    gross_summary = (
        working.groupby("strategy", sort=False)
        .apply(
            lambda frame: pd.Series(
                {
                    "gross_final_equity": float((1.0 + frame["gross_return"]).cumprod().iloc[-1]),
                    "gross_cumulative_return": float((1.0 + frame["gross_return"]).cumprod().iloc[-1] - 1.0),
                }
            ),
            include_groups=False,
        )
        .reset_index()
    )
    cost_summary = (
        cost_frame.groupby("strategy", as_index=False)
        .agg(
            avg_cost_return=("cost_return", "mean"),
            total_cost_return=("cost_return", "sum"),
        )
        .sort_values("strategy")
        .reset_index(drop=True)
    )

    summary = (
        date_summary.merge(metrics, on="strategy", how="inner", validate="one_to_one")
        .merge(gross_summary, on="strategy", how="inner", validate="one_to_one")
        .merge(cost_summary, on="strategy", how="inner", validate="one_to_one")
        .sort_values("strategy")
        .reset_index(drop=True)
    )
    summary["cost_drag"] = summary["gross_cumulative_return"] - summary["cumulative_return"]
    return summary.loc[:, STRATEGY_SUMMARY_COLUMNS]
