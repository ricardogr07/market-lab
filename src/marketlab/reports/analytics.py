from __future__ import annotations

import math

import pandas as pd

from marketlab.backtest.engine import WEIGHT_EPSILON
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
    "avg_long_exposure",
    "avg_short_exposure",
    "avg_gross_exposure",
    "avg_net_exposure",
    "avg_cash_weight",
    "avg_engine_cash_weight",
    "avg_active_positions",
    "max_position_weight",
    "max_group_weight",
    "benchmark_strategy",
    "excess_cumulative_return",
    "annualized_excess_return",
    "tracking_error",
    "information_ratio",
    "correlation_to_benchmark",
    "up_capture",
    "down_capture",
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

COST_SENSITIVITY_COLUMNS = [
    "strategy",
    "bps_per_trade",
    "gross_cumulative_return",
    "cumulative_return",
    "cost_drag",
    "final_equity",
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

DAILY_EXPOSURE_COLUMNS = [
    "date",
    "strategy",
    "long_exposure",
    "short_exposure",
    "gross_exposure",
    "net_exposure",
    "cash_weight",
    "engine_cash_weight",
    "active_positions",
    "max_position_weight",
]

GROUP_EXPOSURE_COLUMNS = [
    "date",
    "strategy",
    "group_name",
    "long_exposure",
    "short_exposure",
    "gross_exposure",
    "net_exposure",
]

BENCHMARK_RELATIVE_COLUMNS = [
    "date",
    "strategy",
    "benchmark_strategy",
    "strategy_net_return",
    "benchmark_net_return",
    "excess_return",
    "strategy_equity",
    "benchmark_equity",
    "relative_equity",
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


def _normalized_daily_holdings(daily_holdings: pd.DataFrame) -> pd.DataFrame:
    _require_columns(
        daily_holdings,
        {"date", "strategy", "symbol", "weight"},
        "Daily holdings frame",
    )
    if daily_holdings.empty:
        return daily_holdings.copy()

    return (
        daily_holdings.copy()
        .assign(date=lambda frame: pd.to_datetime(frame["date"]))
        .sort_values(["strategy", "date", "symbol"])
        .reset_index(drop=True)
    )


def _normalized_daily_cash(daily_cash: pd.DataFrame) -> pd.DataFrame:
    _require_columns(
        daily_cash,
        {"date", "strategy", "engine_cash_weight"},
        "Daily cash frame",
    )
    if daily_cash.empty:
        return daily_cash.copy()

    return (
        daily_cash.copy()
        .assign(date=lambda frame: pd.to_datetime(frame["date"]))
        .sort_values(["strategy", "date"])
        .reset_index(drop=True)
    )


def _normalized_daily_exposure(daily_exposure: pd.DataFrame) -> pd.DataFrame:
    _require_columns(
        daily_exposure,
        set(DAILY_EXPOSURE_COLUMNS),
        "Daily exposure frame",
    )
    if daily_exposure.empty:
        return daily_exposure.copy()

    return (
        daily_exposure.copy()
        .assign(date=lambda frame: pd.to_datetime(frame["date"]))
        .sort_values(["strategy", "date"])
        .reset_index(drop=True)
    )


def _normalized_group_exposure(group_exposure: pd.DataFrame) -> pd.DataFrame:
    _require_columns(
        group_exposure,
        set(GROUP_EXPOSURE_COLUMNS),
        "Group exposure frame",
    )
    if group_exposure.empty:
        return group_exposure.copy()

    return (
        group_exposure.copy()
        .assign(date=lambda frame: pd.to_datetime(frame["date"]))
        .sort_values(["strategy", "date", "group_name"])
        .reset_index(drop=True)
    )


def _normalized_benchmark_relative(benchmark_relative: pd.DataFrame) -> pd.DataFrame:
    _require_columns(
        benchmark_relative,
        set(BENCHMARK_RELATIVE_COLUMNS),
        "Benchmark-relative frame",
    )
    if benchmark_relative.empty:
        return benchmark_relative.copy()

    return (
        benchmark_relative.copy()
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


def _cost_sensitivity_grid(
    base_cost_bps: float,
    sensitivity_bps: list[float] | None,
) -> list[float]:
    grid = {0.0, float(base_cost_bps)}
    if sensitivity_bps is not None:
        grid.update(float(value) for value in sensitivity_bps)
    return sorted(grid)


def _repriced_performance(working: pd.DataFrame, bps_per_trade: float) -> pd.DataFrame:
    repriced = working.loc[:, ["date", "strategy", "gross_return", "turnover"]].copy()
    repriced["net_return"] = repriced["gross_return"] - (repriced["turnover"] * (bps_per_trade / 10_000.0))
    repriced["equity"] = (
        repriced.groupby("strategy", sort=False)["net_return"]
        .transform(lambda values: (1.0 + values).cumprod())
        .astype(float)
    )
    return repriced.loc[:, ["date", "strategy", "gross_return", "net_return", "turnover", "equity"]]


def _compute_repriced_strategy_metrics(repriced: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, float | str]] = []
    for strategy, frame in repriced.groupby("strategy", sort=False):
        ordered = frame.sort_values("date").reset_index(drop=True)
        returns = ordered["net_return"]
        equity = ordered["equity"]
        periods = len(ordered)
        final_equity = float(equity.iloc[-1]) if periods else 1.0

        annualized_return = float("nan")
        if periods and final_equity > 0.0:
            annualized_return = float((final_equity ** (252.0 / periods)) - 1.0)

        annualized_volatility = float(returns.std(ddof=0) * math.sqrt(252.0))
        if annualized_volatility > 0.0 and math.isfinite(annualized_return):
            sharpe_like = annualized_return / annualized_volatility
        elif math.isfinite(annualized_return):
            sharpe_like = 0.0
        else:
            sharpe_like = float("nan")

        drawdown = (equity / equity.cummax()) - 1.0
        max_drawdown = float(drawdown.min()) if not drawdown.empty else 0.0

        rows.append(
            {
                "strategy": strategy,
                "cumulative_return": (final_equity - 1.0) if periods else 0.0,
                "annualized_return": annualized_return,
                "annualized_volatility": annualized_volatility,
                "sharpe_like": float(sharpe_like),
                "max_drawdown": max_drawdown,
                "hit_rate": float((returns > 0.0).mean()) if periods else 0.0,
                "avg_turnover": float(ordered["turnover"].mean()) if periods else 0.0,
                "total_turnover": float(ordered["turnover"].sum()) if periods else 0.0,
            }
        )

    return pd.DataFrame(rows)

def build_cost_sensitivity(
    performance: pd.DataFrame,
    base_cost_bps: float,
    sensitivity_bps: list[float] | None = None,
) -> pd.DataFrame:
    working = _normalized_performance(performance)
    if working.empty:
        return pd.DataFrame(columns=COST_SENSITIVITY_COLUMNS)

    gross_summary = (
        working.groupby("strategy", sort=False)
        .apply(
            lambda frame: pd.Series(
                {
                    "gross_cumulative_return": float((1.0 + frame["gross_return"]).cumprod().iloc[-1] - 1.0),
                }
            ),
            include_groups=False,
        )
        .reset_index()
        .sort_values("strategy")
        .reset_index(drop=True)
    )

    scenario_frames: list[pd.DataFrame] = []
    for bps_per_trade in _cost_sensitivity_grid(base_cost_bps, sensitivity_bps):
        repriced = _repriced_performance(working, bps_per_trade)
        metrics = _compute_repriced_strategy_metrics(repriced)
        final_equity = (
            repriced.groupby("strategy", as_index=False)
            .agg(final_equity=("equity", "last"))
            .sort_values("strategy")
            .reset_index(drop=True)
        )
        cost_summary = (
            build_turnover_costs(repriced)
            .groupby("strategy", as_index=False)
            .agg(
                avg_cost_return=("cost_return", "mean"),
                total_cost_return=("cost_return", "sum"),
            )
            .sort_values("strategy")
            .reset_index(drop=True)
        )

        scenario_summary = (
            metrics.merge(gross_summary, on="strategy", how="inner", validate="one_to_one")
            .merge(final_equity, on="strategy", how="inner", validate="one_to_one")
            .merge(cost_summary, on="strategy", how="inner", validate="one_to_one")
            .sort_values("strategy")
            .reset_index(drop=True)
        )
        scenario_summary["bps_per_trade"] = float(bps_per_trade)
        scenario_summary["cost_drag"] = (
            scenario_summary["gross_cumulative_return"] - scenario_summary["cumulative_return"]
        )
        scenario_frames.append(scenario_summary.loc[:, COST_SENSITIVITY_COLUMNS])

    return (
        pd.concat(scenario_frames, ignore_index=True)
        .sort_values(["strategy", "bps_per_trade"])
        .reset_index(drop=True)
    )


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


def build_daily_exposure(
    daily_holdings: pd.DataFrame,
    daily_cash: pd.DataFrame,
) -> pd.DataFrame:
    holdings = _normalized_daily_holdings(daily_holdings)
    cash = _normalized_daily_cash(daily_cash)
    if holdings.empty or cash.empty:
        return pd.DataFrame(columns=DAILY_EXPOSURE_COLUMNS)

    working = holdings.assign(
        long_component=lambda frame: frame["weight"].clip(lower=0.0),
        short_component=lambda frame: (-frame["weight"].clip(upper=0.0)),
        abs_weight=lambda frame: frame["weight"].abs(),
        active_flag=lambda frame: frame["weight"].abs().gt(WEIGHT_EPSILON).astype(int),
    )

    exposure = (
        working.groupby(["date", "strategy"], as_index=False)
        .agg(
            long_exposure=("long_component", "sum"),
            short_exposure=("short_component", "sum"),
            active_positions=("active_flag", "sum"),
            max_position_weight=("abs_weight", "max"),
        )
        .sort_values(["strategy", "date"])
        .reset_index(drop=True)
    )
    exposure["gross_exposure"] = exposure["long_exposure"] + exposure["short_exposure"]
    exposure["net_exposure"] = exposure["long_exposure"] - exposure["short_exposure"]
    exposure["cash_weight"] = (1.0 - exposure["gross_exposure"]).clip(lower=0.0)

    merged = exposure.merge(
        cash.loc[:, ["date", "strategy", "engine_cash_weight"]],
        on=["date", "strategy"],
        how="inner",
        validate="one_to_one",
    )
    return merged.loc[:, DAILY_EXPOSURE_COLUMNS]


def build_group_exposure(
    daily_holdings: pd.DataFrame,
    symbol_groups: dict[str, str] | None,
) -> pd.DataFrame:
    if not symbol_groups:
        return pd.DataFrame(columns=GROUP_EXPOSURE_COLUMNS)

    holdings = _normalized_daily_holdings(daily_holdings)
    if holdings.empty:
        return pd.DataFrame(columns=GROUP_EXPOSURE_COLUMNS)

    covered_symbols = set(symbol_groups)
    active_symbols = set(holdings["symbol"].unique())
    if not active_symbols.issubset(covered_symbols):
        return pd.DataFrame(columns=GROUP_EXPOSURE_COLUMNS)

    working = holdings.assign(
        group_name=lambda frame: frame["symbol"].map(symbol_groups),
        long_component=lambda frame: frame["weight"].clip(lower=0.0),
        short_component=lambda frame: (-frame["weight"].clip(upper=0.0)),
    )
    grouped = (
        working.groupby(["date", "strategy", "group_name"], as_index=False)
        .agg(
            long_exposure=("long_component", "sum"),
            short_exposure=("short_component", "sum"),
        )
        .sort_values(["strategy", "date", "group_name"])
        .reset_index(drop=True)
    )
    grouped["gross_exposure"] = grouped["long_exposure"] + grouped["short_exposure"]
    grouped["net_exposure"] = grouped["long_exposure"] - grouped["short_exposure"]
    return grouped.loc[:, GROUP_EXPOSURE_COLUMNS]


def build_benchmark_relative(
    performance: pd.DataFrame,
    benchmark_strategy: str,
) -> pd.DataFrame:
    working = _normalized_performance(performance)
    if working.empty or benchmark_strategy == "":
        return pd.DataFrame(columns=BENCHMARK_RELATIVE_COLUMNS)

    strategies = working["strategy"].drop_duplicates().tolist()
    if benchmark_strategy not in set(strategies):
        available = ", ".join(sorted(strategies))
        raise ValueError(
            "evaluation.benchmark_strategy='"
            f"{benchmark_strategy}' is not present in run strategies. Available strategies: {available}"
        )

    benchmark_frame = (
        working.loc[working["strategy"] == benchmark_strategy, ["date", "net_return", "equity"]]
        .rename(
            columns={
                "net_return": "benchmark_net_return",
                "equity": "benchmark_equity",
            }
        )
        .sort_values("date")
        .reset_index(drop=True)
    )

    benchmark_relative = (
        working.loc[:, ["date", "strategy", "net_return", "equity"]]
        .rename(
            columns={
                "net_return": "strategy_net_return",
                "equity": "strategy_equity",
            }
        )
        .merge(
            benchmark_frame,
            on="date",
            how="inner",
            validate="many_to_one",
        )
        .sort_values(["strategy", "date"])
        .reset_index(drop=True)
    )
    benchmark_relative["benchmark_strategy"] = benchmark_strategy
    benchmark_relative["excess_return"] = (
        benchmark_relative["strategy_net_return"] - benchmark_relative["benchmark_net_return"]
    )
    benchmark_relative["relative_equity"] = (
        benchmark_relative["strategy_equity"] / benchmark_relative["benchmark_equity"]
    )
    return benchmark_relative.loc[:, BENCHMARK_RELATIVE_COLUMNS]


def _capture_ratio(frame: pd.DataFrame, *, direction: str) -> float:
    if direction == "up":
        subset = frame.loc[frame["benchmark_net_return"] > 0.0]
    else:
        subset = frame.loc[frame["benchmark_net_return"] < 0.0]

    if subset.empty:
        return float("nan")

    benchmark_return = float((1.0 + subset["benchmark_net_return"]).prod() - 1.0)
    if abs(benchmark_return) <= WEIGHT_EPSILON:
        return float("nan")

    strategy_return = float((1.0 + subset["strategy_net_return"]).prod() - 1.0)
    return strategy_return / benchmark_return


def build_strategy_summary(
    performance: pd.DataFrame,
    daily_exposure: pd.DataFrame | None = None,
    group_exposure: pd.DataFrame | None = None,
    benchmark_relative: pd.DataFrame | None = None,
    benchmark_strategy: str = "",
) -> pd.DataFrame:
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

    if daily_exposure is None:
        exposure_summary = pd.DataFrame(
            {
                "strategy": summary["strategy"],
                "avg_long_exposure": float("nan"),
                "avg_short_exposure": float("nan"),
                "avg_gross_exposure": float("nan"),
                "avg_net_exposure": float("nan"),
                "avg_cash_weight": float("nan"),
                "avg_engine_cash_weight": float("nan"),
                "avg_active_positions": float("nan"),
                "max_position_weight": float("nan"),
            }
        )
    else:
        normalized_daily_exposure = _normalized_daily_exposure(daily_exposure)
        exposure_summary = (
            normalized_daily_exposure.groupby("strategy", as_index=False)
            .agg(
                avg_long_exposure=("long_exposure", "mean"),
                avg_short_exposure=("short_exposure", "mean"),
                avg_gross_exposure=("gross_exposure", "mean"),
                avg_net_exposure=("net_exposure", "mean"),
                avg_cash_weight=("cash_weight", "mean"),
                avg_engine_cash_weight=("engine_cash_weight", "mean"),
                avg_active_positions=("active_positions", "mean"),
                max_position_weight=("max_position_weight", "max"),
            )
            .sort_values("strategy")
            .reset_index(drop=True)
        )

    if group_exposure is None or group_exposure.empty:
        group_summary = pd.DataFrame(
            {
                "strategy": summary["strategy"],
                "max_group_weight": float("nan"),
            }
        )
    else:
        normalized_group_exposure = _normalized_group_exposure(group_exposure).copy()
        normalized_group_exposure["same_side_group_weight"] = normalized_group_exposure[
            ["long_exposure", "short_exposure"]
        ].max(axis=1)
        group_summary = (
            normalized_group_exposure.groupby("strategy", as_index=False)
            .agg(max_group_weight=("same_side_group_weight", "max"))
            .sort_values("strategy")
            .reset_index(drop=True)
        )

    if benchmark_strategy == "" or benchmark_relative is None or benchmark_relative.empty:
        benchmark_summary = pd.DataFrame(
            {
                "strategy": summary["strategy"],
                "benchmark_strategy": "",
                "excess_cumulative_return": float("nan"),
                "annualized_excess_return": float("nan"),
                "tracking_error": float("nan"),
                "information_ratio": float("nan"),
                "correlation_to_benchmark": float("nan"),
                "up_capture": float("nan"),
                "down_capture": float("nan"),
            }
        )
    else:
        normalized_benchmark_relative = _normalized_benchmark_relative(benchmark_relative)

        def _benchmark_summary_row(frame: pd.DataFrame) -> pd.Series:
            frame = frame.sort_values("date").reset_index(drop=True)
            trading_days = len(frame)
            relative_equity = float(frame["relative_equity"].iloc[-1]) if trading_days else float("nan")
            tracking_error = float(frame["excess_return"].std(ddof=0) * math.sqrt(252.0))
            information_ratio = float("nan")
            if tracking_error > WEIGHT_EPSILON:
                information_ratio = float(frame["excess_return"].mean() * 252.0 / tracking_error)

            return pd.Series(
                {
                    "benchmark_strategy": str(frame["benchmark_strategy"].iat[0]),
                    "excess_cumulative_return": relative_equity - 1.0,
                    "annualized_excess_return": (
                        float((relative_equity ** (252.0 / trading_days)) - 1.0)
                        if trading_days and relative_equity > 0.0
                        else float("nan")
                    ),
                    "tracking_error": tracking_error,
                    "information_ratio": information_ratio,
                    "correlation_to_benchmark": float(
                        frame["strategy_net_return"].corr(frame["benchmark_net_return"])
                    ),
                    "up_capture": _capture_ratio(frame, direction="up"),
                    "down_capture": _capture_ratio(frame, direction="down"),
                }
            )

        benchmark_summary = (
            normalized_benchmark_relative.groupby("strategy", sort=False)
            .apply(_benchmark_summary_row, include_groups=False)
            .reset_index()
            .sort_values("strategy")
            .reset_index(drop=True)
        )

    summary = (
        summary.merge(exposure_summary, on="strategy", how="left", validate="one_to_one")
        .merge(group_summary, on="strategy", how="left", validate="one_to_one")
        .merge(benchmark_summary, on="strategy", how="left", validate="one_to_one")
        .sort_values("strategy")
        .reset_index(drop=True)
    )
    return summary.loc[:, STRATEGY_SUMMARY_COLUMNS]

