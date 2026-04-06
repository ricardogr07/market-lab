from __future__ import annotations

import pandas as pd
import pytest

from marketlab.reports.analytics import (
    BENCHMARK_RELATIVE_COLUMNS,
    DAILY_EXPOSURE_COLUMNS,
    GROUP_EXPOSURE_COLUMNS,
    MONTHLY_RETURNS_COLUMNS,
    STRATEGY_SUMMARY_COLUMNS,
    TURNOVER_COSTS_COLUMNS,
    build_benchmark_relative,
    build_daily_exposure,
    build_group_exposure,
    build_monthly_returns,
    build_strategy_summary,
    build_turnover_costs,
)


@pytest.fixture()
def performance() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": pd.to_datetime(
                [
                    "2024-01-30",
                    "2024-01-31",
                    "2024-02-01",
                    "2024-01-30",
                    "2024-01-31",
                    "2024-02-01",
                ]
            ),
            "strategy": ["alpha", "alpha", "alpha", "beta", "beta", "beta"],
            "gross_return": [0.0100, 0.0200, -0.0100, 0.0050, 0.0100, 0.0200],
            "net_return": [0.0090, 0.0180, -0.0110, 0.0045, 0.0090, 0.0190],
            "turnover": [1.0, 2.0, 1.5, 0.5, 0.75, 1.25],
            "equity": [1.0090, 1.027162, 1.015863218, 1.0045, 1.0135405, 1.0327977695],
        }
    )


@pytest.fixture()
def daily_holdings() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": pd.to_datetime(
                [
                    "2024-01-30",
                    "2024-01-30",
                    "2024-01-31",
                    "2024-01-31",
                    "2024-02-01",
                    "2024-02-01",
                    "2024-01-30",
                    "2024-01-30",
                    "2024-01-31",
                    "2024-01-31",
                    "2024-02-01",
                    "2024-02-01",
                ]
            ),
            "strategy": [
                "alpha",
                "alpha",
                "alpha",
                "alpha",
                "alpha",
                "alpha",
                "beta",
                "beta",
                "beta",
                "beta",
                "beta",
                "beta",
            ],
            "symbol": ["AAA", "BBB", "AAA", "BBB", "AAA", "BBB"] * 2,
            "weight": [0.60, 0.40, 0.70, 0.30, 0.65, 0.35, 0.50, -0.50, 0.60, -0.40, 0.55, -0.45],
        }
    )


@pytest.fixture()
def daily_cash() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": pd.to_datetime(
                [
                    "2024-01-30",
                    "2024-01-31",
                    "2024-02-01",
                    "2024-01-30",
                    "2024-01-31",
                    "2024-02-01",
                ]
            ),
            "strategy": ["alpha", "alpha", "alpha", "beta", "beta", "beta"],
            "engine_cash_weight": [0.0, 0.0, 0.0, 1.0, 0.98, 1.02],
        }
    )


def test_build_turnover_costs_derives_daily_cost_return(performance: pd.DataFrame) -> None:
    turnover_costs = build_turnover_costs(performance)

    assert list(turnover_costs.columns) == TURNOVER_COSTS_COLUMNS
    assert turnover_costs["cost_return"].tolist() == pytest.approx([0.001, 0.002, 0.001, 0.0005, 0.001, 0.001])


def test_build_monthly_returns_compounds_by_strategy_and_month(performance: pd.DataFrame) -> None:
    monthly = build_monthly_returns(performance)

    assert list(monthly.columns) == MONTHLY_RETURNS_COLUMNS
    assert monthly["strategy"].tolist() == ["alpha", "alpha", "beta", "beta"]
    assert monthly["month"].tolist() == ["2024-01", "2024-02", "2024-01", "2024-02"]

    alpha_jan = monthly.loc[(monthly["strategy"] == "alpha") & (monthly["month"] == "2024-01")].iloc[0]
    assert alpha_jan["gross_return"] == pytest.approx((1.01 * 1.02) - 1.0)
    assert alpha_jan["net_return"] == pytest.approx((1.009 * 1.018) - 1.0)


def test_build_daily_exposure_derives_both_cash_semantics(
    daily_holdings: pd.DataFrame,
    daily_cash: pd.DataFrame,
) -> None:
    daily_exposure = build_daily_exposure(daily_holdings, daily_cash)

    assert list(daily_exposure.columns) == DAILY_EXPOSURE_COLUMNS

    alpha_row = daily_exposure.loc[
        (daily_exposure["strategy"] == "alpha") & (daily_exposure["date"] == pd.Timestamp("2024-01-31"))
    ].iloc[0]
    assert alpha_row["long_exposure"] == pytest.approx(1.0)
    assert alpha_row["short_exposure"] == pytest.approx(0.0)
    assert alpha_row["gross_exposure"] == pytest.approx(1.0)
    assert alpha_row["net_exposure"] == pytest.approx(1.0)
    assert alpha_row["cash_weight"] == pytest.approx(0.0)
    assert alpha_row["engine_cash_weight"] == pytest.approx(0.0)
    assert alpha_row["active_positions"] == 2
    assert alpha_row["max_position_weight"] == pytest.approx(0.70)

    beta_row = daily_exposure.loc[
        (daily_exposure["strategy"] == "beta") & (daily_exposure["date"] == pd.Timestamp("2024-01-31"))
    ].iloc[0]
    assert beta_row["long_exposure"] == pytest.approx(0.60)
    assert beta_row["short_exposure"] == pytest.approx(0.40)
    assert beta_row["gross_exposure"] == pytest.approx(1.0)
    assert beta_row["net_exposure"] == pytest.approx(0.20)
    assert beta_row["cash_weight"] == pytest.approx(0.0)
    assert beta_row["engine_cash_weight"] == pytest.approx(0.98)


def test_build_group_exposure_keeps_long_and_short_sleeves_separate(
    daily_holdings: pd.DataFrame,
) -> None:
    group_exposure = build_group_exposure(
        daily_holdings,
        {"AAA": "growth", "BBB": "defensive"},
    )

    assert list(group_exposure.columns) == GROUP_EXPOSURE_COLUMNS

    beta_growth = group_exposure.loc[
        (group_exposure["strategy"] == "beta")
        & (group_exposure["date"] == pd.Timestamp("2024-01-31"))
        & (group_exposure["group_name"] == "growth")
    ].iloc[0]
    beta_defensive = group_exposure.loc[
        (group_exposure["strategy"] == "beta")
        & (group_exposure["date"] == pd.Timestamp("2024-01-31"))
        & (group_exposure["group_name"] == "defensive")
    ].iloc[0]
    assert beta_growth["long_exposure"] == pytest.approx(0.60)
    assert beta_growth["short_exposure"] == pytest.approx(0.0)
    assert beta_growth["gross_exposure"] == pytest.approx(0.60)
    assert beta_growth["net_exposure"] == pytest.approx(0.60)
    assert beta_defensive["long_exposure"] == pytest.approx(0.0)
    assert beta_defensive["short_exposure"] == pytest.approx(0.40)
    assert beta_defensive["gross_exposure"] == pytest.approx(0.40)
    assert beta_defensive["net_exposure"] == pytest.approx(-0.40)


def test_build_strategy_summary_aggregates_one_row_per_strategy(
    performance: pd.DataFrame,
    daily_holdings: pd.DataFrame,
    daily_cash: pd.DataFrame,
) -> None:
    daily_exposure = build_daily_exposure(daily_holdings, daily_cash)
    group_exposure = build_group_exposure(
        daily_holdings,
        {"AAA": "growth", "BBB": "defensive"},
    )
    summary = build_strategy_summary(
        performance,
        daily_exposure=daily_exposure,
        group_exposure=group_exposure,
    )

    assert list(summary.columns) == STRATEGY_SUMMARY_COLUMNS
    assert summary["strategy"].tolist() == ["alpha", "beta"]

    alpha_row = summary.loc[summary["strategy"] == "alpha"].iloc[0]
    assert alpha_row["start_date"] == pd.Timestamp("2024-01-30")
    assert alpha_row["end_date"] == pd.Timestamp("2024-02-01")
    assert alpha_row["trading_days"] == 3
    assert alpha_row["final_equity"] == pytest.approx(1.015863218)
    assert alpha_row["gross_final_equity"] == pytest.approx(1.019898)
    assert alpha_row["gross_cumulative_return"] == pytest.approx(0.019898)
    assert alpha_row["cumulative_return"] == pytest.approx(0.015863218)
    assert alpha_row["cost_drag"] == pytest.approx(0.004034782)
    assert alpha_row["avg_cost_return"] == pytest.approx((0.001 + 0.002 + 0.001) / 3.0)
    assert alpha_row["total_cost_return"] == pytest.approx(0.004)
    assert alpha_row["avg_long_exposure"] == pytest.approx(1.0)
    assert alpha_row["avg_short_exposure"] == pytest.approx(0.0)
    assert alpha_row["avg_gross_exposure"] == pytest.approx(1.0)
    assert alpha_row["avg_net_exposure"] == pytest.approx(1.0)
    assert alpha_row["avg_cash_weight"] == pytest.approx(0.0)
    assert alpha_row["avg_engine_cash_weight"] == pytest.approx(0.0)
    assert alpha_row["avg_active_positions"] == pytest.approx(2.0)
    assert alpha_row["max_position_weight"] == pytest.approx(0.70)
    assert alpha_row["max_group_weight"] == pytest.approx(0.70)

    beta_row = summary.loc[summary["strategy"] == "beta"].iloc[0]
    assert beta_row["avg_long_exposure"] == pytest.approx((0.50 + 0.60 + 0.55) / 3.0)
    assert beta_row["avg_short_exposure"] == pytest.approx((0.50 + 0.40 + 0.45) / 3.0)
    assert beta_row["avg_gross_exposure"] == pytest.approx(1.0)
    assert beta_row["avg_cash_weight"] == pytest.approx(0.0)
    assert beta_row["avg_engine_cash_weight"] == pytest.approx((1.0 + 0.98 + 1.02) / 3.0)
    assert beta_row["max_position_weight"] == pytest.approx(0.60)
    assert beta_row["max_group_weight"] == pytest.approx(0.60)


def test_build_benchmark_relative_requires_present_benchmark(performance: pd.DataFrame) -> None:
    with pytest.raises(
        ValueError,
        match="evaluation.benchmark_strategy='missing' is not present in run strategies",
    ):
        build_benchmark_relative(performance, "missing")


def test_build_benchmark_relative_aligns_daily_paths(performance: pd.DataFrame) -> None:
    benchmark_relative = build_benchmark_relative(performance, "beta")

    assert list(benchmark_relative.columns) == BENCHMARK_RELATIVE_COLUMNS
    assert set(benchmark_relative["strategy"]) == {"alpha", "beta"}
    assert benchmark_relative["benchmark_strategy"].eq("beta").all()

    alpha_last = benchmark_relative.loc[
        benchmark_relative["strategy"] == "alpha"
    ].sort_values("date").iloc[-1]
    assert alpha_last["excess_return"] == pytest.approx(-0.03)
    assert alpha_last["relative_equity"] == pytest.approx(
        1.015863218 / 1.0327977695
    )

    beta_rows = benchmark_relative.loc[
        benchmark_relative["strategy"] == "beta"
    ].sort_values("date")
    assert beta_rows["excess_return"].tolist() == pytest.approx([0.0, 0.0, 0.0])
    assert beta_rows["relative_equity"].tolist() == pytest.approx([1.0, 1.0, 1.0])


def test_build_strategy_summary_appends_benchmark_relative_metrics(
    performance: pd.DataFrame,
    daily_holdings: pd.DataFrame,
    daily_cash: pd.DataFrame,
) -> None:
    daily_exposure = build_daily_exposure(daily_holdings, daily_cash)
    group_exposure = build_group_exposure(
        daily_holdings,
        {"AAA": "growth", "BBB": "defensive"},
    )
    benchmark_relative = build_benchmark_relative(performance, "beta")
    summary = build_strategy_summary(
        performance,
        daily_exposure=daily_exposure,
        group_exposure=group_exposure,
        benchmark_relative=benchmark_relative,
        benchmark_strategy="beta",
    )

    assert list(summary.columns) == STRATEGY_SUMMARY_COLUMNS

    alpha_rows = performance.loc[performance["strategy"] == "alpha"].sort_values("date")
    beta_rows = performance.loc[performance["strategy"] == "beta"].sort_values("date")
    alpha_excess = alpha_rows["net_return"].reset_index(drop=True) - beta_rows["net_return"].reset_index(drop=True)
    relative_equity = alpha_rows["equity"].iloc[-1] / beta_rows["equity"].iloc[-1]
    tracking_error = float(alpha_excess.std(ddof=0) * (252.0 ** 0.5))
    information_ratio = float(alpha_excess.mean() * 252.0 / tracking_error)
    up_capture = float(
        (((1.0 + alpha_rows["net_return"]).prod()) - 1.0)
        / (((1.0 + beta_rows["net_return"]).prod()) - 1.0)
    )

    alpha_row = summary.loc[summary["strategy"] == "alpha"].iloc[0]
    assert alpha_row["benchmark_strategy"] == "beta"
    assert alpha_row["excess_cumulative_return"] == pytest.approx(relative_equity - 1.0)
    assert alpha_row["annualized_excess_return"] == pytest.approx(
        (relative_equity ** (252.0 / 3.0)) - 1.0
    )
    assert alpha_row["tracking_error"] == pytest.approx(tracking_error)
    assert alpha_row["information_ratio"] == pytest.approx(information_ratio)
    assert alpha_row["correlation_to_benchmark"] == pytest.approx(
        alpha_rows["net_return"].reset_index(drop=True).corr(
            beta_rows["net_return"].reset_index(drop=True)
        )
    )
    assert alpha_row["up_capture"] == pytest.approx(up_capture)
    assert pd.isna(alpha_row["down_capture"])

    beta_row = summary.loc[summary["strategy"] == "beta"].iloc[0]
    assert beta_row["benchmark_strategy"] == "beta"
    assert beta_row["excess_cumulative_return"] == pytest.approx(0.0)
    assert beta_row["tracking_error"] == pytest.approx(0.0)
    assert pd.isna(beta_row["information_ratio"])
    assert beta_row["correlation_to_benchmark"] == pytest.approx(1.0)
    assert beta_row["up_capture"] == pytest.approx(1.0)
    assert pd.isna(beta_row["down_capture"])

