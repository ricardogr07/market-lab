from __future__ import annotations

import pandas as pd
import pytest

from marketlab.reports.analytics import (
    MONTHLY_RETURNS_COLUMNS,
    STRATEGY_SUMMARY_COLUMNS,
    TURNOVER_COSTS_COLUMNS,
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


def test_build_strategy_summary_aggregates_one_row_per_strategy(performance: pd.DataFrame) -> None:
    summary = build_strategy_summary(performance)

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
