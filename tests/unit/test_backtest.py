from __future__ import annotations

import pandas as pd
import pytest

from marketlab.backtest.engine import run_backtest, run_backtest_detailed


def _rotation_panel() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "symbol": ["AAA", "BBB", "AAA", "BBB"],
            "timestamp": pd.to_datetime(
                ["2024-01-02", "2024-01-02", "2024-01-03", "2024-01-03"]
            ),
            "open": [100.0, 100.0, 110.0, 100.0],
            "high": [110.0, 100.0, 110.0, 110.0],
            "low": [100.0, 100.0, 110.0, 100.0],
            "close": [110.0, 100.0, 110.0, 110.0],
            "volume": [1000, 1000, 1000, 1000],
            "adj_close": [110.0, 100.0, 110.0, 110.0],
            "adj_factor": [1.0, 1.0, 1.0, 1.0],
            "adj_open": [100.0, 100.0, 110.0, 100.0],
            "adj_high": [110.0, 100.0, 110.0, 110.0],
            "adj_low": [100.0, 100.0, 110.0, 100.0],
        }
    ).sort_values(["symbol", "timestamp"]).reset_index(drop=True)


def _rotation_weights() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "strategy": ["rotation"] * 4,
            "effective_date": pd.to_datetime(
                ["2024-01-02", "2024-01-02", "2024-01-03", "2024-01-03"]
            ),
            "symbol": ["AAA", "BBB", "AAA", "BBB"],
            "weight": [1.0, 0.0, 0.0, 1.0],
        }
    )


def test_backtest_applies_turnover_costs_on_rebalance() -> None:
    performance = run_backtest(_rotation_panel(), _rotation_weights(), cost_bps=10.0)

    assert performance.loc[0, "gross_return"] == pytest.approx(0.10)
    assert performance.loc[0, "net_return"] == pytest.approx(0.099)
    assert performance.loc[1, "gross_return"] == pytest.approx(0.10)
    assert performance.loc[1, "turnover"] == pytest.approx(2.0)
    assert performance.loc[1, "net_return"] == pytest.approx(0.098)


def test_backtest_detailed_captures_end_of_day_drifted_holdings_and_cash() -> None:
    result = run_backtest_detailed(_rotation_panel(), _rotation_weights(), cost_bps=10.0)

    assert list(result.performance.columns) == [
        "date",
        "strategy",
        "gross_return",
        "net_return",
        "turnover",
        "equity",
    ]
    assert list(result.daily_holdings.columns) == ["date", "strategy", "symbol", "weight"]
    assert list(result.daily_cash.columns) == ["date", "strategy", "engine_cash_weight"]

    second_day_holdings = result.daily_holdings.loc[
        result.daily_holdings["date"] == pd.Timestamp("2024-01-03")
    ].set_index("symbol")
    assert second_day_holdings.loc["AAA", "weight"] == pytest.approx(0.0)
    assert second_day_holdings.loc["BBB", "weight"] == pytest.approx(1.0)

    second_day_cash = result.daily_cash.loc[
        result.daily_cash["date"] == pd.Timestamp("2024-01-03"), "engine_cash_weight"
    ].iat[0]
    assert second_day_cash == pytest.approx(0.0)
