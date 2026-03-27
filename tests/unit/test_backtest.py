from __future__ import annotations

import pandas as pd
import pytest

from marketlab.backtest.engine import run_backtest


def test_backtest_applies_turnover_costs_on_rebalance() -> None:
    panel = pd.DataFrame(
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

    weights = pd.DataFrame(
        {
            "strategy": ["rotation"] * 4,
            "effective_date": pd.to_datetime(
                ["2024-01-02", "2024-01-02", "2024-01-03", "2024-01-03"]
            ),
            "symbol": ["AAA", "BBB", "AAA", "BBB"],
            "weight": [1.0, 0.0, 0.0, 1.0],
        }
    )

    performance = run_backtest(panel, weights, cost_bps=10.0)

    assert performance.loc[0, "gross_return"] == pytest.approx(0.10)
    assert performance.loc[0, "net_return"] == pytest.approx(0.099)
    assert performance.loc[1, "gross_return"] == pytest.approx(0.10)
    assert performance.loc[1, "turnover"] == pytest.approx(2.0)
    assert performance.loc[1, "net_return"] == pytest.approx(0.098)
