from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from marketlab.backtest.engine import run_backtest
from marketlab.data.panel import load_panel_csv
from marketlab.strategies.allocation import generate_weights as allocation_weights
from marketlab.strategies.buy_hold import generate_weights as buy_hold_weights
from marketlab.strategies.sma import generate_weights as sma_weights


@pytest.fixture()
def panel() -> pd.DataFrame:
    fixture_path = Path(__file__).resolve().parents[1] / "fixtures" / "market_panel.csv"
    return load_panel_csv(fixture_path)


def test_buy_hold_generates_equal_weights(panel: pd.DataFrame) -> None:
    weights = buy_hold_weights(panel)

    assert weights["effective_date"].nunique() == 1
    assert weights["weight"].sum() == pytest.approx(1.0)
    assert weights["weight"].nunique() == 1


def test_allocation_equal_rebalances_from_first_date_and_effective_dates(panel: pd.DataFrame) -> None:
    weights = allocation_weights(panel, frequency="W-FRI", mode="equal")

    unique_dates = sorted(weights["effective_date"].unique())
    assert pd.Timestamp(panel["timestamp"].min()) in unique_dates
    assert len(unique_dates) > 1

    first_rebalance = weights.loc[weights["effective_date"] == unique_dates[0]]
    assert set(first_rebalance["symbol"]) == set(panel["symbol"].unique())
    assert first_rebalance["weight"].sum() == pytest.approx(1.0)
    assert first_rebalance["weight"].nunique() == 1


def test_allocation_group_weights_split_sleeves_equally(panel: pd.DataFrame) -> None:
    weights = allocation_weights(
        panel,
        frequency="W-FRI",
        mode="group_weights",
        symbol_groups={
            "VOO": "broad",
            "QQQ": "growth",
            "SMH": "growth",
            "XLV": "defensive",
            "IEMG": "broad",
        },
        group_weights={
            "broad": 0.40,
            "growth": 0.40,
            "defensive": 0.20,
        },
    )

    first_date = weights["effective_date"].min()
    first_rebalance = weights.loc[weights["effective_date"] == first_date].set_index("symbol")

    assert first_rebalance.loc["VOO", "weight"] == pytest.approx(0.20)
    assert first_rebalance.loc["IEMG", "weight"] == pytest.approx(0.20)
    assert first_rebalance.loc["QQQ", "weight"] == pytest.approx(0.20)
    assert first_rebalance.loc["SMH", "weight"] == pytest.approx(0.20)
    assert first_rebalance.loc["XLV", "weight"] == pytest.approx(0.20)


def test_allocation_equal_is_distinct_from_buy_hold_under_weight_drift() -> None:
    panel = pd.DataFrame(
        {
            "symbol": [
                "AAA",
                "BBB",
                "AAA",
                "BBB",
                "AAA",
                "BBB",
                "AAA",
                "BBB",
            ],
            "timestamp": pd.to_datetime(
                [
                    "2024-01-02",
                    "2024-01-02",
                    "2024-01-05",
                    "2024-01-05",
                    "2024-01-08",
                    "2024-01-08",
                    "2024-01-09",
                    "2024-01-09",
                ]
            ),
            "open": [100.0, 100.0, 100.0, 100.0, 200.0, 100.0, 200.0, 100.0],
            "high": [100.0, 100.0, 200.0, 100.0, 200.0, 100.0, 200.0, 200.0],
            "low": [100.0, 100.0, 100.0, 100.0, 200.0, 100.0, 200.0, 100.0],
            "close": [100.0, 100.0, 200.0, 100.0, 200.0, 100.0, 200.0, 200.0],
            "volume": [1000] * 8,
            "adj_close": [100.0, 100.0, 200.0, 100.0, 200.0, 100.0, 200.0, 200.0],
            "adj_factor": [1.0] * 8,
            "adj_open": [100.0, 100.0, 100.0, 100.0, 200.0, 100.0, 200.0, 100.0],
            "adj_high": [100.0, 100.0, 200.0, 100.0, 200.0, 100.0, 200.0, 200.0],
            "adj_low": [100.0, 100.0, 100.0, 100.0, 200.0, 100.0, 200.0, 100.0],
        }
    ).sort_values(["symbol", "timestamp"]).reset_index(drop=True)

    buy_hold_performance = run_backtest(
        panel,
        buy_hold_weights(panel),
        cost_bps=0.0,
    )
    allocation_performance = run_backtest(
        panel,
        allocation_weights(panel, frequency="W-FRI", mode="equal"),
        cost_bps=0.0,
    )

    assert allocation_performance["turnover"].iloc[2] > 0.0
    assert allocation_performance["equity"].iloc[-1] > buy_hold_performance["equity"].iloc[-1]


def test_sma_uses_week_end_signal_and_next_open_effective_date(panel: pd.DataFrame) -> None:
    weights = sma_weights(panel, fast_window=2, slow_window=3)

    first_effective_date = pd.Timestamp("2024-01-08")
    first_rebalance = weights.loc[weights["effective_date"] == first_effective_date]

    assert not first_rebalance.empty
    assert set(first_rebalance["symbol"]) == set(panel["symbol"].unique())
    assert first_rebalance["weight"].sum() == pytest.approx(1.0)
    assert first_rebalance.loc[first_rebalance["symbol"] == "SMH", "weight"].iat[0] == pytest.approx(0.0)
