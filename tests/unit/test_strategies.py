from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from marketlab.data.panel import load_panel_csv
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


def test_sma_uses_week_end_signal_and_next_open_effective_date(panel: pd.DataFrame) -> None:
    weights = sma_weights(panel, fast_window=2, slow_window=3)

    first_effective_date = pd.Timestamp("2024-01-08")
    first_rebalance = weights.loc[weights["effective_date"] == first_effective_date]

    assert not first_rebalance.empty
    assert set(first_rebalance["symbol"]) == set(panel["symbol"].unique())
    assert first_rebalance["weight"].sum() == pytest.approx(1.0)
    assert first_rebalance.loc[first_rebalance["symbol"] == "SMH", "weight"].iat[0] == pytest.approx(0.0)
