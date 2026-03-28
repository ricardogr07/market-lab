from __future__ import annotations

import pandas as pd
import pytest

from marketlab.strategies.ranking import generate_weights


def _panel(symbols: list[str]) -> pd.DataFrame:
    dates = pd.bdate_range("2024-01-01", "2024-01-31")
    rows = [
        {"symbol": symbol, "timestamp": timestamp}
        for symbol in symbols
        for timestamp in dates
    ]
    return pd.DataFrame(rows)


def test_ranking_generates_market_neutral_weights_with_symbol_tiebreaks() -> None:
    panel = _panel(["AAA", "BBB", "CCC", "DDD"])
    predictions = pd.DataFrame(
        {
            "model_name": ["logistic_regression"] * 4,
            "fold_id": [0] * 4,
            "signal_date": [pd.Timestamp("2024-01-05")] * 4,
            "effective_date": [pd.Timestamp("2024-01-08")] * 4,
            "symbol": ["AAA", "BBB", "CCC", "DDD"],
            "score": [0.9, 0.9, 0.1, 0.2],
        }
    )

    weights = generate_weights(
        predictions=predictions,
        panel=panel,
        long_n=2,
        short_n=2,
    )

    rebalance = weights.loc[weights["effective_date"] == pd.Timestamp("2024-01-08")]
    assert rebalance["symbol"].tolist() == ["AAA", "BBB", "CCC", "DDD"]
    assert rebalance.loc[rebalance["symbol"].isin(["AAA", "BBB"]), "weight"].tolist() == [
        pytest.approx(0.25),
        pytest.approx(0.25),
    ]
    assert rebalance.loc[rebalance["symbol"].isin(["CCC", "DDD"]), "weight"].tolist() == [
        pytest.approx(-0.25),
        pytest.approx(-0.25),
    ]
    assert rebalance["weight"].sum() == pytest.approx(0.0)


def test_ranking_falls_back_to_zero_weights_when_sides_cannot_be_filled() -> None:
    panel = _panel(["AAA", "BBB", "CCC"])
    predictions = pd.DataFrame(
        {
            "model_name": ["logistic_regression"] * 3,
            "fold_id": [0] * 3,
            "signal_date": [pd.Timestamp("2024-01-05")] * 3,
            "effective_date": [pd.Timestamp("2024-01-08")] * 3,
            "symbol": ["AAA", "BBB", "CCC"],
            "score": [0.9, 0.8, 0.1],
        }
    )

    weights = generate_weights(
        predictions=predictions,
        panel=panel,
        long_n=2,
        short_n=2,
    )

    assert weights["weight"].eq(0.0).all()


def test_ranking_adds_zero_boundary_rows_at_next_rebalance_effective_date() -> None:
    panel = _panel(["AAA", "BBB", "CCC", "DDD"])
    predictions = pd.DataFrame(
        {
            "model_name": ["logistic_regression"] * 8,
            "fold_id": [0] * 8,
            "signal_date": [pd.Timestamp("2024-01-05")] * 4
            + [pd.Timestamp("2024-01-12")] * 4,
            "effective_date": [pd.Timestamp("2024-01-08")] * 4
            + [pd.Timestamp("2024-01-15")] * 4,
            "symbol": ["AAA", "BBB", "CCC", "DDD"] * 2,
            "score": [0.9, 0.8, 0.2, 0.1] * 2,
        }
    )

    weights = generate_weights(
        predictions=predictions,
        panel=panel,
        long_n=2,
        short_n=2,
    )

    assert pd.Timestamp("2024-01-16") not in set(weights["effective_date"])
    boundary = weights.loc[weights["effective_date"] == pd.Timestamp("2024-01-22")]
    assert len(boundary) == 4
    assert boundary["weight"].eq(0.0).all()



def test_ranking_skips_boundary_zero_rows_when_next_fold_starts_there() -> None:
    panel = _panel(["AAA", "BBB", "CCC", "DDD"])
    predictions = pd.DataFrame(
        {
            "model_name": ["logistic_regression"] * 12,
            "fold_id": [0] * 8 + [1] * 4,
            "signal_date": [pd.Timestamp("2024-01-05")] * 4
            + [pd.Timestamp("2024-01-12")] * 4
            + [pd.Timestamp("2024-01-19")] * 4,
            "effective_date": [pd.Timestamp("2024-01-08")] * 4
            + [pd.Timestamp("2024-01-15")] * 4
            + [pd.Timestamp("2024-01-22")] * 4,
            "symbol": ["AAA", "BBB", "CCC", "DDD"] * 3,
            "score": [0.9, 0.8, 0.2, 0.1] * 3,
        }
    )

    weights = generate_weights(
        predictions=predictions,
        panel=panel,
        long_n=2,
        short_n=2,
    )

    assert not weights.duplicated(subset=["effective_date", "symbol"]).any()
    jan_22 = weights.loc[weights["effective_date"] == pd.Timestamp("2024-01-22")]
    assert not jan_22.empty
    assert not jan_22["weight"].eq(0.0).all()
