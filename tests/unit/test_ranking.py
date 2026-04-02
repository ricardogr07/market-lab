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


def _predictions(
    symbols: list[str],
    scores: list[float],
    *,
    fold_ids: list[int] | None = None,
    signal_dates: list[pd.Timestamp] | None = None,
    effective_dates: list[pd.Timestamp] | None = None,
) -> pd.DataFrame:
    row_count = len(symbols)
    return pd.DataFrame(
        {
            "model_name": ["logistic_regression"] * row_count,
            "fold_id": fold_ids or ([0] * row_count),
            "signal_date": signal_dates or ([pd.Timestamp("2024-01-05")] * row_count),
            "effective_date": effective_dates or ([pd.Timestamp("2024-01-08")] * row_count),
            "symbol": symbols,
            "score": scores,
        }
    )


def _rebalance(weights: pd.DataFrame, effective_date: str) -> pd.DataFrame:
    return weights.loc[weights["effective_date"] == pd.Timestamp(effective_date)].reset_index(drop=True)


def test_ranking_generates_default_market_neutral_weights_with_symbol_tiebreaks() -> None:
    panel = _panel(["AAA", "BBB", "CCC", "DDD"])
    predictions = _predictions(["AAA", "BBB", "CCC", "DDD"], [0.9, 0.9, 0.1, 0.2])

    weights = generate_weights(
        predictions=predictions,
        panel=panel,
        long_n=2,
        short_n=2,
    )

    rebalance = _rebalance(weights, "2024-01-08")
    assert rebalance["strategy"].nunique() == 1
    assert rebalance["strategy"].iat[0] == "ml_logistic_regression"
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


def test_ranking_generates_long_only_weights() -> None:
    panel = _panel(["AAA", "BBB", "CCC", "DDD"])
    predictions = _predictions(["AAA", "BBB", "CCC", "DDD"], [0.95, 0.85, 0.3, 0.2])

    weights = generate_weights(
        predictions=predictions,
        panel=panel,
        long_n=2,
        short_n=2,
        mode="long_only",
    )

    rebalance = _rebalance(weights, "2024-01-08")
    assert rebalance["strategy"].iat[0] == "ml_logistic_regression__long_only"
    assert rebalance.loc[rebalance["symbol"].isin(["AAA", "BBB"]), "weight"].tolist() == [
        pytest.approx(0.5),
        pytest.approx(0.5),
    ]
    assert rebalance.loc[rebalance["symbol"].isin(["CCC", "DDD"]), "weight"].tolist() == [
        pytest.approx(0.0),
        pytest.approx(0.0),
    ]
    assert rebalance["weight"].sum() == pytest.approx(1.0)


def test_ranking_applies_symmetric_threshold_gating_in_long_short_mode() -> None:
    panel = _panel(["AAA", "BBB", "CCC", "DDD"])
    predictions = _predictions(["AAA", "BBB", "CCC", "DDD"], [0.95, 0.85, 0.15, 0.05])

    weights = generate_weights(
        predictions=predictions,
        panel=panel,
        long_n=1,
        short_n=1,
        mode="long_short",
        min_score_threshold=0.9,
    )

    rebalance = _rebalance(weights, "2024-01-08")
    assert rebalance["strategy"].iat[0] == "ml_logistic_regression__long_short__thr0p90"
    assert rebalance.loc[rebalance["symbol"] == "AAA", "weight"].iat[0] == pytest.approx(0.5)
    assert rebalance.loc[rebalance["symbol"] == "DDD", "weight"].iat[0] == pytest.approx(-0.5)
    assert rebalance.loc[rebalance["symbol"].isin(["BBB", "CCC"]), "weight"].tolist() == [
        pytest.approx(0.0),
        pytest.approx(0.0),
    ]


def test_ranking_preserves_partial_book_as_cash_when_enabled() -> None:
    panel = _panel(["AAA", "BBB", "CCC", "DDD"])
    predictions = _predictions(["AAA", "BBB", "CCC", "DDD"], [0.95, 0.85, 0.45, 0.35])

    weights = generate_weights(
        predictions=predictions,
        panel=panel,
        long_n=2,
        short_n=2,
        mode="long_short",
        min_score_threshold=0.8,
        cash_when_underfilled=True,
    )

    rebalance = _rebalance(weights, "2024-01-08")
    assert rebalance["strategy"].iat[0] == "ml_logistic_regression__long_short__thr0p80__cash"
    assert rebalance.loc[rebalance["symbol"].isin(["AAA", "BBB"]), "weight"].tolist() == [
        pytest.approx(0.25),
        pytest.approx(0.25),
    ]
    assert rebalance.loc[rebalance["symbol"].isin(["CCC", "DDD"]), "weight"].tolist() == [
        pytest.approx(0.0),
        pytest.approx(0.0),
    ]
    assert rebalance["weight"].sum() == pytest.approx(0.5)


def test_ranking_falls_back_to_zero_weights_when_underfilled_cash_is_disabled() -> None:
    panel = _panel(["AAA", "BBB", "CCC", "DDD"])
    predictions = _predictions(["AAA", "BBB", "CCC", "DDD"], [0.95, 0.85, 0.45, 0.35])

    weights = generate_weights(
        predictions=predictions,
        panel=panel,
        long_n=2,
        short_n=2,
        mode="long_short",
        min_score_threshold=0.8,
        cash_when_underfilled=False,
    )

    rebalance = _rebalance(weights, "2024-01-08")
    assert rebalance["weight"].eq(0.0).all()


def test_ranking_adds_zero_boundary_rows_at_next_rebalance_effective_date() -> None:
    panel = _panel(["AAA", "BBB", "CCC", "DDD"])
    predictions = _predictions(
        ["AAA", "BBB", "CCC", "DDD"] * 2,
        [0.9, 0.8, 0.2, 0.1] * 2,
        signal_dates=[pd.Timestamp("2024-01-05")] * 4 + [pd.Timestamp("2024-01-12")] * 4,
        effective_dates=[pd.Timestamp("2024-01-08")] * 4 + [pd.Timestamp("2024-01-15")] * 4,
    )

    weights = generate_weights(
        predictions=predictions,
        panel=panel,
        long_n=2,
        short_n=2,
    )

    assert pd.Timestamp("2024-01-16") not in set(weights["effective_date"])
    boundary = _rebalance(weights, "2024-01-22")
    assert len(boundary) == 4
    assert boundary["weight"].eq(0.0).all()


def test_ranking_skips_boundary_zero_rows_when_next_fold_starts_there() -> None:
    panel = _panel(["AAA", "BBB", "CCC", "DDD"])
    predictions = _predictions(
        ["AAA", "BBB", "CCC", "DDD"] * 3,
        [0.9, 0.8, 0.2, 0.1] * 3,
        fold_ids=[0] * 8 + [1] * 4,
        signal_dates=[pd.Timestamp("2024-01-05")] * 4
        + [pd.Timestamp("2024-01-12")] * 4
        + [pd.Timestamp("2024-01-19")] * 4,
        effective_dates=[pd.Timestamp("2024-01-08")] * 4
        + [pd.Timestamp("2024-01-15")] * 4
        + [pd.Timestamp("2024-01-22")] * 4,
    )

    weights = generate_weights(
        predictions=predictions,
        panel=panel,
        long_n=2,
        short_n=2,
    )

    assert not weights.duplicated(subset=["effective_date", "symbol"]).any()
    jan_22 = _rebalance(weights, "2024-01-22")
    assert not jan_22.empty
    assert not jan_22["weight"].eq(0.0).all()
