from __future__ import annotations

import pandas as pd
import pytest

from marketlab.strategies.indicator_stack import (
    DIAGNOSTIC_COLUMNS,
    build_indicator_frame,
    generate_diagnostics,
    generate_weights,
)


def _panel() -> pd.DataFrame:
    closes = [100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 103.0, 101.0, 99.0, 100.0, 102.0, 104.0]
    rows = []
    for index, close in enumerate(closes):
        timestamp = pd.Timestamp("2024-01-01 00:00:00") + pd.Timedelta(hours=index)
        rows.append(
            {
                "symbol": "BTC-USD",
                "timestamp": timestamp,
                "open": close - 0.5,
                "high": close + 1.0,
                "low": close - 1.0,
                "close": close,
                "volume": 1000 + (index * 100),
                "adj_close": close,
                "adj_factor": 1.0,
                "adj_open": close - 0.5,
                "adj_high": close + 1.0,
                "adj_low": close - 1.0,
            }
        )
    return pd.DataFrame(rows)


def test_build_indicator_frame_derives_explicit_confirmations() -> None:
    indicators = build_indicator_frame(
        _panel(),
        ema_fast_window=2,
        ema_slow_window=4,
        rsi_window=3,
        rsi_min=40.0,
        rsi_max=100.0,
        macd_fast_window=2,
        macd_slow_window=5,
        macd_signal_window=2,
        bollinger_window=3,
        bollinger_std=0.5,
        bollinger_mode="breakout",
        volume_window=2,
        volume_multiplier=1.0,
        vwap_window=3,
        use_vwap=True,
    )

    late_row = indicators.iloc[5]
    assert bool(late_row["ema_confirmed"]) is True
    assert bool(late_row["rsi_confirmed"]) is True
    assert bool(late_row["macd_confirmed"]) is True
    assert bool(late_row["bollinger_confirmed"]) is True
    assert bool(late_row["volume_confirmed"]) is True
    assert bool(late_row["vwap_confirmed"]) is True
    assert late_row["confirmation_count"] == 6


def test_generate_weights_emits_long_cash_rows_on_next_bar() -> None:
    weights = generate_weights(
        _panel(),
        frequency="bar",
        ema_fast_window=2,
        ema_slow_window=4,
        rsi_window=3,
        rsi_min=40.0,
        rsi_max=100.0,
        macd_fast_window=2,
        macd_slow_window=5,
        macd_signal_window=2,
        bollinger_window=3,
        bollinger_std=0.5,
        bollinger_mode="breakout",
        volume_window=2,
        volume_multiplier=1.0,
        vwap_window=3,
        use_vwap=True,
        min_confirmations=3,
    )

    assert list(weights.columns) == ["strategy", "effective_date", "symbol", "weight"]
    assert len(weights) == 11
    assert weights["effective_date"].min() == pd.Timestamp("2024-01-01 01:00:00")
    assert weights["effective_date"].max() == pd.Timestamp("2024-01-01 11:00:00")
    assert weights["weight"].max() == pytest.approx(1.0)
    assert set(weights["weight"]) <= {0.0, 1.0}


def test_generate_diagnostics_aligns_one_to_one_with_weights() -> None:
    kwargs = {
        "frequency": "bar",
        "ema_fast_window": 2,
        "ema_slow_window": 4,
        "rsi_window": 3,
        "rsi_min": 40.0,
        "rsi_max": 100.0,
        "macd_fast_window": 2,
        "macd_slow_window": 5,
        "macd_signal_window": 2,
        "bollinger_window": 3,
        "bollinger_std": 0.5,
        "bollinger_mode": "breakout",
        "volume_window": 2,
        "volume_multiplier": 1.0,
        "vwap_window": 3,
        "use_vwap": True,
        "min_confirmations": 3,
    }

    diagnostics = generate_diagnostics(_panel(), **kwargs)
    weights = generate_weights(_panel(), **kwargs)

    assert list(diagnostics.columns) == DIAGNOSTIC_COLUMNS
    assert len(diagnostics) == len(weights)
    assert diagnostics["timestamp"].min() == pd.Timestamp("2024-01-01 00:00:00")
    assert diagnostics["effective_date"].min() == pd.Timestamp("2024-01-01 01:00:00")
    assert diagnostics["target_weight"].tolist() == pytest.approx(weights["weight"].tolist())
    assert diagnostics["target_weight"].max() == pytest.approx(1.0)
