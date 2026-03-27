from __future__ import annotations

import pandas as pd
import pytest

from marketlab.data.panel import build_market_panel, normalize_ohlcv_frame


def test_normalize_ohlcv_frame_builds_adjusted_columns() -> None:
    raw = pd.DataFrame(
        {
            "Date": ["2024-01-02"],
            "Open": [100.0],
            "High": [101.0],
            "Low": [99.0],
            "Close": [100.0],
            "Adj Close": [50.0],
            "Volume": [1000],
        }
    )

    panel = normalize_ohlcv_frame("VOO", raw)

    assert panel.loc[0, "adj_factor"] == pytest.approx(0.5)
    assert panel.loc[0, "adj_open"] == pytest.approx(50.0)
    assert panel.loc[0, "adj_high"] == pytest.approx(50.5)
    assert panel.loc[0, "adj_low"] == pytest.approx(49.5)


def test_normalize_ohlcv_frame_accepts_multiindex_yfinance_columns() -> None:
    raw = pd.DataFrame(
        [[50.0, 100.0, 101.0, 99.0, 100.0, 1000]],
        index=pd.Index([pd.Timestamp("2024-01-02")], name="Date"),
        columns=pd.MultiIndex.from_tuples(
            [
                ("Adj Close", "VOO"),
                ("Close", "VOO"),
                ("High", "VOO"),
                ("Low", "VOO"),
                ("Open", "VOO"),
                ("Volume", "VOO"),
            ]
        ),
    ).reset_index()

    panel = normalize_ohlcv_frame("VOO", raw)

    assert panel.loc[0, "timestamp"] == pd.Timestamp("2024-01-02")
    assert panel.loc[0, "adj_factor"] == pytest.approx(0.5)


def test_normalize_ohlcv_frame_drops_cached_ticker_header_row() -> None:
    raw = pd.DataFrame(
        {
            "Date": [None, "2024-01-02"],
            "Adj Close": ["VOO", 50.0],
            "Close": ["VOO", 100.0],
            "High": ["VOO", 101.0],
            "Low": ["VOO", 99.0],
            "Open": ["VOO", 100.0],
            "Volume": ["VOO", 1000],
        }
    )

    panel = normalize_ohlcv_frame("VOO", raw)

    assert len(panel) == 1
    assert panel.loc[0, "timestamp"] == pd.Timestamp("2024-01-02")
    assert panel.loc[0, "volume"] == pytest.approx(1000.0)


def test_build_market_panel_rejects_duplicate_symbol_timestamp() -> None:
    raw = pd.DataFrame(
        {
            "Date": ["2024-01-02", "2024-01-02"],
            "Open": [100.0, 100.0],
            "High": [101.0, 101.0],
            "Low": [99.0, 99.0],
            "Close": [100.0, 100.0],
            "Adj Close": [100.0, 100.0],
            "Volume": [1000, 1000],
        }
    )

    with pytest.raises(ValueError, match="duplicate"):
        build_market_panel({"VOO": raw})
