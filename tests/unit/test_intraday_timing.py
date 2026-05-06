from __future__ import annotations

import pandas as pd
import pytest

from marketlab.rebalance import next_effective_dates, rebalance_signal_dates
from marketlab.targets.timing import add_forward_targets, build_rebalance_snapshots


def _hourly_panel() -> pd.DataFrame:
    timestamps = pd.date_range("2024-01-01 00:00:00", periods=5, freq="h")
    rows = []
    for index, timestamp in enumerate(timestamps):
        price = 100.0 + index
        rows.append(
            {
                "symbol": "BTC-USD",
                "timestamp": timestamp,
                "open": price,
                "high": price + 1.0,
                "low": price - 1.0,
                "close": price + 0.5,
                "volume": 1000 + index,
                "adj_close": price + 0.5,
                "adj_factor": 1.0,
                "adj_open": price,
                "adj_high": price + 1.0,
                "adj_low": price - 1.0,
                "feature_x": float(index),
            }
        )
    return pd.DataFrame(rows)


def test_bar_rebalance_signals_every_completed_bar_and_executes_next_bar_open() -> None:
    panel = _hourly_panel()

    signal_dates = rebalance_signal_dates(panel, "bar")
    effective_dates = next_effective_dates(panel, signal_dates)

    assert signal_dates == list(panel["timestamp"])
    assert effective_dates.index.tolist() == list(panel["timestamp"].iloc[:-1])
    assert effective_dates.tolist() == list(panel["timestamp"].iloc[1:])


def test_intraday_forward_targets_use_next_bar_open_without_lookahead() -> None:
    panel = _hourly_panel()
    snapshots = build_rebalance_snapshots(
        panel,
        feature_columns=["feature_x"],
        frequency="bar",
    )

    dataset = add_forward_targets(
        snapshots,
        panel,
        horizon_days=1,
        target_type="direction",
    )

    first = dataset.iloc[0]
    assert first["signal_date"] == pd.Timestamp("2024-01-01 00:00:00")
    assert first["effective_date"] == pd.Timestamp("2024-01-01 01:00:00")
    assert first["target_end_date"] == pd.Timestamp("2024-01-01 01:00:00")
    assert first["forward_return"] == pytest.approx((101.5 / 101.0) - 1.0)
