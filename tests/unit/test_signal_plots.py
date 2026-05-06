from __future__ import annotations

from pathlib import Path

import pandas as pd

from marketlab.reports.plots import (
    plot_pattern_detection_windows,
    plot_signal_confirmations,
    plot_signal_performance_focus,
    plot_signal_price_overlay,
)


def _diagnostics() -> pd.DataFrame:
    timestamps = pd.date_range("2024-01-01 00:00:00", periods=6, freq="15min")
    return pd.DataFrame(
        {
            "strategy": ["indicator_stack"] * len(timestamps),
            "timestamp": timestamps,
            "effective_date": timestamps + pd.Timedelta(minutes=15),
            "symbol": ["BTC-USD"] * len(timestamps),
            "close": [100.0, 101.0, 102.0, 101.5, 103.0, 104.0],
            "ema_fast": [100.0, 100.5, 101.2, 101.4, 102.1, 103.0],
            "ema_slow": [100.0, 100.3, 100.8, 101.0, 101.5, 102.2],
            "rsi": [50.0, 55.0, 62.0, 58.0, 65.0, 68.0],
            "macd": [0.0, 0.1, 0.2, 0.1, 0.3, 0.4],
            "macd_signal": [0.0, 0.05, 0.1, 0.12, 0.2, 0.3],
            "bollinger_mid": [100.0, 100.5, 101.0, 101.3, 102.0, 103.0],
            "bollinger_upper": [101.0, 101.5, 102.0, 102.3, 103.0, 104.0],
            "bollinger_lower": [99.0, 99.5, 100.0, 100.3, 101.0, 102.0],
            "volume_average": [1000, 1050, 1100, 1150, 1200, 1250],
            "vwap": [100.0, 100.4, 100.9, 101.1, 101.8, 102.6],
            "ema_confirmed": [False, True, True, True, True, True],
            "rsi_confirmed": [True, True, True, True, True, True],
            "macd_confirmed": [False, True, True, False, True, True],
            "bollinger_confirmed": [False, False, True, False, True, True],
            "volume_confirmed": [False, True, True, False, True, True],
            "vwap_confirmed": [True, True, True, True, True, True],
            "confirmation_count": [2, 5, 6, 3, 6, 6],
            "target_weight": [0.0, 1.0, 1.0, 0.0, 1.0, 1.0],
        }
    )


def _performance() -> pd.DataFrame:
    timestamps = pd.date_range("2024-01-01 00:15:00", periods=5, freq="15min")
    rows = []
    for strategy, returns in {
        "buy_hold": [0.01, 0.005, -0.003, 0.006, 0.004],
        "indicator_stack": [0.0, 0.005, -0.001, 0.006, 0.002],
    }.items():
        equity = 1.0
        for timestamp, net_return in zip(timestamps, returns):
            equity *= 1.0 + net_return
            rows.append(
                {
                    "date": timestamp,
                    "strategy": strategy,
                    "gross_return": net_return,
                    "net_return": net_return,
                    "turnover": 0.0,
                    "equity": equity,
                }
            )
    return pd.DataFrame(rows)


def _assert_non_empty_png(path: Path) -> None:
    assert path.exists()
    assert path.stat().st_size > 0


def test_signal_plot_artifacts_are_created(tmp_path: Path) -> None:
    price_path = plot_signal_price_overlay(_diagnostics(), tmp_path / "price.png")
    confirmations_path = plot_signal_confirmations(
        _diagnostics(),
        tmp_path / "confirmations.png",
    )
    performance_path = plot_signal_performance_focus(
        _performance(),
        tmp_path / "performance.png",
    )

    _assert_non_empty_png(price_path)
    _assert_non_empty_png(confirmations_path)
    _assert_non_empty_png(performance_path)


def test_pattern_detection_window_plot_is_created(tmp_path: Path) -> None:
    diagnostics = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01 00:00:00", periods=10, freq="15min"),
            "symbol": ["BTC-USD"] * 10,
            "close": [100, 101, 102, 101, 103, 104, 103, 105, 106, 107],
            "target_weight": [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0],
            "ascending_triangle_breakout": [False, False, False, False, True, False, False, False, False, False],
            "symmetrical_triangle_breakout": [False] * 10,
            "bullish_rectangle_breakout": [False] * 10,
            "inverse_head_and_shoulders_breakout": [False] * 10,
            "double_bottom_breakout": [False, False, False, False, False, False, False, True, False, False],
            "triple_bottom_breakout": [False] * 10,
            "falling_wedge_breakout": [False] * 10,
            "bull_flag_breakout": [False] * 10,
            "pennant_breakout": [False] * 10,
            "cup_and_handle_breakout": [False] * 10,
            "ascending_channel_continuation": [False] * 10,
            "megaphone_breakout": [False] * 10,
            "descending_triangle_breakdown": [False] * 10,
            "bearish_rectangle_breakdown": [False] * 10,
            "head_and_shoulders_breakdown": [False] * 10,
            "double_top_breakdown": [False] * 10,
            "triple_top_breakdown": [False] * 10,
            "rising_wedge_breakdown": [False] * 10,
            "bear_flag_breakdown": [False] * 10,
            "descending_channel_breakdown": [False] * 10,
            "diamond_breakdown": [False] * 10,
        }
    )

    output_path = plot_pattern_detection_windows(
        diagnostics,
        tmp_path / "pattern_windows.png",
    )

    _assert_non_empty_png(output_path)
