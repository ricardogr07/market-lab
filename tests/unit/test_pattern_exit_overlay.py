from __future__ import annotations

import pandas as pd
import pytest

from marketlab.strategies.chart_patterns import (
    BEARISH_PATTERN_COLUMNS,
    BULLISH_PATTERN_COLUMNS,
    PATTERN_COLUMNS,
)
from marketlab.strategies.pattern_exit_overlay import (
    generate_diagnostics,
    generate_weights,
)


def _panel(closes: list[float]) -> pd.DataFrame:
    rows = []
    for index, close in enumerate(closes):
        timestamp = pd.Timestamp("2024-01-01 00:00:00") + pd.Timedelta(hours=index)
        rows.append(
            {
                "symbol": "BTC-USD",
                "timestamp": timestamp,
                "open": close,
                "high": close + 1,
                "low": close - 1,
                "close": close,
                "volume": 1000.0,
                "adj_close": close,
                "adj_factor": 1.0,
                "adj_open": close,
                "adj_high": close + 1,
                "adj_low": close - 1,
            }
        )
    return pd.DataFrame(rows)


def _pattern_diagnostics(
    closes: list[float],
    *,
    bearish_indices: set[int] | None = None,
    bullish_indices: set[int] | None = None,
) -> pd.DataFrame:
    bearish = bearish_indices or {1}
    bullish = bullish_indices or {3}
    rows = []
    for index, close in enumerate(closes[:-1]):
        timestamp = pd.Timestamp("2024-01-01 00:00:00") + pd.Timedelta(hours=index)
        effective_date = timestamp + pd.Timedelta(hours=1)
        patterns = {column: False for column in PATTERN_COLUMNS}
        if index in bearish:
            patterns["double_top_breakdown"] = True
        if index in bullish:
            patterns["double_bottom_breakout"] = True
        rows.append(
            {
                "strategy": "chart_patterns",
                "timestamp": timestamp,
                "effective_date": effective_date,
                "symbol": "BTC-USD",
                "close": close,
                "support_level": close - 5,
                "resistance_level": close + 5,
                "volume_average": 1000.0,
                "volume_confirmed": True,
                **patterns,
                "bullish_pattern_count": sum(patterns[column] for column in BULLISH_PATTERN_COLUMNS),
                "bearish_pattern_count": sum(patterns[column] for column in BEARISH_PATTERN_COLUMNS),
                "target_weight": 0.0,
            }
        )
    return pd.DataFrame(rows)


def test_exit_overlay_starts_long_exits_on_bearish_and_reenters_after_clear() -> None:
    panel = _panel([100.0, 99.0, 97.0, 101.0, 103.0])
    diagnostics = generate_diagnostics(
        panel,
        _pattern_diagnostics([100.0, 99.0, 97.0, 101.0, 103.0]),
        min_bearish_patterns=1,
        min_bullish_reentry_patterns=1,
        trend_ema_window=2,
        reentry_clear_bars=1,
    )

    assert diagnostics["target_weight"].tolist() == pytest.approx([1.0, 0.0, 0.0, 1.0])
    assert bool(diagnostics.loc[1, "bearish_exit_candidate"]) is True
    assert bool(diagnostics.loc[3, "bullish_reentry_candidate"]) is True
    assert "exit_blocked_by_trend" in diagnostics.columns
    assert "bars_since_exit" in diagnostics.columns


def test_exit_overlay_weights_preserve_next_bar_effective_dates() -> None:
    diagnostics = generate_diagnostics(
        _panel([100.0, 99.0, 97.0, 101.0, 103.0]),
        _pattern_diagnostics([100.0, 99.0, 97.0, 101.0, 103.0]),
        min_bearish_patterns=1,
        min_bullish_reentry_patterns=1,
        trend_ema_window=2,
        reentry_clear_bars=1,
    )

    weights = generate_weights(diagnostics)

    assert list(weights.columns) == ["strategy", "effective_date", "symbol", "weight"]
    assert weights["effective_date"].tolist() == diagnostics["effective_date"].tolist()
    assert weights["weight"].tolist() == pytest.approx(diagnostics["target_weight"].tolist())


def test_exit_overlay_blocks_exit_when_price_is_above_trend() -> None:
    diagnostics = generate_diagnostics(
        _panel([100.0, 110.0, 111.0, 112.0]),
        _pattern_diagnostics([100.0, 110.0, 111.0, 112.0], bullish_indices=set()),
        min_bearish_patterns=1,
        min_bullish_reentry_patterns=1,
        trend_ema_window=3,
        reentry_clear_bars=1,
        require_price_below_trend_for_exit=True,
    )

    assert diagnostics["target_weight"].tolist() == pytest.approx([1.0, 1.0, 1.0])
    assert bool(diagnostics.loc[1, "exit_blocked_by_trend"]) is True


def test_exit_overlay_requires_bearish_confirmation_window() -> None:
    diagnostics = generate_diagnostics(
        _panel([100.0, 98.0, 96.0, 94.0, 95.0]),
        _pattern_diagnostics(
            [100.0, 98.0, 96.0, 94.0, 95.0],
            bearish_indices={1, 2},
            bullish_indices=set(),
        ),
        min_bearish_patterns=1,
        min_bullish_reentry_patterns=1,
        trend_ema_window=2,
        reentry_clear_bars=1,
        bearish_confirmation_window_bars=2,
    )

    assert diagnostics["target_weight"].tolist() == pytest.approx([1.0, 1.0, 0.0, 0.0])
    assert bool(diagnostics.loc[1, "exit_blocked_by_confirmation"]) is True
    assert bool(diagnostics.loc[2, "exit_blocked_by_confirmation"]) is False


def test_exit_overlay_min_cash_bars_blocks_immediate_reentry() -> None:
    diagnostics = generate_diagnostics(
        _panel([100.0, 98.0, 100.0, 102.0, 104.0]),
        _pattern_diagnostics(
            [100.0, 98.0, 100.0, 102.0, 104.0],
            bearish_indices={1},
            bullish_indices={2, 3},
        ),
        min_bearish_patterns=1,
        min_bullish_reentry_patterns=1,
        trend_ema_window=2,
        reentry_clear_bars=1,
        min_cash_bars=2,
    )

    assert diagnostics["target_weight"].tolist() == pytest.approx([1.0, 0.0, 0.0, 1.0])
    assert bool(diagnostics.loc[2, "reentry_blocked_by_min_cash"]) is True


def test_exit_overlay_cooldown_blocks_rapid_repeated_exit() -> None:
    diagnostics = generate_diagnostics(
        _panel([100.0, 98.0, 100.0, 97.0, 96.0, 95.0]),
        _pattern_diagnostics(
            [100.0, 98.0, 100.0, 97.0, 96.0, 95.0],
            bearish_indices={1, 3},
            bullish_indices={2},
        ),
        min_bearish_patterns=1,
        min_bullish_reentry_patterns=1,
        trend_ema_window=2,
        reentry_clear_bars=1,
        exit_cooldown_bars=3,
    )

    assert diagnostics["target_weight"].tolist() == pytest.approx([1.0, 0.0, 1.0, 1.0, 1.0])
    assert bool(diagnostics.loc[3, "exit_blocked_by_cooldown"]) is True
