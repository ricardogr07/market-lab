from __future__ import annotations

import pandas as pd
import pytest

from marketlab.strategies.chart_patterns import (
    DIAGNOSTIC_COLUMNS,
    build_pattern_frame,
    generate_diagnostics,
    generate_weights,
)
from marketlab.strategies.pattern_catalog import PATTERN_CATALOG


def _panel(
    closes: list[float],
    volumes: list[float] | None = None,
    highs: list[float] | None = None,
    lows: list[float] | None = None,
) -> pd.DataFrame:
    resolved_volumes = volumes or [1000.0] * len(closes)
    resolved_highs = highs or [close + 0.5 for close in closes]
    resolved_lows = lows or [close - 0.5 for close in closes]
    rows = []
    for index, close in enumerate(closes):
        timestamp = pd.Timestamp("2024-01-01 00:00:00") + pd.Timedelta(minutes=15 * index)
        rows.append(
            {
                "symbol": "BTC-USD",
                "timestamp": timestamp,
                "open": close - 0.2,
                "high": resolved_highs[index],
                "low": resolved_lows[index],
                "close": close,
                "volume": resolved_volumes[index],
                "adj_close": close,
                "adj_factor": 1.0,
                "adj_open": close - 0.2,
                "adj_high": resolved_highs[index],
                "adj_low": resolved_lows[index],
            }
        )
    return pd.DataFrame(rows)


def test_build_pattern_frame_detects_bull_flag_breakout_without_lookahead() -> None:
    panel = _panel(
        [100.0, 104.0, 108.0, 110.0, 109.0, 108.5, 111.0, 112.0],
        [1000.0, 1000.0, 1000.0, 1100.0, 1000.0, 1000.0, 1800.0, 1000.0],
    )

    patterns = build_pattern_frame(
        panel,
        lookback_bars=4,
        triangle_slope_min=0.0,
        level_tolerance_pct=0.02,
        breakout_pct=0.001,
        rectangle_max_range_pct=0.08,
        flag_pole_bars=3,
        flag_consolidation_bars=3,
        flag_min_pole_return=0.08,
        flag_max_retrace_pct=0.02,
        volume_window=2,
        volume_multiplier=1.0,
    )

    signal_row = patterns.loc[patterns["timestamp"] == pd.Timestamp("2024-01-01 01:30:00")].iloc[0]
    assert bool(signal_row["bull_flag_breakout"]) is True
    assert signal_row["bullish_pattern_count"] >= 1
    assert bool(patterns.iloc[5]["bull_flag_breakout"]) is False


def test_generate_diagnostics_aligns_pattern_weights_to_next_bar() -> None:
    panel = _panel(
        [100.0, 104.0, 108.0, 110.0, 109.0, 108.5, 111.0, 112.0],
        [1000.0, 1000.0, 1000.0, 1100.0, 1000.0, 1000.0, 1800.0, 1000.0],
    )
    kwargs = {
        "frequency": "bar",
        "lookback_bars": 4,
        "triangle_slope_min": 0.0,
        "level_tolerance_pct": 0.02,
        "breakout_pct": 0.001,
        "rectangle_max_range_pct": 0.08,
        "flag_pole_bars": 3,
        "flag_consolidation_bars": 3,
        "flag_min_pole_return": 0.08,
        "flag_max_retrace_pct": 0.02,
        "volume_window": 2,
        "volume_multiplier": 1.0,
        "min_bullish_patterns": 1,
    }

    diagnostics = generate_diagnostics(panel, **kwargs)
    weights = generate_weights(panel, **kwargs)

    assert list(diagnostics.columns) == DIAGNOSTIC_COLUMNS
    assert len(diagnostics) == len(weights) == 7
    assert diagnostics["effective_date"].min() == pd.Timestamp("2024-01-01 00:15:00")
    assert diagnostics.loc[
        diagnostics["timestamp"] == pd.Timestamp("2024-01-01 01:30:00"),
        "target_weight",
    ].iloc[0] == pytest.approx(1.0)
    assert diagnostics["target_weight"].tolist() == pytest.approx(weights["weight"].tolist())


def test_build_pattern_frame_detects_double_bottom_breakout() -> None:
    panel = _panel(
        [106.0, 100.0, 104.0, 101.0, 105.0, 109.0],
        [1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1800.0],
        highs=[106.5, 101.0, 106.0, 102.0, 106.5, 110.0],
        lows=[105.5, 99.5, 103.5, 100.0, 104.5, 108.5],
    )

    patterns = build_pattern_frame(
        panel,
        lookback_bars=5,
        triangle_slope_min=0.0,
        level_tolerance_pct=0.02,
        breakout_pct=0.001,
        rectangle_max_range_pct=0.20,
        flag_pole_bars=3,
        flag_consolidation_bars=3,
        flag_min_pole_return=0.08,
        flag_max_retrace_pct=0.02,
        volume_window=2,
        volume_multiplier=1.0,
    )

    assert bool(patterns.iloc[-1]["double_bottom_breakout"]) is True


def test_build_pattern_frame_detects_double_top_breakdown() -> None:
    panel = _panel(
        [100.0, 108.0, 103.0, 107.0, 104.0, 99.0],
        [1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1800.0],
        highs=[100.5, 108.5, 103.5, 108.0, 104.5, 99.5],
        lows=[99.5, 107.0, 102.0, 106.5, 103.5, 98.0],
    )

    patterns = build_pattern_frame(
        panel,
        lookback_bars=5,
        triangle_slope_min=0.0,
        level_tolerance_pct=0.02,
        breakout_pct=0.001,
        rectangle_max_range_pct=0.20,
        flag_pole_bars=3,
        flag_consolidation_bars=3,
        flag_min_pole_return=0.08,
        flag_max_retrace_pct=0.02,
        volume_window=2,
        volume_multiplier=1.0,
    )

    assert bool(patterns.iloc[-1]["double_top_breakdown"]) is True


def test_build_pattern_frame_detects_every_catalog_archetype() -> None:
    for spec in PATTERN_CATALOG:
        panel = _panel(
            list(spec.closes),
            [1000.0] * (len(spec.closes) - 1) + [2000.0],
        )
        patterns = build_pattern_frame(
            panel,
            lookback_bars=8,
            triangle_slope_min=0.0,
            level_tolerance_pct=0.02,
            breakout_pct=0.001,
            rectangle_max_range_pct=0.20,
            flag_pole_bars=3,
            flag_consolidation_bars=3,
            flag_min_pole_return=0.08,
            flag_max_retrace_pct=0.03,
            volume_window=2,
            volume_multiplier=1.0,
        )
        final_row = patterns.iloc[-1]

        for column in spec.implemented_columns:
            assert bool(final_row[column]) is True, f"{spec.name} did not trigger {column}"
