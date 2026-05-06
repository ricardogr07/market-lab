from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True, slots=True)
class PatternCatalogSpec:
    name: str
    bias: str
    closes: tuple[float, ...]
    implemented_columns: tuple[str, ...] = ()


PATTERN_CATALOG: tuple[PatternCatalogSpec, ...] = (
    PatternCatalogSpec(
        "ascending_triangle",
        "bullish",
        (100, 104, 102, 105, 103, 105, 104, 106, 108),
        ("ascending_triangle_breakout",),
    ),
    PatternCatalogSpec(
        "descending_triangle",
        "bearish",
        (108, 104, 106, 103, 105, 103, 104, 102, 100),
        ("descending_triangle_breakdown",),
    ),
    PatternCatalogSpec(
        "symmetrical_triangle",
        "neutral_breakout",
        (100, 108, 102, 106, 103, 105, 104, 106, 109),
        ("symmetrical_triangle_breakout",),
    ),
    PatternCatalogSpec(
        "rectangle",
        "neutral_breakout",
        (100, 104, 101, 104, 100, 103, 101, 104, 107),
        ("bullish_rectangle_breakout", "bearish_rectangle_breakdown"),
    ),
    PatternCatalogSpec(
        "head_and_shoulders",
        "bearish",
        (100, 106, 102, 112, 103, 106, 101, 98, 95),
        ("head_and_shoulders_breakdown",),
    ),
    PatternCatalogSpec(
        "inverse_head_and_shoulders",
        "bullish",
        (110, 104, 108, 98, 107, 103, 109, 112, 115),
        ("inverse_head_and_shoulders_breakout",),
    ),
    PatternCatalogSpec(
        "double_top",
        "bearish",
        (100, 108, 103, 109, 104, 108, 102, 99, 96),
        ("double_top_breakdown",),
    ),
    PatternCatalogSpec(
        "double_bottom",
        "bullish",
        (110, 102, 107, 101, 106, 102, 108, 112, 115),
        ("double_bottom_breakout",),
    ),
    PatternCatalogSpec(
        "triple_top",
        "bearish",
        (100, 108, 103, 109, 104, 108, 103, 100, 97),
        ("triple_top_breakdown",),
    ),
    PatternCatalogSpec(
        "triple_bottom",
        "bullish",
        (110, 102, 107, 101, 106, 102, 108, 112, 116),
        ("triple_bottom_breakout",),
    ),
    PatternCatalogSpec(
        "falling_wedge",
        "bullish",
        (112, 108, 110, 105, 107, 103, 105, 106, 110),
        ("falling_wedge_breakout",),
    ),
    PatternCatalogSpec(
        "rising_wedge",
        "bearish",
        (100, 104, 102, 106, 104, 107, 105, 103, 99),
        ("rising_wedge_breakdown",),
    ),
    PatternCatalogSpec(
        "bull_flag",
        "bullish_continuation",
        (100, 104, 108, 111, 110, 109, 110, 113, 116),
        ("bull_flag_breakout",),
    ),
    PatternCatalogSpec(
        "bear_flag",
        "bearish_continuation",
        (116, 112, 108, 105, 106, 107, 106, 103, 100),
        ("bear_flag_breakdown",),
    ),
    PatternCatalogSpec(
        "pennant",
        "continuation",
        (100, 106, 112, 109, 111, 110, 111, 113, 116),
        ("pennant_breakout",),
    ),
    PatternCatalogSpec(
        "cup_and_handle",
        "bullish",
        (110, 106, 102, 100, 102, 106, 110, 108, 113),
        ("cup_and_handle_breakout",),
    ),
    PatternCatalogSpec(
        "ascending_channel",
        "trend_channel",
        (100, 104, 102, 106, 104, 108, 106, 110, 108),
        ("ascending_channel_continuation",),
    ),
    PatternCatalogSpec(
        "descending_channel",
        "trend_channel",
        (110, 106, 108, 104, 106, 102, 104, 100, 102),
        ("descending_channel_breakdown",),
    ),
    PatternCatalogSpec(
        "megaphone",
        "volatility_expansion",
        (105, 100, 110, 98, 112, 96, 115, 94, 118),
        ("megaphone_breakout",),
    ),
    PatternCatalogSpec(
        "diamond",
        "reversal",
        (104, 102, 108, 98, 112, 100, 108, 103, 99),
        ("diamond_breakdown",),
    ),
)


def build_synthetic_pattern_gallery_frame() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    start = pd.Timestamp("2024-01-01 00:00:00")
    for spec in PATTERN_CATALOG:
        for bar_index, close in enumerate(spec.closes):
            rows.append(
                {
                    "pattern": spec.name,
                    "bias": spec.bias,
                    "bar": bar_index,
                    "timestamp": start + pd.Timedelta(minutes=15 * bar_index),
                    "close": float(close),
                    "implemented": bool(spec.implemented_columns),
                    "implemented_columns": ",".join(spec.implemented_columns),
                }
            )
    return pd.DataFrame(rows)
