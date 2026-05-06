from __future__ import annotations

from pathlib import Path

from marketlab.reports.pattern_gallery import plot_synthetic_pattern_gallery
from marketlab.strategies.chart_patterns import PATTERN_COLUMNS
from marketlab.strategies.pattern_catalog import (
    PATTERN_CATALOG,
    build_synthetic_pattern_gallery_frame,
)


def test_pattern_catalog_defines_twenty_unique_archetypes() -> None:
    pattern_names = [spec.name for spec in PATTERN_CATALOG]

    assert len(pattern_names) == 20
    assert len(set(pattern_names)) == 20
    assert pattern_names == [
        "ascending_triangle",
        "descending_triangle",
        "symmetrical_triangle",
        "rectangle",
        "head_and_shoulders",
        "inverse_head_and_shoulders",
        "double_top",
        "double_bottom",
        "triple_top",
        "triple_bottom",
        "falling_wedge",
        "rising_wedge",
        "bull_flag",
        "bear_flag",
        "pennant",
        "cup_and_handle",
        "ascending_channel",
        "descending_channel",
        "megaphone",
        "diamond",
    ]


def test_pattern_catalog_marks_only_strategy_backed_columns_as_implemented() -> None:
    implemented_columns = {
        column
        for spec in PATTERN_CATALOG
        for column in spec.implemented_columns
    }

    assert all(spec.implemented_columns for spec in PATTERN_CATALOG)
    assert implemented_columns == set(PATTERN_COLUMNS)


def test_synthetic_pattern_gallery_frame_covers_each_pattern_evenly() -> None:
    gallery = build_synthetic_pattern_gallery_frame()

    assert set(gallery.columns) == {
        "pattern",
        "bias",
        "bar",
        "timestamp",
        "close",
        "implemented",
        "implemented_columns",
    }
    assert gallery["pattern"].nunique() == 20
    assert gallery.groupby("pattern").size().eq(9).all()
    assert gallery["implemented"].all()


def test_synthetic_pattern_gallery_plot_is_created(tmp_path: Path) -> None:
    gallery = build_synthetic_pattern_gallery_frame()

    output_path = plot_synthetic_pattern_gallery(
        gallery,
        tmp_path / "synthetic_pattern_gallery.png",
    )

    assert output_path.exists()
    assert output_path.stat().st_size > 0
