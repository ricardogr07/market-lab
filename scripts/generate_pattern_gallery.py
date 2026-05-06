from __future__ import annotations

import argparse
from pathlib import Path

from marketlab.reports.pattern_gallery import plot_synthetic_pattern_gallery
from marketlab.strategies.pattern_catalog import build_synthetic_pattern_gallery_frame


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate synthetic chart-pattern gallery artifacts.")
    parser.add_argument(
        "--output-dir",
        default="artifacts/pattern-gallery",
        help="Directory where synthetic gallery CSV and PNG artifacts will be written.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    gallery = build_synthetic_pattern_gallery_frame()
    csv_path = output_dir / "synthetic_pattern_gallery.csv"
    png_path = output_dir / "synthetic_pattern_gallery.png"
    gallery.to_csv(csv_path, index=False)
    plot_synthetic_pattern_gallery(gallery, png_path)
    print(output_dir.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
