from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

SOURCE_PANEL = Path(
    "artifacts/phase87_4h_long_run/data/4h/btc_usd_4h_2023-01-01_2026-05-06.csv"
)
DEFAULT_OUTPUTS = {
    "12h": Path("artifacts/data-btc-phase8-12h/btc_usd_12h_panel.csv"),
    "1d": Path("artifacts/data-btc-phase8-1d/btc_usd_1d_panel.csv"),
}
EXPECTED_4H_BARS = {"12h": 3, "1d": 6}


def _load_source(path: Path) -> pd.DataFrame:
    panel = pd.read_csv(path, parse_dates=["timestamp"])
    required_columns = {
        "symbol",
        "timestamp",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "adj_close",
    }
    missing = required_columns - set(panel.columns)
    if missing:
        joined = ", ".join(sorted(missing))
        raise ValueError(f"Source BTC panel is missing required columns: {joined}")
    return panel.sort_values(["symbol", "timestamp"]).reset_index(drop=True)


def _resample_symbol(symbol_panel: pd.DataFrame, cadence: str) -> pd.DataFrame:
    rule = "12h" if cadence == "12h" else "1D"
    working = symbol_panel.set_index("timestamp").sort_index()
    resampled = working.resample(rule, label="left", closed="left").agg(
        {
            "symbol": "last",
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum",
            "adj_close": "last",
        }
    )
    resampled["source_bar_count"] = working["close"].resample(
        rule,
        label="left",
        closed="left",
    ).count()
    resampled = resampled.loc[
        resampled["source_bar_count"].eq(EXPECTED_4H_BARS[cadence])
    ].copy()
    resampled = resampled.dropna(subset=["symbol", "open", "high", "low", "close", "adj_close"])
    resampled["adj_factor"] = resampled["adj_close"] / resampled["close"]
    resampled["adj_open"] = resampled["open"] * resampled["adj_factor"]
    resampled["adj_high"] = resampled["high"] * resampled["adj_factor"]
    resampled["adj_low"] = resampled["low"] * resampled["adj_factor"]
    return resampled.reset_index().loc[
        :,
        [
            "symbol",
            "timestamp",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "adj_close",
            "adj_factor",
            "adj_open",
            "adj_high",
            "adj_low",
        ],
    ]


def resample_panel(source_panel: pd.DataFrame, cadence: str) -> pd.DataFrame:
    if cadence not in DEFAULT_OUTPUTS:
        allowed = ", ".join(sorted(DEFAULT_OUTPUTS))
        raise ValueError(f"Unsupported cadence {cadence!r}; expected one of: {allowed}")
    frames = [
        _resample_symbol(symbol_panel, cadence)
        for _, symbol_panel in source_panel.groupby("symbol", sort=True)
    ]
    return pd.concat(frames, ignore_index=True).sort_values(
        ["symbol", "timestamp"]
    ).reset_index(drop=True)


def write_panel(panel: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    panel.to_csv(path, index=False)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare lower-cadence BTC Phase 8 panels from the seeded 4h panel."
    )
    parser.add_argument("--source", type=Path, default=SOURCE_PANEL)
    parser.add_argument(
        "--cadence",
        choices=sorted(DEFAULT_OUTPUTS),
        action="append",
        help="Cadence to build. Pass multiple times, or omit to build all.",
    )
    args = parser.parse_args()

    source = _load_source(args.source)
    cadences = args.cadence or sorted(DEFAULT_OUTPUTS)
    for cadence in cadences:
        output = DEFAULT_OUTPUTS[cadence]
        panel = resample_panel(source, cadence)
        write_panel(panel, output)
        print(f"Wrote {len(panel)} rows to {output}")


if __name__ == "__main__":
    main()
