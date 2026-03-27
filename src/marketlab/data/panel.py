from __future__ import annotations

from pathlib import Path

import pandas as pd

PANEL_COLUMNS = [
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
]


def _normalize_column_name(column: object) -> str:
    if isinstance(column, tuple):
        for part in column:
            text = str(part).strip()
            if text and not text.startswith("Unnamed:"):
                return text
        return ""

    return str(column).strip()


def normalize_ohlcv_frame(symbol: str, frame: pd.DataFrame) -> pd.DataFrame:
    working = frame.copy()
    working.columns = [_normalize_column_name(column) for column in working.columns]

    rename_map = {
        "Date": "timestamp",
        "Adj Close": "adj_close",
        "Open": "open",
        "High": "high",
        "Low": "low",
        "Close": "close",
        "Volume": "volume",
    }
    working = working.rename(columns=rename_map)

    required = {"timestamp", "open", "high", "low", "close", "volume"}
    missing = required - set(working.columns)
    if missing:
        joined = ", ".join(sorted(missing))
        raise ValueError(f"OHLCV frame for {symbol} is missing required columns: {joined}")

    working["timestamp"] = pd.to_datetime(working["timestamp"], errors="coerce")
    for column in ("open", "high", "low", "close", "volume", "adj_close"):
        if column in working.columns:
            working[column] = pd.to_numeric(working[column], errors="coerce")

    # yfinance may emit a second ticker-header row when MultiIndex columns are cached to CSV.
    working = working.loc[working["timestamp"].notna()].copy()
    if working.empty:
        raise ValueError(f"OHLCV frame for {symbol} does not contain any timestamped rows.")

    invalid_rows = working[["open", "high", "low", "close", "volume"]].isna().any(axis=1)
    if invalid_rows.any():
        raise ValueError(f"OHLCV frame for {symbol} contains rows with missing OHLCV values.")

    if "adj_close" not in working.columns:
        working["adj_close"] = working["close"]
    elif working["adj_close"].isna().any():
        working["adj_close"] = working["adj_close"].fillna(working["close"])

    working["symbol"] = symbol
    working["adj_factor"] = working["adj_close"] / working["close"]
    working["adj_open"] = working["open"] * working["adj_factor"]
    working["adj_high"] = working["high"] * working["adj_factor"]
    working["adj_low"] = working["low"] * working["adj_factor"]

    panel = working[PANEL_COLUMNS].sort_values("timestamp").reset_index(drop=True)
    return panel


def build_market_panel(symbol_frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    if not symbol_frames:
        raise ValueError("No symbol frames were provided.")

    panel = pd.concat(
        [normalize_ohlcv_frame(symbol, frame) for symbol, frame in symbol_frames.items()],
        ignore_index=True,
    )
    panel = panel.sort_values(["symbol", "timestamp"]).reset_index(drop=True)
    validate_market_panel(panel)
    return panel


def validate_market_panel(panel: pd.DataFrame) -> None:
    missing = [column for column in PANEL_COLUMNS if column not in panel.columns]
    if missing:
        joined = ", ".join(missing)
        raise ValueError(f"Market panel is missing required columns: {joined}")

    duplicates = panel.duplicated(["symbol", "timestamp"])
    if duplicates.any():
        duplicate_rows = panel.loc[duplicates, ["symbol", "timestamp"]]
        raise ValueError(f"Market panel contains duplicate symbol/timestamp rows: {duplicate_rows}")

    sorted_panel = panel.sort_values(["symbol", "timestamp"]).reset_index(drop=True)
    if not sorted_panel.equals(panel.reset_index(drop=True)):
        raise ValueError("Market panel must be sorted by symbol and timestamp.")


def load_panel_csv(path: str | Path) -> pd.DataFrame:
    frame = pd.read_csv(path, parse_dates=["timestamp"])
    frame = frame.sort_values(["symbol", "timestamp"]).reset_index(drop=True)
    validate_market_panel(frame)
    return frame


def save_panel_csv(panel: pd.DataFrame, path: str | Path) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    panel.to_csv(output_path, index=False)
    return output_path
