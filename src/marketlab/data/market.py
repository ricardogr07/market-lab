from __future__ import annotations

import logging
from pathlib import Path
from typing import Protocol

import pandas as pd

from marketlab.config import ExperimentConfig

LOGGER = logging.getLogger(__name__)


def _flatten_download_columns(frame: pd.DataFrame) -> pd.DataFrame:
    working = frame.copy()

    if isinstance(working.columns, pd.MultiIndex):
        flattened: list[str] = []
        for column in working.columns.to_flat_index():
            parts = [str(part).strip() for part in column if str(part).strip()]
            flattened.append(parts[0] if parts else "")
        working.columns = flattened
    else:
        working.columns = [str(column).strip() for column in working.columns]

    return working


class MarketDataProvider(Protocol):
    def download_symbol_history(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
        interval: str,
    ) -> pd.DataFrame: ...


class YFinanceMarketDataProvider:
    def download_symbol_history(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
        interval: str,
    ) -> pd.DataFrame:
        try:
            import yfinance as yf
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "yfinance is required to download market data. "
                "Install it or provide a prepared panel cache."
            ) from exc

        frame = yf.download(
            symbol,
            start=start_date,
            end=end_date,
            interval=interval,
            auto_adjust=False,
            progress=False,
        )
        if frame.empty:
            raise RuntimeError(f"No market data returned for {symbol}.")

        return _flatten_download_columns(frame.reset_index())


def raw_cache_path(cache_dir: str | Path, symbol: str) -> Path:
    return Path(cache_dir) / f"{symbol}.csv"


def load_symbol_frames(
    config: ExperimentConfig,
    provider: MarketDataProvider | None = None,
) -> dict[str, pd.DataFrame]:
    cache_dir = config.cache_dir
    cache_dir.mkdir(parents=True, exist_ok=True)
    provider = provider or YFinanceMarketDataProvider()

    frames: dict[str, pd.DataFrame] = {}
    for symbol in config.data.symbols:
        symbol_cache = raw_cache_path(cache_dir, symbol)
        if symbol_cache.exists():
            LOGGER.info("Loading cached raw data for %s from %s", symbol, symbol_cache)
            frames[symbol] = pd.read_csv(symbol_cache)
            continue

        LOGGER.info("Downloading raw data for %s", symbol)
        frame = provider.download_symbol_history(
            symbol=symbol,
            start_date=config.data.start_date,
            end_date=config.data.end_date,
            interval=config.data.interval,
        )
        frame.to_csv(symbol_cache, index=False)
        frames[symbol] = frame

    return frames
