from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import urlopen

import pandas as pd

from marketlab.config import default_periods_per_year, load_config
from marketlab.data.market import (
    YFinanceMarketDataProvider,
    load_symbol_frames,
    raw_cache_path,
)
from marketlab.data.panel import build_market_panel, load_panel_csv, save_panel_csv
from marketlab.pipeline import backtest

DEFAULT_INTERVALS = ("1m", "5m", "15m", "30m", "45m", "1h")
RESAMPLED_INTERVAL_RULES = {
    "15m": "15min",
    "30m": "30min",
    "45m": "45min",
    "1h": "1h",
    "2h": "2h",
    "4h": "4h",
    "6h": "6h",
    "8h": "8h",
    "12h": "12h",
    "1d": "1D",
}
BARS_PER_HOUR = {
    "1m": 60.0,
    "5m": 12.0,
    "15m": 4.0,
    "30m": 2.0,
    "45m": 4.0 / 3.0,
    "1h": 1.0,
    "2h": 0.5,
    "4h": 0.25,
    "6h": 1.0 / 6.0,
    "8h": 0.125,
    "12h": 1.0 / 12.0,
    "1d": 1.0 / 24.0,
}
BINANCE_INTERVAL_MS = {
    "1h": 60 * 60 * 1000,
}
BINANCE_SYMBOLS = {
    "BTC-USD": "BTCUSDT",
}
CRYPTO_PERIODS_PER_YEAR = {
    "1d": 365.0,
}
BASE_CONFIG_PATH = Path("configs/experiment.crypto_5m_patterns_day.yaml")
SUMMARY_COLUMNS = [
    "interval",
    "status",
    "run_dir",
    "panel_path",
    "panel_rows",
    "diagnostic_rows",
    "buy_rows",
    "pattern_hit_rows",
    "buy_hold_cumulative_return",
    "chart_patterns_cumulative_return",
    "excess_cumulative_return",
    "chart_patterns_cost_drag",
    "chart_patterns_total_turnover",
    "chart_patterns_max_drawdown",
    "error",
]


@dataclass(frozen=True, slots=True)
class MetaRunSettings:
    start_date: str
    end_date: str
    focus_start: str
    focus_end: str
    output_root: Path
    experiment_suffix: str
    intervals: tuple[str, ...]
    source: str


def _resample_interval(panel_5m: pd.DataFrame, interval: str) -> pd.DataFrame:
    rule = RESAMPLED_INTERVAL_RULES[interval]
    frames: list[pd.DataFrame] = []
    for symbol, symbol_frame in panel_5m.groupby("symbol", sort=False):
        working = symbol_frame.sort_values("timestamp").set_index("timestamp")
        resampled = working.resample(rule, label="left", closed="left").agg(
            {
                "open": "first",
                "high": "max",
                "low": "min",
                "close": "last",
                "volume": "sum",
                "adj_close": "last",
                "adj_factor": "last",
                "adj_open": "first",
                "adj_high": "max",
                "adj_low": "min",
            }
        )
        resampled = resampled.dropna(subset=["open", "high", "low", "close", "adj_close"])
        resampled["symbol"] = symbol
        frames.append(resampled.reset_index())
    return (
        pd.concat(frames, ignore_index=True)
        .loc[
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
        .sort_values(["timestamp", "symbol"])
        .reset_index(drop=True)
    )


def _timestamp_column(frame: pd.DataFrame) -> str:
    for column in ("Datetime", "Date", "timestamp"):
        if column in frame.columns:
            return column
    raise ValueError("Downloaded frame is missing a timestamp column.")


def _load_chunked_1m_symbol_frames(config) -> dict[str, pd.DataFrame]:
    cache_dir = config.cache_dir
    cache_dir.mkdir(parents=True, exist_ok=True)
    provider = YFinanceMarketDataProvider()
    start = pd.Timestamp(config.data.start_date)
    end = pd.Timestamp(config.data.end_date)
    if start >= end:
        raise ValueError("start_date must be before end_date for 1m chunked downloads.")

    frames: dict[str, pd.DataFrame] = {}
    for symbol in config.data.symbols:
        symbol_cache = raw_cache_path(cache_dir, symbol)
        if symbol_cache.exists():
            frames[symbol] = pd.read_csv(symbol_cache)
            continue

        chunks: list[pd.DataFrame] = []
        cursor = start
        while cursor < end:
            chunk_end = min(cursor + pd.Timedelta(days=7), end)
            chunk = provider.download_symbol_history(
                symbol=symbol,
                start_date=cursor.strftime("%Y-%m-%d"),
                end_date=chunk_end.strftime("%Y-%m-%d"),
                interval="1m",
            )
            chunks.append(chunk)
            cursor = chunk_end

        combined = pd.concat(chunks, ignore_index=True)
        timestamp_column = _timestamp_column(combined)
        combined = (
            combined.drop_duplicates(subset=[timestamp_column])
            .sort_values(timestamp_column)
            .reset_index(drop=True)
        )
        combined.to_csv(symbol_cache, index=False)
        frames[symbol] = combined

    return frames


def _timestamp_ms(value: str) -> int:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize("UTC")
    else:
        timestamp = timestamp.tz_convert("UTC")
    return int(timestamp.timestamp() * 1000)


def _download_binance_klines(
    symbol: str,
    interval: str,
    start_ms: int,
    end_ms: int,
) -> list[list[object]]:
    params = urlencode(
        {
            "symbol": symbol,
            "interval": interval,
            "startTime": start_ms,
            "endTime": end_ms - 1,
            "limit": 1000,
        }
    )
    with urlopen(f"https://api.binance.com/api/v3/klines?{params}", timeout=30) as response:
        return json.loads(response.read().decode("utf-8"))


def _load_binance_symbol_frames(config, interval: str) -> dict[str, pd.DataFrame]:
    if interval not in BINANCE_INTERVAL_MS:
        raise ValueError(f"Binance source is only wired for native intervals: {sorted(BINANCE_INTERVAL_MS)}")

    cache_dir = config.cache_dir
    cache_dir.mkdir(parents=True, exist_ok=True)
    start_ms = _timestamp_ms(config.data.start_date)
    end_ms = _timestamp_ms(config.data.end_date)
    if start_ms >= end_ms:
        raise ValueError("start_date must be before end_date for Binance downloads.")

    frames: dict[str, pd.DataFrame] = {}
    interval_ms = BINANCE_INTERVAL_MS[interval]
    for symbol in config.data.symbols:
        binance_symbol = BINANCE_SYMBOLS.get(symbol)
        if binance_symbol is None:
            raise ValueError(f"No Binance symbol mapping configured for {symbol}.")

        symbol_cache = raw_cache_path(cache_dir, symbol)
        if symbol_cache.exists():
            frames[symbol] = pd.read_csv(symbol_cache)
            continue

        rows: list[list[object]] = []
        cursor = start_ms
        while cursor < end_ms:
            chunk = _download_binance_klines(binance_symbol, interval, cursor, end_ms)
            if not chunk:
                break
            rows.extend(chunk)
            cursor = int(chunk[-1][0]) + interval_ms

        if not rows:
            raise RuntimeError(f"No Binance market data returned for {binance_symbol}.")

        frame = pd.DataFrame(
            rows,
            columns=[
                "open_time",
                "Open",
                "High",
                "Low",
                "Close",
                "Volume",
                "close_time",
                "quote_asset_volume",
                "number_of_trades",
                "taker_buy_base_volume",
                "taker_buy_quote_volume",
                "ignore",
            ],
        )
        frame["Datetime"] = pd.to_datetime(frame["open_time"], unit="ms", utc=True)
        frame = frame.loc[:, ["Datetime", "Open", "High", "Low", "Close", "Volume"]]
        frame = frame.drop_duplicates(subset=["Datetime"]).sort_values("Datetime")
        frame.to_csv(symbol_cache, index=False)
        frames[symbol] = frame

    return frames


def _prepare_native_panel(interval: str, settings: MetaRunSettings) -> tuple[pd.DataFrame, Path]:
    config = load_config(BASE_CONFIG_PATH)
    config.data.interval = interval
    config.data.start_date = settings.start_date
    config.data.end_date = settings.end_date
    config.data.cache_dir = f"artifacts/data-crypto-{interval}-{settings.experiment_suffix}"
    config.data.prepared_panel_filename = (
        f"btc_usd_{interval}_{settings.start_date}_{settings.end_date}.csv"
    )
    panel_path = config.prepared_panel_path
    if panel_path.exists():
        return load_panel_csv(panel_path), panel_path

    if settings.source == "binance":
        frames = _load_binance_symbol_frames(config, interval)
    elif interval == "1m":
        frames = _load_chunked_1m_symbol_frames(config)
    else:
        frames = load_symbol_frames(config)
    panel = build_market_panel(frames)
    return panel, save_panel_csv(panel, panel_path)


def _prepare_resampled_panel(
    interval: str,
    base_panel: pd.DataFrame,
    settings: MetaRunSettings,
) -> tuple[pd.DataFrame, Path]:
    panel_path = Path(
        f"artifacts/data-crypto-{interval}-{settings.experiment_suffix}/"
        f"btc_usd_{interval}_{settings.start_date}_{settings.end_date}.csv"
    )
    if panel_path.exists():
        return load_panel_csv(panel_path), panel_path
    panel = _resample_interval(base_panel, interval)
    return panel, save_panel_csv(panel, panel_path)


def _scaled_pattern_settings(config, interval: str) -> None:
    bars_per_hour = BARS_PER_HOUR[interval]
    chart_patterns = config.baselines.chart_patterns
    chart_patterns.lookback_bars = max(8, int(round(24 * bars_per_hour)))
    chart_patterns.flag_pole_bars = max(3, int(round(4 * bars_per_hour)))
    chart_patterns.flag_consolidation_bars = max(3, int(round(3 * bars_per_hour)))
    chart_patterns.volume_window = max(2, int(round(5 * bars_per_hour)))


def _run_interval(interval: str, panel_path: Path, settings: MetaRunSettings) -> dict[str, object]:
    config = load_config(BASE_CONFIG_PATH)
    config.experiment_name = f"crypto_{interval}_patterns_{settings.experiment_suffix}"
    config.data.interval = interval
    config.data.start_date = settings.start_date
    config.data.end_date = settings.end_date
    config.data.cache_dir = str(panel_path.parent)
    config.data.prepared_panel_filename = panel_path.name
    config.evaluation.periods_per_year = CRYPTO_PERIODS_PER_YEAR.get(
        interval,
        default_periods_per_year(interval),
    )
    config.evaluation.focus_start = settings.focus_start
    config.evaluation.focus_end = settings.focus_end
    _scaled_pattern_settings(config, interval)
    artifacts = backtest(config)

    summary = pd.read_csv(artifacts.strategy_summary_path)
    diagnostics = pd.read_csv(artifacts.pattern_diagnostics_path) if artifacts.pattern_diagnostics_path else pd.DataFrame()
    chart_row = summary.loc[summary["strategy"] == "chart_patterns"].iloc[0]
    buy_hold_row = summary.loc[summary["strategy"] == "buy_hold"].iloc[0]
    return {
        "interval": interval,
        "status": "ok",
        "run_dir": str(artifacts.run_dir),
        "panel_path": str(panel_path),
        "panel_rows": len(load_panel_csv(panel_path)),
        "diagnostic_rows": len(diagnostics),
        "buy_rows": int((diagnostics["target_weight"] > 0.0).sum()) if not diagnostics.empty else 0,
        "pattern_hit_rows": int(
            (
                (diagnostics["bullish_pattern_count"] > 0)
                | (diagnostics["bearish_pattern_count"] > 0)
            ).sum()
        )
        if not diagnostics.empty
        else 0,
        "buy_hold_cumulative_return": float(buy_hold_row["cumulative_return"]),
        "chart_patterns_cumulative_return": float(chart_row["cumulative_return"]),
        "excess_cumulative_return": float(chart_row["cumulative_return"])
        - float(buy_hold_row["cumulative_return"]),
        "chart_patterns_cost_drag": float(chart_row["cost_drag"]),
        "chart_patterns_total_turnover": float(chart_row["total_turnover"]),
        "chart_patterns_max_drawdown": float(chart_row["max_drawdown"]),
        "error": "",
    }


def _failed_interval(interval: str, exc: Exception) -> dict[str, object]:
    return {
        "interval": interval,
        "status": "failed",
        "run_dir": "",
        "panel_path": "",
        "panel_rows": 0,
        "diagnostic_rows": 0,
        "buy_rows": 0,
        "pattern_hit_rows": 0,
        "buy_hold_cumulative_return": pd.NA,
        "chart_patterns_cumulative_return": pd.NA,
        "excess_cumulative_return": pd.NA,
        "chart_patterns_cost_drag": pd.NA,
        "chart_patterns_total_turnover": pd.NA,
        "chart_patterns_max_drawdown": pd.NA,
        "error": str(exc).replace("\n", " "),
    }


def _markdown_table(frame: pd.DataFrame) -> str:
    values = frame.astype("string").fillna("")
    widths = {
        column: max(len(column), *(len(value) for value in values[column].tolist()))
        for column in values.columns
    }
    header = "| " + " | ".join(column.ljust(widths[column]) for column in values.columns) + " |"
    separator = "| " + " | ".join("-" * widths[column] for column in values.columns) + " |"
    rows = [
        "| "
        + " | ".join(str(row[column]).ljust(widths[column]) for column in values.columns)
        + " |"
        for _, row in values.iterrows()
    ]
    return "\n".join([header, separator, *rows])


def _write_markdown(summary: pd.DataFrame, path: Path, settings: MetaRunSettings) -> None:
    display = summary.copy()
    numeric_columns = display.select_dtypes(include="number").columns
    display.loc[:, numeric_columns] = display.loc[:, numeric_columns].round(6)
    lines = [
        "# Crypto Pattern Interval Meta-Analysis",
        "",
        f"- Window: `{settings.start_date}` to `{settings.end_date}`",
        "- Strategy: `chart_patterns` versus `buy_hold`",
        "- Baseline cost: `10 bps` per turnover event",
        f"- Source: `{settings.source}`",
        f"- Intervals: `{', '.join(settings.intervals)}`",
        "- Non-native intervals are OHLCV-resampled from the finest native panel required by the interval grid.",
        "- `1m` uses native provider data when available; it is not synthesized from 5m.",
        "",
        _markdown_table(display.loc[:, SUMMARY_COLUMNS]),
        "",
    ]
    successful = summary.loc[summary["status"] == "ok"].copy()
    if successful.empty:
        lines.extend(
            [
                "## Verdict",
                "",
                "- No interval completed successfully.",
            ]
        )
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return
    best = successful.sort_values(
        ["chart_patterns_cumulative_return", "interval"],
        ascending=[False, True],
    ).iloc[0]
    lines.extend(
        [
            "## Verdict",
            "",
            f"- Best chart-pattern interval by cumulative return: `{best['interval']}` "
            f"({best['chart_patterns_cumulative_return']:.6f}).",
            "- The strategy must beat `buy_hold` after costs before any paper-shadow work is considered.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run chart-pattern crypto meta-analysis across intraday intervals.",
    )
    parser.add_argument("--start-date", default="2026-04-20")
    parser.add_argument("--end-date", default="2026-04-27")
    parser.add_argument("--focus-start", default="")
    parser.add_argument("--focus-end", default="")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("artifacts/runs/crypto_pattern_interval_meta"),
    )
    parser.add_argument("--experiment-suffix", default="week_meta")
    parser.add_argument("--source", choices=["yfinance", "binance"], default="yfinance")
    parser.add_argument(
        "--intervals",
        nargs="+",
        default=list(DEFAULT_INTERVALS),
        help="Intervals to run, e.g. --intervals 1h 2h 4h 6h 12h.",
    )
    return parser.parse_args()


def _normalize_intervals(values: list[str]) -> tuple[str, ...]:
    intervals: list[str] = []
    for value in values:
        normalized = value.lower()
        default_periods_per_year(normalized)
        if normalized not in {"1m", "5m"} and normalized not in RESAMPLED_INTERVAL_RULES:
            raise ValueError(f"Interval {value!r} is not supported by the meta-run script.")
        if normalized not in intervals:
            intervals.append(normalized)
    return tuple(intervals)


def _settings_from_args(args: argparse.Namespace) -> MetaRunSettings:
    return MetaRunSettings(
        start_date=args.start_date,
        end_date=args.end_date,
        focus_start=args.focus_start or f"{args.start_date} 00:00:00",
        focus_end=args.focus_end or f"{args.end_date} 00:00:00",
        output_root=args.output_root,
        experiment_suffix=args.experiment_suffix,
        intervals=_normalize_intervals(args.intervals),
        source=args.source,
    )


def _base_interval_for_grid(intervals: tuple[str, ...]) -> str | None:
    if all(interval == "1m" for interval in intervals):
        return None
    if any(interval in {"5m", "15m", "30m", "45m"} for interval in intervals):
        return "5m"
    return "1h"


def main() -> int:
    settings = _settings_from_args(_parse_args())
    settings.output_root.mkdir(parents=True, exist_ok=True)
    panel_by_interval: dict[str, tuple[pd.DataFrame, Path]] = {}
    rows: list[dict[str, object]] = []
    base_interval = _base_interval_for_grid(settings.intervals)
    if base_interval is not None:
        panel_by_interval[base_interval] = _prepare_native_panel(base_interval, settings)
    if "1m" in settings.intervals:
        try:
            panel_by_interval["1m"] = _prepare_native_panel("1m", settings)
        except Exception as exc:  # pragma: no cover - depends on provider retention.
            rows.append(_failed_interval("1m", exc))
    if base_interval is not None:
        base_panel = panel_by_interval[base_interval][0]
        for interval in settings.intervals:
            if interval in {"1m", base_interval}:
                continue
            if interval not in panel_by_interval:
                panel_by_interval[interval] = _prepare_resampled_panel(interval, base_panel, settings)

    for interval in settings.intervals:
        if interval not in panel_by_interval:
            continue
        try:
            rows.append(_run_interval(interval, panel_by_interval[interval][1], settings))
        except Exception as exc:
            rows.append(_failed_interval(interval, exc))
    summary = pd.DataFrame(rows).loc[:, SUMMARY_COLUMNS]
    summary_path = settings.output_root / "interval_meta_summary.csv"
    report_path = settings.output_root / "interval_meta_report.md"
    summary.to_csv(summary_path, index=False)
    _write_markdown(summary, report_path, settings)
    print(settings.output_root.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
