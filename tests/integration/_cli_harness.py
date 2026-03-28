from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pandas as pd
import yaml

from marketlab.backtest.metrics import compute_strategy_metrics
from marketlab.reports.summary import (
    FOLD_SUMMARY_COLUMNS as _FOLD_SUMMARY_COLUMNS,
)
from marketlab.reports.summary import (
    MODEL_SUMMARY_COLUMNS as _MODEL_SUMMARY_COLUMNS,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
LAUNCHER_PATH = REPO_ROOT / "scripts" / "run_marketlab.py"
FIXTURE_PANEL_PATH = REPO_ROOT / "tests" / "fixtures" / "market_panel.csv"
DEFAULT_FIXTURE_SYMBOLS = ["VOO", "QQQ", "SMH", "XLV", "IEMG"]
FOLD_SUMMARY_COLUMNS = list(_FOLD_SUMMARY_COLUMNS)
MODEL_SUMMARY_COLUMNS = list(_MODEL_SUMMARY_COLUMNS)

EXPECTED_METRICS_COLUMNS = list(
    compute_strategy_metrics(
        pd.DataFrame(
            [
                {
                    "strategy": "placeholder",
                    "date": pd.Timestamp("2024-01-01"),
                    "net_return": 0.0,
                    "equity": 1.0,
                    "turnover": 0.0,
                }
            ]
        )
    ).columns
)


def build_synthetic_panel(
    symbol_specs: tuple[tuple[str, float, float], ...],
    *,
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    trading_dates = pd.bdate_range(start_date, end_date)
    rows: list[dict[str, object]] = []

    for symbol_index, (symbol, base_price, amplitude) in enumerate(symbol_specs):
        close_price = base_price
        for row_index, timestamp in enumerate(trading_dates):
            week_ordinal = timestamp.to_period("W-FRI").ordinal + symbol_index
            direction = 1.0 if week_ordinal % 2 == 0 else -1.0
            open_price = close_price
            close_price = max(5.0, open_price + (amplitude * direction))
            high_price = max(open_price, close_price) + 0.2
            low_price = min(open_price, close_price) - 0.2

            rows.append(
                {
                    "symbol": symbol,
                    "timestamp": timestamp,
                    "open": round(open_price, 4),
                    "high": round(high_price, 4),
                    "low": round(low_price, 4),
                    "close": round(close_price, 4),
                    "volume": 1_000_000 + (symbol_index * 10_000) + row_index,
                    "adj_close": round(close_price, 4),
                    "adj_factor": 1.0,
                    "adj_open": round(open_price, 4),
                    "adj_high": round(high_price, 4),
                    "adj_low": round(low_price, 4),
                }
            )

    return pd.DataFrame(rows).sort_values(["symbol", "timestamp"]).reset_index(drop=True)


def load_fixture_panel() -> pd.DataFrame:
    return pd.read_csv(FIXTURE_PANEL_PATH, parse_dates=["timestamp"])


def write_raw_symbol_cache(
    cache_dir: Path,
    *,
    panel: pd.DataFrame | None = None,
    symbols: list[str] | None = None,
) -> dict[str, Path]:
    working_panel = panel if panel is not None else load_fixture_panel()
    selected_symbols = symbols or DEFAULT_FIXTURE_SYMBOLS

    cache_dir.mkdir(parents=True, exist_ok=True)
    written_paths: dict[str, Path] = {}
    rename_map = {
        "timestamp": "Date",
        "open": "Open",
        "high": "High",
        "low": "Low",
        "close": "Close",
        "volume": "Volume",
        "adj_close": "Adj Close",
    }

    for symbol in selected_symbols:
        symbol_rows = working_panel.loc[working_panel["symbol"] == symbol].copy()
        if symbol_rows.empty:
            raise ValueError(f"Fixture panel does not contain symbol {symbol}.")

        raw_frame = (
            symbol_rows.loc[:, list(rename_map)]
            .rename(columns=rename_map)
            .sort_values("Date")
            .reset_index(drop=True)
        )
        raw_path = cache_dir / f"{symbol}.csv"
        raw_frame.to_csv(raw_path, index=False)
        written_paths[symbol] = raw_path

    return written_paths


def write_yaml_config(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return path


def run_launcher_command(command: str, config_path: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(LAUNCHER_PATH), command, "--config", str(config_path)],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )


def run_marketlab_cli(command: str, config_path: Path) -> subprocess.CompletedProcess[str]:
    return run_launcher_command(command, config_path)


def assert_command_ok(result: subprocess.CompletedProcess[str]) -> None:
    assert result.returncode == 0, (
        f"Expected successful command, got {result.returncode}\n"
        f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
    )


def printed_path(result: subprocess.CompletedProcess[str]) -> Path:
    output_lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    assert output_lines, f"Command did not print a path.\nSTDERR:\n{result.stderr}"
    return Path(output_lines[-1]).resolve()


def stdout_path(result: subprocess.CompletedProcess[str]) -> Path:
    return printed_path(result)


def latest_run_dir(run_root: Path) -> Path:
    run_dirs = sorted(path for path in run_root.iterdir() if path.is_dir())
    assert run_dirs, f"No run directories exist under {run_root}"
    return run_dirs[-1]

