from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from marketlab.rebalance import signal_effective_dates

COVARIANCE_ESTIMATORS = {"diagonal_shrinkage", "ewma", "external_csv", "sample"}
EXPECTED_RETURN_SOURCES = {"external_csv", "historical_mean"}
EWMA_LAMBDA = 0.94
DIAGONAL_SHRINKAGE_WEIGHT = 0.10
REQUIRED_PANEL_COLUMNS = {"adj_close", "symbol", "timestamp"}


@dataclass(slots=True)
class OptimizerWindow:
    signal_date: pd.Timestamp
    effective_date: pd.Timestamp
    symbols: list[str]
    returns: pd.DataFrame


@dataclass(slots=True)
class OptimizerInput:
    signal_date: pd.Timestamp
    effective_date: pd.Timestamp
    symbols: list[str]
    returns: pd.DataFrame
    covariance: pd.DataFrame
    expected_returns: pd.Series


def _require_columns(panel: pd.DataFrame) -> None:
    missing = REQUIRED_PANEL_COLUMNS - set(panel.columns)
    if missing:
        joined = ", ".join(sorted(missing))
        raise ValueError(f"Panel is missing required columns for optimizer inputs: {joined}")


def _normalized_panel(panel: pd.DataFrame) -> pd.DataFrame:
    _require_columns(panel)
    if panel.empty:
        return panel.copy()

    return (
        panel.loc[:, ["symbol", "timestamp", "adj_close"]]
        .copy()
        .assign(timestamp=lambda frame: pd.to_datetime(frame["timestamp"]))
        .sort_values(["symbol", "timestamp"])
        .reset_index(drop=True)
    )


def _validated_symbols(panel: pd.DataFrame, symbols: list[str]) -> list[str]:
    missing_symbols = sorted(set(symbols) - set(panel["symbol"].unique()))
    if missing_symbols:
        joined = ", ".join(missing_symbols)
        raise ValueError(f"Panel is missing symbols required for optimizer inputs: {joined}")
    return list(symbols)


def _common_daily_returns(panel: pd.DataFrame, symbols: list[str]) -> pd.DataFrame:
    working = _normalized_panel(panel)
    if working.empty:
        return pd.DataFrame(columns=symbols, dtype=float)

    ordered_symbols = _validated_symbols(working, symbols)
    adj_close = (
        working.loc[working["symbol"].isin(ordered_symbols)]
        .pivot(index="timestamp", columns="symbol", values="adj_close")
        .reindex(columns=ordered_symbols)
        .sort_index()
    )
    daily_returns = adj_close.pct_change(fill_method=None).dropna(how="any")
    return daily_returns.astype(float)


def build_optimizer_windows(
    panel: pd.DataFrame,
    *,
    symbols: list[str],
    lookback_days: int,
    frequency: str = "W-FRI",
) -> list[OptimizerWindow]:
    if lookback_days < 2:
        raise ValueError("lookback_days must be at least 2.")

    working = _normalized_panel(panel)
    if working.empty:
        return []

    ordered_symbols = _validated_symbols(working, symbols)
    effective_dates = signal_effective_dates(working, frequency)
    common_returns = _common_daily_returns(working, ordered_symbols)
    windows: list[OptimizerWindow] = []

    for signal_date, effective_date in effective_dates.items():
        signal_timestamp = pd.Timestamp(signal_date)
        if signal_timestamp not in common_returns.index:
            continue

        trailing_returns = common_returns.loc[common_returns.index <= signal_timestamp].tail(lookback_days)
        if len(trailing_returns) < lookback_days:
            continue
        if pd.Timestamp(trailing_returns.index[-1]) != signal_timestamp:
            continue

        windows.append(
            OptimizerWindow(
                signal_date=signal_timestamp,
                effective_date=pd.Timestamp(effective_date),
                symbols=ordered_symbols,
                returns=trailing_returns.loc[:, ordered_symbols].copy(),
            )
        )

    return windows


def load_external_covariance(path: Path | str, *, symbols: list[str]) -> pd.DataFrame:
    covariance = pd.read_csv(Path(path), index_col=0)
    if covariance.shape[0] != covariance.shape[1]:
        raise ValueError("External covariance CSV must be a square matrix.")

    covariance.index = covariance.index.map(str)
    covariance.columns = covariance.columns.map(str)
    ordered_symbols = list(symbols)
    symbol_set = set(ordered_symbols)
    if set(covariance.index) != symbol_set or set(covariance.columns) != symbol_set:
        raise ValueError("External covariance CSV must contain exactly the configured symbols.")

    covariance = covariance.loc[ordered_symbols, ordered_symbols]
    covariance = covariance.apply(pd.to_numeric, errors="raise")
    if not np.isfinite(covariance.to_numpy(dtype=float)).all():
        raise ValueError("External covariance CSV must contain only finite numeric values.")
    return covariance.astype(float)


def estimate_covariance_matrix(
    returns: pd.DataFrame,
    *,
    method: str = "sample",
    external_path: Path | str | None = None,
) -> pd.DataFrame:
    ordered_symbols = list(returns.columns)
    if method not in COVARIANCE_ESTIMATORS:
        allowed = ", ".join(sorted(COVARIANCE_ESTIMATORS))
        raise ValueError(f"Unsupported covariance estimator: {method}. Expected one of: {allowed}")

    if method == "external_csv":
        if external_path is None:
            raise ValueError("external_path is required when method='external_csv'.")
        return load_external_covariance(external_path, symbols=ordered_symbols)

    if external_path is not None:
        raise ValueError("external_path must be omitted unless method='external_csv'.")

    if returns.empty:
        return pd.DataFrame(index=ordered_symbols, columns=ordered_symbols, dtype=float)

    if method == "sample":
        return returns.cov().loc[ordered_symbols, ordered_symbols].astype(float)

    if method == "ewma":
        values = returns.loc[:, ordered_symbols].to_numpy(dtype=float)
        row_count = len(values)
        weights = (1.0 - EWMA_LAMBDA) * (EWMA_LAMBDA ** np.arange(row_count - 1, -1, -1))
        weights = weights / weights.sum()
        mean = np.average(values, axis=0, weights=weights)
        centered = values - mean
        covariance = centered.T @ (centered * weights[:, None])
        return pd.DataFrame(covariance, index=ordered_symbols, columns=ordered_symbols, dtype=float)

    sample_covariance = returns.cov().loc[ordered_symbols, ordered_symbols].astype(float)
    diagonal_covariance = pd.DataFrame(
        np.diag(np.diag(sample_covariance.to_numpy(dtype=float))),
        index=ordered_symbols,
        columns=ordered_symbols,
        dtype=float,
    )
    return ((1.0 - DIAGONAL_SHRINKAGE_WEIGHT) * sample_covariance) + (
        DIAGONAL_SHRINKAGE_WEIGHT * diagonal_covariance
    )


def load_external_expected_returns(path: Path | str, *, symbols: list[str]) -> pd.Series:
    expected_returns = pd.read_csv(Path(path))
    expected_columns = ["symbol", "expected_return"]
    if list(expected_returns.columns) != expected_columns:
        raise ValueError(
            "External expected returns CSV must have exactly these columns: symbol, expected_return."
        )

    expected_returns["symbol"] = expected_returns["symbol"].astype(str)
    ordered_symbols = list(symbols)
    symbol_set = set(ordered_symbols)
    if set(expected_returns["symbol"]) != symbol_set:
        raise ValueError("External expected returns CSV must contain exactly the configured symbols.")
    if expected_returns["symbol"].duplicated().any():
        raise ValueError("External expected returns CSV must contain one row per symbol.")

    series = pd.to_numeric(
        expected_returns.set_index("symbol").loc[ordered_symbols, "expected_return"],
        errors="raise",
    )
    if not np.isfinite(series.to_numpy(dtype=float)).all():
        raise ValueError("External expected returns CSV must contain only finite numeric values.")
    series.name = "expected_return"
    return series.astype(float)


def estimate_expected_returns(
    returns: pd.DataFrame,
    *,
    source: str = "historical_mean",
    external_path: Path | str | None = None,
) -> pd.Series:
    ordered_symbols = list(returns.columns)
    if source not in EXPECTED_RETURN_SOURCES:
        allowed = ", ".join(sorted(EXPECTED_RETURN_SOURCES))
        raise ValueError(f"Unsupported expected return source: {source}. Expected one of: {allowed}")

    if source == "external_csv":
        if external_path is None:
            raise ValueError("external_path is required when source='external_csv'.")
        return load_external_expected_returns(external_path, symbols=ordered_symbols)

    if external_path is not None:
        raise ValueError("external_path must be omitted unless source='external_csv'.")

    return returns.loc[:, ordered_symbols].mean(axis=0).astype(float).rename("expected_return")


def build_optimizer_inputs(
    panel: pd.DataFrame,
    *,
    symbols: list[str],
    lookback_days: int,
    frequency: str = "W-FRI",
    covariance_estimator: str = "sample",
    external_covariance_path: Path | str | None = None,
    expected_return_source: str = "historical_mean",
    external_expected_returns_path: Path | str | None = None,
) -> list[OptimizerInput]:
    windows = build_optimizer_windows(
        panel,
        symbols=symbols,
        lookback_days=lookback_days,
        frequency=frequency,
    )
    if not windows:
        return []

    shared_covariance: pd.DataFrame | None = None
    if covariance_estimator == "external_csv":
        if external_covariance_path is None:
            raise ValueError("external_covariance_path is required when covariance_estimator='external_csv'.")
        shared_covariance = load_external_covariance(
            external_covariance_path,
            symbols=windows[0].symbols,
        )

    shared_expected_returns: pd.Series | None = None
    if expected_return_source == "external_csv":
        if external_expected_returns_path is None:
            raise ValueError(
                "external_expected_returns_path is required when expected_return_source='external_csv'."
            )
        shared_expected_returns = load_external_expected_returns(
            external_expected_returns_path,
            symbols=windows[0].symbols,
        )

    inputs: list[OptimizerInput] = []
    for window in windows:
        covariance = (
            shared_covariance.copy()
            if shared_covariance is not None
            else estimate_covariance_matrix(window.returns, method=covariance_estimator)
        )
        expected_returns = (
            shared_expected_returns.copy()
            if shared_expected_returns is not None
            else estimate_expected_returns(window.returns, source=expected_return_source)
        )
        inputs.append(
            OptimizerInput(
                signal_date=window.signal_date,
                effective_date=window.effective_date,
                symbols=window.symbols,
                returns=window.returns.copy(),
                covariance=covariance.loc[window.symbols, window.symbols].copy(),
                expected_returns=expected_returns.loc[window.symbols].copy(),
            )
        )

    return inputs

