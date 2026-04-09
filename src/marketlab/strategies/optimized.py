from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from marketlab.rebalance import signal_effective_dates

COVARIANCE_ESTIMATORS = {"diagonal_shrinkage", "ewma", "external_csv", "sample"}
EXPECTED_RETURN_SOURCES = {"external_csv", "historical_mean"}
EWMA_LAMBDA = 0.94
DIAGONAL_SHRINKAGE_WEIGHT = 0.10
COVARIANCE_REGULARIZATION = 1e-8
CONSTRAINT_TOLERANCE = 1e-6
WEIGHT_EPSILON = 1e-12
REQUIRED_PANEL_COLUMNS = {"adj_close", "symbol", "timestamp"}
WEIGHTS_COLUMNS = ["strategy", "effective_date", "symbol", "weight"]
MEAN_VARIANCE_STRATEGY_NAME = "mean_variance"


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


def _empty_weights_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=WEIGHTS_COLUMNS)


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
    ordered_symbols = list(symbols)

    shared_covariance: pd.DataFrame | None = None
    if covariance_estimator == "external_csv":
        if external_covariance_path is None:
            raise ValueError("external_covariance_path is required when covariance_estimator='external_csv'.")
        shared_covariance = load_external_covariance(
            external_covariance_path,
            symbols=ordered_symbols,
        )
    else:
        estimate_covariance_matrix(
            pd.DataFrame(columns=ordered_symbols, dtype=float),
            method=covariance_estimator,
            external_path=external_covariance_path,
        )

    shared_expected_returns: pd.Series | None = None
    if expected_return_source == "external_csv":
        if external_expected_returns_path is None:
            raise ValueError(
                "external_expected_returns_path is required when expected_return_source='external_csv'."
            )
        shared_expected_returns = load_external_expected_returns(
            external_expected_returns_path,
            symbols=ordered_symbols,
        )
    else:
        estimate_expected_returns(
            pd.DataFrame(columns=ordered_symbols, dtype=float),
            source=expected_return_source,
            external_path=external_expected_returns_path,
        )

    windows = build_optimizer_windows(
        panel,
        symbols=ordered_symbols,
        lookback_days=lookback_days,
        frequency=frequency,
    )
    if not windows:
        return []

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


def _unsupported_method_error(method: str) -> RuntimeError:
    return RuntimeError(f"baselines.optimized.method='{method}' is not implemented yet.")


def _validated_symbol_groups(
    symbols: list[str],
    symbol_groups: dict[str, str] | None,
    max_group_weight: float | None,
) -> dict[str, str]:
    if max_group_weight is None:
        return {}

    resolved_symbol_groups = symbol_groups or {}
    missing_group_symbols = sorted(set(symbols) - set(resolved_symbol_groups))
    if missing_group_symbols:
        joined = ", ".join(missing_group_symbols)
        raise ValueError(
            "Optimized max_group_weight requires symbol_groups for all symbols: "
            f"{joined}"
        )
    return {symbol: resolved_symbol_groups[symbol] for symbol in symbols}


def _group_indices(symbols: list[str], symbol_groups: dict[str, str]) -> dict[str, np.ndarray]:
    group_names = sorted(set(symbol_groups.values()))
    return {
        group_name: np.array(
            [index for index, symbol in enumerate(symbols) if symbol_groups[symbol] == group_name],
            dtype=int,
        )
        for group_name in group_names
    }


def _position_upper_bounds(
    *,
    symbols: list[str],
    achieved_target_exposure: float,
    max_position_weight: float | None,
    max_group_weight: float | None,
) -> np.ndarray:
    upper_bounds: list[float] = []
    for _symbol in symbols:
        upper_bound = achieved_target_exposure
        if max_position_weight is not None:
            upper_bound = min(upper_bound, max_position_weight)
        if max_group_weight is not None:
            upper_bound = min(upper_bound, max_group_weight)
        upper_bounds.append(float(upper_bound))
    return np.asarray(upper_bounds, dtype=float)


def _build_feasible_initializer(
    *,
    symbols: list[str],
    target_gross_exposure: float,
    max_position_weight: float | None,
    symbol_groups: dict[str, str] | None,
    max_group_weight: float | None,
) -> tuple[np.ndarray, float]:
    resolved_symbol_groups = _validated_symbol_groups(symbols, symbol_groups, max_group_weight)
    symbol_weights = {symbol: 0.0 for symbol in symbols}
    position_remaining = {
        symbol: min(target_gross_exposure, max_position_weight)
        if max_position_weight is not None
        else target_gross_exposure
        for symbol in symbols
    }
    group_remaining = {
        group_name: max_group_weight
        for group_name in sorted(set(resolved_symbol_groups.values()))
    }

    remaining_exposure = float(target_gross_exposure)
    while remaining_exposure > WEIGHT_EPSILON:
        eligible_symbols: list[str] = []
        for symbol in symbols:
            symbol_capacity = position_remaining[symbol]
            if max_group_weight is not None:
                symbol_capacity = min(symbol_capacity, group_remaining[resolved_symbol_groups[symbol]])
            if symbol_capacity > WEIGHT_EPSILON:
                eligible_symbols.append(symbol)

        if not eligible_symbols:
            break

        equal_slice = remaining_exposure / len(eligible_symbols)
        deployed_exposure = 0.0
        for symbol in eligible_symbols:
            capacity = position_remaining[symbol]
            if max_group_weight is not None:
                capacity = min(capacity, group_remaining[resolved_symbol_groups[symbol]])
            delta = min(equal_slice, capacity)
            if delta <= WEIGHT_EPSILON:
                continue
            symbol_weights[symbol] += delta
            position_remaining[symbol] -= delta
            if max_group_weight is not None:
                group_remaining[resolved_symbol_groups[symbol]] -= delta
            deployed_exposure += delta

        if deployed_exposure <= WEIGHT_EPSILON:
            break
        remaining_exposure -= deployed_exposure

    weight_vector = np.asarray([symbol_weights[symbol] for symbol in symbols], dtype=float)
    return weight_vector, float(weight_vector.sum())


def _optimizer_failure(
    *,
    method: str,
    signal_date: pd.Timestamp,
    effective_date: pd.Timestamp,
    message: str,
) -> RuntimeError:
    return RuntimeError(
        "Optimized baseline failed "
        f"for method='{method}' signal_date={signal_date.date()} "
        f"effective_date={effective_date.date()}: {message}"
    )


def _solve_mean_variance_weights(
    optimizer_input: OptimizerInput,
    *,
    target_gross_exposure: float,
    risk_aversion: float,
    max_position_weight: float | None,
    symbol_groups: dict[str, str] | None,
    max_group_weight: float | None,
) -> pd.Series:
    symbols = optimizer_input.symbols
    feasible_weights, achieved_target_exposure = _build_feasible_initializer(
        symbols=symbols,
        target_gross_exposure=target_gross_exposure,
        max_position_weight=max_position_weight,
        symbol_groups=symbol_groups,
        max_group_weight=max_group_weight,
    )
    if achieved_target_exposure <= WEIGHT_EPSILON:
        return pd.Series(0.0, index=symbols, dtype=float)

    if len(symbols) == 1:
        return pd.Series(feasible_weights, index=symbols, dtype=float)

    covariance = optimizer_input.covariance.loc[symbols, symbols].to_numpy(dtype=float)
    covariance = (covariance + covariance.T) / 2.0
    covariance += np.eye(len(symbols), dtype=float) * COVARIANCE_REGULARIZATION
    expected_returns = optimizer_input.expected_returns.loc[symbols].to_numpy(dtype=float)
    upper_bounds = _position_upper_bounds(
        symbols=symbols,
        achieved_target_exposure=achieved_target_exposure,
        max_position_weight=max_position_weight,
        max_group_weight=max_group_weight,
    )
    resolved_symbol_groups = _validated_symbol_groups(symbols, symbol_groups, max_group_weight)
    grouped_indices = _group_indices(symbols, resolved_symbol_groups)

    def objective(weights: np.ndarray) -> float:
        return float(0.5 * risk_aversion * (weights @ covariance @ weights) - (expected_returns @ weights))

    constraints: list[dict[str, object]] = [
        {
            "type": "eq",
            "fun": lambda weights: float(weights.sum() - achieved_target_exposure),
        }
    ]
    for indices in grouped_indices.values():
        constraints.append(
            {
                "type": "ineq",
                "fun": lambda weights, idx=indices: float(max_group_weight - weights[idx].sum()),
            }
        )

    result = minimize(
        objective,
        x0=feasible_weights,
        method="SLSQP",
        bounds=[(0.0, float(bound)) for bound in upper_bounds],
        constraints=constraints,
        options={"ftol": 1e-9, "maxiter": 500},
    )
    if not result.success:
        raise _optimizer_failure(
            method="mean_variance",
            signal_date=optimizer_input.signal_date,
            effective_date=optimizer_input.effective_date,
            message=result.message,
        )

    weights = np.asarray(result.x, dtype=float)
    if not np.isfinite(weights).all():
        raise _optimizer_failure(
            method="mean_variance",
            signal_date=optimizer_input.signal_date,
            effective_date=optimizer_input.effective_date,
            message="solver returned non-finite weights",
        )

    weights = np.clip(weights, 0.0, upper_bounds)
    if abs(float(weights.sum()) - achieved_target_exposure) > CONSTRAINT_TOLERANCE:
        raise _optimizer_failure(
            method="mean_variance",
            signal_date=optimizer_input.signal_date,
            effective_date=optimizer_input.effective_date,
            message="solver returned weights that violate the target exposure constraint",
        )
    if (weights > upper_bounds + CONSTRAINT_TOLERANCE).any():
        raise _optimizer_failure(
            method="mean_variance",
            signal_date=optimizer_input.signal_date,
            effective_date=optimizer_input.effective_date,
            message="solver returned weights that violate the position cap constraint",
        )
    for group_name, indices in grouped_indices.items():
        group_weight = float(weights[indices].sum())
        if group_weight > max_group_weight + CONSTRAINT_TOLERANCE:
            raise _optimizer_failure(
                method="mean_variance",
                signal_date=optimizer_input.signal_date,
                effective_date=optimizer_input.effective_date,
                message=(
                    "solver returned weights that violate the group cap constraint "
                    f"for group '{group_name}'"
                ),
            )

    return pd.Series(weights, index=symbols, dtype=float)


def _weights_to_frame(
    strategy_name: str,
    effective_date: pd.Timestamp,
    symbols: list[str],
    weights: pd.Series,
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "strategy": strategy_name,
            "effective_date": pd.Timestamp(effective_date),
            "symbol": symbols,
            "weight": [float(weights.loc[symbol]) for symbol in symbols],
        }
    )


def generate_weights(
    panel: pd.DataFrame,
    *,
    symbols: list[str],
    method: str,
    lookback_days: int,
    frequency: str = "W-FRI",
    covariance_estimator: str = "sample",
    external_covariance_path: Path | str | None = None,
    expected_return_source: str = "historical_mean",
    external_expected_returns_path: Path | str | None = None,
    long_only: bool = True,
    target_gross_exposure: float = 1.0,
    risk_aversion: float = 1.0,
    symbol_groups: dict[str, str] | None = None,
    max_position_weight: float | None = None,
    max_group_weight: float | None = None,
) -> pd.DataFrame:
    if method != "mean_variance":
        raise _unsupported_method_error(method)
    if not long_only:
        raise ValueError("Mean-variance optimized baseline currently supports long_only=True only.")
    if panel.empty:
        return _empty_weights_frame()

    optimizer_inputs = build_optimizer_inputs(
        panel,
        symbols=list(symbols),
        lookback_days=lookback_days,
        frequency=frequency,
        covariance_estimator=covariance_estimator,
        external_covariance_path=external_covariance_path,
        expected_return_source=expected_return_source,
        external_expected_returns_path=external_expected_returns_path,
    )
    if not optimizer_inputs:
        return _empty_weights_frame()

    weight_frames = []
    for optimizer_input in optimizer_inputs:
        weights = _solve_mean_variance_weights(
            optimizer_input,
            target_gross_exposure=target_gross_exposure,
            risk_aversion=risk_aversion,
            max_position_weight=max_position_weight,
            symbol_groups=symbol_groups,
            max_group_weight=max_group_weight,
        )
        weight_frames.append(
            _weights_to_frame(
                MEAN_VARIANCE_STRATEGY_NAME,
                optimizer_input.effective_date,
                optimizer_input.symbols,
                weights,
            )
        )

    return pd.concat(weight_frames, ignore_index=True).sort_values(
        ["effective_date", "symbol"]
    ).reset_index(drop=True)
