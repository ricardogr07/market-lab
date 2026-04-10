from __future__ import annotations

from collections.abc import Mapping
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
RISK_PARITY_STRATEGY_NAME = "risk_parity"
BLACK_LITTERMAN_STRATEGY_NAME = "black_litterman"
BLACK_LITTERMAN_ASSUMPTIONS_COLUMNS = [
    "signal_date",
    "effective_date",
    "symbol",
    "equilibrium_weight",
    "implied_prior_return",
    "posterior_expected_return",
    "tau",
]
EXECUTABLE_METHODS = frozenset(
    {MEAN_VARIANCE_STRATEGY_NAME, RISK_PARITY_STRATEGY_NAME, BLACK_LITTERMAN_STRATEGY_NAME}
)


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
    expected_returns: pd.Series | None = None


@dataclass(slots=True)
class LongOnlyProblem:
    symbols: list[str]
    feasible_weights: np.ndarray
    achieved_target_exposure: float
    covariance: np.ndarray
    upper_bounds: np.ndarray
    grouped_indices: dict[str, np.ndarray]


@dataclass(slots=True)
class BlackLittermanOutput:
    weights: pd.DataFrame
    assumptions: pd.DataFrame


@dataclass(slots=True)
class CovarianceDiagnosticWindow:
    strategy: str
    signal_date: pd.Timestamp
    effective_date: pd.Timestamp
    symbols: list[str]
    covariance: pd.DataFrame


def _empty_weights_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=WEIGHTS_COLUMNS)


def is_executable_method(method: str) -> bool:
    return method in EXECUTABLE_METHODS


def strategy_name_for_method(method: str) -> str:
    if method == "mean_variance":
        return MEAN_VARIANCE_STRATEGY_NAME
    if method == "risk_parity":
        return RISK_PARITY_STRATEGY_NAME
    if method == "black_litterman":
        return BLACK_LITTERMAN_STRATEGY_NAME
    raise _unsupported_method_error(method)


def generate_cash_only_weights(
    method: str,
    *,
    effective_date: pd.Timestamp,
    symbols: list[str],
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "strategy": strategy_name_for_method(method),
            "effective_date": pd.Timestamp(effective_date),
            "symbol": symbols,
            "weight": [0.0] * len(symbols),
        }
    )


def _black_litterman_view_field(view: object, field: str) -> object:
    if isinstance(view, dict):
        return view.get(field)
    return getattr(view, field, None)


def _validated_black_litterman_inputs(
    *,
    symbols: list[str],
    equilibrium_weights: dict[str, float] | None,
    tau: float,
    views: list[object] | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    symbol_set = set(symbols)
    symbol_positions = {symbol: index for index, symbol in enumerate(symbols)}
    if not isinstance(equilibrium_weights, Mapping) or set(equilibrium_weights) != symbol_set:
        raise ValueError(
            "baselines.optimized.equilibrium_weights must match data.symbols exactly when "
            "baselines.optimized.method='black_litterman'."
        )

    try:
        equilibrium_vector = np.asarray(
            [float(equilibrium_weights[symbol]) for symbol in symbols],
            dtype=float,
        )
    except (TypeError, ValueError):
        raise ValueError("baselines.optimized.equilibrium_weights must contain only finite numeric values.")

    if not np.isfinite(equilibrium_vector).all():
        raise ValueError("baselines.optimized.equilibrium_weights must contain only finite numeric values.")
    if (equilibrium_vector < 0.0).any():
        raise ValueError("baselines.optimized.equilibrium_weights must contain non-negative weights.")
    if abs(float(equilibrium_vector.sum()) - 1.0) > CONSTRAINT_TOLERANCE:
        raise ValueError("baselines.optimized.equilibrium_weights must sum to 1.0.")
    if not np.isfinite(tau) or tau <= 0.0:
        raise ValueError("baselines.optimized.tau must be a finite positive value.")
    if not views:
        raise ValueError(
            "baselines.optimized.views must be non-empty when "
            "baselines.optimized.method='black_litterman'."
        )

    view_rows: list[np.ndarray] = []
    view_returns: list[float] = []
    for index, view in enumerate(views):
        label = f"baselines.optimized.views[{index}]"
        view_name = _black_litterman_view_field(view, "name")
        if not view_name:
            raise ValueError(f"{label}.name must be non-empty.")

        view_weights = _black_litterman_view_field(view, "weights")
        if not isinstance(view_weights, Mapping):
            raise ValueError(f"{label}.weights must be a mapping of symbol to coefficient.")

        view_weights = dict(view_weights)
        unknown_view_symbols = sorted(set(view_weights) - symbol_set)
        if unknown_view_symbols:
            joined = ", ".join(unknown_view_symbols)
            raise ValueError(f"{label}.weights contains unknown symbols: {joined}")
        if not view_weights:
            raise ValueError(f"{label}.weights must not be empty.")

        view_row = np.zeros(len(symbols), dtype=float)
        non_zero_weights = 0
        for symbol, coefficient in view_weights.items():
            try:
                coefficient_value = float(coefficient)
            except (TypeError, ValueError):
                raise ValueError(f"{label}.weights[{symbol}] must be finite.")
            if not np.isfinite(coefficient_value):
                raise ValueError(f"{label}.weights[{symbol}] must be finite.")
            view_row[symbol_positions[symbol]] = coefficient_value
            if abs(coefficient_value) > WEIGHT_EPSILON:
                non_zero_weights += 1

        if non_zero_weights == 0:
            raise ValueError(f"{label}.weights must contain at least one non-zero coefficient.")

        view_return = _black_litterman_view_field(view, "view_return")
        if view_return is None:
            raise ValueError(f"{label}.view_return must be finite.")
        try:
            view_return_value = float(view_return)
        except (TypeError, ValueError):
            raise ValueError(f"{label}.view_return must be finite.")
        if not np.isfinite(view_return_value):
            raise ValueError(f"{label}.view_return must be finite.")

        view_rows.append(view_row)
        view_returns.append(view_return_value)

    return (
        equilibrium_vector,
        np.asarray(view_rows, dtype=float),
        np.asarray(view_returns, dtype=float),
        float(tau),
    )


def _black_litterman_expected_returns(
    covariance: np.ndarray,
    *,
    symbols: list[str],
    risk_aversion: float,
    equilibrium_weights: np.ndarray,
    tau: float,
    view_matrix: np.ndarray,
    view_returns: np.ndarray,
) -> pd.Series:
    covariance = np.asarray(covariance, dtype=float)
    if not np.isfinite(covariance).all():
        raise ValueError("Black-Litterman covariance contains non-finite values.")

    tau_covariance = tau * covariance
    implied_returns = risk_aversion * (covariance @ equilibrium_weights)
    projected_variance = view_matrix @ tau_covariance @ view_matrix.T
    omega_diagonal = np.diag(projected_variance).astype(float)
    omega_diagonal = np.maximum(omega_diagonal, COVARIANCE_REGULARIZATION)
    omega = np.diag(omega_diagonal)
    middle = projected_variance + omega
    rhs = view_returns - (view_matrix @ implied_returns)

    try:
        adjustment = np.linalg.solve(middle, rhs)
    except np.linalg.LinAlgError:
        adjustment = np.linalg.pinv(middle) @ rhs

    posterior_expected_returns = implied_returns + (tau_covariance @ view_matrix.T @ adjustment)
    if not np.isfinite(posterior_expected_returns).all():
        raise ValueError("Black-Litterman posterior expected returns contain non-finite values.")

    return (
        pd.Series(implied_returns, index=symbols, dtype=float, name="implied_prior_return"),
        pd.Series(
            posterior_expected_returns,
            index=symbols,
            dtype=float,
            name="expected_return",
        ),
    )


def _black_litterman_assumptions_frame(
    optimizer_input: OptimizerInput,
    *,
    equilibrium_weights: pd.Series,
    prior_returns: pd.Series,
    posterior_expected_returns: pd.Series,
    tau: float,
) -> pd.DataFrame:
    rows = []
    for symbol in optimizer_input.symbols:
        rows.append(
            {
                "signal_date": pd.Timestamp(optimizer_input.signal_date),
                "effective_date": pd.Timestamp(optimizer_input.effective_date),
                "symbol": symbol,
                "equilibrium_weight": float(equilibrium_weights.loc[symbol]),
                "implied_prior_return": float(prior_returns.loc[symbol]),
                "posterior_expected_return": float(posterior_expected_returns.loc[symbol]),
                "tau": float(tau),
            }
        )
    return pd.DataFrame(rows, columns=BLACK_LITTERMAN_ASSUMPTIONS_COLUMNS)


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


def build_covariance_inputs(
    panel: pd.DataFrame,
    *,
    symbols: list[str],
    lookback_days: int,
    frequency: str = "W-FRI",
    covariance_estimator: str = "sample",
    external_covariance_path: Path | str | None = None,
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
        inputs.append(
            OptimizerInput(
                signal_date=window.signal_date,
                effective_date=window.effective_date,
                symbols=window.symbols,
                returns=window.returns.copy(),
                covariance=covariance.loc[window.symbols, window.symbols].copy(),
            )
        )

    return inputs


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

    covariance_inputs = build_covariance_inputs(
        panel,
        symbols=ordered_symbols,
        lookback_days=lookback_days,
        frequency=frequency,
        covariance_estimator=covariance_estimator,
        external_covariance_path=external_covariance_path,
    )
    if not covariance_inputs:
        return []

    inputs: list[OptimizerInput] = []
    for optimizer_input in covariance_inputs:
        expected_returns = (
            shared_expected_returns.copy()
            if shared_expected_returns is not None
            else estimate_expected_returns(
                optimizer_input.returns,
                source=expected_return_source,
            )
        )
        inputs.append(
            OptimizerInput(
                signal_date=optimizer_input.signal_date,
                effective_date=optimizer_input.effective_date,
                symbols=optimizer_input.symbols,
                returns=optimizer_input.returns.copy(),
                covariance=optimizer_input.covariance.copy(),
                expected_returns=expected_returns.loc[optimizer_input.symbols].copy(),
            )
        )

    return inputs


def generate_black_litterman_output(
    panel: pd.DataFrame,
    *,
    symbols: list[str],
    lookback_days: int,
    frequency: str = "W-FRI",
    covariance_estimator: str = "sample",
    external_covariance_path: Path | str | None = None,
    long_only: bool = True,
    target_gross_exposure: float = 1.0,
    risk_aversion: float = 1.0,
    symbol_groups: dict[str, str] | None = None,
    max_position_weight: float | None = None,
    max_group_weight: float | None = None,
    equilibrium_weights: dict[str, float] | None = None,
    tau: float = 0.05,
    views: list[object] | None = None,
) -> BlackLittermanOutput:
    if not long_only:
        raise ValueError(
            f"{strategy_name_for_method(BLACK_LITTERMAN_STRATEGY_NAME)} optimized baseline currently supports long_only=True only."
        )
    if panel.empty:
        return BlackLittermanOutput(
            weights=_empty_weights_frame(),
            assumptions=pd.DataFrame(columns=BLACK_LITTERMAN_ASSUMPTIONS_COLUMNS),
        )

    (
        bl_equilibrium_weights,
        bl_view_matrix,
        bl_view_returns,
        bl_tau,
    ) = _validated_black_litterman_inputs(
        symbols=list(symbols),
        equilibrium_weights=equilibrium_weights,
        tau=tau,
        views=views,
    )
    equilibrium_weight_series = pd.Series(
        bl_equilibrium_weights,
        index=list(symbols),
        dtype=float,
        name="equilibrium_weight",
    )

    optimizer_inputs = build_covariance_inputs(
        panel,
        symbols=list(symbols),
        lookback_days=lookback_days,
        frequency=frequency,
        covariance_estimator=covariance_estimator,
        external_covariance_path=external_covariance_path,
    )
    if not optimizer_inputs:
        return BlackLittermanOutput(
            weights=_empty_weights_frame(),
            assumptions=pd.DataFrame(columns=BLACK_LITTERMAN_ASSUMPTIONS_COLUMNS),
        )

    weight_frames: list[pd.DataFrame] = []
    assumptions_frames: list[pd.DataFrame] = []
    for optimizer_input in optimizer_inputs:
        problem = _prepare_long_only_problem(
            optimizer_input,
            method=BLACK_LITTERMAN_STRATEGY_NAME,
            target_gross_exposure=target_gross_exposure,
            max_position_weight=max_position_weight,
            symbol_groups=symbol_groups,
            max_group_weight=max_group_weight,
        )
        prior_returns, posterior_expected_returns = _black_litterman_expected_returns(
            problem.covariance,
            symbols=problem.symbols,
            risk_aversion=risk_aversion,
            equilibrium_weights=bl_equilibrium_weights,
            tau=bl_tau,
            view_matrix=bl_view_matrix,
            view_returns=bl_view_returns,
        )
        weights = _solve_long_only_expected_return_weights(
            optimizer_input,
            method=BLACK_LITTERMAN_STRATEGY_NAME,
            problem=problem,
            expected_returns=posterior_expected_returns,
            risk_aversion=risk_aversion,
            max_position_weight=max_position_weight,
            max_group_weight=max_group_weight,
        )
        weight_frames.append(
            _weights_to_frame(
                BLACK_LITTERMAN_STRATEGY_NAME,
                optimizer_input.effective_date,
                optimizer_input.symbols,
                weights,
            )
        )
        assumptions_frames.append(
            _black_litterman_assumptions_frame(
                optimizer_input,
                equilibrium_weights=equilibrium_weight_series,
                prior_returns=prior_returns,
                posterior_expected_returns=posterior_expected_returns,
                tau=bl_tau,
            )
        )

    return BlackLittermanOutput(
        weights=pd.concat(weight_frames, ignore_index=True)
        .sort_values(["effective_date", "symbol"])
        .reset_index(drop=True),
        assumptions=pd.concat(assumptions_frames, ignore_index=True)
        .sort_values(["signal_date", "effective_date", "symbol"])
        .reset_index(drop=True),
    )


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


def _regularized_covariance_matrix(
    optimizer_input: OptimizerInput,
    *,
    method: str,
) -> np.ndarray:
    covariance = optimizer_input.covariance.loc[
        optimizer_input.symbols,
        optimizer_input.symbols,
    ].to_numpy(dtype=float)
    if not np.isfinite(covariance).all():
        raise _optimizer_failure(
            method=method,
            signal_date=optimizer_input.signal_date,
            effective_date=optimizer_input.effective_date,
            message="covariance contains non-finite values",
        )
    covariance = (covariance + covariance.T) / 2.0
    covariance += np.eye(len(optimizer_input.symbols), dtype=float) * COVARIANCE_REGULARIZATION
    if (np.diag(covariance) <= 0.0).any():
        raise _optimizer_failure(
            method=method,
            signal_date=optimizer_input.signal_date,
            effective_date=optimizer_input.effective_date,
            message="covariance contains non-positive diagonal entries after regularization",
        )
    return covariance


def _prepare_long_only_problem(
    optimizer_input: OptimizerInput,
    *,
    method: str,
    target_gross_exposure: float,
    max_position_weight: float | None,
    symbol_groups: dict[str, str] | None,
    max_group_weight: float | None,
) -> LongOnlyProblem:
    symbols = optimizer_input.symbols
    feasible_weights, achieved_target_exposure = _build_feasible_initializer(
        symbols=symbols,
        target_gross_exposure=target_gross_exposure,
        max_position_weight=max_position_weight,
        symbol_groups=symbol_groups,
        max_group_weight=max_group_weight,
    )
    covariance = _regularized_covariance_matrix(
        optimizer_input,
        method=method,
    )
    upper_bounds = _position_upper_bounds(
        symbols=symbols,
        achieved_target_exposure=achieved_target_exposure,
        max_position_weight=max_position_weight,
        max_group_weight=max_group_weight,
    )
    resolved_symbol_groups = _validated_symbol_groups(
        symbols,
        symbol_groups,
        max_group_weight,
    )
    grouped_indices = _group_indices(symbols, resolved_symbol_groups)
    return LongOnlyProblem(
        symbols=symbols,
        feasible_weights=feasible_weights,
        achieved_target_exposure=achieved_target_exposure,
        covariance=covariance,
        upper_bounds=upper_bounds,
        grouped_indices=grouped_indices,
    )


def _constraints(
    *,
    achieved_target_exposure: float,
    grouped_indices: dict[str, np.ndarray],
    max_group_weight: float | None,
) -> list[dict[str, object]]:
    constraints: list[dict[str, object]] = [
        {
            "type": "eq",
            "fun": lambda weights, target=achieved_target_exposure: float(
                weights.sum() - target
            ),
        }
    ]
    if max_group_weight is None:
        return constraints

    for indices in grouped_indices.values():
        constraints.append(
            {
                "type": "ineq",
                "fun": lambda weights, idx=indices, cap=max_group_weight: float(
                    cap - weights[idx].sum()
                ),
            }
        )
    return constraints


def _finalize_optimized_weights(
    *,
    method: str,
    optimizer_input: OptimizerInput,
    problem: LongOnlyProblem,
    weights: np.ndarray,
    max_group_weight: float | None,
    ) -> pd.Series:
    if not np.isfinite(weights).all():
        raise _optimizer_failure(
            method=method,
            signal_date=optimizer_input.signal_date,
            effective_date=optimizer_input.effective_date,
            message="solver returned non-finite weights",
        )

    weights = np.clip(np.asarray(weights, dtype=float), 0.0, problem.upper_bounds)
    if abs(float(weights.sum()) - problem.achieved_target_exposure) > CONSTRAINT_TOLERANCE:
        raise _optimizer_failure(
            method=method,
            signal_date=optimizer_input.signal_date,
            effective_date=optimizer_input.effective_date,
            message="solver returned weights that violate the target exposure constraint",
        )
    if (weights > problem.upper_bounds + CONSTRAINT_TOLERANCE).any():
        raise _optimizer_failure(
            method=method,
            signal_date=optimizer_input.signal_date,
            effective_date=optimizer_input.effective_date,
            message="solver returned weights that violate the position cap constraint",
        )
    if (weights < -CONSTRAINT_TOLERANCE).any():
        raise _optimizer_failure(
            method=method,
            signal_date=optimizer_input.signal_date,
            effective_date=optimizer_input.effective_date,
            message="solver returned weights that violate the long-only constraint",
        )
    if max_group_weight is not None:
        for group_name, indices in problem.grouped_indices.items():
            group_weight = float(weights[indices].sum())
            if group_weight > max_group_weight + CONSTRAINT_TOLERANCE:
                raise _optimizer_failure(
                    method=method,
                    signal_date=optimizer_input.signal_date,
                    effective_date=optimizer_input.effective_date,
                    message=(
                        "solver returned weights that violate the group cap constraint "
                        f"for group '{group_name}'"
                    ),
                )
    return pd.Series(weights, index=problem.symbols, dtype=float)


def _solve_long_only_expected_return_weights(
    optimizer_input: OptimizerInput,
    *,
    method: str,
    problem: LongOnlyProblem,
    expected_returns: pd.Series,
    risk_aversion: float,
    max_position_weight: float | None,
    max_group_weight: float | None,
) -> pd.Series:
    if problem.achieved_target_exposure <= WEIGHT_EPSILON:
        return pd.Series(0.0, index=problem.symbols, dtype=float)

    if len(problem.symbols) == 1:
        return pd.Series(problem.feasible_weights, index=problem.symbols, dtype=float)

    expected_returns_array = expected_returns.loc[problem.symbols].to_numpy(dtype=float)
    if not np.isfinite(expected_returns_array).all():
        raise _optimizer_failure(
            method=method,
            signal_date=optimizer_input.signal_date,
            effective_date=optimizer_input.effective_date,
            message="expected returns contain non-finite values",
        )

    def objective(weights: np.ndarray) -> float:
        return float(
            0.5 * risk_aversion * (weights @ problem.covariance @ weights)
            - (expected_returns_array @ weights)
        )

    result = minimize(
        objective,
        x0=problem.feasible_weights,
        method="SLSQP",
        bounds=[(0.0, float(bound)) for bound in problem.upper_bounds],
        constraints=_constraints(
            achieved_target_exposure=problem.achieved_target_exposure,
            grouped_indices=problem.grouped_indices,
            max_group_weight=max_group_weight,
        ),
        options={"ftol": 1e-9, "maxiter": 500},
    )
    if not result.success:
        raise _optimizer_failure(
            method=method,
            signal_date=optimizer_input.signal_date,
            effective_date=optimizer_input.effective_date,
            message=result.message,
        )

    return _finalize_optimized_weights(
        method=method,
        optimizer_input=optimizer_input,
        problem=problem,
        weights=np.asarray(result.x, dtype=float),
        max_group_weight=max_group_weight,
    )


def _solve_mean_variance_weights(
    optimizer_input: OptimizerInput,
    *,
    target_gross_exposure: float,
    risk_aversion: float,
    max_position_weight: float | None,
    symbol_groups: dict[str, str] | None,
    max_group_weight: float | None,
    method: str = "mean_variance",
) -> pd.Series:
    if optimizer_input.expected_returns is None:
        raise ValueError("Mean-variance optimized inputs require expected_returns.")

    problem = _prepare_long_only_problem(
        optimizer_input,
        method="mean_variance",
        target_gross_exposure=target_gross_exposure,
        max_position_weight=max_position_weight,
        symbol_groups=symbol_groups,
        max_group_weight=max_group_weight,
    )
    return _solve_long_only_expected_return_weights(
        optimizer_input,
        method=method,
        problem=problem,
        expected_returns=optimizer_input.expected_returns,
        risk_aversion=risk_aversion,
        max_position_weight=max_position_weight,
        max_group_weight=max_group_weight,
    )


def _solve_risk_parity_weights(
    optimizer_input: OptimizerInput,
    *,
    target_gross_exposure: float,
    max_position_weight: float | None,
    symbol_groups: dict[str, str] | None,
    max_group_weight: float | None,
) -> pd.Series:
    problem = _prepare_long_only_problem(
        optimizer_input,
        method="risk_parity",
        target_gross_exposure=target_gross_exposure,
        max_position_weight=max_position_weight,
        symbol_groups=symbol_groups,
        max_group_weight=max_group_weight,
    )
    if problem.achieved_target_exposure <= WEIGHT_EPSILON:
        return pd.Series(0.0, index=problem.symbols, dtype=float)

    if len(problem.symbols) == 1:
        return pd.Series(problem.feasible_weights, index=problem.symbols, dtype=float)

    target_share = 1.0 / len(problem.symbols)

    def objective(weights: np.ndarray) -> float:
        portfolio_variance = float(weights @ problem.covariance @ weights)
        if portfolio_variance <= 0.0:
            raise _optimizer_failure(
                method="risk_parity",
                signal_date=optimizer_input.signal_date,
                effective_date=optimizer_input.effective_date,
                message="covariance produced non-positive portfolio variance during optimization",
            )
        marginal_contributions = problem.covariance @ weights
        contribution_shares = (weights * marginal_contributions) / portfolio_variance
        return float(np.square(contribution_shares - target_share).sum())

    result = minimize(
        objective,
        x0=problem.feasible_weights,
        method="SLSQP",
        bounds=[(0.0, float(bound)) for bound in problem.upper_bounds],
        constraints=_constraints(
            achieved_target_exposure=problem.achieved_target_exposure,
            grouped_indices=problem.grouped_indices,
            max_group_weight=max_group_weight,
        ),
        options={"ftol": 1e-9, "maxiter": 500},
    )
    if not result.success:
        raise _optimizer_failure(
            method="risk_parity",
            signal_date=optimizer_input.signal_date,
            effective_date=optimizer_input.effective_date,
            message=result.message,
        )

    weights = _finalize_optimized_weights(
        method="risk_parity",
        optimizer_input=optimizer_input,
        problem=problem,
        weights=np.asarray(result.x, dtype=float),
        max_group_weight=max_group_weight,
    )
    portfolio_variance = float(
        weights.to_numpy(dtype=float) @ problem.covariance @ weights.to_numpy(dtype=float)
    )
    if portfolio_variance <= 0.0:
        raise _optimizer_failure(
            method="risk_parity",
            signal_date=optimizer_input.signal_date,
            effective_date=optimizer_input.effective_date,
            message="covariance produced non-positive portfolio variance for the final solution",
        )
    return weights


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
    equilibrium_weights: dict[str, float] | None = None,
    tau: float = 0.05,
    views: list[object] | None = None,
) -> pd.DataFrame:
    if method not in EXECUTABLE_METHODS:
        raise _unsupported_method_error(method)
    if not long_only:
        raise ValueError(
            f"{strategy_name_for_method(method)} optimized baseline currently supports long_only=True only."
        )
    if method == BLACK_LITTERMAN_STRATEGY_NAME:
        if expected_return_source != "historical_mean":
            raise ValueError(
                "Black-Litterman optimized baseline does not use external expected returns; "
                "expected_return_source must remain 'historical_mean'."
            )
        if external_expected_returns_path is not None:
            raise ValueError(
                "Black-Litterman optimized baseline does not use external expected returns; "
                "external_expected_returns_path must be omitted."
            )
        _validated_black_litterman_inputs(
            symbols=list(symbols),
            equilibrium_weights=equilibrium_weights,
            tau=tau,
            views=views,
        )
    if method == "risk_parity":
        if expected_return_source != "historical_mean":
            raise ValueError(
                "Risk-parity optimized baseline does not use expected returns; "
                "expected_return_source must remain 'historical_mean'."
            )
        if external_expected_returns_path is not None:
            raise ValueError(
                "Risk-parity optimized baseline does not use expected returns; "
                "external_expected_returns_path must be omitted."
            )
    if panel.empty:
        return _empty_weights_frame()
    if method == BLACK_LITTERMAN_STRATEGY_NAME:
        return generate_black_litterman_output(
            panel,
            symbols=list(symbols),
            lookback_days=lookback_days,
            frequency=frequency,
            covariance_estimator=covariance_estimator,
            external_covariance_path=external_covariance_path,
            long_only=long_only,
            target_gross_exposure=target_gross_exposure,
            risk_aversion=risk_aversion,
            symbol_groups=symbol_groups,
            max_position_weight=max_position_weight,
            max_group_weight=max_group_weight,
            equilibrium_weights=equilibrium_weights,
            tau=tau,
            views=views,
        ).weights

    strategy_name = strategy_name_for_method(method)
    if method == "mean_variance":
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
    else:
        optimizer_inputs = build_covariance_inputs(
            panel,
            symbols=list(symbols),
            lookback_days=lookback_days,
            frequency=frequency,
            covariance_estimator=covariance_estimator,
            external_covariance_path=external_covariance_path,
        )
    if not optimizer_inputs:
        return _empty_weights_frame()

    weight_frames = []
    for optimizer_input in optimizer_inputs:
        if method == "mean_variance":
            weights = _solve_mean_variance_weights(
                optimizer_input,
                target_gross_exposure=target_gross_exposure,
                risk_aversion=risk_aversion,
                max_position_weight=max_position_weight,
                symbol_groups=symbol_groups,
                max_group_weight=max_group_weight,
                method="mean_variance",
            )
        else:
            weights = _solve_risk_parity_weights(
                optimizer_input,
                target_gross_exposure=target_gross_exposure,
                max_position_weight=max_position_weight,
                symbol_groups=symbol_groups,
                max_group_weight=max_group_weight,
            )
        weight_frames.append(
            _weights_to_frame(
                strategy_name,
                optimizer_input.effective_date,
                optimizer_input.symbols,
                weights,
            )
        )

    return pd.concat(weight_frames, ignore_index=True).sort_values(["effective_date", "symbol"]).reset_index(
        drop=True
    )


def generate_covariance_diagnostic_windows(
    panel: pd.DataFrame,
    *,
    symbols: list[str],
    method: str,
    lookback_days: int,
    frequency: str = "W-FRI",
    covariance_estimator: str = "sample",
    external_covariance_path: Path | str | None = None,
) -> list[CovarianceDiagnosticWindow]:
    if method not in EXECUTABLE_METHODS:
        raise _unsupported_method_error(method)
    if panel.empty:
        return []

    optimizer_inputs = build_covariance_inputs(
        panel,
        symbols=list(symbols),
        lookback_days=lookback_days,
        frequency=frequency,
        covariance_estimator=covariance_estimator,
        external_covariance_path=external_covariance_path,
    )
    if not optimizer_inputs:
        return []

    strategy_name = strategy_name_for_method(method)
    windows: list[CovarianceDiagnosticWindow] = []
    for optimizer_input in optimizer_inputs:
        covariance = _regularized_covariance_matrix(
            optimizer_input,
            method=method,
        )
        windows.append(
            CovarianceDiagnosticWindow(
                strategy=strategy_name,
                signal_date=pd.Timestamp(optimizer_input.signal_date),
                effective_date=pd.Timestamp(optimizer_input.effective_date),
                symbols=list(optimizer_input.symbols),
                covariance=pd.DataFrame(
                    covariance,
                    index=optimizer_input.symbols,
                    columns=optimizer_input.symbols,
                    dtype=float,
                ),
            )
        )
    return windows
