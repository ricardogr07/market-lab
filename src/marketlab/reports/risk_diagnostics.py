from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pandas as pd

from marketlab.strategies.optimized import CovarianceDiagnosticWindow

FACTOR_DIAGNOSTICS_COLUMNS = [
    "strategy",
    "start_date",
    "end_date",
    "observations",
    "factor",
    "beta_like_exposure",
    "mean_factor_return",
    "mean_factor_contribution",
    "alpha_like_intercept",
    "mean_strategy_return",
    "modeled_mean_return",
    "r_squared",
]
FACTOR_SUMMARY_COLUMNS = [
    "strategy",
    "start_date",
    "end_date",
    "observations",
    "alpha_like_intercept",
    "total_mean_factor_contribution",
    "mean_strategy_return",
    "modeled_mean_return",
    "r_squared",
]
COVARIANCE_DIAGNOSTICS_COLUMNS = [
    "strategy",
    "signal_date",
    "effective_date",
    "row_symbol",
    "column_symbol",
    "covariance",
    "correlation",
]
COVARIANCE_SUMMARY_COLUMNS = [
    "strategy",
    "rebalance_windows",
    "avg_variance",
    "avg_pairwise_correlation",
    "max_pairwise_correlation",
    "min_eigenvalue",
    "worst_condition_number",
]
FACTOR_REGRESSION_EPSILON = 1e-12


def _empty_factor_diagnostics() -> pd.DataFrame:
    return pd.DataFrame(columns=FACTOR_DIAGNOSTICS_COLUMNS)


def _empty_factor_summary() -> pd.DataFrame:
    return pd.DataFrame(columns=FACTOR_SUMMARY_COLUMNS)


def _empty_covariance_diagnostics() -> pd.DataFrame:
    return pd.DataFrame(columns=COVARIANCE_DIAGNOSTICS_COLUMNS)


def _empty_covariance_summary() -> pd.DataFrame:
    return pd.DataFrame(columns=COVARIANCE_SUMMARY_COLUMNS)


def _factor_header(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle)
        try:
            header = next(reader)
        except StopIteration as exc:
            raise ValueError("Factor model CSV must contain a header row.") from exc
    return [str(value) for value in header]


def load_factor_returns(path: Path | str) -> pd.DataFrame:
    factor_path = Path(path)
    header = _factor_header(factor_path)
    if not header:
        raise ValueError("Factor model CSV must contain a header row.")
    if len(header) != len(set(header)):
        raise ValueError("Factor model CSV must not contain duplicate column names.")
    if "date" not in header:
        raise ValueError("Factor model CSV must contain a 'date' column.")

    factor_columns = [column for column in header if column != "date"]
    if not factor_columns:
        raise ValueError("Factor model CSV must contain at least one factor column.")

    frame = pd.read_csv(factor_path)
    frame.columns = [str(column) for column in frame.columns]
    frame["date"] = pd.to_datetime(frame["date"], errors="raise")
    if frame["date"].isna().any():
        raise ValueError("Factor model CSV must contain only valid dates.")
    if frame["date"].duplicated().any():
        raise ValueError("Factor model CSV must not contain duplicate dates.")

    for column in factor_columns:
        numeric = pd.to_numeric(frame[column], errors="raise")
        if not np.isfinite(numeric.to_numpy(dtype=float)).all():
            raise ValueError("Factor model CSV must contain only finite numeric factor values.")
        frame.loc[:, column] = numeric.astype(float)

    if frame.loc[:, factor_columns].isna().any().any():
        raise ValueError("Factor model CSV must not contain missing factor values.")

    return (
        frame.loc[:, ["date", *factor_columns]]
        .sort_values("date")
        .reset_index(drop=True)
    )


def build_factor_diagnostics(
    performance: pd.DataFrame,
    factor_returns: pd.DataFrame,
) -> pd.DataFrame:
    if performance.empty:
        return _empty_factor_diagnostics()
    if factor_returns.empty:
        raise ValueError("Factor model CSV must contain at least one factor row.")

    factor_columns = [column for column in factor_returns.columns if column != "date"]
    factor_frame = factor_returns.copy()
    factor_frame["date"] = pd.to_datetime(factor_frame["date"])

    rows: list[dict[str, object]] = []
    for strategy, strategy_rows in performance.groupby("strategy", sort=True):
        aligned = strategy_rows.loc[:, ["date", "net_return"]].copy()
        aligned["date"] = pd.to_datetime(aligned["date"])
        aligned = aligned.merge(factor_frame, on="date", how="inner")
        if aligned.empty:
            raise ValueError(
                f"Factor model CSV has no overlapping dates with strategy '{strategy}'."
            )

        design_factors = aligned.loc[:, factor_columns].to_numpy(dtype=float)
        design_matrix = np.column_stack([np.ones(len(aligned), dtype=float), design_factors])
        if design_matrix.shape[0] < design_matrix.shape[1]:
            raise ValueError(
                f"Factor model regression is underdetermined for strategy '{strategy}'."
            )
        if np.linalg.matrix_rank(design_matrix) < design_matrix.shape[1]:
            raise ValueError(
                f"Factor model regression is rank-deficient for strategy '{strategy}'."
            )

        response = aligned["net_return"].to_numpy(dtype=float)
        coefficients, *_ = np.linalg.lstsq(design_matrix, response, rcond=None)
        intercept = float(coefficients[0])
        betas = coefficients[1:]
        modeled_returns = design_matrix @ coefficients
        mean_strategy_return = float(response.mean())
        modeled_mean_return = float(modeled_returns.mean())
        residuals = response - modeled_returns
        total_sum_squares = float(np.square(response - mean_strategy_return).sum())
        residual_sum_squares = float(np.square(residuals).sum())
        if abs(total_sum_squares) <= FACTOR_REGRESSION_EPSILON:
            r_squared = float("nan")
        else:
            r_squared = float(1.0 - (residual_sum_squares / total_sum_squares))

        mean_factor_returns = aligned.loc[:, factor_columns].mean()
        start_date = pd.Timestamp(aligned["date"].min())
        end_date = pd.Timestamp(aligned["date"].max())
        observations = int(len(aligned))
        for index, factor in enumerate(factor_columns):
            beta = float(betas[index])
            mean_factor_return = float(mean_factor_returns.loc[factor])
            rows.append(
                {
                    "strategy": strategy,
                    "start_date": start_date,
                    "end_date": end_date,
                    "observations": observations,
                    "factor": factor,
                    "beta_like_exposure": beta,
                    "mean_factor_return": mean_factor_return,
                    "mean_factor_contribution": float(beta * mean_factor_return),
                    "alpha_like_intercept": intercept,
                    "mean_strategy_return": mean_strategy_return,
                    "modeled_mean_return": modeled_mean_return,
                    "r_squared": r_squared,
                }
            )

    diagnostics = pd.DataFrame(rows, columns=FACTOR_DIAGNOSTICS_COLUMNS)
    if diagnostics.empty:
        return _empty_factor_diagnostics()
    return diagnostics.sort_values(["strategy", "factor"]).reset_index(drop=True)


def build_factor_summary(factor_diagnostics: pd.DataFrame) -> pd.DataFrame:
    if factor_diagnostics.empty:
        return _empty_factor_summary()

    summary = (
        factor_diagnostics.groupby("strategy", as_index=False)
        .agg(
            start_date=("start_date", "first"),
            end_date=("end_date", "first"),
            observations=("observations", "first"),
            alpha_like_intercept=("alpha_like_intercept", "first"),
            total_mean_factor_contribution=("mean_factor_contribution", "sum"),
            mean_strategy_return=("mean_strategy_return", "first"),
            modeled_mean_return=("modeled_mean_return", "first"),
            r_squared=("r_squared", "first"),
        )
        .sort_values("strategy")
        .reset_index(drop=True)
    )
    return summary.loc[:, FACTOR_SUMMARY_COLUMNS]


def build_covariance_diagnostics(
    windows: list[CovarianceDiagnosticWindow],
) -> pd.DataFrame:
    if not windows:
        return _empty_covariance_diagnostics()

    rows: list[dict[str, object]] = []
    for window in windows:
        covariance = window.covariance.loc[window.symbols, window.symbols].astype(float)
        covariance_values = covariance.to_numpy(dtype=float)
        volatility = np.sqrt(np.diag(covariance_values))
        denominator = np.outer(volatility, volatility)
        correlation = np.divide(
            covariance_values,
            denominator,
            out=np.full_like(covariance_values, np.nan, dtype=float),
            where=denominator > 0.0,
        )
        correlation = np.clip(correlation, -1.0, 1.0)

        for row_index, row_symbol in enumerate(window.symbols):
            for column_index, column_symbol in enumerate(window.symbols):
                rows.append(
                    {
                        "strategy": window.strategy,
                        "signal_date": pd.Timestamp(window.signal_date),
                        "effective_date": pd.Timestamp(window.effective_date),
                        "row_symbol": row_symbol,
                        "column_symbol": column_symbol,
                        "covariance": float(covariance_values[row_index, column_index]),
                        "correlation": float(correlation[row_index, column_index]),
                    }
                )

    diagnostics = pd.DataFrame(rows, columns=COVARIANCE_DIAGNOSTICS_COLUMNS)
    if diagnostics.empty:
        return _empty_covariance_diagnostics()
    return diagnostics.sort_values(
        ["strategy", "effective_date", "signal_date", "row_symbol", "column_symbol"]
    ).reset_index(drop=True)


def build_covariance_summary(covariance_diagnostics: pd.DataFrame) -> pd.DataFrame:
    if covariance_diagnostics.empty:
        return _empty_covariance_summary()

    per_window_rows: list[dict[str, object]] = []
    grouped = covariance_diagnostics.groupby(
        ["strategy", "signal_date", "effective_date"],
        sort=True,
        dropna=False,
    )
    for (strategy, signal_date, effective_date), frame in grouped:
        covariance_matrix = (
            frame.pivot(index="row_symbol", columns="column_symbol", values="covariance")
            .sort_index(axis=0)
            .sort_index(axis=1)
        )
        correlation_matrix = (
            frame.pivot(index="row_symbol", columns="column_symbol", values="correlation")
            .sort_index(axis=0)
            .sort_index(axis=1)
        )
        covariance_values = covariance_matrix.to_numpy(dtype=float)
        correlation_values = correlation_matrix.to_numpy(dtype=float)
        upper_triangle_indices = np.triu_indices_from(correlation_values, k=1)
        pairwise = correlation_values[upper_triangle_indices]
        eigenvalues = np.linalg.eigvalsh(covariance_values)
        per_window_rows.append(
            {
                "strategy": strategy,
                "signal_date": pd.Timestamp(signal_date),
                "effective_date": pd.Timestamp(effective_date),
                "avg_variance": float(np.diag(covariance_values).mean()),
                "avg_pairwise_correlation": float(pairwise.mean()) if len(pairwise) > 0 else float("nan"),
                "max_pairwise_correlation": float(pairwise.max()) if len(pairwise) > 0 else float("nan"),
                "min_eigenvalue": float(eigenvalues.min()),
                "condition_number": float(np.linalg.cond(covariance_values)),
            }
        )

    per_window = pd.DataFrame(per_window_rows)
    summary = (
        per_window.groupby("strategy", as_index=False)
        .agg(
            rebalance_windows=("effective_date", "size"),
            avg_variance=("avg_variance", "mean"),
            avg_pairwise_correlation=("avg_pairwise_correlation", "mean"),
            max_pairwise_correlation=("max_pairwise_correlation", "max"),
            min_eigenvalue=("min_eigenvalue", "min"),
            worst_condition_number=("condition_number", "max"),
        )
        .sort_values("strategy")
        .reset_index(drop=True)
    )
    return summary.loc[:, COVARIANCE_SUMMARY_COLUMNS]
