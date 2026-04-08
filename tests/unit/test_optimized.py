from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from marketlab.strategies.optimized import (
    DIAGONAL_SHRINKAGE_WEIGHT,
    EWMA_LAMBDA,
    build_optimizer_inputs,
    build_optimizer_windows,
    estimate_covariance_matrix,
    estimate_expected_returns,
    load_external_covariance,
    load_external_expected_returns,
)


def _build_panel(*, missing_bbb_on: str | None = None) -> pd.DataFrame:
    trading_dates = pd.bdate_range("2024-01-01", "2024-01-19")
    rows: list[dict[str, object]] = []
    for symbol, base_price, daily_step in (
        ("AAA", 100.0, 1.0),
        ("BBB", 120.0, 0.5),
    ):
        for index, timestamp in enumerate(trading_dates):
            if symbol == "BBB" and missing_bbb_on is not None and timestamp == pd.Timestamp(missing_bbb_on):
                continue
            close_price = base_price + (daily_step * index)
            rows.append(
                {
                    "symbol": symbol,
                    "timestamp": timestamp,
                    "adj_close": close_price,
                }
            )

    return pd.DataFrame(rows).sort_values(["symbol", "timestamp"]).reset_index(drop=True)


def test_build_optimizer_windows_uses_signal_dates_and_common_history() -> None:
    windows = build_optimizer_windows(
        _build_panel(),
        symbols=["AAA", "BBB"],
        lookback_days=3,
        frequency="W-FRI",
    )

    assert len(windows) >= 2
    first_window = windows[0]
    assert first_window.signal_date == pd.Timestamp("2024-01-05")
    assert first_window.effective_date == pd.Timestamp("2024-01-08")
    assert first_window.symbols == ["AAA", "BBB"]
    assert list(first_window.returns.columns) == ["AAA", "BBB"]
    assert len(first_window.returns) == 3
    assert first_window.returns.index.min() == pd.Timestamp("2024-01-03")
    assert first_window.returns.index.max() == pd.Timestamp("2024-01-05")


def test_build_optimizer_windows_skips_signal_dates_without_common_signal_return() -> None:
    windows = build_optimizer_windows(
        _build_panel(missing_bbb_on="2024-01-05"),
        symbols=["AAA", "BBB"],
        lookback_days=3,
        frequency="W-FRI",
    )

    assert [window.signal_date for window in windows] == [pd.Timestamp("2024-01-12")]


def test_estimate_covariance_matrix_sample_matches_pandas_covariance() -> None:
    returns = pd.DataFrame(
        {
            "AAA": [0.01, 0.02, -0.01],
            "BBB": [0.00, 0.01, 0.03],
        }
    )

    covariance = estimate_covariance_matrix(returns, method="sample")

    pd.testing.assert_frame_equal(covariance, returns.cov())


def test_estimate_covariance_matrix_ewma_matches_weighted_formula() -> None:
    returns = pd.DataFrame(
        {
            "AAA": [0.01, 0.02, -0.01],
            "BBB": [0.00, 0.01, 0.03],
        }
    )

    covariance = estimate_covariance_matrix(returns, method="ewma")

    values = returns.to_numpy(dtype=float)
    weights = (1.0 - EWMA_LAMBDA) * (EWMA_LAMBDA ** np.arange(len(values) - 1, -1, -1))
    weights = weights / weights.sum()
    mean = np.average(values, axis=0, weights=weights)
    centered = values - mean
    expected = centered.T @ (centered * weights[:, None])
    expected_frame = pd.DataFrame(expected, index=returns.columns, columns=returns.columns)

    pd.testing.assert_frame_equal(covariance, expected_frame)


def test_estimate_covariance_matrix_diagonal_shrinkage_blends_sample_and_diagonal() -> None:
    returns = pd.DataFrame(
        {
            "AAA": [0.01, 0.02, -0.01],
            "BBB": [0.00, 0.01, 0.03],
        }
    )

    covariance = estimate_covariance_matrix(returns, method="diagonal_shrinkage")
    sample_covariance = returns.cov()
    diagonal_only = pd.DataFrame(
        np.diag(np.diag(sample_covariance.to_numpy(dtype=float))),
        index=returns.columns,
        columns=returns.columns,
        dtype=float,
    )
    expected = ((1.0 - DIAGONAL_SHRINKAGE_WEIGHT) * sample_covariance) + (
        DIAGONAL_SHRINKAGE_WEIGHT * diagonal_only
    )

    pd.testing.assert_frame_equal(covariance, expected)


def test_load_external_covariance_reorders_symbols_and_rejects_mismatches(tmp_path: Path) -> None:
    covariance_path = tmp_path / "covariance.csv"
    pd.DataFrame(
        {
            "symbol": ["BBB", "AAA"],
            "BBB": [0.04, 0.01],
            "AAA": [0.01, 0.09],
        }
    ).to_csv(covariance_path, index=False)

    covariance = load_external_covariance(covariance_path, symbols=["AAA", "BBB"])

    assert list(covariance.index) == ["AAA", "BBB"]
    assert list(covariance.columns) == ["AAA", "BBB"]
    assert covariance.loc["AAA", "BBB"] == pytest.approx(0.01)

    mismatch_path = tmp_path / "covariance_bad.csv"
    pd.DataFrame(
        {
            "symbol": ["AAA", "CCC"],
            "AAA": [0.09, 0.01],
            "CCC": [0.01, 0.04],
        }
    ).to_csv(mismatch_path, index=False)

    with pytest.raises(ValueError, match="External covariance CSV must contain exactly the configured symbols"):
        load_external_covariance(mismatch_path, symbols=["AAA", "BBB"])


def test_load_external_covariance_rejects_non_square_files(tmp_path: Path) -> None:
    covariance_path = tmp_path / "covariance_nonsquare.csv"
    pd.DataFrame(
        {
            "symbol": ["AAA", "BBB"],
            "AAA": [0.09, 0.01],
            "BBB": [0.01, 0.04],
            "CCC": [0.00, 0.00],
        }
    ).to_csv(covariance_path, index=False)

    with pytest.raises(ValueError, match="External covariance CSV must be a square matrix"):
        load_external_covariance(covariance_path, symbols=["AAA", "BBB"])


def test_estimate_expected_returns_historical_mean_and_external_loader(tmp_path: Path) -> None:
    returns = pd.DataFrame(
        {
            "AAA": [0.01, 0.02, -0.01],
            "BBB": [0.00, 0.01, 0.03],
        }
    )

    historical = estimate_expected_returns(returns, source="historical_mean")
    pd.testing.assert_series_equal(
        historical,
        returns.mean(axis=0).rename("expected_return"),
    )

    expected_path = tmp_path / "expected.csv"
    pd.DataFrame(
        {
            "symbol": ["BBB", "AAA"],
            "expected_return": [0.02, 0.01],
        }
    ).to_csv(expected_path, index=False)

    expected_returns = load_external_expected_returns(expected_path, symbols=["AAA", "BBB"])

    assert list(expected_returns.index) == ["AAA", "BBB"]
    assert expected_returns.loc["AAA"] == pytest.approx(0.01)
    assert expected_returns.loc["BBB"] == pytest.approx(0.02)

    invalid_path = tmp_path / "expected_invalid.csv"
    pd.DataFrame(
        {
            "symbol": ["AAA", "BBB"],
            "expected": [0.01, 0.02],
        }
    ).to_csv(invalid_path, index=False)

    with pytest.raises(
        ValueError,
        match="External expected returns CSV must have exactly these columns: symbol, expected_return",
    ):
        load_external_expected_returns(invalid_path, symbols=["AAA", "BBB"])


def test_build_optimizer_inputs_supports_external_sources(tmp_path: Path) -> None:
    panel = _build_panel()
    covariance_path = tmp_path / "covariance.csv"
    pd.DataFrame(
        {
            "symbol": ["AAA", "BBB"],
            "AAA": [0.09, 0.01],
            "BBB": [0.01, 0.04],
        }
    ).to_csv(covariance_path, index=False)
    expected_path = tmp_path / "expected.csv"
    pd.DataFrame(
        {
            "symbol": ["AAA", "BBB"],
            "expected_return": [0.01, 0.02],
        }
    ).to_csv(expected_path, index=False)

    optimizer_inputs = build_optimizer_inputs(
        panel,
        symbols=["AAA", "BBB"],
        lookback_days=3,
        covariance_estimator="external_csv",
        external_covariance_path=covariance_path,
        expected_return_source="external_csv",
        external_expected_returns_path=expected_path,
    )

    assert len(optimizer_inputs) >= 1
    first_input = optimizer_inputs[0]
    assert first_input.signal_date == pd.Timestamp("2024-01-05")
    assert first_input.effective_date == pd.Timestamp("2024-01-08")
    assert list(first_input.covariance.index) == ["AAA", "BBB"]
    assert list(first_input.expected_returns.index) == ["AAA", "BBB"]
    assert first_input.expected_returns.loc["AAA"] == pytest.approx(0.01)
    assert first_input.covariance.loc["BBB", "BBB"] == pytest.approx(0.04)


