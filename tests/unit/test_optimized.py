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
    generate_weights,
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

def test_build_optimizer_inputs_validates_external_requirements_before_window_generation() -> None:
    empty_panel = pd.DataFrame(columns=["symbol", "timestamp", "adj_close"])

    with pytest.raises(
        ValueError,
        match="external_covariance_path is required when covariance_estimator='external_csv'",
    ):
        build_optimizer_inputs(
            empty_panel,
            symbols=["AAA", "BBB"],
            lookback_days=3,
            covariance_estimator="external_csv",
        )


def test_build_optimizer_inputs_rejects_unused_external_paths() -> None:
    empty_panel = pd.DataFrame(columns=["symbol", "timestamp", "adj_close"])

    with pytest.raises(
        ValueError,
        match="external_path must be omitted unless method='external_csv'",
    ):
        build_optimizer_inputs(
            empty_panel,
            symbols=["AAA", "BBB"],
            lookback_days=3,
            covariance_estimator="sample",
            external_covariance_path="covariance.csv",
        )

    with pytest.raises(
        ValueError,
        match="external_path must be omitted unless source='external_csv'",
    ):
        build_optimizer_inputs(
            empty_panel,
            symbols=["AAA", "BBB"],
            lookback_days=3,
            expected_return_source="historical_mean",
            external_expected_returns_path="expected.csv",
        )

def _build_optimizer_panel(
    symbol_specs: tuple[tuple[str, float, float], ...],
    *,
    start_date: str = "2024-01-01",
    end_date: str = "2024-01-26",
) -> pd.DataFrame:
    trading_dates = pd.bdate_range(start_date, end_date)
    rows: list[dict[str, object]] = []
    for symbol, base_price, daily_step in symbol_specs:
        for index, timestamp in enumerate(trading_dates):
            rows.append(
                {
                    "symbol": symbol,
                    "timestamp": timestamp,
                    "adj_close": base_price + (daily_step * index),
                }
            )

    return pd.DataFrame(rows).sort_values(["symbol", "timestamp"]).reset_index(drop=True)


def _write_external_covariance(path: Path, covariance: pd.DataFrame) -> None:
    covariance.reset_index(names="symbol").to_csv(path, index=False)


def _write_external_expected_returns(path: Path, expected_returns: pd.Series) -> None:
    expected_returns.rename("expected_return").rename_axis("symbol").reset_index().to_csv(
        path,
        index=False,
    )


def _first_effective_weights(weights: pd.DataFrame) -> pd.Series:
    first_effective_date = pd.to_datetime(weights["effective_date"]).min()
    first_frame = weights.loc[pd.to_datetime(weights["effective_date"]) == first_effective_date]
    return first_frame.set_index("symbol")["weight"].astype(float)


def test_generate_weights_produces_symmetric_mean_variance_solution(tmp_path: Path) -> None:
    panel = _build_optimizer_panel((("AAA", 100.0, 1.0), ("BBB", 120.0, 1.0)))
    covariance = pd.DataFrame(
        [[0.04, 0.0], [0.0, 0.04]],
        index=["AAA", "BBB"],
        columns=["AAA", "BBB"],
        dtype=float,
    )
    expected_returns = pd.Series({"AAA": 0.01, "BBB": 0.01}, dtype=float)
    covariance_path = tmp_path / "covariance.csv"
    expected_returns_path = tmp_path / "expected.csv"
    _write_external_covariance(covariance_path, covariance)
    _write_external_expected_returns(expected_returns_path, expected_returns)

    weights = generate_weights(
        panel,
        symbols=["AAA", "BBB"],
        method="mean_variance",
        lookback_days=3,
        covariance_estimator="external_csv",
        external_covariance_path=covariance_path,
        expected_return_source="external_csv",
        external_expected_returns_path=expected_returns_path,
    )

    assert set(weights["strategy"]) == {"mean_variance"}
    first_weights = _first_effective_weights(weights)
    assert first_weights.loc["AAA"] == pytest.approx(0.5, abs=1e-6)
    assert first_weights.loc["BBB"] == pytest.approx(0.5, abs=1e-6)


def test_generate_weights_prefers_higher_expected_return_assets(tmp_path: Path) -> None:
    panel = _build_optimizer_panel((("AAA", 100.0, 1.0), ("BBB", 120.0, 1.0)))
    covariance = pd.DataFrame(
        [[0.04, 0.0], [0.0, 0.04]],
        index=["AAA", "BBB"],
        columns=["AAA", "BBB"],
        dtype=float,
    )
    expected_returns = pd.Series({"AAA": 0.03, "BBB": 0.01}, dtype=float)
    covariance_path = tmp_path / "covariance.csv"
    expected_returns_path = tmp_path / "expected.csv"
    _write_external_covariance(covariance_path, covariance)
    _write_external_expected_returns(expected_returns_path, expected_returns)

    weights = generate_weights(
        panel,
        symbols=["AAA", "BBB"],
        method="mean_variance",
        lookback_days=3,
        covariance_estimator="external_csv",
        external_covariance_path=covariance_path,
        expected_return_source="external_csv",
        external_expected_returns_path=expected_returns_path,
    )

    first_weights = _first_effective_weights(weights)
    assert first_weights.loc["AAA"] == pytest.approx(0.75, abs=1e-3)
    assert first_weights.loc["BBB"] == pytest.approx(0.25, abs=1e-3)


def test_generate_weights_respects_target_gross_exposure(tmp_path: Path) -> None:
    panel = _build_optimizer_panel((("AAA", 100.0, 1.0), ("BBB", 120.0, 1.0)))
    covariance = pd.DataFrame(
        [[0.04, 0.0], [0.0, 0.04]],
        index=["AAA", "BBB"],
        columns=["AAA", "BBB"],
        dtype=float,
    )
    expected_returns = pd.Series({"AAA": 0.01, "BBB": 0.01}, dtype=float)
    covariance_path = tmp_path / "covariance.csv"
    expected_returns_path = tmp_path / "expected.csv"
    _write_external_covariance(covariance_path, covariance)
    _write_external_expected_returns(expected_returns_path, expected_returns)

    weights = generate_weights(
        panel,
        symbols=["AAA", "BBB"],
        method="mean_variance",
        lookback_days=3,
        covariance_estimator="external_csv",
        external_covariance_path=covariance_path,
        expected_return_source="external_csv",
        external_expected_returns_path=expected_returns_path,
        target_gross_exposure=0.6,
    )

    first_weights = _first_effective_weights(weights)
    assert float(first_weights.sum()) == pytest.approx(0.6, abs=1e-6)
    assert first_weights.loc["AAA"] == pytest.approx(0.3, abs=1e-6)
    assert first_weights.loc["BBB"] == pytest.approx(0.3, abs=1e-6)


def test_generate_weights_respects_position_caps(tmp_path: Path) -> None:
    panel = _build_optimizer_panel((("AAA", 100.0, 1.0), ("BBB", 120.0, 1.0)))
    covariance = pd.DataFrame(
        [[0.04, 0.0], [0.0, 0.04]],
        index=["AAA", "BBB"],
        columns=["AAA", "BBB"],
        dtype=float,
    )
    expected_returns = pd.Series({"AAA": 0.03, "BBB": 0.01}, dtype=float)
    covariance_path = tmp_path / "covariance.csv"
    expected_returns_path = tmp_path / "expected.csv"
    _write_external_covariance(covariance_path, covariance)
    _write_external_expected_returns(expected_returns_path, expected_returns)

    weights = generate_weights(
        panel,
        symbols=["AAA", "BBB"],
        method="mean_variance",
        lookback_days=3,
        covariance_estimator="external_csv",
        external_covariance_path=covariance_path,
        expected_return_source="external_csv",
        external_expected_returns_path=expected_returns_path,
        max_position_weight=0.6,
    )

    first_weights = _first_effective_weights(weights)
    assert first_weights.loc["AAA"] == pytest.approx(0.6, abs=1e-5)
    assert first_weights.loc["BBB"] == pytest.approx(0.4, abs=1e-5)


def test_generate_weights_respects_group_caps(tmp_path: Path) -> None:
    panel = _build_optimizer_panel(
        (("AAA", 100.0, 1.0), ("BBB", 120.0, 1.0), ("CCC", 140.0, 1.0))
    )
    covariance = pd.DataFrame(
        np.diag([0.04, 0.04, 0.04]),
        index=["AAA", "BBB", "CCC"],
        columns=["AAA", "BBB", "CCC"],
        dtype=float,
    )
    expected_returns = pd.Series({"AAA": 0.03, "BBB": 0.03, "CCC": 0.0}, dtype=float)
    covariance_path = tmp_path / "covariance.csv"
    expected_returns_path = tmp_path / "expected.csv"
    _write_external_covariance(covariance_path, covariance)
    _write_external_expected_returns(expected_returns_path, expected_returns)

    weights = generate_weights(
        panel,
        symbols=["AAA", "BBB", "CCC"],
        method="mean_variance",
        lookback_days=3,
        covariance_estimator="external_csv",
        external_covariance_path=covariance_path,
        expected_return_source="external_csv",
        external_expected_returns_path=expected_returns_path,
        symbol_groups={"AAA": "growth", "BBB": "growth", "CCC": "defensive"},
        max_group_weight=0.4,
    )

    first_weights = _first_effective_weights(weights)
    assert float(first_weights.loc[["AAA", "BBB"]].sum()) == pytest.approx(0.4, abs=1e-5)
    assert first_weights.loc["CCC"] == pytest.approx(0.6, abs=1e-5)


def test_generate_weights_rejects_unimplemented_methods() -> None:
    panel = _build_optimizer_panel((("AAA", 100.0, 1.0), ("BBB", 120.0, 1.0)))

    with pytest.raises(RuntimeError, match="baselines.optimized.method='risk_parity' is not implemented yet"):
        generate_weights(
            panel,
            symbols=["AAA", "BBB"],
            method="risk_parity",
            lookback_days=3,
        )
