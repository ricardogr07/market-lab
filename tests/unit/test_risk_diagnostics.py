from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from marketlab.reports.risk_diagnostics import (
    COVARIANCE_DIAGNOSTICS_COLUMNS,
    FACTOR_DIAGNOSTICS_COLUMNS,
    build_covariance_diagnostics,
    build_covariance_summary,
    build_factor_diagnostics,
    build_factor_summary,
    load_factor_returns,
)
from marketlab.strategies.optimized import CovarianceDiagnosticWindow


@pytest.mark.parametrize(
    ("contents", "message"),
    [
        ("MKT,SMB\n0.01,0.02\n", "Factor model CSV must contain a 'date' column."),
        ("date\n2024-01-02\n", "Factor model CSV must contain at least one factor column."),
        (
            "date,MKT,MKT\n2024-01-02,0.01,0.02\n",
            "Factor model CSV must not contain duplicate column names.",
        ),
        (
            "date,MKT\n2024-01-02,0.01\n2024-01-02,0.02\n",
            "Factor model CSV must not contain duplicate dates.",
        ),
        (
            "date,MKT\n2024-01-02,\n",
            "Factor model CSV must contain only finite numeric factor values.",
        ),
    ],
)
def test_load_factor_returns_rejects_invalid_contract(
    tmp_path: Path,
    contents: str,
    message: str,
) -> None:
    factor_path = tmp_path / "factor_returns.csv"
    factor_path.write_text(contents, encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        load_factor_returns(factor_path)


def test_build_factor_diagnostics_matches_known_factor_regression() -> None:
    factor_returns = pd.DataFrame(
        {
            "date": pd.to_datetime(
                ["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"]
            ),
            "MKT": [0.01, -0.02, 0.03, -0.01],
        }
    )
    performance = pd.DataFrame(
        {
            "date": list(factor_returns["date"]) * 2,
            "strategy": ["alpha"] * 4 + ["beta"] * 4,
            "net_return": [
                0.001 + (1.5 * value) for value in factor_returns["MKT"]
            ]
            + [0.002 + (-0.5 * value) for value in factor_returns["MKT"]],
        }
    )

    diagnostics = build_factor_diagnostics(performance, factor_returns)
    summary = build_factor_summary(diagnostics)

    assert list(diagnostics.columns) == FACTOR_DIAGNOSTICS_COLUMNS
    assert set(diagnostics["strategy"]) == {"alpha", "beta"}
    assert len(diagnostics) == 2

    alpha_row = diagnostics.loc[diagnostics["strategy"] == "alpha"].iloc[0]
    beta_row = diagnostics.loc[diagnostics["strategy"] == "beta"].iloc[0]
    assert alpha_row["beta_like_exposure"] == pytest.approx(1.5)
    assert alpha_row["alpha_like_intercept"] == pytest.approx(0.001)
    assert alpha_row["r_squared"] == pytest.approx(1.0)
    assert beta_row["beta_like_exposure"] == pytest.approx(-0.5)
    assert beta_row["alpha_like_intercept"] == pytest.approx(0.002)
    assert beta_row["r_squared"] == pytest.approx(1.0)

    summary = summary.set_index("strategy")
    assert summary.loc["alpha", "total_mean_factor_contribution"] == pytest.approx(
        alpha_row["mean_factor_contribution"]
    )
    assert summary.loc["beta", "modeled_mean_return"] == pytest.approx(
        beta_row["modeled_mean_return"]
    )


def test_build_factor_diagnostics_rejects_underdetermined_regression() -> None:
    factor_returns = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-02", "2024-01-03"]),
            "MKT": [0.01, 0.02],
            "SMB": [0.00, 0.01],
        }
    )
    performance = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-02", "2024-01-03"]),
            "strategy": ["alpha", "alpha"],
            "net_return": [0.01, 0.02],
        }
    )

    with pytest.raises(ValueError, match="underdetermined"):
        build_factor_diagnostics(performance, factor_returns)


def test_build_covariance_diagnostics_and_summary_match_known_windows() -> None:
    covariance_one = pd.DataFrame(
        [[0.04, 0.01], [0.01, 0.09]],
        index=["AAA", "BBB"],
        columns=["AAA", "BBB"],
        dtype=float,
    )
    covariance_two = pd.DataFrame(
        [[0.01, 0.002], [0.002, 0.04]],
        index=["AAA", "BBB"],
        columns=["AAA", "BBB"],
        dtype=float,
    )
    windows = [
        CovarianceDiagnosticWindow(
            strategy="mean_variance",
            signal_date=pd.Timestamp("2024-01-05"),
            effective_date=pd.Timestamp("2024-01-08"),
            symbols=["AAA", "BBB"],
            covariance=covariance_one,
        ),
        CovarianceDiagnosticWindow(
            strategy="mean_variance",
            signal_date=pd.Timestamp("2024-01-12"),
            effective_date=pd.Timestamp("2024-01-15"),
            symbols=["AAA", "BBB"],
            covariance=covariance_two,
        ),
    ]

    diagnostics = build_covariance_diagnostics(windows)
    summary = build_covariance_summary(diagnostics)

    assert list(diagnostics.columns) == COVARIANCE_DIAGNOSTICS_COLUMNS
    assert len(diagnostics) == 8

    first_pair = diagnostics.loc[
        (diagnostics["effective_date"] == pd.Timestamp("2024-01-08"))
        & (diagnostics["row_symbol"] == "AAA")
        & (diagnostics["column_symbol"] == "BBB")
    ].iloc[0]
    assert first_pair["covariance"] == pytest.approx(0.01)
    assert first_pair["correlation"] == pytest.approx(0.01 / np.sqrt(0.04 * 0.09))

    expected_avg_variance = np.mean(
        [
            np.mean(np.diag(covariance_one.to_numpy(dtype=float))),
            np.mean(np.diag(covariance_two.to_numpy(dtype=float))),
        ]
    )
    expected_avg_pairwise = np.mean(
        [
            0.01 / np.sqrt(0.04 * 0.09),
            0.002 / np.sqrt(0.01 * 0.04),
        ]
    )
    expected_max_pairwise = max(
        0.01 / np.sqrt(0.04 * 0.09),
        0.002 / np.sqrt(0.01 * 0.04),
    )
    expected_min_eigenvalue = min(
        np.linalg.eigvalsh(covariance_one.to_numpy(dtype=float)).min(),
        np.linalg.eigvalsh(covariance_two.to_numpy(dtype=float)).min(),
    )
    expected_worst_condition = max(
        np.linalg.cond(covariance_one.to_numpy(dtype=float)),
        np.linalg.cond(covariance_two.to_numpy(dtype=float)),
    )

    summary_row = summary.iloc[0]
    assert summary_row["strategy"] == "mean_variance"
    assert summary_row["rebalance_windows"] == 2
    assert summary_row["avg_variance"] == pytest.approx(expected_avg_variance)
    assert summary_row["avg_pairwise_correlation"] == pytest.approx(expected_avg_pairwise)
    assert summary_row["max_pairwise_correlation"] == pytest.approx(expected_max_pairwise)
    assert summary_row["min_eigenvalue"] == pytest.approx(expected_min_eigenvalue)
    assert summary_row["worst_condition_number"] == pytest.approx(expected_worst_condition)
