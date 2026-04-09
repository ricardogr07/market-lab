from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from marketlab.config import load_config


def _write_config(
    path: Path,
    *,
    data: dict[str, object] | None = None,
    portfolio: dict[str, object] | None = None,
    baselines: dict[str, object] | None = None,
    evaluation: dict[str, object] | None = None,
) -> Path:
    payload = {
        "experiment_name": "config_fixture",
        "data": {
            "symbols": ["AAA", "BBB"],
            "start_date": "2024-01-01",
            "end_date": "2024-03-31",
            "cache_dir": "artifacts/test-cache",
            "prepared_panel_filename": "panel.csv",
        },
    }
    if data is not None:
        payload["data"].update(data)
    if portfolio is not None:
        payload["portfolio"] = portfolio
    if baselines is not None:
        payload["baselines"] = baselines
    if evaluation is not None:
        payload["evaluation"] = evaluation

    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return path


def test_load_config_preserves_backward_compatible_allocation_defaults(tmp_path: Path) -> None:
    config_path = _write_config(tmp_path / "config.yaml")

    config = load_config(config_path)

    assert config.data.symbol_groups == {}
    assert config.baselines.allocation.enabled is False
    assert config.baselines.allocation.mode == "equal"
    assert config.baselines.allocation.symbol_weights == {}
    assert config.baselines.allocation.group_weights == {}
    assert config.baselines.optimized.enabled is False
    assert config.baselines.optimized.method == "mean_variance"
    assert config.baselines.optimized.lookback_days == 252
    assert config.baselines.optimized.rebalance_frequency == "W-FRI"
    assert config.baselines.optimized.covariance_estimator == "sample"
    assert config.baselines.optimized.external_covariance_path == ""
    assert config.baselines.optimized.expected_return_source == "historical_mean"
    assert config.baselines.optimized.external_expected_returns_path == ""
    assert config.baselines.optimized.long_only is True
    assert config.baselines.optimized.target_gross_exposure == pytest.approx(1.0)
    assert config.baselines.optimized.risk_aversion == pytest.approx(1.0)
    assert config.optimized_external_covariance_path is None
    assert config.optimized_external_expected_returns_path is None
    assert config.portfolio.risk.max_position_weight is None
    assert config.portfolio.risk.max_group_weight is None
    assert config.portfolio.risk.max_long_exposure is None
    assert config.portfolio.risk.max_short_exposure is None
    assert config.evaluation.cost_sensitivity_bps == []


def test_load_config_normalizes_nullable_mapping_sections(tmp_path: Path) -> None:
    config_path = _write_config(
        tmp_path / "config.yaml",
        data={"symbol_groups": None},
        baselines={
            "allocation": {
                "enabled": False,
                "symbol_weights": None,
                "group_weights": None,
            },
            "optimized": {
                "external_covariance_path": None,
                "external_expected_returns_path": None,
            },
        },
        evaluation={"cost_sensitivity_bps": None},
    )

    config = load_config(config_path)

    assert config.data.symbol_groups == {}
    assert config.baselines.allocation.symbol_weights == {}
    assert config.baselines.allocation.group_weights == {}
    assert config.baselines.optimized.external_covariance_path == ""
    assert config.baselines.optimized.external_expected_returns_path == ""
    assert config.evaluation.cost_sensitivity_bps == []


def test_load_config_rejects_unknown_symbol_group_entries(tmp_path: Path) -> None:
    config_path = _write_config(
        tmp_path / "config.yaml",
        data={"symbol_groups": {"CCC": "growth"}},
    )

    with pytest.raises(ValueError, match="data.symbol_groups contains unknown symbols: CCC"):
        load_config(config_path)


def test_load_config_rejects_symbol_weights_that_do_not_match_symbols(tmp_path: Path) -> None:
    config_path = _write_config(
        tmp_path / "config.yaml",
        baselines={
            "allocation": {
                "enabled": True,
                "mode": "symbol_weights",
                "symbol_weights": {"AAA": 1.0},
            }
        },
    )

    with pytest.raises(
        ValueError,
        match="baselines.allocation.symbol_weights must match data.symbols exactly",
    ):
        load_config(config_path)


def test_load_config_accepts_valid_group_weight_allocations(tmp_path: Path) -> None:
    config_path = _write_config(
        tmp_path / "config.yaml",
        data={
            "symbols": ["AAA", "BBB", "CCC", "DDD"],
            "symbol_groups": {
                "AAA": "growth",
                "BBB": "growth",
                "CCC": "defensive",
                "DDD": "defensive",
            },
        },
        baselines={
            "allocation": {
                "enabled": True,
                "mode": "group_weights",
                "group_weights": {"growth": 0.75, "defensive": 0.25},
            }
        },
    )

    config = load_config(config_path)

    assert config.baselines.allocation.enabled is True
    assert config.baselines.allocation.mode == "group_weights"
    assert config.baselines.allocation.group_weights == {
        "growth": 0.75,
        "defensive": 0.25,
    }


def test_load_config_rejects_risk_caps_outside_unit_interval(tmp_path: Path) -> None:
    config_path = _write_config(
        tmp_path / "config.yaml",
        portfolio={
            "risk": {
                "max_position_weight": 1.2,
            }
        },
    )

    with pytest.raises(
        ValueError,
        match="portfolio.risk.max_position_weight must be between 0.0 and 1.0",
    ):
        load_config(config_path)


def test_load_config_rejects_group_cap_without_full_symbol_groups(tmp_path: Path) -> None:
    config_path = _write_config(
        tmp_path / "config.yaml",
        data={
            "symbol_groups": {
                "AAA": "growth",
            }
        },
        portfolio={
            "risk": {
                "max_group_weight": 0.30,
            }
        },
    )

    with pytest.raises(
        ValueError,
        match="portfolio.risk.max_group_weight requires data.symbol_groups for all data.symbols: BBB",
    ):
        load_config(config_path)


def test_load_config_rejects_short_cap_in_long_only_mode(tmp_path: Path) -> None:
    config_path = _write_config(
        tmp_path / "config.yaml",
        portfolio={
            "ranking": {
                "mode": "long_only",
            },
            "risk": {
                "max_short_exposure": 0.25,
            },
        },
    )

    with pytest.raises(
        ValueError,
        match="portfolio.risk.max_short_exposure is not allowed when portfolio.ranking.mode='long_only'",
    ):
        load_config(config_path)


def test_load_config_accepts_valid_risk_caps(tmp_path: Path) -> None:
    config_path = _write_config(
        tmp_path / "config.yaml",
        data={
            "symbol_groups": {
                "AAA": "growth",
                "BBB": "defensive",
            }
        },
        portfolio={
            "risk": {
                "max_position_weight": 0.35,
                "max_group_weight": 0.40,
                "max_long_exposure": 0.60,
                "max_short_exposure": 0.45,
            }
        },
    )

    config = load_config(config_path)

    assert config.portfolio.risk.max_position_weight == pytest.approx(0.35)
    assert config.portfolio.risk.max_group_weight == pytest.approx(0.40)
    assert config.portfolio.risk.max_long_exposure == pytest.approx(0.60)
    assert config.portfolio.risk.max_short_exposure == pytest.approx(0.45)


def test_load_config_accepts_cost_sensitivity_bps(tmp_path: Path) -> None:
    config_path = _write_config(
        tmp_path / "config.yaml",
        evaluation={"cost_sensitivity_bps": [25.0, 5.0]},
    )

    config = load_config(config_path)

    assert config.evaluation.cost_sensitivity_bps == [25.0, 5.0]


@pytest.mark.parametrize("values", [[-1.0], [float("inf")], [float("nan")]])
def test_load_config_rejects_invalid_cost_sensitivity_bps(
    tmp_path: Path,
    values: list[float],
) -> None:
    config_path = _write_config(
        tmp_path / "config.yaml",
        evaluation={"cost_sensitivity_bps": values},
    )

    with pytest.raises(
        ValueError,
        match="evaluation.cost_sensitivity_bps must contain only finite non-negative values",
    ):
        load_config(config_path)


@pytest.mark.parametrize(
    ("optimized", "message"),
    [
        (
            {"method": "unknown"},
            "baselines.optimized.method must be one of",
        ),
        (
            {"covariance_estimator": "unknown"},
            "baselines.optimized.covariance_estimator must be one of",
        ),
        (
            {"expected_return_source": "unknown"},
            "baselines.optimized.expected_return_source must be one of",
        ),
        (
            {"lookback_days": 1},
            "baselines.optimized.lookback_days must be at least 2",
        ),
        (
            {"target_gross_exposure": 0.0},
            "baselines.optimized.target_gross_exposure must be a finite positive value",
        ),
        (
            {"risk_aversion": 0.0},
            "baselines.optimized.risk_aversion must be a finite positive value",
        ),
        (
            {"method": "mean_variance", "long_only": False},
            "baselines.optimized.long_only must be true when baselines.optimized.method='mean_variance'",
        ),
        (
            {"method": "mean_variance", "target_gross_exposure": 1.2},
            "baselines.optimized.target_gross_exposure must be less than or equal to 1.0 when baselines.optimized.method='mean_variance'",
        ),
        (
            {"covariance_estimator": "external_csv"},
            "baselines.optimized.external_covariance_path is required when baselines.optimized.covariance_estimator='external_csv'",
        ),
        (
            {"covariance_estimator": "sample", "external_covariance_path": "cov.csv"},
            "baselines.optimized.external_covariance_path must be empty unless baselines.optimized.covariance_estimator='external_csv'",
        ),
        (
            {"expected_return_source": "external_csv"},
            "baselines.optimized.external_expected_returns_path is required when baselines.optimized.expected_return_source='external_csv'",
        ),
        (
            {"expected_return_source": "historical_mean", "external_expected_returns_path": "expected.csv"},
            "baselines.optimized.external_expected_returns_path must be empty unless baselines.optimized.expected_return_source='external_csv'",
        ),
    ],
)
def test_load_config_rejects_invalid_optimized_scaffold_settings(
    tmp_path: Path,
    optimized: dict[str, object],
    message: str,
) -> None:
    config_path = _write_config(
        tmp_path / "config.yaml",
        baselines={"optimized": optimized},
    )

    with pytest.raises(ValueError, match=message):
        load_config(config_path)


def test_load_config_resolves_relative_optimized_external_paths(tmp_path: Path) -> None:
    config_dir = tmp_path / "nested"
    config_dir.mkdir(parents=True, exist_ok=True)
    config_path = _write_config(
        config_dir / "config.yaml",
        baselines={
            "optimized": {
                "covariance_estimator": "external_csv",
                "external_covariance_path": "inputs/covariance.csv",
                "expected_return_source": "external_csv",
                "external_expected_returns_path": "inputs/expected.csv",
            }
        },
    )

    config = load_config(config_path)

    assert config.optimized_external_covariance_path == (config_dir / "inputs" / "covariance.csv").resolve()
    assert config.optimized_external_expected_returns_path == (config_dir / "inputs" / "expected.csv").resolve()
