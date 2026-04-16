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
    assert config.baselines.optimized.equilibrium_weights == {}
    assert config.baselines.optimized.tau == pytest.approx(0.05)
    assert config.baselines.optimized.views == []
    assert config.optimized_external_covariance_path is None
    assert config.optimized_external_expected_returns_path is None
    assert config.portfolio.risk.max_position_weight is None
    assert config.portfolio.risk.max_group_weight is None
    assert config.portfolio.risk.max_long_exposure is None
    assert config.portfolio.risk.max_short_exposure is None
    assert config.evaluation.cost_sensitivity_bps == []
    assert config.evaluation.factor_model_path == ""
    assert config.factor_model_path is None


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
                "equilibrium_weights": None,
                "views": None,
            },
        },
        evaluation={"cost_sensitivity_bps": None, "factor_model_path": None},
    )

    config = load_config(config_path)

    assert config.data.symbol_groups == {}
    assert config.baselines.allocation.symbol_weights == {}
    assert config.baselines.allocation.group_weights == {}
    assert config.baselines.optimized.external_covariance_path == ""
    assert config.baselines.optimized.external_expected_returns_path == ""
    assert config.baselines.optimized.equilibrium_weights == {}
    assert config.baselines.optimized.views == []
    assert config.evaluation.cost_sensitivity_bps == []
    assert config.evaluation.factor_model_path == ""
    assert config.factor_model_path is None


def test_load_config_resolves_factor_model_path_relative_to_config(tmp_path: Path) -> None:
    config_dir = tmp_path / "configs"
    factors_path = tmp_path / "inputs" / "factor_returns.csv"
    config_dir.mkdir(parents=True, exist_ok=True)
    factors_path.parent.mkdir(parents=True, exist_ok=True)
    factors_path.write_text("date,MKT\n2024-01-02,0.01\n", encoding="utf-8")
    config_path = _write_config(
        config_dir / "config.yaml",
        evaluation={"factor_model_path": "inputs/factor_returns.csv"},
    )

    config = load_config(config_path)

    assert config.factor_model_path == factors_path.resolve()


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


def test_load_config_accepts_valid_black_litterman_settings(tmp_path: Path) -> None:
    config_path = _write_config(
        tmp_path / "config.yaml",
        data={"symbols": ["AAA", "BBB", "CCC"]},
        baselines={
            "optimized": {
                "enabled": True,
                "method": "black_litterman",
                "covariance_estimator": "sample",
                "equilibrium_weights": {"AAA": 0.4, "BBB": 0.4, "CCC": 0.2},
                "tau": 0.10,
                "views": [
                    {
                        "name": "growth_over_defensive",
                        "weights": {"AAA": 1.0, "BBB": 1.0, "CCC": -1.0},
                        "view_return": 0.0025,
                    }
                ],
            }
        },
    )

    config = load_config(config_path)

    optimized = config.baselines.optimized
    assert optimized.method == "black_litterman"
    assert optimized.equilibrium_weights == {"AAA": 0.4, "BBB": 0.4, "CCC": 0.2}
    assert optimized.tau == pytest.approx(0.10)
    assert len(optimized.views) == 1
    view = optimized.views[0]
    assert view.name == "growth_over_defensive"
    assert view.weights == {"AAA": 1.0, "BBB": 1.0, "CCC": -1.0}
    assert view.view_return == pytest.approx(0.0025)


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
            {"method": "risk_parity", "long_only": False},
            "baselines.optimized.long_only must be true when baselines.optimized.method='risk_parity'",
        ),
        (
            {"method": "risk_parity", "target_gross_exposure": 1.2},
            "baselines.optimized.target_gross_exposure must be less than or equal to 1.0 when baselines.optimized.method='risk_parity'",
        ),
        (
            {"method": "risk_parity", "expected_return_source": "external_csv"},
            "baselines.optimized.expected_return_source must remain 'historical_mean' when baselines.optimized.method='risk_parity'",
        ),
        (
            {"method": "risk_parity", "external_expected_returns_path": "expected.csv"},
            "baselines.optimized.external_expected_returns_path must be empty when baselines.optimized.method='risk_parity'",
        ),
        (
            {"tau": 0.0},
            "baselines.optimized.tau must be a finite positive value",
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
        (
            {"method": "black_litterman", "long_only": False},
            "baselines.optimized.long_only must be true when baselines.optimized.method='black_litterman'",
        ),
        (
            {"method": "black_litterman", "target_gross_exposure": 1.2},
            "baselines.optimized.target_gross_exposure must be less than or equal to 1.0 when baselines.optimized.method='black_litterman'",
        ),
        (
            {"method": "black_litterman", "expected_return_source": "external_csv"},
            "baselines.optimized.expected_return_source must remain 'historical_mean' when baselines.optimized.method='black_litterman'",
        ),
        (
            {"method": "black_litterman", "external_expected_returns_path": "expected.csv"},
            "baselines.optimized.external_expected_returns_path must be empty when baselines.optimized.method='black_litterman'",
        ),
        (
            {"method": "black_litterman"},
            "baselines.optimized.equilibrium_weights must match data.symbols exactly when baselines.optimized.method='black_litterman'",
        ),
        (
            {
                "method": "black_litterman",
                "equilibrium_weights": {"AAA": float("nan"), "BBB": 1.0},
                "views": [{"name": "good", "weights": {"AAA": 1.0}, "view_return": 0.01}],
            },
            "baselines.optimized.equilibrium_weights must contain only finite numeric values",
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


@pytest.mark.parametrize(
    ("optimized", "message"),
    [
        (
            {
                "method": "black_litterman",
                "equilibrium_weights": {"AAA": 0.5, "BBB": 0.5},
            },
            "baselines.optimized.views must be non-empty when baselines.optimized.method='black_litterman'",
        ),
        (
            {
                "method": "black_litterman",
                "equilibrium_weights": {"AAA": 0.5, "BBB": 0.5},
                "views": [{"name": "", "weights": {"AAA": 1.0}, "view_return": 0.01}],
            },
            r"baselines\.optimized\.views\[0\]\.name must be non-empty",
        ),
        (
            {
                "method": "black_litterman",
                "equilibrium_weights": {"AAA": 0.5, "BBB": 0.5},
                "views": [{"name": "bad", "weights": {"CCC": 1.0}, "view_return": 0.01}],
            },
            r"baselines\.optimized\.views\[0\]\.weights contains unknown symbols: CCC",
        ),
        (
            {
                "method": "black_litterman",
                "equilibrium_weights": {"AAA": 0.5, "BBB": 0.5},
                "views": [{"name": "bad", "weights": {}, "view_return": 0.01}],
            },
            r"baselines\.optimized\.views\[0\]\.weights must not be empty",
        ),
        (
            {
                "method": "black_litterman",
                "equilibrium_weights": {"AAA": 0.5, "BBB": 0.5},
                "views": [{"name": "bad", "weights": {"AAA": 0.0}, "view_return": 0.01}],
            },
            r"baselines\.optimized\.views\[0\]\.weights must contain at least one non-zero coefficient",
        ),
        (
            {
                "method": "black_litterman",
                "equilibrium_weights": {"AAA": 0.5, "BBB": 0.5},
                "views": [{"name": "bad", "weights": {"AAA": float("inf")}, "view_return": 0.01}],
            },
            r"baselines\.optimized\.views\[0\]\.weights\[AAA\] must be finite",
        ),
        (
            {
                "method": "black_litterman",
                "equilibrium_weights": {"AAA": 0.5, "BBB": 0.5},
                "views": [{"name": "bad", "weights": {"AAA": 1.0}, "view_return": float("nan")}],
            },
            r"baselines\.optimized\.views\[0\]\.view_return must be finite",
        ),
    ],
)
def test_load_config_rejects_invalid_black_litterman_view_settings(
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


def test_load_config_accepts_phase7_paper_settings(tmp_path: Path) -> None:
    config_path = _write_config(
        tmp_path / "config.yaml",
        data={
            "symbols": ["VOO"],
            "interval": "1d",
        },
    )
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    payload["target"] = {"horizon_days": 1, "type": "direction"}
    payload["portfolio"] = {
        "ranking": {
            "long_n": 1,
            "short_n": 1,
            "rebalance_frequency": "D",
            "mode": "long_only",
        }
    }
    payload["models"] = [
        {"name": "logistic_regression"},
        {"name": "logistic_l1"},
        {"name": "random_forest"},
        {"name": "extra_trees"},
        {"name": "gradient_boosting"},
        {"name": "hist_gradient_boosting"},
    ]
    payload["paper"] = {
        "enabled": True,
        "data_provider": "alpaca",
        "broker": "alpaca",
        "execution_mode": "agent_approval",
        "agent_backend": "openai",
        "agent_model": "gpt-4o-mini",
        "agent_timeout_seconds": 45,
        "agent_fallback_backend": "deterministic_consensus",
        "consensus_min_long_votes": 4,
        "schedule_timezone": "America/New_York",
        "decision_time": "16:10",
        "submission_time": "19:05",
        "order_type": "day_market",
        "position_sizing": "full_equity_fractional",
        "approval_inbox_dir": "artifacts/paper/inbox",
        "state_dir": "artifacts/paper/state",
        "poll_interval_seconds": 15,
        "notifications": {
            "telegram": {
                "enabled": True,
            }
        },
    }
    config_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

    config = load_config(config_path)

    assert config.paper.enabled is True
    assert config.paper.execution_mode == "agent_approval"
    assert config.paper.agent_backend == "openai"
    assert config.paper.agent_model == "gpt-4o-mini"
    assert config.paper.agent_timeout_seconds == 45
    assert config.paper.consensus_min_long_votes == 4
    assert config.paper.notifications.telegram.enabled is True
    assert config.paper_approval_inbox_dir == (tmp_path / "artifacts" / "paper" / "inbox").resolve()
    assert config.paper_state_dir == (tmp_path / "artifacts" / "paper" / "state").resolve()


def test_load_config_defaults_paper_telegram_notifications_to_disabled(tmp_path: Path) -> None:
    config_path = _write_config(
        tmp_path / "config.yaml",
        data={"symbols": ["VOO"], "interval": "1d"},
    )
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    payload["target"] = {"horizon_days": 1, "type": "direction"}
    payload["portfolio"] = {
        "ranking": {
            "long_n": 1,
            "short_n": 1,
            "rebalance_frequency": "D",
            "mode": "long_only",
        }
    }
    payload["models"] = [
        {"name": "logistic_regression"},
        {"name": "logistic_l1"},
        {"name": "random_forest"},
        {"name": "extra_trees"},
        {"name": "gradient_boosting"},
        {"name": "hist_gradient_boosting"},
    ]
    payload["paper"] = {
        "enabled": True,
        "execution_mode": "agent_approval",
    }
    config_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

    config = load_config(config_path)

    assert config.paper.notifications.telegram.enabled is False


def test_load_config_rejects_unknown_paper_agent_backend(tmp_path: Path) -> None:
    config_path = _write_config(
        tmp_path / "config.yaml",
        data={"symbols": ["QQQ"], "interval": "1d"},
    )
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    payload["paper"] = {
        "enabled": True,
        "agent_backend": "unknown",
    }
    config_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

    with pytest.raises(ValueError, match="paper.agent_backend must be one of"):
        load_config(config_path)


def test_load_config_rejects_openai_backend_without_model(tmp_path: Path) -> None:
    config_path = _write_config(
        tmp_path / "config.yaml",
        data={"symbols": ["QQQ"], "interval": "1d"},
    )
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    payload["paper"] = {
        "enabled": True,
        "agent_backend": "openai",
        "agent_model": "",
    }
    config_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

    with pytest.raises(ValueError, match="paper.agent_model must be set"):
        load_config(config_path)
