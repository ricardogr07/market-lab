from __future__ import annotations

from pathlib import Path

import pytest

from marketlab.config import load_config

SCENARIO_CONFIG_PATHS = {
    "phase5_allocation_equal": Path("configs/experiment.phase5.allocation_equal.yaml"),
    "phase5_allocation_group": Path("configs/experiment.phase5.allocation_group.yaml"),
    "phase5_ranking_default": Path("configs/experiment.phase5.ranking_default.yaml"),
    "phase5_ranking_capped": Path("configs/experiment.phase5.ranking_capped.yaml"),
    "phase5_mean_variance": Path("configs/experiment.phase5.mean_variance.yaml"),
    "phase5_risk_parity": Path("configs/experiment.phase5.risk_parity.yaml"),
    "phase5_black_litterman": Path("configs/experiment.phase5.black_litterman.yaml"),
}
SHARED_SYMBOLS = ["VOO", "QQQ", "SMH", "XLV", "IEMG"]
SHARED_SYMBOL_GROUPS = {
    "VOO": "broad_market",
    "QQQ": "growth",
    "SMH": "growth",
    "XLV": "defensive",
    "IEMG": "broad_market",
}
SHARED_MODEL_NAMES = [
    "logistic_regression",
    "logistic_l1",
    "random_forest",
    "extra_trees",
    "gradient_boosting",
    "hist_gradient_boosting",
]


def _load_scenario(name: str):
    return load_config(SCENARIO_CONFIG_PATHS[name])


def _assert_shared_phase5_frame(config) -> None:
    assert config.data.symbols == SHARED_SYMBOLS
    assert config.data.start_date == "2018-01-01"
    assert config.data.end_date == "2025-12-31"
    assert config.data.interval == "1d"
    assert config.data.cache_dir == "artifacts/data"
    assert config.data.prepared_panel_filename == "panel.csv"
    assert config.data.symbol_groups == SHARED_SYMBOL_GROUPS

    assert config.features.return_windows == [5, 10, 20, 40]
    assert config.features.ma_windows == [10, 20, 50]
    assert config.features.vol_windows == [10, 20]
    assert config.features.momentum_window == 20

    assert config.target.horizon_days == 5
    assert config.target.type == "direction"

    ranking = config.portfolio.ranking
    assert ranking.long_n == 2
    assert ranking.short_n == 2
    assert ranking.rebalance_frequency == "W-FRI"
    assert ranking.weighting == "equal"
    assert ranking.mode == "long_short"
    assert ranking.min_score_threshold == pytest.approx(0.0)
    assert ranking.cash_when_underfilled is False

    assert config.portfolio.costs.bps_per_trade == pytest.approx(10.0)
    assert config.baselines.buy_hold is True
    assert config.baselines.sma.enabled is True
    assert config.baselines.sma.fast_window == 20
    assert config.baselines.sma.slow_window == 50
    assert [model.name for model in config.models] == SHARED_MODEL_NAMES

    walk_forward = config.evaluation.walk_forward
    assert walk_forward.train_years == 3
    assert walk_forward.test_months == 3
    assert walk_forward.step_months == 3
    assert walk_forward.min_train_rows == 100
    assert walk_forward.min_test_rows == 20
    assert walk_forward.min_train_positive_rate == pytest.approx(0.05)
    assert walk_forward.min_test_positive_rate == pytest.approx(0.05)
    assert walk_forward.embargo_periods == 1
    assert config.evaluation.benchmark_strategy == "buy_hold"
    assert config.evaluation.cost_sensitivity_bps == [5.0, 25.0]
    assert config.evaluation.factor_model_path == ""
    assert config.factor_model_path is None

    assert config.artifacts.output_dir == "artifacts/runs"
    assert config.artifacts.save_predictions is True
    assert config.artifacts.save_metrics_csv is True
    assert config.artifacts.save_report_md is True
    assert config.artifacts.save_plots is True


@pytest.mark.parametrize("scenario_name", list(SCENARIO_CONFIG_PATHS))
def test_phase5_scenario_configs_share_the_same_comparison_frame(scenario_name: str) -> None:
    config = _load_scenario(scenario_name)

    assert config.experiment_name == scenario_name
    _assert_shared_phase5_frame(config)


def test_phase5_ranking_scenarios_define_only_the_intended_risk_delta() -> None:
    default = _load_scenario("phase5_ranking_default")
    capped = _load_scenario("phase5_ranking_capped")

    assert default.baselines.allocation.enabled is False
    assert default.baselines.optimized.enabled is False
    assert default.portfolio.risk.max_position_weight is None
    assert default.portfolio.risk.max_group_weight is None
    assert default.portfolio.risk.max_long_exposure is None
    assert default.portfolio.risk.max_short_exposure is None

    assert capped.baselines.allocation.enabled is False
    assert capped.baselines.optimized.enabled is False
    assert capped.portfolio.risk.max_position_weight == pytest.approx(0.30)
    assert capped.portfolio.risk.max_group_weight == pytest.approx(0.35)
    assert capped.portfolio.risk.max_long_exposure == pytest.approx(0.60)
    assert capped.portfolio.risk.max_short_exposure == pytest.approx(0.60)


def test_phase5_allocation_scenarios_enable_the_expected_baselines() -> None:
    equal = _load_scenario("phase5_allocation_equal")
    grouped = _load_scenario("phase5_allocation_group")

    assert equal.baselines.allocation.enabled is True
    assert equal.baselines.allocation.mode == "equal"
    assert equal.baselines.allocation.symbol_weights == {}
    assert equal.baselines.allocation.group_weights == {}
    assert equal.baselines.optimized.enabled is False

    assert grouped.baselines.allocation.enabled is True
    assert grouped.baselines.allocation.mode == "group_weights"
    assert grouped.baselines.allocation.symbol_weights == {}
    assert grouped.baselines.allocation.group_weights == {
        "broad_market": pytest.approx(0.50),
        "growth": pytest.approx(0.30),
        "defensive": pytest.approx(0.20),
    }
    assert grouped.baselines.optimized.enabled is False


def test_phase5_optimized_scenarios_enable_the_expected_methods() -> None:
    mean_variance = _load_scenario("phase5_mean_variance")
    risk_parity = _load_scenario("phase5_risk_parity")
    black_litterman = _load_scenario("phase5_black_litterman")

    for config, method in (
        (mean_variance, "mean_variance"),
        (risk_parity, "risk_parity"),
        (black_litterman, "black_litterman"),
    ):
        optimized = config.baselines.optimized
        assert config.baselines.allocation.enabled is False
        assert optimized.enabled is True
        assert optimized.method == method
        assert optimized.lookback_days == 252
        assert optimized.rebalance_frequency == "W-FRI"
        assert optimized.covariance_estimator == "sample"
        assert optimized.expected_return_source == "historical_mean"
        assert optimized.long_only is True
        assert optimized.target_gross_exposure == pytest.approx(1.0)
        assert optimized.risk_aversion == pytest.approx(1.0)

    assert mean_variance.baselines.optimized.equilibrium_weights == {}
    assert mean_variance.baselines.optimized.views == []
    assert risk_parity.baselines.optimized.equilibrium_weights == {}
    assert risk_parity.baselines.optimized.views == []

    optimized = black_litterman.baselines.optimized
    assert optimized.equilibrium_weights == {
        "VOO": pytest.approx(0.20),
        "QQQ": pytest.approx(0.20),
        "SMH": pytest.approx(0.20),
        "XLV": pytest.approx(0.20),
        "IEMG": pytest.approx(0.20),
    }
    assert optimized.tau == pytest.approx(0.05)
    assert len(optimized.views) == 2
    assert optimized.views[0].name == "growth_over_defensive"
    assert optimized.views[0].weights == {
        "QQQ": pytest.approx(1.0),
        "SMH": pytest.approx(1.0),
        "XLV": pytest.approx(-1.0),
    }
    assert optimized.views[0].view_return == pytest.approx(0.0010)
    assert optimized.views[1].name == "core_over_international"
    assert optimized.views[1].weights == {
        "VOO": pytest.approx(1.0),
        "IEMG": pytest.approx(-1.0),
    }
    assert optimized.views[1].view_return == pytest.approx(0.0005)
