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
    "crypto_hourly_trend": Path("configs/experiment.crypto_hourly_trend.yaml"),
    "crypto_15m_signal_week": Path("configs/experiment.crypto_15m_signal_week.yaml"),
    "crypto_15m_patterns_week": Path("configs/experiment.crypto_15m_patterns_week.yaml"),
    "crypto_15m_patterns_day": Path("configs/experiment.crypto_15m_patterns_day.yaml"),
    "crypto_5m_patterns_day": Path("configs/experiment.crypto_5m_patterns_day.yaml"),
    "crypto_pattern_exit_tuned_2024_ytd": Path(
        "configs/experiment.crypto_pattern_exit_tuned_2024_ytd.yaml"
    ),
    "crypto_pattern_meta_label_2024_ytd": Path(
        "configs/experiment.crypto_pattern_meta_label_2024_ytd.yaml"
    ),
    "crypto_pattern_meta_tuned_2024_ytd": Path(
        "configs/experiment.crypto_pattern_meta_tuned_2024_ytd.yaml"
    ),
    "crypto_ts_ml_2024_ytd": Path("configs/experiment.crypto_ts_ml_2024_ytd.yaml"),
    "crypto_indicator_ml_tuned_6h_2024_ytd": Path(
        "configs/experiment.crypto_indicator_ml_tuned_6h_2024_ytd.yaml"
    ),
    "crypto_indicator_ml_tuned_12h_2024_ytd": Path(
        "configs/experiment.crypto_indicator_ml_tuned_12h_2024_ytd.yaml"
    ),
    "crypto_indicator_ml_tuned_24h_2024_ytd": Path(
        "configs/experiment.crypto_indicator_ml_tuned_24h_2024_ytd.yaml"
    ),
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


@pytest.mark.parametrize("scenario_name", [name for name in SCENARIO_CONFIG_PATHS if name.startswith("phase5_")])
def test_phase5_scenario_configs_share_the_same_comparison_frame(scenario_name: str) -> None:
    config = _load_scenario(scenario_name)

    assert config.experiment_name == scenario_name
    _assert_shared_phase5_frame(config)


def test_crypto_hourly_trend_config_is_research_first_and_paper_disabled() -> None:
    config = _load_scenario("crypto_hourly_trend")

    assert config.experiment_name == "crypto_hourly_trend"
    assert config.data.symbols == ["BTC-USD"]
    assert config.data.interval == "1h"
    assert config.portfolio.ranking.rebalance_frequency == "bar"
    assert config.portfolio.ranking.mode == "long_only"
    assert config.baselines.buy_hold is True
    assert config.baselines.sma.enabled is False
    assert config.baselines.indicator_stack.enabled is True
    assert config.evaluation.benchmark_strategy == "buy_hold"
    assert config.evaluation.periods_per_year == pytest.approx(8760.0)
    assert config.portfolio.costs.bps_per_trade == pytest.approx(10.0)
    assert config.evaluation.cost_sensitivity_bps == [7.5, 10.0, 20.0, 40.0, 75.0]
    assert config.paper.enabled is False


def test_crypto_15m_signal_week_config_enables_visual_inspection_only() -> None:
    config = _load_scenario("crypto_15m_signal_week")

    assert config.experiment_name == "crypto_15m_signal_week"
    assert config.data.symbols == ["BTC-USD"]
    assert config.data.interval == "15m"
    assert config.portfolio.ranking.rebalance_frequency == "bar"
    assert config.portfolio.ranking.mode == "long_only"
    assert config.baselines.buy_hold is True
    assert config.baselines.sma.enabled is False
    assert config.baselines.indicator_stack.enabled is True
    assert config.evaluation.benchmark_strategy == "buy_hold"
    assert config.evaluation.periods_per_year == pytest.approx(35040.0)
    assert config.portfolio.costs.bps_per_trade == pytest.approx(10.0)
    assert config.evaluation.cost_sensitivity_bps == [7.5, 10.0, 20.0, 40.0, 75.0]
    assert config.evaluation.focus_start == "2024-01-01 00:00:00"
    assert config.evaluation.focus_end == "2024-01-07 23:59:59"
    assert config.evaluation.visualize_signals is True
    assert config.paper.enabled is False


def test_crypto_15m_patterns_week_config_enables_pattern_research_only() -> None:
    config = _load_scenario("crypto_15m_patterns_week")

    assert config.experiment_name == "crypto_15m_patterns_week"
    assert config.data.symbols == ["BTC-USD"]
    assert config.data.interval == "15m"
    assert config.portfolio.ranking.rebalance_frequency == "bar"
    assert config.baselines.buy_hold is True
    assert config.baselines.sma.enabled is False
    assert config.baselines.indicator_stack.enabled is False
    assert config.baselines.chart_patterns.enabled is True
    assert config.baselines.chart_patterns.lookback_bars == 96
    assert config.evaluation.benchmark_strategy == "buy_hold"
    assert config.evaluation.periods_per_year == pytest.approx(35040.0)
    assert config.portfolio.costs.bps_per_trade == pytest.approx(10.0)
    assert config.evaluation.cost_sensitivity_bps == [7.5, 10.0, 20.0, 40.0, 75.0]
    assert config.evaluation.visualize_signals is True
    assert config.paper.enabled is False


def test_crypto_15m_patterns_day_config_focuses_single_inspection_day() -> None:
    config = _load_scenario("crypto_15m_patterns_day")

    assert config.experiment_name == "crypto_15m_patterns_day"
    assert config.data.symbols == ["BTC-USD"]
    assert config.data.interval == "15m"
    assert config.data.start_date == "2026-04-20"
    assert config.data.end_date == "2026-04-27"
    assert config.baselines.chart_patterns.enabled is True
    assert config.portfolio.costs.bps_per_trade == pytest.approx(10.0)
    assert config.evaluation.cost_sensitivity_bps == [7.5, 10.0, 20.0, 40.0, 75.0]
    assert config.evaluation.focus_start == "2026-04-22 00:00:00"
    assert config.evaluation.focus_end == "2026-04-22 23:59:59"
    assert config.evaluation.visualize_signals is True
    assert config.paper.enabled is False


def test_crypto_5m_patterns_day_config_focuses_single_inspection_day() -> None:
    config = _load_scenario("crypto_5m_patterns_day")

    assert config.experiment_name == "crypto_5m_patterns_day"
    assert config.data.symbols == ["BTC-USD"]
    assert config.data.interval == "5m"
    assert config.data.start_date == "2026-04-20"
    assert config.data.end_date == "2026-04-23"
    assert config.baselines.chart_patterns.enabled is True
    assert config.baselines.chart_patterns.lookback_bars == 288
    assert config.evaluation.periods_per_year == pytest.approx(105120.0)
    assert config.portfolio.costs.bps_per_trade == pytest.approx(10.0)
    assert config.evaluation.cost_sensitivity_bps == [7.5, 10.0, 20.0, 40.0, 75.0]
    assert config.evaluation.focus_start == "2026-04-22 00:00:00"
    assert config.evaluation.focus_end == "2026-04-22 23:59:59"
    assert config.evaluation.visualize_signals is True
    assert config.paper.enabled is False


def test_crypto_pattern_exit_tuned_2024_ytd_config_is_research_only() -> None:
    config = _load_scenario("crypto_pattern_exit_tuned_2024_ytd")

    assert config.experiment_name == "crypto_pattern_exit_tuned_2024_ytd"
    assert config.data.symbols == ["BTC-USD"]
    assert config.data.interval == "1h"
    assert config.baselines.chart_patterns.enabled is True
    assert config.baselines.pattern_exit_overlay.enabled is True
    assert config.baselines.pattern_exit_overlay.trend_ema_window == 168
    assert config.baselines.pattern_exit_overlay.reentry_clear_bars == 6
    assert config.baselines.pattern_exit_overlay.require_price_below_trend_for_exit is True
    assert config.baselines.pattern_exit_overlay.bearish_confirmation_window_bars == 6
    assert config.baselines.pattern_exit_overlay.min_cash_bars == 12
    assert config.baselines.pattern_exit_overlay.exit_cooldown_bars == 48
    assert config.baselines.pattern_meta_label.enabled is True
    assert config.baselines.pattern_meta_label.exit_probability_threshold == pytest.approx(0.7)
    assert config.baselines.pattern_meta_label.exit_probability_threshold_grid == [
        0.55,
        0.6,
        0.65,
        0.7,
        0.75,
        0.8,
        0.85,
        0.9,
    ]
    assert config.evaluation.benchmark_strategy == "buy_hold"
    assert config.portfolio.costs.bps_per_trade == pytest.approx(10.0)
    assert config.paper.enabled is False


def test_crypto_pattern_meta_label_2024_ytd_config_is_research_only() -> None:
    config = _load_scenario("crypto_pattern_meta_label_2024_ytd")

    assert config.experiment_name == "crypto_pattern_meta_label_2024_ytd"
    assert config.data.symbols == ["BTC-USD"]
    assert config.data.interval == "1h"
    assert config.baselines.chart_patterns.enabled is True
    assert config.baselines.pattern_exit_overlay.enabled is True
    assert config.baselines.pattern_meta_label.enabled is True
    assert config.baselines.pattern_meta_label.label_horizon_bars == 12
    assert config.baselines.pattern_meta_label.exit_probability_threshold == pytest.approx(
        0.55
    )
    assert config.evaluation.benchmark_strategy == "buy_hold"
    assert config.portfolio.costs.bps_per_trade == pytest.approx(10.0)
    assert config.paper.enabled is False


def test_crypto_pattern_meta_tuned_2024_ytd_config_enables_nested_research() -> None:
    config = _load_scenario("crypto_pattern_meta_tuned_2024_ytd")

    assert config.experiment_name == "crypto_pattern_meta_tuned_2024_ytd"
    assert config.data.symbols == ["BTC-USD"]
    assert config.data.interval == "1h"
    assert config.baselines.chart_patterns.enabled is True
    assert config.baselines.pattern_exit_overlay.enabled is True
    assert config.baselines.pattern_meta_label.enabled is True
    assert config.baselines.pattern_meta_label.tuning_mode == "nested_walk_forward"
    assert config.baselines.pattern_meta_label.min_oos_exit_count == 1
    assert config.baselines.pattern_meta_label.max_average_exposure_for_active == pytest.approx(
        0.999
    )
    assert config.baselines.pattern_partial_exposure_overlay.enabled is True
    assert config.baselines.pattern_partial_exposure_overlay.partial_weight == pytest.approx(
        0.5
    )
    assert config.evaluation.benchmark_strategy == "buy_hold"
    assert config.portfolio.costs.bps_per_trade == pytest.approx(10.0)
    assert config.paper.enabled is False


def test_crypto_ts_ml_2024_ytd_config_enables_direct_time_series_research() -> None:
    config = _load_scenario("crypto_ts_ml_2024_ytd")

    assert config.experiment_name == "crypto_ts_ml_2024_ytd"
    assert config.data.symbols == ["BTC-USD"]
    assert config.data.interval == "1h"
    assert config.features.crypto_time_series_enabled is True
    assert config.features.crypto_return_windows == [1, 3, 6, 12, 24, 72, 168]
    assert config.target.horizon_days == 12
    assert config.portfolio.ranking.rebalance_frequency == "bar"
    assert config.portfolio.ranking.mode == "long_only"
    assert config.portfolio.ranking.long_n == 1
    assert config.portfolio.ranking.cash_when_underfilled is True
    assert config.portfolio.ranking.min_score_threshold == pytest.approx(0.55)
    assert [model.name for model in config.models] == [
        "logistic_l1",
        "random_forest",
        "extra_trees",
        "gradient_boosting",
        "hist_gradient_boosting",
    ]
    ml_sweep = config.evaluation.ml_strategy_threshold_sweep
    assert ml_sweep.enabled is True
    assert ml_sweep.thresholds == [0.50, 0.52, 0.55, 0.58, 0.60]
    assert ml_sweep.min_exposure_changes == 5
    assert ml_sweep.max_average_exposure_for_active == pytest.approx(0.995)
    assert config.baselines.chart_patterns.enabled is True
    assert config.baselines.pattern_meta_label.enabled is True
    assert config.baselines.pattern_partial_exposure_overlay.enabled is True
    assert config.evaluation.benchmark_strategy == "buy_hold"
    assert config.portfolio.costs.bps_per_trade == pytest.approx(10.0)
    assert config.paper.enabled is False


@pytest.mark.parametrize(
    ("scenario_name", "horizon"),
    [
        ("crypto_indicator_ml_tuned_6h_2024_ytd", 6),
        ("crypto_indicator_ml_tuned_12h_2024_ytd", 12),
        ("crypto_indicator_ml_tuned_24h_2024_ytd", 24),
    ],
)
def test_crypto_indicator_ml_tuned_configs_enable_phase_8_7_research(
    scenario_name: str,
    horizon: int,
) -> None:
    config = _load_scenario(scenario_name)

    assert config.data.symbols == ["BTC-USD"]
    assert config.data.interval == "1h"
    assert config.features.indicator_stack_ml_features_enabled is True
    assert config.features.crypto_time_series_enabled is True
    assert config.target.horizon_days == horizon
    assert config.portfolio.ranking.rebalance_frequency == "bar"
    assert config.portfolio.ranking.mode == "long_only"
    assert config.portfolio.ranking.cash_when_underfilled is True
    assert config.baselines.indicator_stack.enabled is True
    assert [model.name for model in config.models] == [
        "logistic_l1",
        "random_forest",
        "extra_trees",
        "gradient_boosting",
        "hist_gradient_boosting",
    ]
    assert config.evaluation.ml_strategy_tuning.enabled is True
    assert config.evaluation.ml_strategy_tuning.thresholds == [
        0.50,
        0.52,
        0.55,
        0.58,
        0.60,
        0.62,
        0.65,
    ]
    assert config.evaluation.ml_strategy_tuning.objective == (
        "net_return_and_risk_vs_buy_hold"
    )
    assert config.evaluation.benchmark_strategy == "buy_hold"
    assert config.paper.enabled is False


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
