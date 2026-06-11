from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from marketlab.config import default_periods_per_year, load_config


def _write_config(
    path: Path,
    *,
    data: dict[str, object] | None = None,
    features: dict[str, object] | None = None,
    target: dict[str, object] | None = None,
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
    if features is not None:
        payload["features"] = features
    if target is not None:
        payload["target"] = target
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
    assert config.baselines.partial_allocation_benchmarks.enabled is False
    assert config.baselines.partial_allocation_benchmarks.weights == []
    assert config.baselines.rebalanced_partial_allocation_benchmarks.enabled is False
    assert config.baselines.rebalanced_partial_allocation_benchmarks.weights == []
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
    assert config.evaluation.periods_per_year == pytest.approx(252.0)
    assert config.evaluation.focus_start == ""
    assert config.evaluation.focus_end == ""
    assert config.evaluation.visualize_signals is False
    assert config.evaluation.ml_strategy_threshold_sweep.enabled is False
    assert config.evaluation.ml_strategy_threshold_sweep.thresholds == [
        0.50,
        0.52,
        0.55,
        0.58,
        0.60,
    ]
    assert config.evaluation.ml_strategy_tuning.enabled is False
    assert config.evaluation.ml_strategy_tuning.thresholds == [
        0.50,
        0.52,
        0.55,
        0.58,
        0.60,
        0.62,
        0.65,
    ]
    assert config.evaluation.ml_strategy_tuning.rolling_train_bars_grid == []
    assert config.evaluation.ml_strategy_tuning.min_holding_period_bars_grid == [0]
    assert config.evaluation.ml_strategy_tuning.hysteresis_margin_grid == [0.0]
    assert config.evaluation.ml_strategy_tuning.max_annualized_turnover is None
    assert config.evaluation.ml_strategy_tuning.selection_policy == "strict"
    assert config.evaluation.ml_strategy_tuning.no_candidate_fallback_regime_policy is None
    assert config.evaluation.ml_strategy_tuning.no_valid_candidate_regime_fallback is None
    assert (
        config.evaluation.ml_strategy_tuning.allocation_score_policy
        == "expected_allocation"
    )
    assert (
        config.evaluation.ml_strategy_tuning.allocation_score_policy_prob100_threshold
        == pytest.approx(0.20)
    )
    assert config.evaluation.ml_strategy_tuning.allocation_score_policy_prob100_threshold_grid == []
    assert [
        (
            transform.name,
            transform.bull_multiplier,
            transform.bull_addend,
            transform.risk_off_score_cap,
            transform.non_bull_score_cap,
        )
        for transform in config.evaluation.ml_strategy_tuning.allocation_score_transforms
    ] == [("identity", 1.0, 0.0, None, None)]
    assert [
        (
            policy.name,
            policy.bull_floor,
            policy.sideways_floor,
            policy.bear_floor,
            policy.risk_off_cap,
            policy.gate_bull_floor,
        )
        for policy in config.evaluation.ml_strategy_tuning.regime_participation_policies
    ] == [("model_only", 0.0, 0.0, 0.0, 0.25, None)]
    assert config.baselines.pattern_exit_overlay.enabled is False
    assert config.baselines.pattern_meta_label.enabled is False


def test_load_config_accepts_crypto_time_series_features_and_ml_sweep(
    tmp_path: Path,
) -> None:
    config_path = _write_config(
        tmp_path / "config.yaml",
        features={
            "return_windows": [12, 24],
            "ma_windows": [24],
            "vol_windows": [24],
            "momentum_window": 24,
            "indicator_stack_ml_features_enabled": True,
            "crypto_time_series_enabled": True,
            "crypto_return_windows": [1, 12, 24],
            "crypto_vol_windows": [12, 24],
            "crypto_ma_windows": [24],
            "crypto_rsi_window": 14,
            "crypto_macd_fast_window": 12,
            "crypto_macd_slow_window": 26,
            "crypto_macd_signal_window": 9,
            "crypto_bollinger_window": 20,
            "crypto_bollinger_std": 2.0,
            "crypto_volume_window": 24,
            "crypto_time_features": True,
            "crypto_regime_signal_features_enabled": True,
        },
        evaluation={
            "ml_strategy_threshold_sweep": {
                "enabled": True,
                "thresholds": [0.5, 0.55],
                "min_exposure_changes": 5,
                "max_average_exposure_for_active": 0.995,
            },
            "ml_strategy_tuning": {
                "enabled": True,
                "thresholds": [0.5, 0.55, 0.6],
                "validation_months": 2,
                "min_validation_rows": 50,
                "min_exposure_changes": 3,
                "max_average_exposure_for_active": 0.99,
                "rolling_train_bars_grid": [540, 1095],
                "min_holding_period_bars_grid": [0, 6],
                "hysteresis_margin_grid": [0.0, 0.02],
                "max_annualized_turnover": 24.0,
                "objective": "net_return_and_risk_vs_buy_hold",
            },
        },
    )

    config = load_config(config_path)

    assert config.features.indicator_stack_ml_features_enabled is True
    assert config.features.crypto_time_series_enabled is True
    assert config.features.crypto_regime_signal_features_enabled is True
    assert config.features.crypto_return_windows == [1, 12, 24]
    assert config.features.crypto_ma_windows == [24]
    assert config.evaluation.ml_strategy_threshold_sweep.enabled is True
    assert config.evaluation.ml_strategy_threshold_sweep.thresholds == [0.5, 0.55]
    assert config.evaluation.ml_strategy_threshold_sweep.min_exposure_changes == 5
    assert config.evaluation.ml_strategy_threshold_sweep.max_average_exposure_for_active == pytest.approx(
        0.995
    )
    assert config.evaluation.ml_strategy_tuning.enabled is True
    assert config.evaluation.ml_strategy_tuning.thresholds == [0.5, 0.55, 0.6]
    assert config.evaluation.ml_strategy_tuning.validation_months == 2
    assert config.evaluation.ml_strategy_tuning.min_validation_rows == 50
    assert config.evaluation.ml_strategy_tuning.min_exposure_changes == 3
    assert config.evaluation.ml_strategy_tuning.max_average_exposure_for_active == pytest.approx(
        0.99
    )
    assert config.evaluation.ml_strategy_tuning.rolling_train_bars_grid == [540, 1095]
    assert config.evaluation.ml_strategy_tuning.min_holding_period_bars_grid == [0, 6]
    assert config.evaluation.ml_strategy_tuning.hysteresis_margin_grid == [0.0, 0.02]
    assert config.evaluation.ml_strategy_tuning.max_annualized_turnover == pytest.approx(24.0)
    assert config.evaluation.ml_strategy_tuning.selection_policy == "strict"
    assert config.evaluation.ml_strategy_tuning.selection_benchmark_strategies == []
    assert config.target.allocation_utility_risk_penalty_power == pytest.approx(2.0)


def test_load_config_accepts_no_candidate_fallback_regime_policy(
    tmp_path: Path,
) -> None:
    config_path = _write_config(
        tmp_path / "config.yaml",
        evaluation={
            "ml_strategy_tuning": {
                "enabled": True,
                "no_candidate_fallback_regime_policy": "bull100_sideways25",
                "regime_participation_policies": [
                    {
                        "name": "bull100_sideways25",
                        "bull_floor": 1.0,
                        "sideways_floor": 0.25,
                        "bear_floor": 0.0,
                        "risk_off_cap": 0.25,
                    }
                ],
            },
        },
    )

    config = load_config(config_path)

    assert (
        config.evaluation.ml_strategy_tuning.no_candidate_fallback_regime_policy
        == "bull100_sideways25"
    )
    assert (
        config.evaluation.ml_strategy_tuning.no_valid_candidate_regime_fallback
        == "bull100_sideways25"
    )


def test_load_config_accepts_regime_gate_bull_floor(tmp_path: Path) -> None:
    config_path = _write_config(
        tmp_path / "config.yaml",
        evaluation={
            "ml_strategy_tuning": {
                "enabled": True,
                "regime_participation_policies": [
                    {
                        "name": "gate_bull_override",
                        "bull_floor": 0.75,
                        "sideways_floor": 0.25,
                        "bear_floor": 0.0,
                        "risk_off_cap": 0.25,
                        "gate_bull_floor": 1.0,
                    }
                ],
            },
        },
    )

    config = load_config(config_path)

    policy = config.evaluation.ml_strategy_tuning.regime_participation_policies[0]
    assert policy.gate_bull_floor == pytest.approx(1.0)


def test_load_config_rejects_invalid_regime_gate_bull_floor(tmp_path: Path) -> None:
    config_path = _write_config(
        tmp_path / "config.yaml",
        evaluation={
            "ml_strategy_tuning": {
                "enabled": True,
                "regime_participation_policies": [
                    {
                        "name": "bad_gate_bull_floor",
                        "gate_bull_floor": 0.33,
                    }
                ],
            },
        },
    )

    with pytest.raises(ValueError, match="gate_bull_floor"):
        load_config(config_path)


def test_load_config_accepts_no_valid_candidate_regime_fallback_alias(
    tmp_path: Path,
) -> None:
    config_path = _write_config(
        tmp_path / "config.yaml",
        evaluation={
            "ml_strategy_tuning": {
                "enabled": True,
                "no_valid_candidate_regime_fallback": "bull100_sideways25",
                "allocation_score_policy": "bull_prob100_threshold",
                "allocation_score_policy_prob100_threshold_grid": [0.2, 0.36],
                "regime_participation_policies": [
                    {
                        "name": "bull100_sideways25",
                        "bull_floor": 1.0,
                        "sideways_floor": 0.25,
                        "bear_floor": 0.0,
                        "risk_off_cap": 0.25,
                    }
                ],
            },
        },
    )

    config = load_config(config_path)

    assert (
        config.evaluation.ml_strategy_tuning.no_candidate_fallback_regime_policy
        == "bull100_sideways25"
    )
    assert (
        config.evaluation.ml_strategy_tuning.no_valid_candidate_regime_fallback
        == "bull100_sideways25"
    )
    assert config.evaluation.ml_strategy_tuning.allocation_score_policy_prob100_threshold_grid == [
        0.2,
        0.36,
    ]


def test_load_config_accepts_allocation_score_transforms(tmp_path: Path) -> None:
    config_path = _write_config(
        tmp_path / "config.yaml",
        evaluation={
            "ml_strategy_tuning": {
                "enabled": True,
                "allocation_score_transforms": [
                    {"name": "identity"},
                    {
                        "name": "bull_shift_18",
                        "bull_multiplier": 1.0,
                        "bull_addend": 0.18,
                        "risk_off_score_cap": 0.25,
                        "non_bull_score_cap": 0.50,
                    },
                ],
            },
        },
    )

    config = load_config(config_path)

    transforms = config.evaluation.ml_strategy_tuning.allocation_score_transforms
    assert [(transform.name, transform.bull_addend) for transform in transforms] == [
        ("identity", 0.0),
        ("bull_shift_18", 0.18),
    ]
    assert transforms[1].bull_multiplier == pytest.approx(1.0)
    assert transforms[1].risk_off_score_cap == pytest.approx(0.25)
    assert transforms[1].non_bull_score_cap == pytest.approx(0.50)


@pytest.mark.parametrize(
    "allocation_score_transforms",
    [
        [{"name": ""}],
        [{"name": "duplicate"}, {"name": "duplicate"}],
        [{"name": "bad_multiplier", "bull_multiplier": float("nan")}],
        [{"name": "bad_multiplier", "bull_multiplier": -0.01}],
        [{"name": "bad_addend", "bull_addend": float("inf")}],
        [{"name": "bad_cap", "risk_off_score_cap": 1.10}],
        [{"name": "bad_cap", "non_bull_score_cap": -0.01}],
    ],
)
def test_load_config_rejects_invalid_allocation_score_transforms(
    tmp_path: Path,
    allocation_score_transforms: list[dict[str, object]],
) -> None:
    config_path = _write_config(
        tmp_path / "config.yaml",
        evaluation={
            "ml_strategy_tuning": {
                "enabled": True,
                "allocation_score_transforms": allocation_score_transforms,
            },
        },
    )

    with pytest.raises(ValueError, match="allocation_score_transforms"):
        load_config(config_path)


def test_load_config_accepts_retained_btc_phase8_shadow_handoff_config() -> None:
    config = load_config(
        "configs/experiment.btc_phase8_guarded_gate_bull_risk_off_override_partial_support.yaml"
    )
    tuning = config.evaluation.ml_strategy_tuning

    assert config.experiment_name == (
        "btc_phase8_guarded_gate_bull_risk_off_override_partial_support"
    )
    assert config.data.symbols == ["BTC-USD"]
    assert config.data.interval == "1d"
    assert config.data.cache_dir == "artifacts/data-btc-phase8-1d-long"
    assert config.target.type == "allocation_utility"
    assert config.paper.enabled is False
    assert tuning.allocation_score_policy == "gate_bull_prob100_threshold"
    assert tuning.selection_validation_cost_bps == [35.0, 50.0]
    assert tuning.guarded_gate_bull_risk_off_override is True
    assert (
        config.target.allocation_utility_drawdown_penalty,
        config.target.allocation_utility_volatility_penalty,
        config.target.allocation_utility_risk_penalty_power,
    ) == pytest.approx((0.75, 0.25, 2.0))
    assert config.evaluation.strict_research_gate.enabled is True
    assert config.evaluation.strict_research_gate.required_partial_target_weights == [
        0.25,
        0.50,
    ]
    assert config.evaluation.strict_research_gate.required_predicted_target_weights == [
        0.25,
        0.50,
    ]


def test_load_config_accepts_isolated_btc_paper_daily_config() -> None:
    config = load_config("configs/experiment.btc_paper_daily.yaml")

    assert config.experiment_name == "btc_paper_daily"
    assert config.data.symbols == ["BTC/USD"]
    assert config.paper.enabled is True
    assert config.paper.order_type == "crypto_market_gtc"
    assert config.paper.position_sizing == "target_weight_fractional"
    assert config.paper_approval_inbox_dir.as_posix().endswith("artifacts/btc-paper/inbox")
    assert config.paper_state_dir.as_posix().endswith("artifacts/btc-paper/state")
    assert config.output_dir.as_posix().endswith("artifacts/btc-paper/runs")


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        ({"thresholds": [0.5, 1.2]}, "ml_strategy_tuning.thresholds"),
        ({"validation_months": 0}, "validation_months"),
        ({"min_validation_rows": 0}, "min_validation_rows"),
        ({"min_exposure_changes": -1}, "min_exposure_changes"),
        ({"max_average_exposure_for_active": 1.5}, "max_average_exposure_for_active"),
        ({"rolling_train_bars_grid": [0]}, "rolling_train_bars_grid"),
        ({"min_holding_period_bars_grid": [-1]}, "min_holding_period_bars_grid"),
        ({"hysteresis_margin_grid": [-0.01]}, "hysteresis_margin_grid"),
        ({"hysteresis_margin_grid": [0.50]}, "hysteresis_margin_grid"),
        ({"max_annualized_turnover": 0.0}, "max_annualized_turnover"),
        ({"selection_validation_cost_bps": [-1.0]}, "selection_validation_cost_bps"),
        (
            {"selection_validation_cost_bps": [35.0, 35.0]},
            "selection_validation_cost_bps",
        ),
        (
            {"guarded_gate_bull_risk_off_override": "true"},
            "guarded_gate_bull_risk_off_override",
        ),
        ({"objective": "auc"}, "ml_strategy_tuning.objective"),
        ({"selection_policy": "loose"}, "ml_strategy_tuning.selection_policy"),
        ({"allocation_score_policy": "magic_score"}, "allocation_score_policy"),
        (
            {"allocation_score_policy_prob100_threshold": 1.2},
            "allocation_score_policy_prob100_threshold",
        ),
        (
            {"allocation_score_policy_prob100_threshold": -0.1},
            "allocation_score_policy_prob100_threshold",
        ),
        (
            {"allocation_score_policy_prob100_threshold_grid": [0.2, 1.2]},
            "allocation_score_policy_prob100_threshold_grid",
        ),
        (
            {"allocation_score_policy_prob100_threshold_grid": [-0.1]},
            "allocation_score_policy_prob100_threshold_grid",
        ),
        (
            {
                "no_candidate_fallback_regime_policy": "model_only",
                "no_valid_candidate_regime_fallback": "other_policy",
                "regime_participation_policies": [
                    {"name": "model_only"},
                    {"name": "other_policy"},
                ],
            },
            "no_candidate_fallback_regime_policy",
        ),
        (
            {"objective": "net_return_and_risk_vs_required_benchmarks"},
            "selection_benchmark_strategies",
        ),
        ({"selection_benchmark_strategies": ["buy_hold", ""]}, "selection_benchmark_strategies"),
        (
            {"selection_benchmark_strategies": ["buy_hold", "buy_hold"]},
            "selection_benchmark_strategies",
        ),
        (
            {"no_candidate_fallback_regime_policy": ""},
            "no_candidate_fallback_regime_policy",
        ),
        (
            {"no_candidate_fallback_regime_policy": "missing_policy"},
            "no_candidate_fallback_regime_policy",
        ),
        (
            {
                "no_candidate_fallback_regime_policy": "risk_off_uncapped",
                "regime_participation_policies": [
                    {"name": "risk_off_uncapped", "risk_off_cap": None}
                ],
            },
            "risk_off_cap",
        ),
        (
            {
                "allocation_utility_profiles": [
                    {
                        "name": "duplicate",
                        "drawdown_penalty": 0.5,
                        "volatility_penalty": 0.25,
                        "risk_penalty_power": 2.0,
                    },
                    {
                        "name": "duplicate",
                        "drawdown_penalty": 0.5,
                        "volatility_penalty": 0.25,
                        "risk_penalty_power": 2.5,
                    },
                ]
            },
            "allocation_utility_profiles",
        ),
        (
            {
                "allocation_utility_profiles": [
                    {
                        "name": "bad_power",
                        "drawdown_penalty": 0.5,
                        "volatility_penalty": 0.25,
                        "risk_penalty_power": 0.5,
                    }
                ]
            },
            "risk_penalty_power",
        ),
        ({"allocation_class_weighting": "rare_magic"}, "allocation_class_weighting"),
        (
            {"allocation_partial_class_weight_multiplier": 0.0},
            "allocation_partial_class_weight_multiplier",
        ),
        ({"allocation_probability_calibration": "isotonic"}, "allocation_probability_calibration"),
        ({"allocation_calibration_cv": 1}, "allocation_calibration_cv"),
        (
            {"regime_participation_policies": [{"name": ""}]},
            "regime_participation_policies",
        ),
        (
            {
                "regime_participation_policies": [
                    {"name": "duplicate"},
                    {"name": "duplicate"},
                ]
            },
            "regime_participation_policies",
        ),
        (
            {
                "regime_participation_policies": [
                    {"name": "bad_floor", "bull_floor": 0.80},
                ]
            },
            "bull_floor",
        ),
        (
            {
                "regime_participation_policies": [
                    {"name": "bad_cap", "risk_off_cap": 0.80},
                ]
            },
            "risk_off_cap",
        ),
    ],
)
def test_load_config_rejects_invalid_ml_strategy_tuning_values(
    tmp_path: Path,
    payload: dict[str, object],
    message: str,
) -> None:
    config_path = _write_config(
        tmp_path / "config.yaml",
        evaluation={"ml_strategy_tuning": {"enabled": True, **payload}},
    )

    with pytest.raises(ValueError, match=message):
        load_config(config_path)


@pytest.mark.parametrize(
    ("target_payload", "message"),
    [
        ({"allocation_utility_risk_penalty_power": 0.5}, "risk_penalty_power"),
        ({"allocation_utility_risk_penalty_power": "nan"}, "risk_penalty_power"),
    ],
)
def test_load_config_rejects_invalid_allocation_utility_target_values(
    tmp_path: Path,
    target_payload: dict[str, object],
    message: str,
) -> None:
    config_path = _write_config(tmp_path / "config.yaml", target=target_payload)

    with pytest.raises(ValueError, match=message):
        load_config(config_path)


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        (
            {"required_benchmark_strategies": [""]},
            "required_benchmark_strategies",
        ),
        (
            {"min_selected_fold_fraction": -0.1},
            "min_selected_fold_fraction",
        ),
        (
            {"min_selected_fold_fraction": 1.1},
            "min_selected_fold_fraction",
        ),
        (
            {"required_partial_target_weights": [0.0]},
            "required_partial_target_weights",
        ),
        (
            {"required_partial_target_weights": [0.25, 0.25]},
            "required_partial_target_weights",
        ),
        (
            {"min_partial_target_fraction": -0.1},
            "min_partial_target_fraction",
        ),
        (
            {"min_partial_target_fold_fraction": 1.1},
            "min_partial_target_fold_fraction",
        ),
        (
            {"required_predicted_target_weights": [0.0]},
            "required_predicted_target_weights",
        ),
        (
            {"required_predicted_target_weights": [0.25, 0.25]},
            "required_predicted_target_weights",
        ),
        (
            {"min_predicted_target_fraction": -0.1},
            "min_predicted_target_fraction",
        ),
        (
            {"min_predicted_target_fold_fraction": 1.1},
            "min_predicted_target_fold_fraction",
        ),
    ],
)
def test_load_config_rejects_invalid_strict_research_gate_values(
    tmp_path: Path,
    payload: dict[str, object],
    message: str,
) -> None:
    config_path = _write_config(
        tmp_path / "config.yaml",
        evaluation={"strict_research_gate": {"enabled": True, **payload}},
    )

    with pytest.raises(ValueError, match=message):
        load_config(config_path)


@pytest.mark.parametrize(
        ("interval", "expected_periods"),
    [
        ("1d", 252.0),
        ("12h", 730.0),
        ("8h", 1095.0),
        ("6h", 1460.0),
        ("4h", 2190.0),
        ("2h", 4380.0),
        ("1h", 8760.0),
        ("45m", 11680.0),
        ("30m", 17520.0),
        ("15m", 35040.0),
        ("5m", 105120.0),
        ("1m", 525600.0),
    ],
)
def test_default_periods_per_year_supports_intraday_intervals(
    interval: str,
    expected_periods: float,
) -> None:
    assert default_periods_per_year(interval) == pytest.approx(expected_periods)


def test_load_config_infers_periods_per_year_from_interval(tmp_path: Path) -> None:
    config_path = _write_config(tmp_path / "config.yaml", data={"interval": "15m"})

    config = load_config(config_path)

    assert config.evaluation.periods_per_year == pytest.approx(35040.0)


def test_load_config_accepts_focus_window_and_visual_signal_flag(tmp_path: Path) -> None:
    config_path = _write_config(
        tmp_path / "config.yaml",
        data={"interval": "15m"},
        evaluation={
            "focus_start": "2024-01-01 00:00:00",
            "focus_end": "2024-01-07 23:59:59",
            "visualize_signals": True,
        },
    )

    config = load_config(config_path)

    assert config.evaluation.focus_start == "2024-01-01 00:00:00"
    assert config.evaluation.focus_end == "2024-01-07 23:59:59"
    assert config.evaluation.visualize_signals is True


def test_load_config_rejects_unknown_interval(tmp_path: Path) -> None:
    config_path = _write_config(tmp_path / "config.yaml", data={"interval": "2m"})

    with pytest.raises(ValueError, match="data.interval must be one of"):
        load_config(config_path)


def test_load_config_rejects_inverted_focus_window(tmp_path: Path) -> None:
    config_path = _write_config(
        tmp_path / "config.yaml",
        evaluation={
            "focus_start": "2024-01-08 00:00:00",
            "focus_end": "2024-01-07 23:59:59",
        },
    )

    with pytest.raises(ValueError, match="evaluation.focus_start must be before"):
        load_config(config_path)


def test_load_config_accepts_pattern_exit_overlay_and_meta_label(tmp_path: Path) -> None:
    config_path = _write_config(
        tmp_path / "config.yaml",
        baselines={
            "chart_patterns": {"enabled": True},
            "pattern_exit_overlay": {
                "enabled": True,
                "min_bearish_patterns": 2,
                "min_bullish_reentry_patterns": 1,
                "trend_ema_window": 20,
                "reentry_clear_bars": 2,
                "require_price_below_trend_for_exit": True,
                "bearish_confirmation_window_bars": 3,
                "min_cash_bars": 2,
                "exit_cooldown_bars": 4,
                "reentry_requires_price_above_trend": True,
            },
            "pattern_meta_label": {
                "enabled": True,
                "label_horizon_bars": 6,
                "exit_probability_threshold": 0.6,
                "exit_probability_threshold_grid": [0.5, 0.6, 0.7],
                "tuning_mode": "nested_walk_forward",
                "min_oos_exit_count": 2,
                "max_average_exposure_for_active": 0.995,
                "models": ["logistic_l1"],
            },
            "pattern_partial_exposure_overlay": {
                "enabled": True,
                "partial_weight": 0.5,
                "partial_exit_probability_threshold_grid": [0.55, 0.6],
                "full_exit_probability_threshold_grid": [0.75, 0.8],
            },
        },
    )

    config = load_config(config_path)

    assert config.baselines.pattern_exit_overlay.enabled is True
    assert config.baselines.pattern_exit_overlay.min_bearish_patterns == 2
    assert config.baselines.pattern_exit_overlay.require_price_below_trend_for_exit is True
    assert config.baselines.pattern_exit_overlay.bearish_confirmation_window_bars == 3
    assert config.baselines.pattern_exit_overlay.min_cash_bars == 2
    assert config.baselines.pattern_exit_overlay.exit_cooldown_bars == 4
    assert config.baselines.pattern_exit_overlay.reentry_requires_price_above_trend is True
    assert config.baselines.pattern_meta_label.enabled is True
    assert config.baselines.pattern_meta_label.label_horizon_bars == 6
    assert config.baselines.pattern_meta_label.exit_probability_threshold_grid == [0.5, 0.6, 0.7]
    assert config.baselines.pattern_meta_label.tuning_mode == "nested_walk_forward"
    assert config.baselines.pattern_meta_label.min_oos_exit_count == 2
    assert config.baselines.pattern_meta_label.max_average_exposure_for_active == pytest.approx(
        0.995
    )
    assert config.baselines.pattern_meta_label.models == ["logistic_l1"]
    assert config.baselines.pattern_partial_exposure_overlay.enabled is True
    assert config.baselines.pattern_partial_exposure_overlay.partial_weight == pytest.approx(0.5)
    assert config.baselines.pattern_partial_exposure_overlay.partial_exit_probability_threshold_grid == [
        0.55,
        0.6,
    ]


def test_load_config_rejects_pattern_meta_label_without_exit_overlay(tmp_path: Path) -> None:
    config_path = _write_config(
        tmp_path / "config.yaml",
        baselines={
            "chart_patterns": {"enabled": True},
            "pattern_meta_label": {"enabled": True},
        },
    )

    with pytest.raises(ValueError, match="pattern_exit_overlay.enabled must be true"):
        load_config(config_path)


def test_load_config_rejects_unsupported_pattern_meta_model(tmp_path: Path) -> None:
    config_path = _write_config(
        tmp_path / "config.yaml",
        baselines={
            "chart_patterns": {"enabled": True},
            "pattern_exit_overlay": {"enabled": True},
            "pattern_meta_label": {
                "enabled": True,
                "models": ["not_a_model"],
            },
        },
    )

    with pytest.raises(ValueError, match="unsupported models"):
        load_config(config_path)


def test_load_config_rejects_invalid_pattern_exit_tuning_values(tmp_path: Path) -> None:
    config_path = _write_config(
        tmp_path / "config.yaml",
        baselines={
            "chart_patterns": {"enabled": True},
            "pattern_exit_overlay": {
                "enabled": True,
                "bearish_confirmation_window_bars": 0,
            },
        },
    )

    with pytest.raises(ValueError, match="bearish_confirmation_window_bars"):
        load_config(config_path)


def test_load_config_rejects_invalid_pattern_meta_threshold_grid(tmp_path: Path) -> None:
    config_path = _write_config(
        tmp_path / "config.yaml",
        baselines={
            "chart_patterns": {"enabled": True},
            "pattern_exit_overlay": {"enabled": True},
            "pattern_meta_label": {
                "enabled": True,
                "exit_probability_threshold_grid": [0.4, 1.2],
            },
        },
    )

    with pytest.raises(ValueError, match="exit_probability_threshold_grid"):
        load_config(config_path)


def test_load_config_rejects_invalid_partial_exposure_overlay(tmp_path: Path) -> None:
    config_path = _write_config(
        tmp_path / "config.yaml",
        baselines={
            "chart_patterns": {"enabled": True},
            "pattern_exit_overlay": {"enabled": True},
            "pattern_meta_label": {"enabled": True},
            "pattern_partial_exposure_overlay": {
                "enabled": True,
                "partial_weight": 1.0,
            },
        },
    )

    with pytest.raises(ValueError, match="partial_weight"):
        load_config(config_path)


def test_load_config_accepts_partial_allocation_benchmarks(tmp_path: Path) -> None:
    config_path = _write_config(
        tmp_path / "config.yaml",
        data={"symbols": ["BTC-USD"]},
        baselines={
            "partial_allocation_benchmarks": {
                "enabled": True,
                "weights": [0.25, 0.50, 0.75],
            }
        },
    )

    config = load_config(config_path)

    assert config.baselines.partial_allocation_benchmarks.enabled is True
    assert config.baselines.partial_allocation_benchmarks.weights == [0.25, 0.50, 0.75]


def test_load_config_accepts_rebalanced_partial_allocation_benchmarks(
    tmp_path: Path,
) -> None:
    config_path = _write_config(
        tmp_path / "config.yaml",
        data={"symbols": ["BTC-USD"]},
        baselines={
            "rebalanced_partial_allocation_benchmarks": {
                "enabled": True,
                "weights": [0.25, 0.50, 0.75],
            }
        },
    )

    config = load_config(config_path)

    assert config.baselines.rebalanced_partial_allocation_benchmarks.enabled is True
    assert config.baselines.rebalanced_partial_allocation_benchmarks.weights == [
        0.25,
        0.50,
        0.75,
    ]


@pytest.mark.parametrize(
    ("data", "payload", "message"),
    [
        (
            {"symbols": ["BTC-USD"]},
            {"enabled": True, "weights": []},
            "must contain at least one value",
        ),
        (
            {"symbols": ["BTC-USD"]},
            {"enabled": True, "weights": [0.0]},
            "greater than 0.0 and less than 1.0",
        ),
        (
            {"symbols": ["BTC-USD"]},
            {"enabled": True, "weights": [0.25, 0.25]},
            "must not contain duplicate values",
        ),
        (
            {"symbols": ["BTC-USD", "ETH-USD"]},
            {"enabled": True, "weights": [0.25]},
            "requires exactly one data symbol",
        ),
    ],
)
def test_load_config_rejects_invalid_partial_allocation_benchmarks(
    tmp_path: Path,
    data: dict[str, object],
    payload: dict[str, object],
    message: str,
) -> None:
    config_path = _write_config(
        tmp_path / "config.yaml",
        data=data,
        baselines={"partial_allocation_benchmarks": payload},
    )

    with pytest.raises(ValueError, match=message):
        load_config(config_path)


def test_load_config_rejects_invalid_rebalanced_partial_allocation_benchmarks(
    tmp_path: Path,
) -> None:
    config_path = _write_config(
        tmp_path / "config.yaml",
        data={"symbols": ["BTC-USD"]},
        baselines={
            "rebalanced_partial_allocation_benchmarks": {
                "enabled": True,
                "weights": [0.25, 0.25],
            }
        },
    )

    with pytest.raises(ValueError, match="rebalanced_partial_allocation_benchmarks"):
        load_config(config_path)


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
            "partial_allocation_benchmarks": {
                "enabled": False,
                "weights": None,
            },
            "rebalanced_partial_allocation_benchmarks": {
                "enabled": False,
                "weights": None,
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
    assert config.baselines.partial_allocation_benchmarks.weights == []
    assert config.baselines.rebalanced_partial_allocation_benchmarks.weights == []
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


def test_load_config_accepts_crypto_hourly_trend_settings(tmp_path: Path) -> None:
    config_path = _write_config(
        tmp_path / "config.yaml",
        data={"symbols": ["BTC-USD"], "interval": "1h"},
        portfolio={
            "ranking": {
                "long_n": 1,
                "short_n": 1,
                "rebalance_frequency": "bar",
                "mode": "long_only",
                "cash_when_underfilled": True,
            },
            "costs": {"bps_per_trade": 20},
        },
        baselines={
            "buy_hold": True,
            "sma": {"enabled": False},
            "indicator_stack": {
                "enabled": True,
                "ema_fast_window": 3,
                "ema_slow_window": 8,
                "min_confirmations": 3,
                "use_vwap": True,
            },
        },
        evaluation={"benchmark_strategy": "buy_hold", "periods_per_year": 8760},
    )

    config = load_config(config_path)

    assert config.data.interval == "1h"
    assert config.portfolio.ranking.rebalance_frequency == "bar"
    assert config.baselines.indicator_stack.enabled is True
    assert config.baselines.indicator_stack.ema_fast_window == 3
    assert config.baselines.indicator_stack.ema_slow_window == 8
    assert config.baselines.indicator_stack.use_vwap is True
    assert config.evaluation.periods_per_year == pytest.approx(8760.0)


def test_load_config_accepts_chart_pattern_settings(tmp_path: Path) -> None:
    config_path = _write_config(
        tmp_path / "config.yaml",
        data={"symbols": ["BTC-USD"], "interval": "15m"},
        portfolio={
            "ranking": {
                "long_n": 1,
                "short_n": 1,
                "rebalance_frequency": "bar",
                "mode": "long_only",
                "cash_when_underfilled": True,
            },
        },
        baselines={
            "buy_hold": True,
            "sma": {"enabled": False},
            "chart_patterns": {
                "enabled": True,
                "lookback_bars": 16,
                "level_tolerance_pct": 0.02,
                "breakout_pct": 0.001,
                "min_bullish_patterns": 1,
            },
        },
        evaluation={"benchmark_strategy": "buy_hold", "visualize_signals": True},
    )

    config = load_config(config_path)

    assert config.baselines.chart_patterns.enabled is True
    assert config.baselines.chart_patterns.lookback_bars == 16
    assert config.baselines.chart_patterns.level_tolerance_pct == pytest.approx(0.02)
    assert config.baselines.chart_patterns.breakout_pct == pytest.approx(0.001)
    assert config.evaluation.periods_per_year == pytest.approx(35040.0)


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
        "persistence_backend": "sqlite",
        "sqlite_db_path": "artifacts/paper/state/paper-control.db",
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
    assert config.paper.persistence_backend == "sqlite"
    assert config.paper.notifications.telegram.enabled is True
    assert config.paper_approval_inbox_dir == (tmp_path / "artifacts" / "paper" / "inbox").resolve()
    assert config.paper_state_dir == (tmp_path / "artifacts" / "paper" / "state").resolve()
    assert config.paper_sqlite_db_path == (
        tmp_path / "artifacts" / "paper" / "state" / "paper-control.db"
    ).resolve()


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


def test_load_config_rejects_unknown_paper_persistence_backend(tmp_path: Path) -> None:
    config_path = _write_config(
        tmp_path / "config.yaml",
        data={"symbols": ["QQQ"], "interval": "1d"},
    )
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    payload["paper"] = {
        "enabled": True,
        "persistence_backend": "unknown",
    }
    config_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

    with pytest.raises(ValueError, match="paper.persistence_backend must be one of"):
        load_config(config_path)


def test_load_config_rejects_sqlite_backend_without_db_path(tmp_path: Path) -> None:
    config_path = _write_config(
        tmp_path / "config.yaml",
        data={"symbols": ["QQQ"], "interval": "1d"},
    )
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    payload["paper"] = {
        "enabled": True,
        "persistence_backend": "sqlite",
        "sqlite_db_path": "",
    }
    config_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

    with pytest.raises(ValueError, match="paper.sqlite_db_path must be set"):
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


def test_load_config_parses_phase9_shadow_metadata() -> None:
    root = Path(__file__).resolve().parents[2]
    config = load_config(root / "configs" / "experiment.btc_phase9_shadow_daily.yaml")

    assert config.shadow is not None
    assert config.shadow.candidate_id == "btc-phase9-shadow-v1"
    assert config.shadow.behavior_version == "btc-phase8-guarded-gate-v1"
    assert config.shadow.protocol_start == "2026-06-03"
    assert config.shadow.protocol_end == "2027-06-02"
    assert config.shadow.earliest_final_evaluation == "2027-06-16"
    assert config.shadow.maturity_lag_bars == 14
    assert config.shadow.code_lock == "ce01124"
    assert config.shadow.artifact_root == "artifacts/phase9-shadow"
