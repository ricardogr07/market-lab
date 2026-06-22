from __future__ import annotations

import math
from dataclasses import dataclass, field, fields
from datetime import datetime
from pathlib import Path
from typing import Any, TypeVar, cast

import yaml

ALLOCATION_MODES = {"equal", "group_weights", "symbol_weights"}
OPTIMIZED_METHODS = {"black_litterman", "mean_variance", "risk_parity"}
COVARIANCE_ESTIMATORS = {"diagonal_shrinkage", "ewma", "external_csv", "sample"}
EXPECTED_RETURN_SOURCES = {"external_csv", "historical_mean"}
PAPER_DATA_PROVIDERS = {"alpaca"}
PAPER_BROKERS = {"alpaca"}
PAPER_PERSISTENCE_BACKENDS = {"filesystem", "postgres", "sqlite"}
PAPER_EXECUTION_MODES = {"autonomous", "agent_approval", "manual_approval"}
PAPER_ORDER_TYPES = {"crypto_market_gtc", "crypto_market_ioc", "day_market"}
PAPER_POSITION_SIZING = {"full_equity_fractional", "target_weight_fractional"}
PAPER_AGENT_BACKENDS = {"claude", "deterministic_consensus", "openai"}
PAPER_AZURE_ARTIFACT_BACKENDS = {"filesystem", "azure_blob"}
PAPER_AZURE_SECRET_BACKENDS = {"environment", "key_vault"}
PAPER_AZURE_SERVICE_BUS_BACKENDS = {"disabled", "in_memory", "azure_service_bus"}
TARGET_TYPES = {"allocation_utility", "direction", "regime_state", "return"}
ML_STRATEGY_ALLOCATION_MODES = {"binary", "direct_tiered", "tiered"}
ML_STRATEGY_TUNING_OBJECTIVES = {
    "net_return_and_risk_vs_buy_hold",
    "net_return_and_risk_vs_required_benchmarks",
    "net_return_risk_score_validity_vs_required_benchmarks",
}
ML_STRATEGY_SELECTION_POLICIES = {"best_active_fallback", "strict"}
ML_STRATEGY_ALLOCATION_SCORE_POLICIES = {
    "bull_prob100_threshold",
    "expected_allocation",
    "gate_bull_prob100_threshold",
}
ALLOCATION_CLASS_WEIGHTING_MODES = {
    "balanced",
    "balanced_partial_boost",
    "none",
}
ALLOCATION_PROBABILITY_CALIBRATION_MODES = {"none", "sigmoid"}
REGIME_PARTICIPATION_POLICY_TIERS = {0.0, 0.25, 0.50, 0.75, 1.0}
WEIGHT_TOLERANCE = 1e-6
INTERVAL_PERIODS_PER_YEAR = {
    "1d": 252.0,
    "12h": 730.0,
    "8h": 1095.0,
    "6h": 1460.0,
    "4h": 2190.0,
    "2h": 4380.0,
    "1h": 8760.0,
    "45m": 11680.0,
    "30m": 17520.0,
    "15m": 35040.0,
    "5m": 105120.0,
    "1m": 525600.0,
}
_ConfigSectionT = TypeVar("_ConfigSectionT")


@dataclass(slots=True)
class DataConfig:
    symbols: list[str] = field(
        default_factory=lambda: ["VOO", "QQQ", "SMH", "XLV", "IEMG"]
    )
    start_date: str = "2018-01-01"
    end_date: str = "2025-12-31"
    interval: str = "1d"
    cache_dir: str = "artifacts/data"
    prepared_panel_filename: str = "panel.csv"
    symbol_groups: dict[str, str] = field(default_factory=dict)


@dataclass(slots=True)
class FeaturesConfig:
    return_windows: list[int] = field(default_factory=lambda: [5, 10, 20, 40])
    ma_windows: list[int] = field(default_factory=lambda: [10, 20, 50])
    vol_windows: list[int] = field(default_factory=lambda: [10, 20])
    momentum_window: int = 20
    indicator_stack_ml_features_enabled: bool = False
    crypto_time_series_enabled: bool = False
    crypto_return_windows: list[int] = field(default_factory=lambda: [1, 3, 6, 12, 24, 72, 168])
    crypto_vol_windows: list[int] = field(default_factory=lambda: [12, 24, 72, 168])
    crypto_ma_windows: list[int] = field(default_factory=lambda: [12, 24, 72, 168])
    crypto_rsi_window: int = 14
    crypto_macd_fast_window: int = 12
    crypto_macd_slow_window: int = 26
    crypto_macd_signal_window: int = 9
    crypto_bollinger_window: int = 20
    crypto_bollinger_std: float = 2.0
    crypto_volume_window: int = 24
    crypto_time_features: bool = True
    crypto_regime_features_enabled: bool = False
    crypto_regime_trend_windows: list[int] = field(default_factory=lambda: [42, 126, 252])
    crypto_regime_volatility_window: int = 42
    crypto_regime_percentile_window: int = 252
    crypto_regime_drawdown_window: int = 252
    crypto_regime_volume_window: int = 42
    crypto_regime_signal_features_enabled: bool = False


@dataclass(slots=True)
class TargetConfig:
    horizon_days: int = 5
    type: str = "direction"
    allocation_utility_drawdown_penalty: float = 0.50
    allocation_utility_volatility_penalty: float = 0.25
    allocation_utility_risk_penalty_power: float = 2.0


@dataclass(slots=True)
class RankingConfig:
    long_n: int = 2
    short_n: int = 2
    rebalance_frequency: str = "W-FRI"
    weighting: str = "equal"
    mode: str = "long_short"
    min_score_threshold: float = 0.0
    cash_when_underfilled: bool = False


@dataclass(slots=True)
class RiskConfig:
    max_position_weight: float | None = None
    max_group_weight: float | None = None
    max_long_exposure: float | None = None
    max_short_exposure: float | None = None


@dataclass(slots=True)
class CostsConfig:
    bps_per_trade: float = 10.0


@dataclass(slots=True)
class PortfolioConfig:
    ranking: RankingConfig = field(default_factory=RankingConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
    costs: CostsConfig = field(default_factory=CostsConfig)


@dataclass(slots=True)
class SMAConfig:
    enabled: bool = True
    fast_window: int = 20
    slow_window: int = 50


@dataclass(slots=True)
class IndicatorStackConfig:
    enabled: bool = False
    ema_fast_window: int = 12
    ema_slow_window: int = 26
    rsi_window: int = 14
    rsi_min: float = 45.0
    rsi_max: float = 70.0
    macd_fast_window: int = 12
    macd_slow_window: int = 26
    macd_signal_window: int = 9
    bollinger_window: int = 20
    bollinger_std: float = 2.0
    bollinger_mode: str = "breakout"
    volume_window: int = 20
    volume_multiplier: float = 1.0
    vwap_window: int = 24
    use_vwap: bool = False
    min_confirmations: int = 3


@dataclass(slots=True)
class ChartPatternsConfig:
    enabled: bool = False
    lookback_bars: int = 96
    triangle_slope_min: float = 0.0
    level_tolerance_pct: float = 0.01
    breakout_pct: float = 0.001
    rectangle_max_range_pct: float = 0.05
    flag_pole_bars: int = 16
    flag_consolidation_bars: int = 12
    flag_min_pole_return: float = 0.015
    flag_max_retrace_pct: float = 0.012
    volume_window: int = 20
    volume_multiplier: float = 1.0
    min_bullish_patterns: int = 1


@dataclass(slots=True)
class PatternExitOverlayConfig:
    enabled: bool = False
    min_bearish_patterns: int = 1
    min_bullish_reentry_patterns: int = 1
    trend_ema_window: int = 50
    reentry_clear_bars: int = 1
    require_price_below_trend_for_exit: bool = False
    bearish_confirmation_window_bars: int = 1
    min_cash_bars: int = 0
    exit_cooldown_bars: int = 0
    reentry_requires_price_above_trend: bool = False


@dataclass(slots=True)
class PatternMetaLabelConfig:
    enabled: bool = False
    label_horizon_bars: int = 12
    exit_probability_threshold: float = 0.55
    exit_probability_threshold_grid: list[float] = field(default_factory=list)
    tuning_mode: str = "fixed"
    tuning_objective: str = "net_return_and_drawdown_vs_buy_hold"
    min_oos_exit_count: int = 1
    max_average_exposure_for_active: float = 0.999
    models: list[str] = field(default_factory=lambda: ["logistic_l1", "gradient_boosting"])


@dataclass(slots=True)
class PatternPartialExposureOverlayConfig:
    enabled: bool = False
    partial_weight: float = 0.5
    partial_exit_probability_threshold_grid: list[float] = field(default_factory=list)
    full_exit_probability_threshold_grid: list[float] = field(default_factory=list)


@dataclass(slots=True)
class AllocationConfig:
    enabled: bool = False
    mode: str = "equal"
    symbol_weights: dict[str, float] = field(default_factory=dict)
    group_weights: dict[str, float] = field(default_factory=dict)


@dataclass(slots=True)
class PartialAllocationBenchmarksConfig:
    enabled: bool = False
    weights: list[float] = field(default_factory=list)


@dataclass(slots=True)
class BlackLittermanViewConfig:
    name: str = ""
    weights: dict[str, float] = field(default_factory=dict)
    view_return: float = 0.0


@dataclass(slots=True)
class OptimizedConfig:
    enabled: bool = False
    method: str = "mean_variance"
    lookback_days: int = 252
    rebalance_frequency: str = "W-FRI"
    covariance_estimator: str = "sample"
    external_covariance_path: str = ""
    expected_return_source: str = "historical_mean"
    external_expected_returns_path: str = ""
    long_only: bool = True
    target_gross_exposure: float = 1.0
    risk_aversion: float = 1.0
    equilibrium_weights: dict[str, float] = field(default_factory=dict)
    tau: float = 0.05
    views: list[BlackLittermanViewConfig] = field(default_factory=list)


@dataclass(slots=True)
class BaselinesConfig:
    buy_hold: bool = True
    sma: SMAConfig = field(default_factory=SMAConfig)
    indicator_stack: IndicatorStackConfig = field(default_factory=IndicatorStackConfig)
    chart_patterns: ChartPatternsConfig = field(default_factory=ChartPatternsConfig)
    pattern_exit_overlay: PatternExitOverlayConfig = field(
        default_factory=PatternExitOverlayConfig
    )
    pattern_meta_label: PatternMetaLabelConfig = field(default_factory=PatternMetaLabelConfig)
    pattern_partial_exposure_overlay: PatternPartialExposureOverlayConfig = field(
        default_factory=PatternPartialExposureOverlayConfig
    )
    partial_allocation_benchmarks: PartialAllocationBenchmarksConfig = field(
        default_factory=PartialAllocationBenchmarksConfig
    )
    rebalanced_partial_allocation_benchmarks: PartialAllocationBenchmarksConfig = field(
        default_factory=PartialAllocationBenchmarksConfig
    )
    allocation: AllocationConfig = field(default_factory=AllocationConfig)
    optimized: OptimizedConfig = field(default_factory=OptimizedConfig)


@dataclass(slots=True)
class ModelSpec:
    name: str


@dataclass(slots=True)
class WalkForwardConfig:
    train_years: int = 3
    test_months: int = 3
    step_months: int = 3
    min_train_rows: int = 0
    min_test_rows: int = 0
    min_train_positive_rate: float = 0.0
    min_test_positive_rate: float = 0.0
    embargo_periods: int = 0


@dataclass(slots=True)
class MLStrategyThresholdSweepConfig:
    enabled: bool = False
    thresholds: list[float] = field(default_factory=lambda: [0.50, 0.52, 0.55, 0.58, 0.60])
    min_exposure_changes: int = 5
    max_average_exposure_for_active: float = 0.995


@dataclass(slots=True)
class AllocationUtilityProfileConfig:
    name: str = "default"
    drawdown_penalty: float = 0.50
    volatility_penalty: float = 0.25
    risk_penalty_power: float = 2.0


@dataclass(slots=True)
class RegimeParticipationPolicyConfig:
    name: str = "model_only"
    bull_floor: float = 0.0
    sideways_floor: float = 0.0
    bear_floor: float = 0.0
    risk_off_cap: float | None = 0.25
    gate_bull_floor: float | None = None


@dataclass(slots=True)
class AllocationScoreTransformConfig:
    name: str = "identity"
    bull_multiplier: float = 1.0
    bull_addend: float = 0.0
    risk_off_score_cap: float | None = None
    non_bull_score_cap: float | None = None


@dataclass(slots=True)
class MLStrategyTuningConfig:
    enabled: bool = False
    thresholds: list[float] = field(
        default_factory=lambda: [0.50, 0.52, 0.55, 0.58, 0.60, 0.62, 0.65]
    )
    validation_months: int = 3
    min_validation_rows: int = 200
    min_exposure_changes: int = 5
    min_average_exposure_for_active: float = 0.0
    max_average_exposure_for_active: float = 0.995
    selection_policy: str = "strict"
    objective: str = "net_return_and_risk_vs_buy_hold"
    selection_benchmark_strategies: list[str] = field(default_factory=list)
    allocation_mode: str = "binary"
    tier_thresholds: list[float] = field(default_factory=lambda: [0.50, 0.55, 0.62])
    tier_threshold_sets: list[list[float]] = field(default_factory=list)
    rolling_train_bars_grid: list[int] = field(default_factory=list)
    min_holding_period_bars_grid: list[int] = field(default_factory=lambda: [0])
    hysteresis_margin_grid: list[float] = field(default_factory=lambda: [0.0])
    max_annualized_turnover: float | None = None
    allocation_utility_profiles: list[AllocationUtilityProfileConfig] = field(default_factory=list)
    regime_participation_policies: list[RegimeParticipationPolicyConfig] = field(
        default_factory=lambda: [RegimeParticipationPolicyConfig()]
    )
    no_candidate_fallback_regime_policy: str | None = None
    no_valid_candidate_regime_fallback: str | None = None
    allocation_class_weighting: str = "none"
    allocation_partial_class_weight_multiplier: float = 1.0
    allocation_probability_calibration: str = "none"
    allocation_calibration_cv: int = 3
    allocation_score_policy: str = "expected_allocation"
    allocation_score_policy_prob100_threshold: float = 0.20
    allocation_score_policy_prob100_threshold_grid: list[float] = field(default_factory=list)
    selection_validation_cost_bps: list[float] = field(default_factory=list)
    guarded_gate_bull_risk_off_override: bool = False
    allocation_score_transforms: list[AllocationScoreTransformConfig] = field(
        default_factory=lambda: [AllocationScoreTransformConfig()]
    )


@dataclass(slots=True)
class StrictResearchGateConfig:
    enabled: bool = False
    strategy_name: str = "ml_indicator_tuned__long_only__cash"
    benchmark_strategy: str = "buy_hold"
    required_benchmark_strategies: list[str] = field(default_factory=list)
    cost_gate_bps: float = 35.0
    acceptable_cost_bps: float = 50.0
    min_positive_regime_slices: int = 3
    min_average_exposure: float = 0.20
    max_average_exposure: float = 0.85
    min_selected_fold_fraction: float = 0.75
    recent_window_months: int = 6
    required_partial_target_weights: list[float] = field(default_factory=list)
    min_partial_target_fraction: float = 0.05
    min_partial_target_fold_fraction: float = 0.60
    required_predicted_target_weights: list[float] = field(
        default_factory=lambda: [0.25, 0.50]
    )
    min_predicted_target_fraction: float = 0.03
    min_predicted_target_fold_fraction: float = 0.50


@dataclass(slots=True)
class EvaluationConfig:
    walk_forward: WalkForwardConfig = field(default_factory=WalkForwardConfig)
    ml_strategy_threshold_sweep: MLStrategyThresholdSweepConfig = field(
        default_factory=MLStrategyThresholdSweepConfig
    )
    ml_strategy_tuning: MLStrategyTuningConfig = field(default_factory=MLStrategyTuningConfig)
    strict_research_gate: StrictResearchGateConfig = field(
        default_factory=StrictResearchGateConfig
    )
    benchmark_strategy: str = ""
    cost_sensitivity_bps: list[float] = field(default_factory=list)
    factor_model_path: str = ""
    periods_per_year: float = 252.0
    focus_start: str = ""
    focus_end: str = ""
    visualize_signals: bool = False


@dataclass(slots=True)
class ArtifactsConfig:
    output_dir: str = "artifacts/runs"
    save_predictions: bool = True
    save_metrics_csv: bool = True
    save_report_md: bool = True
    save_plots: bool = True


@dataclass(slots=True)
class TelegramNotificationsConfig:
    enabled: bool = False


@dataclass(slots=True)
class PaperNotificationsConfig:
    telegram: TelegramNotificationsConfig = field(default_factory=TelegramNotificationsConfig)


@dataclass(slots=True)
class PaperAzureConfig:
    artifact_backend: str = "filesystem"
    secret_backend: str = "environment"
    service_bus_backend: str = "disabled"
    blob_account_url: str = ""
    blob_container_name: str = ""
    artifact_environment: str = ""
    artifact_deployment_id: str = ""
    key_vault_url: str = ""
    service_bus_namespace: str = ""
    service_bus_queue_name: str = ""


@dataclass(slots=True)
class PaperConfig:
    enabled: bool = False
    data_provider: str = "alpaca"
    broker: str = "alpaca"
    persistence_backend: str = "filesystem"
    sqlite_db_path: str = "artifacts/paper/state/control.db"
    execution_mode: str = "agent_approval"
    agent_backend: str = "deterministic_consensus"
    agent_model: str = ""
    agent_timeout_seconds: int = 30
    agent_fallback_backend: str = "deterministic_consensus"
    consensus_min_long_votes: int = 4
    schedule_timezone: str = "America/New_York"
    decision_time: str = "16:10"
    submission_time: str = "19:05"
    order_type: str = "day_market"
    position_sizing: str = "full_equity_fractional"
    approval_inbox_dir: str = "artifacts/paper/inbox"
    state_dir: str = "artifacts/paper/state"
    poll_interval_seconds: int = 30
    azure: PaperAzureConfig = field(default_factory=PaperAzureConfig)
    notifications: PaperNotificationsConfig = field(default_factory=PaperNotificationsConfig)


@dataclass(slots=True)
class ShadowConfig:
    candidate_id: str = ""
    behavior_version: str = ""
    protocol_start: str = ""
    protocol_end: str = ""
    earliest_final_evaluation: str = ""
    maturity_lag_bars: int = 0
    code_lock: str = ""
    artifact_root: str = ""
    config_hash: str = ""
    behavior_hash: str = ""


@dataclass(slots=True)
class ExperimentConfig:
    experiment_name: str = "weekly_rank_v1"
    data: DataConfig = field(default_factory=DataConfig)
    features: FeaturesConfig = field(default_factory=FeaturesConfig)
    target: TargetConfig = field(default_factory=TargetConfig)
    portfolio: PortfolioConfig = field(default_factory=PortfolioConfig)
    baselines: BaselinesConfig = field(default_factory=BaselinesConfig)
    models: list[ModelSpec] = field(
        default_factory=lambda: [
            ModelSpec("logistic_regression"),
            ModelSpec("random_forest"),
            ModelSpec("gradient_boosting"),
        ]
    )
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    artifacts: ArtifactsConfig = field(default_factory=ArtifactsConfig)
    paper: PaperConfig = field(default_factory=PaperConfig)
    shadow: ShadowConfig | None = None
    base_dir: Path = field(default_factory=Path.cwd, repr=False)

    def resolve_path(self, value: str | Path) -> Path:
        path = Path(value)
        if path.is_absolute():
            return path
        return (self.base_dir / path).resolve()

    @property
    def cache_dir(self) -> Path:
        return self.resolve_path(self.data.cache_dir)

    @property
    def prepared_panel_path(self) -> Path:
        return self.cache_dir / self.data.prepared_panel_filename

    @property
    def output_dir(self) -> Path:
        return self.resolve_path(self.artifacts.output_dir)

    @property
    def paper_approval_inbox_dir(self) -> Path:
        return self.resolve_path(self.paper.approval_inbox_dir)

    @property
    def paper_state_dir(self) -> Path:
        return self.resolve_path(self.paper.state_dir)

    @property
    def paper_sqlite_db_path(self) -> Path:
        return self.resolve_path(self.paper.sqlite_db_path)

    @property
    def optimized_external_covariance_path(self) -> Path | None:
        path = self.baselines.optimized.external_covariance_path
        if path == "":
            return None
        return self.resolve_path(path)

    @property
    def optimized_external_expected_returns_path(self) -> Path | None:
        path = self.baselines.optimized.external_expected_returns_path
        if path == "":
            return None
        return self.resolve_path(path)

    @property
    def factor_model_path(self) -> Path | None:
        path = self.evaluation.factor_model_path
        if path == "":
            return None
        return self.resolve_path(path)


def _section(
    cls: type[_ConfigSectionT],
    data: dict[str, Any] | None,
) -> _ConfigSectionT:
    values = data or {}
    allowed = {item.name for item in fields(cast(Any, cls))}
    filtered = {key: value for key, value in values.items() if key in allowed}
    return cls(**filtered)


def _config_base_dir(path: Path) -> Path:
    if path.parent.name == "configs":
        return path.parent.parent.resolve()
    return path.parent.resolve()


def _normalize_mapping_sections(config: ExperimentConfig) -> None:
    if config.data.symbol_groups is None:
        config.data.symbol_groups = {}

    if config.baselines.allocation.symbol_weights is None:
        config.baselines.allocation.symbol_weights = {}

    if config.baselines.allocation.group_weights is None:
        config.baselines.allocation.group_weights = {}
    if config.baselines.partial_allocation_benchmarks.weights is None:
        config.baselines.partial_allocation_benchmarks.weights = []
    if config.baselines.rebalanced_partial_allocation_benchmarks.weights is None:
        config.baselines.rebalanced_partial_allocation_benchmarks.weights = []

    if config.evaluation.cost_sensitivity_bps is None:
        config.evaluation.cost_sensitivity_bps = []
    if config.evaluation.ml_strategy_threshold_sweep.thresholds is None:
        config.evaluation.ml_strategy_threshold_sweep.thresholds = []
    if config.evaluation.ml_strategy_tuning.thresholds is None:
        config.evaluation.ml_strategy_tuning.thresholds = []
    if config.evaluation.ml_strategy_tuning.tier_thresholds is None:
        config.evaluation.ml_strategy_tuning.tier_thresholds = []
    if config.evaluation.ml_strategy_tuning.tier_threshold_sets is None:
        config.evaluation.ml_strategy_tuning.tier_threshold_sets = []
    if config.evaluation.ml_strategy_tuning.selection_benchmark_strategies is None:
        config.evaluation.ml_strategy_tuning.selection_benchmark_strategies = []
    if config.evaluation.ml_strategy_tuning.rolling_train_bars_grid is None:
        config.evaluation.ml_strategy_tuning.rolling_train_bars_grid = []
    if config.evaluation.ml_strategy_tuning.min_holding_period_bars_grid is None:
        config.evaluation.ml_strategy_tuning.min_holding_period_bars_grid = [0]
    if config.evaluation.ml_strategy_tuning.hysteresis_margin_grid is None:
        config.evaluation.ml_strategy_tuning.hysteresis_margin_grid = [0.0]
    if config.evaluation.ml_strategy_tuning.allocation_score_policy_prob100_threshold_grid is None:
        config.evaluation.ml_strategy_tuning.allocation_score_policy_prob100_threshold_grid = []
    if config.evaluation.ml_strategy_tuning.allocation_score_transforms is None:
        config.evaluation.ml_strategy_tuning.allocation_score_transforms = [
            AllocationScoreTransformConfig()
        ]
    else:
        config.evaluation.ml_strategy_tuning.allocation_score_transforms = [
            transform
            if isinstance(transform, AllocationScoreTransformConfig)
            else _section(AllocationScoreTransformConfig, transform)
            for transform in config.evaluation.ml_strategy_tuning.allocation_score_transforms
        ] or [AllocationScoreTransformConfig()]
    if config.evaluation.ml_strategy_tuning.allocation_utility_profiles is None:
        config.evaluation.ml_strategy_tuning.allocation_utility_profiles = []
    else:
        config.evaluation.ml_strategy_tuning.allocation_utility_profiles = [
            profile
            if isinstance(profile, AllocationUtilityProfileConfig)
            else _section(AllocationUtilityProfileConfig, profile)
            for profile in config.evaluation.ml_strategy_tuning.allocation_utility_profiles
        ]
    if config.evaluation.ml_strategy_tuning.regime_participation_policies is None:
        config.evaluation.ml_strategy_tuning.regime_participation_policies = [
            RegimeParticipationPolicyConfig()
        ]
    else:
        config.evaluation.ml_strategy_tuning.regime_participation_policies = [
            policy
            if isinstance(policy, RegimeParticipationPolicyConfig)
            else _section(RegimeParticipationPolicyConfig, policy)
            for policy in config.evaluation.ml_strategy_tuning.regime_participation_policies
        ] or [RegimeParticipationPolicyConfig()]
    if config.evaluation.strict_research_gate.required_benchmark_strategies is None:
        config.evaluation.strict_research_gate.required_benchmark_strategies = []
    if config.evaluation.strict_research_gate.required_partial_target_weights is None:
        config.evaluation.strict_research_gate.required_partial_target_weights = []
    if config.evaluation.strict_research_gate.required_predicted_target_weights is None:
        config.evaluation.strict_research_gate.required_predicted_target_weights = []
    if config.evaluation.factor_model_path is None:
        config.evaluation.factor_model_path = ""
    if config.evaluation.focus_start is None:
        config.evaluation.focus_start = ""
    if config.evaluation.focus_end is None:
        config.evaluation.focus_end = ""
    if config.baselines.pattern_meta_label.models is None:
        config.baselines.pattern_meta_label.models = []
    if config.baselines.pattern_meta_label.exit_probability_threshold_grid is None:
        config.baselines.pattern_meta_label.exit_probability_threshold_grid = []
    partial_overlay = config.baselines.pattern_partial_exposure_overlay
    if partial_overlay.partial_exit_probability_threshold_grid is None:
        partial_overlay.partial_exit_probability_threshold_grid = []
    if partial_overlay.full_exit_probability_threshold_grid is None:
        partial_overlay.full_exit_probability_threshold_grid = []

    optimized = config.baselines.optimized
    if optimized.external_covariance_path is None:
        optimized.external_covariance_path = ""
    if optimized.external_expected_returns_path is None:
        optimized.external_expected_returns_path = ""
    if optimized.equilibrium_weights is None:
        optimized.equilibrium_weights = {}
    if optimized.views is None:
        optimized.views = []


def _validate_weights(label: str, weights: dict[str, float]) -> None:
    for value in weights.values():
        if not math.isfinite(value):
            raise ValueError(f"{label} must contain only finite numeric values.")
        if value < 0.0:
            raise ValueError(f"{label} must contain non-negative weights.")

    if abs(sum(weights.values()) - 1.0) > WEIGHT_TOLERANCE:
        raise ValueError(f"{label} must sum to 1.0.")


def _validate_cap(label: str, value: float | None) -> None:
    if value is None:
        return
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{label} must be between 0.0 and 1.0.")


def _validate_optional_score_cap(label: str, value: float | None) -> float | None:
    if value is None:
        return None
    try:
        numeric_value = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be a finite value between 0.0 and 1.0.") from exc
    if not math.isfinite(numeric_value) or not 0.0 <= numeric_value <= 1.0:
        raise ValueError(f"{label} must be a finite value between 0.0 and 1.0.")
    return numeric_value


def _validate_score_transform_float(label: str, value: float) -> float:
    try:
        numeric_value = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be finite.") from exc
    if not math.isfinite(numeric_value):
        raise ValueError(f"{label} must be finite.")
    return numeric_value


def _validate_non_negative_bps_list(label: str, values: list[float]) -> None:
    for value in values:
        if not math.isfinite(value) or value < 0.0:
            raise ValueError(f"{label} must contain only finite non-negative values.")


def _validate_positive_float(label: str, value: float) -> None:
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"{label} must be a finite positive value.")


def _validate_positive_int_list(label: str, values: list[int]) -> None:
    for value in values:
        if int(value) < 1:
            raise ValueError(f"{label} must contain only positive integers.")


def _validate_non_negative_int_list(label: str, values: list[int]) -> None:
    for value in values:
        if int(value) < 0:
            raise ValueError(f"{label} must contain only non-negative integers.")


def _validate_tier_thresholds(label: str, values: list[float]) -> None:
    if len(values) != 3:
        raise ValueError(f"{label} must contain exactly three thresholds.")
    previous = -math.inf
    for value in values:
        if not math.isfinite(value) or value < 0.0 or value > 1.0:
            raise ValueError(f"{label} must contain values between 0.0 and 1.0.")
        if value < previous:
            raise ValueError(f"{label} must be sorted in ascending order.")
        previous = value


def _validate_regime_policy_tier(label: str, value: float) -> float:
    try:
        numeric_value = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{label} must be one of: 0.0, 0.25, 0.5, 1.0."
        ) from exc
    if not math.isfinite(numeric_value) or not any(
        abs(numeric_value - allowed) <= WEIGHT_TOLERANCE
        for allowed in REGIME_PARTICIPATION_POLICY_TIERS
    ):
        raise ValueError(f"{label} must be one of: 0.0, 0.25, 0.5, 1.0.")
    return min(
        REGIME_PARTICIPATION_POLICY_TIERS,
        key=lambda allowed: abs(numeric_value - allowed),
    )


def default_periods_per_year(interval: str) -> float:
    normalized = interval.lower()
    if normalized not in INTERVAL_PERIODS_PER_YEAR:
        allowed = ", ".join(sorted(INTERVAL_PERIODS_PER_YEAR))
        raise ValueError(f"data.interval must be one of: {allowed}")
    return INTERVAL_PERIODS_PER_YEAR[normalized]


def _validate_optional_datetime(label: str, value: str) -> datetime | None:
    if value == "":
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"{label} must be an ISO-8601 date or datetime.") from exc


def _validate_clock_string(label: str, value: str) -> None:
    parts = value.split(":")
    if len(parts) != 2 or not all(part.isdigit() for part in parts):
        raise ValueError(f"{label} must use HH:MM 24-hour format.")
    hour, minute = (int(part) for part in parts)
    if hour not in range(24) or minute not in range(60):
        raise ValueError(f"{label} must use HH:MM 24-hour format.")


def _validate_config(config: ExperimentConfig) -> None:
    default_periods_per_year(config.data.interval)

    if config.target.horizon_days < 1:
        raise ValueError("target.horizon_days must be at least 1.")
    if config.target.type not in TARGET_TYPES:
        allowed = ", ".join(sorted(TARGET_TYPES))
        raise ValueError(f"target.type must be one of: {allowed}.")
    _validate_positive_float(
        "target.allocation_utility_drawdown_penalty",
        config.target.allocation_utility_drawdown_penalty,
    )
    _validate_positive_float(
        "target.allocation_utility_volatility_penalty",
        config.target.allocation_utility_volatility_penalty,
    )
    try:
        allocation_utility_risk_penalty_power = float(
            config.target.allocation_utility_risk_penalty_power
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "target.allocation_utility_risk_penalty_power must be a finite value greater than or equal to 1.0."
        ) from exc
    if (
        not math.isfinite(allocation_utility_risk_penalty_power)
        or allocation_utility_risk_penalty_power < 1.0
    ):
        raise ValueError(
            "target.allocation_utility_risk_penalty_power must be a finite value greater than or equal to 1.0."
        )
    config.target.allocation_utility_risk_penalty_power = (
        allocation_utility_risk_penalty_power
    )

    features = config.features
    if features.crypto_time_series_enabled:
        _validate_positive_int_list("features.crypto_return_windows", features.crypto_return_windows)
        _validate_positive_int_list("features.crypto_vol_windows", features.crypto_vol_windows)
        _validate_positive_int_list("features.crypto_ma_windows", features.crypto_ma_windows)
        if features.crypto_rsi_window < 1:
            raise ValueError("features.crypto_rsi_window must be at least 1.")
        if features.crypto_macd_fast_window < 1:
            raise ValueError("features.crypto_macd_fast_window must be at least 1.")
        if features.crypto_macd_slow_window <= features.crypto_macd_fast_window:
            raise ValueError(
                "features.crypto_macd_slow_window must be greater than "
                "features.crypto_macd_fast_window."
            )
        if features.crypto_macd_signal_window < 1:
            raise ValueError("features.crypto_macd_signal_window must be at least 1.")
        if features.crypto_bollinger_window < 2:
            raise ValueError("features.crypto_bollinger_window must be at least 2.")
        _validate_positive_float(
            "features.crypto_bollinger_std",
            features.crypto_bollinger_std,
        )
        if features.crypto_volume_window < 1:
            raise ValueError("features.crypto_volume_window must be at least 1.")
    if features.crypto_regime_features_enabled:
        _validate_positive_int_list(
            "features.crypto_regime_trend_windows",
            features.crypto_regime_trend_windows,
        )
        if features.crypto_regime_volatility_window < 2:
            raise ValueError("features.crypto_regime_volatility_window must be at least 2.")
        if features.crypto_regime_percentile_window < 2:
            raise ValueError("features.crypto_regime_percentile_window must be at least 2.")
        if features.crypto_regime_drawdown_window < 2:
            raise ValueError("features.crypto_regime_drawdown_window must be at least 2.")
        if features.crypto_regime_volume_window < 2:
            raise ValueError("features.crypto_regime_volume_window must be at least 2.")

    symbols = list(config.data.symbols)
    symbol_set = set(symbols)
    group_symbol_keys = set(config.data.symbol_groups)

    unknown_group_symbols = sorted(group_symbol_keys - symbol_set)
    if unknown_group_symbols:
        joined = ", ".join(unknown_group_symbols)
        raise ValueError(f"data.symbol_groups contains unknown symbols: {joined}")

    risk = config.portfolio.risk
    _validate_cap("portfolio.risk.max_position_weight", risk.max_position_weight)
    _validate_cap("portfolio.risk.max_group_weight", risk.max_group_weight)
    _validate_cap("portfolio.risk.max_long_exposure", risk.max_long_exposure)
    _validate_cap("portfolio.risk.max_short_exposure", risk.max_short_exposure)

    if config.portfolio.ranking.mode == "long_only" and risk.max_short_exposure is not None:
        raise ValueError(
            "portfolio.risk.max_short_exposure is not allowed when portfolio.ranking.mode='long_only'."
        )

    if risk.max_group_weight is not None:
        missing_group_symbols = sorted(symbol_set - group_symbol_keys)
        if missing_group_symbols:
            joined = ", ".join(missing_group_symbols)
            raise ValueError(
                "portfolio.risk.max_group_weight requires data.symbol_groups for all "
                f"data.symbols: {joined}"
            )

    _validate_non_negative_bps_list(
        "evaluation.cost_sensitivity_bps",
        config.evaluation.cost_sensitivity_bps,
    )
    ml_sweep = config.evaluation.ml_strategy_threshold_sweep
    for threshold in ml_sweep.thresholds:
        if not math.isfinite(threshold) or threshold < 0.0 or threshold > 1.0:
            raise ValueError(
                "evaluation.ml_strategy_threshold_sweep.thresholds must contain values between 0.0 and 1.0."
            )
    if ml_sweep.min_exposure_changes < 0:
        raise ValueError(
            "evaluation.ml_strategy_threshold_sweep.min_exposure_changes must be non-negative."
        )
    max_exposure = ml_sweep.max_average_exposure_for_active
    if not math.isfinite(max_exposure) or max_exposure < 0.0 or max_exposure > 1.0:
        raise ValueError(
            "evaluation.ml_strategy_threshold_sweep.max_average_exposure_for_active must be between 0.0 and 1.0."
        )
    ml_tuning = config.evaluation.ml_strategy_tuning
    for threshold in ml_tuning.thresholds:
        if not math.isfinite(threshold) or threshold < 0.0 or threshold > 1.0:
            raise ValueError(
                "evaluation.ml_strategy_tuning.thresholds must contain values between 0.0 and 1.0."
            )
    if ml_tuning.allocation_mode not in ML_STRATEGY_ALLOCATION_MODES:
        allowed = ", ".join(sorted(ML_STRATEGY_ALLOCATION_MODES))
        raise ValueError(f"evaluation.ml_strategy_tuning.allocation_mode must be one of: {allowed}")
    _validate_tier_thresholds(
        "evaluation.ml_strategy_tuning.tier_thresholds",
        ml_tuning.tier_thresholds,
    )
    for index, threshold_set in enumerate(ml_tuning.tier_threshold_sets):
        _validate_tier_thresholds(
            f"evaluation.ml_strategy_tuning.tier_threshold_sets[{index}]",
            threshold_set,
        )
    if ml_tuning.validation_months < 1:
        raise ValueError("evaluation.ml_strategy_tuning.validation_months must be at least 1.")
    if ml_tuning.min_validation_rows < 1:
        raise ValueError("evaluation.ml_strategy_tuning.min_validation_rows must be at least 1.")
    if ml_tuning.min_exposure_changes < 0:
        raise ValueError(
            "evaluation.ml_strategy_tuning.min_exposure_changes must be non-negative."
        )
    min_exposure = ml_tuning.min_average_exposure_for_active
    if not math.isfinite(min_exposure) or min_exposure < 0.0 or min_exposure > 1.0:
        raise ValueError(
            "evaluation.ml_strategy_tuning.min_average_exposure_for_active must be between 0.0 and 1.0."
        )
    max_exposure = ml_tuning.max_average_exposure_for_active
    if not math.isfinite(max_exposure) or max_exposure < 0.0 or max_exposure > 1.0:
        raise ValueError(
            "evaluation.ml_strategy_tuning.max_average_exposure_for_active must be between 0.0 and 1.0."
        )
    if min_exposure > max_exposure:
        raise ValueError(
            "evaluation.ml_strategy_tuning.min_average_exposure_for_active must be less than or equal to max_average_exposure_for_active."
        )
    _validate_positive_int_list(
        "evaluation.ml_strategy_tuning.rolling_train_bars_grid",
        ml_tuning.rolling_train_bars_grid,
    )
    _validate_non_negative_int_list(
        "evaluation.ml_strategy_tuning.min_holding_period_bars_grid",
        ml_tuning.min_holding_period_bars_grid,
    )
    for margin in ml_tuning.hysteresis_margin_grid:
        if not math.isfinite(margin) or margin < 0.0 or margin > 0.25:
            raise ValueError(
                "evaluation.ml_strategy_tuning.hysteresis_margin_grid must contain values between 0.0 and 0.25."
            )
    if ml_tuning.max_annualized_turnover is not None and (
        not math.isfinite(ml_tuning.max_annualized_turnover)
        or ml_tuning.max_annualized_turnover <= 0.0
    ):
        raise ValueError(
            "evaluation.ml_strategy_tuning.max_annualized_turnover must be a finite positive value."
        )
    if ml_tuning.objective not in ML_STRATEGY_TUNING_OBJECTIVES:
        allowed = ", ".join(sorted(ML_STRATEGY_TUNING_OBJECTIVES))
        raise ValueError(
            f"evaluation.ml_strategy_tuning.objective must be one of: {allowed}."
        )
    ml_tuning.selection_policy = str(ml_tuning.selection_policy).strip()
    if ml_tuning.selection_policy not in ML_STRATEGY_SELECTION_POLICIES:
        allowed = ", ".join(sorted(ML_STRATEGY_SELECTION_POLICIES))
        raise ValueError(
            f"evaluation.ml_strategy_tuning.selection_policy must be one of: {allowed}."
        )
    ml_tuning.allocation_score_policy = str(ml_tuning.allocation_score_policy).strip()
    if ml_tuning.allocation_score_policy not in ML_STRATEGY_ALLOCATION_SCORE_POLICIES:
        allowed = ", ".join(sorted(ML_STRATEGY_ALLOCATION_SCORE_POLICIES))
        raise ValueError(
            f"evaluation.ml_strategy_tuning.allocation_score_policy must be one of: {allowed}."
        )
    try:
        allocation_score_policy_prob100_threshold = float(
            ml_tuning.allocation_score_policy_prob100_threshold
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "evaluation.ml_strategy_tuning.allocation_score_policy_prob100_threshold must be a finite value between 0.0 and 1.0."
        ) from exc
    if (
        not math.isfinite(allocation_score_policy_prob100_threshold)
        or allocation_score_policy_prob100_threshold < 0.0
        or allocation_score_policy_prob100_threshold > 1.0
    ):
        raise ValueError(
            "evaluation.ml_strategy_tuning.allocation_score_policy_prob100_threshold must be a finite value between 0.0 and 1.0."
        )
    ml_tuning.allocation_score_policy_prob100_threshold = (
        allocation_score_policy_prob100_threshold
    )
    validated_prob100_threshold_grid: list[float] = []
    for threshold in ml_tuning.allocation_score_policy_prob100_threshold_grid:
        try:
            threshold_value = float(threshold)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "evaluation.ml_strategy_tuning.allocation_score_policy_prob100_threshold_grid must contain finite values between 0.0 and 1.0."
            ) from exc
        if (
            not math.isfinite(threshold_value)
            or threshold_value < 0.0
            or threshold_value > 1.0
        ):
            raise ValueError(
                "evaluation.ml_strategy_tuning.allocation_score_policy_prob100_threshold_grid must contain finite values between 0.0 and 1.0."
            )
        validated_prob100_threshold_grid.append(threshold_value)
    ml_tuning.allocation_score_policy_prob100_threshold_grid = (
        validated_prob100_threshold_grid
    )
    validated_selection_validation_cost_bps: list[float] = []
    seen_selection_validation_cost_bps: set[float] = set()
    for cost_bps in ml_tuning.selection_validation_cost_bps:
        try:
            cost_value = float(cost_bps)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "evaluation.ml_strategy_tuning.selection_validation_cost_bps must contain finite non-negative values."
            ) from exc
        if not math.isfinite(cost_value) or cost_value < 0.0:
            raise ValueError(
                "evaluation.ml_strategy_tuning.selection_validation_cost_bps must contain finite non-negative values."
            )
        if cost_value in seen_selection_validation_cost_bps:
            raise ValueError(
                "evaluation.ml_strategy_tuning.selection_validation_cost_bps must not contain duplicate values."
            )
        seen_selection_validation_cost_bps.add(cost_value)
        validated_selection_validation_cost_bps.append(cost_value)
    ml_tuning.selection_validation_cost_bps = validated_selection_validation_cost_bps
    if not isinstance(ml_tuning.guarded_gate_bull_risk_off_override, bool):
        raise ValueError(
            "evaluation.ml_strategy_tuning.guarded_gate_bull_risk_off_override must be a boolean."
        )
    seen_score_transform_names: set[str] = set()
    for transform in ml_tuning.allocation_score_transforms:
        transform.name = str(transform.name).strip()
        if transform.name == "":
            raise ValueError(
                "evaluation.ml_strategy_tuning.allocation_score_transforms names must be non-empty."
            )
        if transform.name in seen_score_transform_names:
            raise ValueError(
                "evaluation.ml_strategy_tuning.allocation_score_transforms must not contain duplicate names."
            )
        seen_score_transform_names.add(transform.name)
        transform.bull_multiplier = _validate_score_transform_float(
            f"evaluation.ml_strategy_tuning.allocation_score_transforms[{transform.name}].bull_multiplier",
            transform.bull_multiplier,
        )
        if transform.bull_multiplier < 0.0:
            raise ValueError(
                "evaluation.ml_strategy_tuning.allocation_score_transforms bull_multiplier must be non-negative."
            )
        transform.bull_addend = _validate_score_transform_float(
            f"evaluation.ml_strategy_tuning.allocation_score_transforms[{transform.name}].bull_addend",
            transform.bull_addend,
        )
        transform.risk_off_score_cap = _validate_optional_score_cap(
            f"evaluation.ml_strategy_tuning.allocation_score_transforms[{transform.name}].risk_off_score_cap",
            transform.risk_off_score_cap,
        )
        transform.non_bull_score_cap = _validate_optional_score_cap(
            f"evaluation.ml_strategy_tuning.allocation_score_transforms[{transform.name}].non_bull_score_cap",
            transform.non_bull_score_cap,
        )
    seen_selection_benchmarks: set[str] = set()
    for benchmark_strategy in ml_tuning.selection_benchmark_strategies:
        benchmark_name = str(benchmark_strategy).strip()
        if benchmark_name == "":
            raise ValueError(
                "evaluation.ml_strategy_tuning.selection_benchmark_strategies must not contain empty values."
            )
        if benchmark_name in seen_selection_benchmarks:
            raise ValueError(
                "evaluation.ml_strategy_tuning.selection_benchmark_strategies must not contain duplicate values."
            )
        seen_selection_benchmarks.add(benchmark_name)
    if (
        ml_tuning.objective
        in {
            "net_return_and_risk_vs_required_benchmarks",
            "net_return_risk_score_validity_vs_required_benchmarks",
        }
        and not ml_tuning.selection_benchmark_strategies
    ):
        raise ValueError(
            "evaluation.ml_strategy_tuning.selection_benchmark_strategies must contain at least one strategy when objective is net_return_and_risk_vs_required_benchmarks."
        )
    seen_profile_names: set[str] = set()
    for profile in ml_tuning.allocation_utility_profiles:
        profile.name = str(profile.name).strip()
        if profile.name == "":
            raise ValueError(
                "evaluation.ml_strategy_tuning.allocation_utility_profiles names must be non-empty."
            )
        if profile.name in seen_profile_names:
            raise ValueError(
                "evaluation.ml_strategy_tuning.allocation_utility_profiles must not contain duplicate names."
            )
        seen_profile_names.add(profile.name)
        _validate_positive_float(
            f"evaluation.ml_strategy_tuning.allocation_utility_profiles[{profile.name}].drawdown_penalty",
            profile.drawdown_penalty,
        )
        _validate_positive_float(
            f"evaluation.ml_strategy_tuning.allocation_utility_profiles[{profile.name}].volatility_penalty",
            profile.volatility_penalty,
        )
        try:
            risk_penalty_power = float(profile.risk_penalty_power)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "evaluation.ml_strategy_tuning.allocation_utility_profiles risk_penalty_power must be a finite value greater than or equal to 1.0."
            ) from exc
        if not math.isfinite(risk_penalty_power) or risk_penalty_power < 1.0:
            raise ValueError(
                "evaluation.ml_strategy_tuning.allocation_utility_profiles risk_penalty_power must be a finite value greater than or equal to 1.0."
            )
        profile.risk_penalty_power = risk_penalty_power
    seen_policy_names: set[str] = set()
    for policy in ml_tuning.regime_participation_policies:
        policy.name = str(policy.name).strip()
        if policy.name == "":
            raise ValueError(
                "evaluation.ml_strategy_tuning.regime_participation_policies names must be non-empty."
            )
        if policy.name in seen_policy_names:
            raise ValueError(
                "evaluation.ml_strategy_tuning.regime_participation_policies must not contain duplicate names."
            )
        seen_policy_names.add(policy.name)
        policy.bull_floor = _validate_regime_policy_tier(
            f"evaluation.ml_strategy_tuning.regime_participation_policies[{policy.name}].bull_floor",
            policy.bull_floor,
        )
        policy.sideways_floor = _validate_regime_policy_tier(
            f"evaluation.ml_strategy_tuning.regime_participation_policies[{policy.name}].sideways_floor",
            policy.sideways_floor,
        )
        policy.bear_floor = _validate_regime_policy_tier(
            f"evaluation.ml_strategy_tuning.regime_participation_policies[{policy.name}].bear_floor",
            policy.bear_floor,
        )
        if policy.risk_off_cap is not None:
            policy.risk_off_cap = _validate_regime_policy_tier(
                f"evaluation.ml_strategy_tuning.regime_participation_policies[{policy.name}].risk_off_cap",
                policy.risk_off_cap,
            )
        if policy.gate_bull_floor is not None:
            policy.gate_bull_floor = _validate_regime_policy_tier(
                f"evaluation.ml_strategy_tuning.regime_participation_policies[{policy.name}].gate_bull_floor",
                policy.gate_bull_floor,
            )
    configured_fallback_names = [
        str(value).strip()
        for value in (
            ml_tuning.no_candidate_fallback_regime_policy,
            ml_tuning.no_valid_candidate_regime_fallback,
        )
        if value is not None
    ]
    if (
        len(configured_fallback_names) == 2
        and configured_fallback_names[0] != configured_fallback_names[1]
    ):
        raise ValueError(
            "evaluation.ml_strategy_tuning.no_candidate_fallback_regime_policy "
            "and no_valid_candidate_regime_fallback must match when both are configured."
        )
    fallback_policy_name = configured_fallback_names[0] if configured_fallback_names else None
    if fallback_policy_name is not None:
        if fallback_policy_name == "":
            raise ValueError(
                "evaluation.ml_strategy_tuning.no_candidate_fallback_regime_policy/no_valid_candidate_regime_fallback must be non-empty when configured."
            )
        if fallback_policy_name not in seen_policy_names:
            allowed = ", ".join(sorted(seen_policy_names))
            raise ValueError(
                "evaluation.ml_strategy_tuning.no_candidate_fallback_regime_policy/no_valid_candidate_regime_fallback "
                f"must reference one of regime_participation_policies: {allowed}."
            )
        fallback_policy = next(
            policy
            for policy in ml_tuning.regime_participation_policies
            if policy.name == fallback_policy_name
        )
        if fallback_policy.risk_off_cap is None:
            raise ValueError(
                "evaluation.ml_strategy_tuning.no_candidate_fallback_regime_policy/no_valid_candidate_regime_fallback "
                "requires the referenced policy to define risk_off_cap."
            )
        ml_tuning.no_candidate_fallback_regime_policy = fallback_policy_name
        ml_tuning.no_valid_candidate_regime_fallback = fallback_policy_name
    if ml_tuning.allocation_class_weighting not in ALLOCATION_CLASS_WEIGHTING_MODES:
        allowed = ", ".join(sorted(ALLOCATION_CLASS_WEIGHTING_MODES))
        raise ValueError(
            f"evaluation.ml_strategy_tuning.allocation_class_weighting must be one of: {allowed}."
        )
    _validate_positive_float(
        "evaluation.ml_strategy_tuning.allocation_partial_class_weight_multiplier",
        ml_tuning.allocation_partial_class_weight_multiplier,
    )
    if ml_tuning.allocation_probability_calibration not in ALLOCATION_PROBABILITY_CALIBRATION_MODES:
        allowed = ", ".join(sorted(ALLOCATION_PROBABILITY_CALIBRATION_MODES))
        raise ValueError(
            f"evaluation.ml_strategy_tuning.allocation_probability_calibration must be one of: {allowed}."
        )
    if int(ml_tuning.allocation_calibration_cv) < 2:
        raise ValueError(
            "evaluation.ml_strategy_tuning.allocation_calibration_cv must be at least 2."
        )
    ml_tuning.allocation_calibration_cv = int(ml_tuning.allocation_calibration_cv)
    _validate_positive_float("evaluation.periods_per_year", config.evaluation.periods_per_year)
    gate = config.evaluation.strict_research_gate
    if gate.enabled:
        if gate.strategy_name.strip() == "":
            raise ValueError("evaluation.strict_research_gate.strategy_name must be non-empty.")
        if gate.benchmark_strategy.strip() == "":
            raise ValueError("evaluation.strict_research_gate.benchmark_strategy must be non-empty.")
        for benchmark_strategy in gate.required_benchmark_strategies:
            if str(benchmark_strategy).strip() == "":
                raise ValueError(
                    "evaluation.strict_research_gate.required_benchmark_strategies must not contain empty values."
                )
        if not math.isfinite(gate.cost_gate_bps) or gate.cost_gate_bps < 0.0:
            raise ValueError("evaluation.strict_research_gate.cost_gate_bps must be non-negative.")
        if not math.isfinite(gate.acceptable_cost_bps) or gate.acceptable_cost_bps < 0.0:
            raise ValueError(
                "evaluation.strict_research_gate.acceptable_cost_bps must be non-negative."
            )
        if gate.min_positive_regime_slices < 1:
            raise ValueError(
                "evaluation.strict_research_gate.min_positive_regime_slices must be at least 1."
            )
        _validate_cap(
            "evaluation.strict_research_gate.min_average_exposure",
            gate.min_average_exposure,
        )
        _validate_cap(
            "evaluation.strict_research_gate.max_average_exposure",
            gate.max_average_exposure,
        )
        if gate.min_average_exposure > gate.max_average_exposure:
            raise ValueError(
                "evaluation.strict_research_gate.min_average_exposure must be less than or equal to max_average_exposure."
            )
        if (
            not math.isfinite(gate.min_selected_fold_fraction)
            or gate.min_selected_fold_fraction < 0.0
            or gate.min_selected_fold_fraction > 1.0
        ):
            raise ValueError(
                "evaluation.strict_research_gate.min_selected_fold_fraction must be between 0.0 and 1.0."
            )
        if gate.recent_window_months < 1:
            raise ValueError("evaluation.strict_research_gate.recent_window_months must be at least 1.")
        seen_partial_target_weights: set[float] = set()
        for target_weight in gate.required_partial_target_weights:
            numeric_weight = float(target_weight)
            if (
                not math.isfinite(numeric_weight)
                or numeric_weight <= 0.0
                or numeric_weight >= 1.0
            ):
                raise ValueError(
                    "evaluation.strict_research_gate.required_partial_target_weights must contain values greater than 0.0 and less than 1.0."
                )
            if numeric_weight in seen_partial_target_weights:
                raise ValueError(
                    "evaluation.strict_research_gate.required_partial_target_weights must not contain duplicate values."
                )
            seen_partial_target_weights.add(numeric_weight)
        seen_predicted_target_weights: set[float] = set()
        for target_weight in gate.required_predicted_target_weights:
            numeric_weight = float(target_weight)
            if (
                not math.isfinite(numeric_weight)
                or numeric_weight <= 0.0
                or numeric_weight >= 1.0
            ):
                raise ValueError(
                    "evaluation.strict_research_gate.required_predicted_target_weights must contain values greater than 0.0 and less than 1.0."
                )
            if numeric_weight in seen_predicted_target_weights:
                raise ValueError(
                    "evaluation.strict_research_gate.required_predicted_target_weights must not contain duplicate values."
                )
            seen_predicted_target_weights.add(numeric_weight)
        for label, value in {
            "evaluation.strict_research_gate.min_partial_target_fraction": gate.min_partial_target_fraction,
            "evaluation.strict_research_gate.min_partial_target_fold_fraction": gate.min_partial_target_fold_fraction,
            "evaluation.strict_research_gate.min_predicted_target_fraction": gate.min_predicted_target_fraction,
            "evaluation.strict_research_gate.min_predicted_target_fold_fraction": gate.min_predicted_target_fold_fraction,
        }.items():
            if not math.isfinite(value) or value < 0.0 or value > 1.0:
                raise ValueError(f"{label} must be between 0.0 and 1.0.")
    focus_start = _validate_optional_datetime(
        "evaluation.focus_start",
        config.evaluation.focus_start,
    )
    focus_end = _validate_optional_datetime("evaluation.focus_end", config.evaluation.focus_end)
    if focus_start is not None and focus_end is not None and focus_start > focus_end:
        raise ValueError("evaluation.focus_start must be before or equal to evaluation.focus_end.")
    paper = config.paper
    if paper.data_provider not in PAPER_DATA_PROVIDERS:
        allowed = ", ".join(sorted(PAPER_DATA_PROVIDERS))
        raise ValueError(f"paper.data_provider must be one of: {allowed}")
    if paper.broker not in PAPER_BROKERS:
        allowed = ", ".join(sorted(PAPER_BROKERS))
        raise ValueError(f"paper.broker must be one of: {allowed}")
    if paper.persistence_backend not in PAPER_PERSISTENCE_BACKENDS:
        allowed = ", ".join(sorted(PAPER_PERSISTENCE_BACKENDS))
        raise ValueError(f"paper.persistence_backend must be one of: {allowed}")
    if paper.persistence_backend == "sqlite" and paper.sqlite_db_path.strip() == "":
        raise ValueError("paper.sqlite_db_path must be set when paper.persistence_backend='sqlite'.")
    if paper.execution_mode not in PAPER_EXECUTION_MODES:
        allowed = ", ".join(sorted(PAPER_EXECUTION_MODES))
        raise ValueError(f"paper.execution_mode must be one of: {allowed}")
    if paper.agent_backend not in PAPER_AGENT_BACKENDS:
        allowed = ", ".join(sorted(PAPER_AGENT_BACKENDS))
        raise ValueError(f"paper.agent_backend must be one of: {allowed}")
    if paper.agent_fallback_backend not in PAPER_AGENT_BACKENDS:
        allowed = ", ".join(sorted(PAPER_AGENT_BACKENDS))
        raise ValueError(f"paper.agent_fallback_backend must be one of: {allowed}")
    if paper.agent_fallback_backend != "deterministic_consensus":
        raise ValueError(
            "paper.agent_fallback_backend must remain 'deterministic_consensus' in Phase 7.1."
        )
    if paper.agent_backend in {"openai", "claude"} and paper.agent_model.strip() == "":
        raise ValueError(
            "paper.agent_model must be set when paper.agent_backend is 'openai' or 'claude'."
        )
    if paper.order_type not in PAPER_ORDER_TYPES:
        allowed = ", ".join(sorted(PAPER_ORDER_TYPES))
        raise ValueError(f"paper.order_type must be one of: {allowed}")
    if paper.position_sizing not in PAPER_POSITION_SIZING:
        allowed = ", ".join(sorted(PAPER_POSITION_SIZING))
        raise ValueError(f"paper.position_sizing must be one of: {allowed}")
    _validate_clock_string("paper.decision_time", paper.decision_time)
    _validate_clock_string("paper.submission_time", paper.submission_time)
    if paper.agent_timeout_seconds < 1:
        raise ValueError("paper.agent_timeout_seconds must be at least 1.")
    if paper.consensus_min_long_votes < 1:
        raise ValueError("paper.consensus_min_long_votes must be at least 1.")
    if paper.poll_interval_seconds < 1:
        raise ValueError("paper.poll_interval_seconds must be at least 1.")

    paper_azure = paper.azure
    if paper_azure.artifact_backend not in PAPER_AZURE_ARTIFACT_BACKENDS:
        allowed = ", ".join(sorted(PAPER_AZURE_ARTIFACT_BACKENDS))
        raise ValueError(f"paper.azure.artifact_backend must be one of: {allowed}")
    if paper_azure.secret_backend not in PAPER_AZURE_SECRET_BACKENDS:
        allowed = ", ".join(sorted(PAPER_AZURE_SECRET_BACKENDS))
        raise ValueError(f"paper.azure.secret_backend must be one of: {allowed}")
    if paper_azure.service_bus_backend not in PAPER_AZURE_SERVICE_BUS_BACKENDS:
        allowed = ", ".join(sorted(PAPER_AZURE_SERVICE_BUS_BACKENDS))
        raise ValueError(f"paper.azure.service_bus_backend must be one of: {allowed}")
    if paper_azure.artifact_backend == "azure_blob":
        required_blob_fields = (
            "blob_account_url",
            "blob_container_name",
            "artifact_environment",
            "artifact_deployment_id",
        )
        for field_name in required_blob_fields:
            if str(getattr(paper_azure, field_name)).strip() == "":
                raise ValueError(
                    "paper.azure."
                    f"{field_name} must be set when paper.azure.artifact_backend='azure_blob'."
                )
        if not paper_azure.blob_account_url.startswith("https://"):
            raise ValueError(
                "paper.azure.blob_account_url must use https when "
                "paper.azure.artifact_backend='azure_blob'."
            )
    if paper_azure.secret_backend == "key_vault" and paper_azure.key_vault_url.strip() == "":
        raise ValueError(
            "paper.azure.key_vault_url must be set when paper.azure.secret_backend='key_vault'."
        )
    if (
        paper_azure.service_bus_backend == "azure_service_bus"
        and (
            paper_azure.service_bus_namespace.strip() == ""
            or paper_azure.service_bus_queue_name.strip() == ""
        )
    ):
        raise ValueError(
            "paper.azure.service_bus_namespace and paper.azure.service_bus_queue_name must be set "
            "when paper.azure.service_bus_backend='azure_service_bus'."
        )

    optimized = config.baselines.optimized
    if optimized.method not in OPTIMIZED_METHODS:
        allowed = ", ".join(sorted(OPTIMIZED_METHODS))
        raise ValueError(f"baselines.optimized.method must be one of: {allowed}")
    if optimized.covariance_estimator not in COVARIANCE_ESTIMATORS:
        allowed = ", ".join(sorted(COVARIANCE_ESTIMATORS))
        raise ValueError(f"baselines.optimized.covariance_estimator must be one of: {allowed}")
    if optimized.expected_return_source not in EXPECTED_RETURN_SOURCES:
        allowed = ", ".join(sorted(EXPECTED_RETURN_SOURCES))
        raise ValueError(f"baselines.optimized.expected_return_source must be one of: {allowed}")
    if optimized.lookback_days < 2:
        raise ValueError("baselines.optimized.lookback_days must be at least 2.")
    _validate_positive_float(
        "baselines.optimized.target_gross_exposure",
        optimized.target_gross_exposure,
    )
    _validate_positive_float(
        "baselines.optimized.risk_aversion",
        optimized.risk_aversion,
    )
    _validate_positive_float("baselines.optimized.tau", optimized.tau)
    if optimized.method == "mean_variance":
        if not optimized.long_only:
            raise ValueError(
                "baselines.optimized.long_only must be true when "
                "baselines.optimized.method='mean_variance'."
            )
        if optimized.target_gross_exposure > 1.0:
            raise ValueError(
                "baselines.optimized.target_gross_exposure must be less than or equal to 1.0 "
                "when baselines.optimized.method='mean_variance'."
            )
    if optimized.method == "risk_parity":
        if not optimized.long_only:
            raise ValueError(
                "baselines.optimized.long_only must be true when "
                "baselines.optimized.method='risk_parity'."
            )
        if optimized.target_gross_exposure > 1.0:
            raise ValueError(
                "baselines.optimized.target_gross_exposure must be less than or equal to 1.0 "
                "when baselines.optimized.method='risk_parity'."
            )
        if optimized.expected_return_source != "historical_mean":
            raise ValueError(
                "baselines.optimized.expected_return_source must remain 'historical_mean' "
                "when baselines.optimized.method='risk_parity'."
            )
        if optimized.external_expected_returns_path != "":
            raise ValueError(
                "baselines.optimized.external_expected_returns_path must be empty when "
                "baselines.optimized.method='risk_parity'."
            )
    if optimized.method == "black_litterman":
        if not optimized.long_only:
            raise ValueError(
                "baselines.optimized.long_only must be true when "
                "baselines.optimized.method='black_litterman'."
            )
        if optimized.target_gross_exposure > 1.0:
            raise ValueError(
                "baselines.optimized.target_gross_exposure must be less than or equal to 1.0 "
                "when baselines.optimized.method='black_litterman'."
            )
        if optimized.expected_return_source != "historical_mean":
            raise ValueError(
                "baselines.optimized.expected_return_source must remain 'historical_mean' "
                "when baselines.optimized.method='black_litterman'."
            )
        if optimized.external_expected_returns_path != "":
            raise ValueError(
                "baselines.optimized.external_expected_returns_path must be empty when "
                "baselines.optimized.method='black_litterman'."
            )
        if set(optimized.equilibrium_weights) != symbol_set:
            raise ValueError(
                "baselines.optimized.equilibrium_weights must match data.symbols exactly "
                "when baselines.optimized.method='black_litterman'."
            )
        _validate_weights(
            "baselines.optimized.equilibrium_weights",
            optimized.equilibrium_weights,
        )
        if not optimized.views:
            raise ValueError(
                "baselines.optimized.views must be non-empty when "
                "baselines.optimized.method='black_litterman'."
            )
        for index, view in enumerate(optimized.views):
            label = f"baselines.optimized.views[{index}]"
            if not view.name:
                raise ValueError(f"{label}.name must be non-empty.")
            unknown_view_symbols = sorted(set(view.weights) - symbol_set)
            if unknown_view_symbols:
                joined = ", ".join(unknown_view_symbols)
                raise ValueError(f"{label}.weights contains unknown symbols: {joined}")
            if not view.weights:
                raise ValueError(f"{label}.weights must not be empty.")
            if not math.isfinite(view.view_return):
                raise ValueError(f"{label}.view_return must be finite.")
            non_zero_weights = 0
            for symbol, coefficient in view.weights.items():
                if not math.isfinite(coefficient):
                    raise ValueError(f"{label}.weights[{symbol}] must be finite.")
                if abs(coefficient) > WEIGHT_TOLERANCE:
                    non_zero_weights += 1
            if non_zero_weights == 0:
                raise ValueError(f"{label}.weights must contain at least one non-zero coefficient.")
    if optimized.covariance_estimator == "external_csv":
        if optimized.external_covariance_path == "":
            raise ValueError(
                "baselines.optimized.external_covariance_path is required when "
                "baselines.optimized.covariance_estimator='external_csv'."
            )
    elif optimized.external_covariance_path != "":
        raise ValueError(
            "baselines.optimized.external_covariance_path must be empty unless "
            "baselines.optimized.covariance_estimator='external_csv'."
        )
    if optimized.expected_return_source == "external_csv":
        if optimized.external_expected_returns_path == "":
            raise ValueError(
                "baselines.optimized.external_expected_returns_path is required when "
                "baselines.optimized.expected_return_source='external_csv'."
            )
    elif optimized.external_expected_returns_path != "":
        raise ValueError(
            "baselines.optimized.external_expected_returns_path must be empty unless "
            "baselines.optimized.expected_return_source='external_csv'."
        )

    allocation = config.baselines.allocation
    if allocation.mode not in ALLOCATION_MODES:
        allowed = ", ".join(sorted(ALLOCATION_MODES))
        raise ValueError(f"baselines.allocation.mode must be one of: {allowed}")

    indicator_stack = config.baselines.indicator_stack
    if indicator_stack.enabled:
        if indicator_stack.ema_fast_window < 1:
            raise ValueError("baselines.indicator_stack.ema_fast_window must be at least 1.")
        if indicator_stack.ema_slow_window <= indicator_stack.ema_fast_window:
            raise ValueError(
                "baselines.indicator_stack.ema_slow_window must be greater than "
                "baselines.indicator_stack.ema_fast_window."
            )
        if indicator_stack.rsi_window < 1:
            raise ValueError("baselines.indicator_stack.rsi_window must be at least 1.")
        if not 0.0 <= indicator_stack.rsi_min <= indicator_stack.rsi_max <= 100.0:
            raise ValueError(
                "baselines.indicator_stack.rsi_min and rsi_max must satisfy "
                "0 <= rsi_min <= rsi_max <= 100."
            )
        if indicator_stack.macd_fast_window < 1:
            raise ValueError("baselines.indicator_stack.macd_fast_window must be at least 1.")
        if indicator_stack.macd_slow_window <= indicator_stack.macd_fast_window:
            raise ValueError(
                "baselines.indicator_stack.macd_slow_window must be greater than "
                "baselines.indicator_stack.macd_fast_window."
            )
        if indicator_stack.macd_signal_window < 1:
            raise ValueError("baselines.indicator_stack.macd_signal_window must be at least 1.")
        if indicator_stack.bollinger_window < 2:
            raise ValueError("baselines.indicator_stack.bollinger_window must be at least 2.")
        _validate_positive_float(
            "baselines.indicator_stack.bollinger_std",
            indicator_stack.bollinger_std,
        )
        if indicator_stack.bollinger_mode not in {"breakout", "mean_reversion"}:
            raise ValueError(
                "baselines.indicator_stack.bollinger_mode must be one of: breakout, mean_reversion."
            )
        if indicator_stack.volume_window < 1:
            raise ValueError("baselines.indicator_stack.volume_window must be at least 1.")
        _validate_positive_float(
            "baselines.indicator_stack.volume_multiplier",
            indicator_stack.volume_multiplier,
        )
        if indicator_stack.vwap_window < 1:
            raise ValueError("baselines.indicator_stack.vwap_window must be at least 1.")
        if indicator_stack.min_confirmations < 1 or indicator_stack.min_confirmations > 6:
            raise ValueError(
                "baselines.indicator_stack.min_confirmations must be between 1 and 6."
            )

    chart_patterns = config.baselines.chart_patterns
    if chart_patterns.enabled:
        if chart_patterns.lookback_bars < 8:
            raise ValueError("baselines.chart_patterns.lookback_bars must be at least 8.")
        _validate_positive_float(
            "baselines.chart_patterns.level_tolerance_pct",
            chart_patterns.level_tolerance_pct,
        )
        _validate_positive_float(
            "baselines.chart_patterns.breakout_pct",
            chart_patterns.breakout_pct,
        )
        _validate_positive_float(
            "baselines.chart_patterns.rectangle_max_range_pct",
            chart_patterns.rectangle_max_range_pct,
        )
        if chart_patterns.flag_pole_bars < 2:
            raise ValueError("baselines.chart_patterns.flag_pole_bars must be at least 2.")
        if chart_patterns.flag_consolidation_bars < 2:
            raise ValueError(
                "baselines.chart_patterns.flag_consolidation_bars must be at least 2."
            )
        _validate_positive_float(
            "baselines.chart_patterns.flag_min_pole_return",
            chart_patterns.flag_min_pole_return,
        )
        _validate_positive_float(
            "baselines.chart_patterns.flag_max_retrace_pct",
            chart_patterns.flag_max_retrace_pct,
        )
        if chart_patterns.volume_window < 1:
            raise ValueError("baselines.chart_patterns.volume_window must be at least 1.")
        _validate_positive_float(
            "baselines.chart_patterns.volume_multiplier",
            chart_patterns.volume_multiplier,
        )
        if chart_patterns.min_bullish_patterns < 1 or chart_patterns.min_bullish_patterns > 4:
            raise ValueError(
                "baselines.chart_patterns.min_bullish_patterns must be between 1 and 4."
            )

    pattern_exit_overlay = config.baselines.pattern_exit_overlay
    pattern_meta_label = config.baselines.pattern_meta_label
    pattern_partial_overlay = config.baselines.pattern_partial_exposure_overlay
    if pattern_exit_overlay.enabled or pattern_meta_label.enabled or pattern_partial_overlay.enabled:
        if not chart_patterns.enabled:
            raise ValueError(
                "baselines.chart_patterns.enabled must be true when pattern exit overlays are enabled."
            )
    if pattern_exit_overlay.enabled:
        if pattern_exit_overlay.min_bearish_patterns < 1:
            raise ValueError(
                "baselines.pattern_exit_overlay.min_bearish_patterns must be at least 1."
            )
        if pattern_exit_overlay.min_bullish_reentry_patterns < 1:
            raise ValueError(
                "baselines.pattern_exit_overlay.min_bullish_reentry_patterns must be at least 1."
            )
        if pattern_exit_overlay.trend_ema_window < 2:
            raise ValueError(
                "baselines.pattern_exit_overlay.trend_ema_window must be at least 2."
            )
        if pattern_exit_overlay.reentry_clear_bars < 1:
            raise ValueError(
                "baselines.pattern_exit_overlay.reentry_clear_bars must be at least 1."
            )
        if pattern_exit_overlay.bearish_confirmation_window_bars < 1:
            raise ValueError(
                "baselines.pattern_exit_overlay.bearish_confirmation_window_bars must be at least 1."
            )
        if pattern_exit_overlay.min_cash_bars < 0:
            raise ValueError(
                "baselines.pattern_exit_overlay.min_cash_bars must be non-negative."
            )
        if pattern_exit_overlay.exit_cooldown_bars < 0:
            raise ValueError(
                "baselines.pattern_exit_overlay.exit_cooldown_bars must be non-negative."
            )
    if pattern_meta_label.enabled:
        if not pattern_exit_overlay.enabled:
            raise ValueError(
                "baselines.pattern_exit_overlay.enabled must be true when pattern_meta_label is enabled."
            )
        if pattern_meta_label.label_horizon_bars < 1:
            raise ValueError(
                "baselines.pattern_meta_label.label_horizon_bars must be at least 1."
            )
        threshold = pattern_meta_label.exit_probability_threshold
        if not math.isfinite(threshold) or threshold < 0.0 or threshold > 1.0:
            raise ValueError(
                "baselines.pattern_meta_label.exit_probability_threshold must be between 0.0 and 1.0."
            )
        if pattern_meta_label.tuning_mode not in {"fixed", "nested_walk_forward"}:
            raise ValueError(
                "baselines.pattern_meta_label.tuning_mode must be one of: fixed, nested_walk_forward."
            )
        if (
            pattern_meta_label.tuning_objective
            != "net_return_and_drawdown_vs_buy_hold"
        ):
            raise ValueError(
                "baselines.pattern_meta_label.tuning_objective must be net_return_and_drawdown_vs_buy_hold."
            )
        if pattern_meta_label.min_oos_exit_count < 0:
            raise ValueError(
                "baselines.pattern_meta_label.min_oos_exit_count must be non-negative."
            )
        max_exposure = pattern_meta_label.max_average_exposure_for_active
        if not math.isfinite(max_exposure) or max_exposure < 0.0 or max_exposure > 1.0:
            raise ValueError(
                "baselines.pattern_meta_label.max_average_exposure_for_active must be between 0.0 and 1.0."
            )
        for grid_threshold in pattern_meta_label.exit_probability_threshold_grid:
            if (
                not math.isfinite(grid_threshold)
                or grid_threshold < 0.0
                or grid_threshold > 1.0
            ):
                raise ValueError(
                    "baselines.pattern_meta_label.exit_probability_threshold_grid must contain values between 0.0 and 1.0."
                )
        if not pattern_meta_label.models:
            raise ValueError("baselines.pattern_meta_label.models must not be empty.")
        from marketlab.models.registry import supported_model_names

        supported_models = set(supported_model_names())
        unknown_models = sorted(set(pattern_meta_label.models) - supported_models)
        if unknown_models:
            joined = ", ".join(unknown_models)
            allowed = ", ".join(sorted(supported_models))
            raise ValueError(
                "baselines.pattern_meta_label.models contains unsupported models: "
                f"{joined}. Supported models: {allowed}"
            )
    if pattern_partial_overlay.enabled:
        if not pattern_meta_label.enabled:
            raise ValueError(
                "baselines.pattern_meta_label.enabled must be true when pattern_partial_exposure_overlay is enabled."
            )
        partial_weight = pattern_partial_overlay.partial_weight
        if not math.isfinite(partial_weight) or partial_weight <= 0.0 or partial_weight >= 1.0:
            raise ValueError(
                "baselines.pattern_partial_exposure_overlay.partial_weight must be between 0.0 and 1.0."
            )
        for grid_name, grid_values in {
            "partial_exit_probability_threshold_grid": pattern_partial_overlay.partial_exit_probability_threshold_grid,
            "full_exit_probability_threshold_grid": pattern_partial_overlay.full_exit_probability_threshold_grid,
        }.items():
            for grid_threshold in grid_values:
                if (
                    not math.isfinite(grid_threshold)
                    or grid_threshold < 0.0
                    or grid_threshold > 1.0
                ):
                    raise ValueError(
                        f"baselines.pattern_partial_exposure_overlay.{grid_name} must contain values between 0.0 and 1.0."
                    )

    for section_name, partial_benchmarks in {
        "partial_allocation_benchmarks": config.baselines.partial_allocation_benchmarks,
        "rebalanced_partial_allocation_benchmarks": config.baselines.rebalanced_partial_allocation_benchmarks,
    }.items():
        seen_partial_weights: set[float] = set()
        for weight in partial_benchmarks.weights:
            numeric_weight = float(weight)
            if (
                not math.isfinite(numeric_weight)
                or numeric_weight <= 0.0
                or numeric_weight >= 1.0
            ):
                raise ValueError(
                    f"baselines.{section_name}.weights must contain values greater than 0.0 and less than 1.0."
                )
            if numeric_weight in seen_partial_weights:
                raise ValueError(
                    f"baselines.{section_name}.weights must not contain duplicate values."
                )
            seen_partial_weights.add(numeric_weight)
        if partial_benchmarks.enabled:
            if len(symbols) != 1:
                raise ValueError(
                    f"baselines.{section_name} requires exactly one data symbol."
                )
            if not partial_benchmarks.weights:
                raise ValueError(
                    f"baselines.{section_name}.weights must contain at least one value when enabled."
                )

    if not allocation.enabled:
        return

    if allocation.mode == "equal":
        return

    if allocation.mode == "symbol_weights":
        if set(allocation.symbol_weights) != symbol_set:
            raise ValueError(
                "baselines.allocation.symbol_weights must match data.symbols exactly."
            )
        _validate_weights(
            "baselines.allocation.symbol_weights",
            allocation.symbol_weights,
        )
        return

    missing_group_symbols = sorted(symbol_set - group_symbol_keys)
    if missing_group_symbols:
        joined = ", ".join(missing_group_symbols)
        raise ValueError(
            "baselines.allocation.group_weights requires symbol_groups for all "
            f"data.symbols: {joined}"
        )

    configured_groups = {config.data.symbol_groups[symbol] for symbol in symbols}
    if set(allocation.group_weights) != configured_groups:
        raise ValueError(
            "baselines.allocation.group_weights must match configured symbol "
            "groups exactly."
        )
    _validate_weights(
        "baselines.allocation.group_weights",
        allocation.group_weights,
    )


def load_config(path: str | Path) -> ExperimentConfig:
    config_path = Path(path).resolve()
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    baselines_payload = payload.get("baselines") or {}
    optimized_payload = baselines_payload.get("optimized") or {}
    evaluation_payload = payload.get("evaluation") or {}
    paper_payload = payload.get("paper") or {}
    paper_notifications_payload = paper_payload.get("notifications") or {}
    paper_azure_payload = paper_payload.get("azure") or {}
    paper_defaults = PaperConfig()

    config = ExperimentConfig(
        experiment_name=payload.get("experiment_name", "weekly_rank_v1"),
        data=_section(DataConfig, payload.get("data")),
        features=_section(FeaturesConfig, payload.get("features")),
        target=_section(TargetConfig, payload.get("target")),
        portfolio=PortfolioConfig(
            ranking=_section(
                RankingConfig,
                (payload.get("portfolio") or {}).get("ranking"),
            ),
            risk=_section(
                RiskConfig,
                (payload.get("portfolio") or {}).get("risk"),
            ),
            costs=_section(
                CostsConfig,
                (payload.get("portfolio") or {}).get("costs"),
            ),
        ),
        baselines=BaselinesConfig(
            buy_hold=baselines_payload.get("buy_hold", True),
            sma=_section(SMAConfig, baselines_payload.get("sma")),
            indicator_stack=_section(
                IndicatorStackConfig,
                baselines_payload.get("indicator_stack"),
            ),
            chart_patterns=_section(
                ChartPatternsConfig,
                baselines_payload.get("chart_patterns"),
            ),
            pattern_exit_overlay=_section(
                PatternExitOverlayConfig,
                baselines_payload.get("pattern_exit_overlay"),
            ),
            pattern_meta_label=_section(
                PatternMetaLabelConfig,
                baselines_payload.get("pattern_meta_label"),
            ),
            pattern_partial_exposure_overlay=_section(
                PatternPartialExposureOverlayConfig,
                baselines_payload.get("pattern_partial_exposure_overlay"),
            ),
            partial_allocation_benchmarks=_section(
                PartialAllocationBenchmarksConfig,
                baselines_payload.get("partial_allocation_benchmarks"),
            ),
            rebalanced_partial_allocation_benchmarks=_section(
                PartialAllocationBenchmarksConfig,
                baselines_payload.get("rebalanced_partial_allocation_benchmarks"),
            ),
            allocation=_section(
                AllocationConfig,
                baselines_payload.get("allocation"),
            ),
            optimized=_section(OptimizedConfig, optimized_payload),
        ),
        models=[
            _section(ModelSpec, item)
            for item in payload.get("models", [{"name": "logistic_regression"}])
        ],
        evaluation=EvaluationConfig(
            walk_forward=_section(
                WalkForwardConfig,
                evaluation_payload.get("walk_forward"),
            ),
            ml_strategy_threshold_sweep=_section(
                MLStrategyThresholdSweepConfig,
                evaluation_payload.get("ml_strategy_threshold_sweep"),
            ),
            ml_strategy_tuning=_section(
                MLStrategyTuningConfig,
                evaluation_payload.get("ml_strategy_tuning"),
            ),
            strict_research_gate=_section(
                StrictResearchGateConfig,
                evaluation_payload.get("strict_research_gate"),
            ),
            benchmark_strategy=evaluation_payload.get("benchmark_strategy", ""),
            cost_sensitivity_bps=evaluation_payload.get("cost_sensitivity_bps", []),
            factor_model_path=evaluation_payload.get("factor_model_path", ""),
            periods_per_year=evaluation_payload.get(
                "periods_per_year",
                default_periods_per_year((payload.get("data") or {}).get("interval", "1d")),
            ),
            focus_start=evaluation_payload.get("focus_start", ""),
            focus_end=evaluation_payload.get("focus_end", ""),
            visualize_signals=evaluation_payload.get("visualize_signals", False),
        ),
        artifacts=_section(ArtifactsConfig, payload.get("artifacts")),
        shadow=(
            _section(ShadowConfig, payload.get("shadow"))
            if payload.get("shadow") is not None
            else None
        ),
        paper=PaperConfig(
            enabled=paper_payload.get("enabled", paper_defaults.enabled),
            data_provider=paper_payload.get("data_provider", paper_defaults.data_provider),
            broker=paper_payload.get("broker", paper_defaults.broker),
            persistence_backend=paper_payload.get(
                "persistence_backend",
                paper_defaults.persistence_backend,
            ),
            sqlite_db_path=paper_payload.get(
                "sqlite_db_path",
                paper_defaults.sqlite_db_path,
            ),
            execution_mode=paper_payload.get("execution_mode", paper_defaults.execution_mode),
            agent_backend=paper_payload.get("agent_backend", paper_defaults.agent_backend),
            agent_model=paper_payload.get("agent_model", paper_defaults.agent_model),
            agent_timeout_seconds=paper_payload.get(
                "agent_timeout_seconds",
                paper_defaults.agent_timeout_seconds,
            ),
            agent_fallback_backend=paper_payload.get(
                "agent_fallback_backend",
                paper_defaults.agent_fallback_backend,
            ),
            consensus_min_long_votes=paper_payload.get(
                "consensus_min_long_votes",
                paper_defaults.consensus_min_long_votes,
            ),
            schedule_timezone=paper_payload.get(
                "schedule_timezone",
                paper_defaults.schedule_timezone,
            ),
            decision_time=paper_payload.get("decision_time", paper_defaults.decision_time),
            submission_time=paper_payload.get("submission_time", paper_defaults.submission_time),
            order_type=paper_payload.get("order_type", paper_defaults.order_type),
            position_sizing=paper_payload.get(
                "position_sizing",
                paper_defaults.position_sizing,
            ),
            approval_inbox_dir=paper_payload.get(
                "approval_inbox_dir",
                paper_defaults.approval_inbox_dir,
            ),
            state_dir=paper_payload.get("state_dir", paper_defaults.state_dir),
            poll_interval_seconds=paper_payload.get(
                "poll_interval_seconds",
                paper_defaults.poll_interval_seconds,
            ),
            azure=_section(PaperAzureConfig, paper_azure_payload),
            notifications=PaperNotificationsConfig(
                telegram=_section(
                    TelegramNotificationsConfig,
                    paper_notifications_payload.get("telegram"),
                )
            ),
        ),
        base_dir=_config_base_dir(config_path),
    )
    _normalize_mapping_sections(config)
    config.baselines.optimized.views = [
        _section(BlackLittermanViewConfig, view)
        for view in optimized_payload.get("views") or []
    ]
    _validate_config(config)
    return config
