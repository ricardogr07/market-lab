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
PAPER_PERSISTENCE_BACKENDS = {"filesystem", "sqlite"}
PAPER_EXECUTION_MODES = {"autonomous", "agent_approval", "manual_approval"}
PAPER_ORDER_TYPES = {"day_market"}
PAPER_POSITION_SIZING = {"full_equity_fractional"}
PAPER_AGENT_BACKENDS = {"claude", "deterministic_consensus", "openai"}
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


@dataclass(slots=True)
class TargetConfig:
    horizon_days: int = 5
    type: str = "direction"


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
class MLStrategyTuningConfig:
    enabled: bool = False
    thresholds: list[float] = field(
        default_factory=lambda: [0.50, 0.52, 0.55, 0.58, 0.60, 0.62, 0.65]
    )
    validation_months: int = 3
    min_validation_rows: int = 200
    min_exposure_changes: int = 5
    max_average_exposure_for_active: float = 0.995
    objective: str = "net_return_and_risk_vs_buy_hold"


@dataclass(slots=True)
class EvaluationConfig:
    walk_forward: WalkForwardConfig = field(default_factory=WalkForwardConfig)
    ml_strategy_threshold_sweep: MLStrategyThresholdSweepConfig = field(
        default_factory=MLStrategyThresholdSweepConfig
    )
    ml_strategy_tuning: MLStrategyTuningConfig = field(default_factory=MLStrategyTuningConfig)
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
    notifications: PaperNotificationsConfig = field(default_factory=PaperNotificationsConfig)


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

    if config.evaluation.cost_sensitivity_bps is None:
        config.evaluation.cost_sensitivity_bps = []
    if config.evaluation.ml_strategy_threshold_sweep.thresholds is None:
        config.evaluation.ml_strategy_threshold_sweep.thresholds = []
    if config.evaluation.ml_strategy_tuning.thresholds is None:
        config.evaluation.ml_strategy_tuning.thresholds = []
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
    if ml_tuning.validation_months < 1:
        raise ValueError("evaluation.ml_strategy_tuning.validation_months must be at least 1.")
    if ml_tuning.min_validation_rows < 1:
        raise ValueError("evaluation.ml_strategy_tuning.min_validation_rows must be at least 1.")
    if ml_tuning.min_exposure_changes < 0:
        raise ValueError(
            "evaluation.ml_strategy_tuning.min_exposure_changes must be non-negative."
        )
    max_exposure = ml_tuning.max_average_exposure_for_active
    if not math.isfinite(max_exposure) or max_exposure < 0.0 or max_exposure > 1.0:
        raise ValueError(
            "evaluation.ml_strategy_tuning.max_average_exposure_for_active must be between 0.0 and 1.0."
        )
    if ml_tuning.objective != "net_return_and_risk_vs_buy_hold":
        raise ValueError(
            "evaluation.ml_strategy_tuning.objective must be net_return_and_risk_vs_buy_hold."
        )
    _validate_positive_float("evaluation.periods_per_year", config.evaluation.periods_per_year)
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
