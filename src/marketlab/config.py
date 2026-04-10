from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

ALLOCATION_MODES = {"equal", "group_weights", "symbol_weights"}
OPTIMIZED_METHODS = {"black_litterman", "mean_variance", "risk_parity"}
COVARIANCE_ESTIMATORS = {"diagonal_shrinkage", "ewma", "external_csv", "sample"}
EXPECTED_RETURN_SOURCES = {"external_csv", "historical_mean"}
WEIGHT_TOLERANCE = 1e-6


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
class EvaluationConfig:
    walk_forward: WalkForwardConfig = field(default_factory=WalkForwardConfig)
    benchmark_strategy: str = ""
    cost_sensitivity_bps: list[float] = field(default_factory=list)


@dataclass(slots=True)
class ArtifactsConfig:
    output_dir: str = "artifacts/runs"
    save_predictions: bool = True
    save_metrics_csv: bool = True
    save_report_md: bool = True
    save_plots: bool = True


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


def _section(cls: type[Any], data: dict[str, Any] | None) -> Any:
    values = data or {}
    allowed = {field.name for field in cls.__dataclass_fields__.values()}
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


def _validate_config(config: ExperimentConfig) -> None:
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
            buy_hold=(payload.get("baselines") or {}).get("buy_hold", True),
            sma=_section(SMAConfig, (payload.get("baselines") or {}).get("sma")),
            allocation=_section(
                AllocationConfig,
                (payload.get("baselines") or {}).get("allocation"),
            ),
            optimized=_section(
                OptimizedConfig,
                (payload.get("baselines") or {}).get("optimized"),
            ),
        ),
        models=[
            _section(ModelSpec, item)
            for item in payload.get("models", [{"name": "logistic_regression"}])
        ],
        evaluation=EvaluationConfig(
            walk_forward=_section(
                WalkForwardConfig,
                (payload.get("evaluation") or {}).get("walk_forward"),
            ),
            benchmark_strategy=(payload.get("evaluation") or {}).get("benchmark_strategy", ""),
            cost_sensitivity_bps=(payload.get("evaluation") or {}).get("cost_sensitivity_bps", []),
        ),
        artifacts=_section(ArtifactsConfig, payload.get("artifacts")),
        base_dir=_config_base_dir(config_path),
    )
    _normalize_mapping_sections(config)
    config.baselines.optimized.views = [
        _section(BlackLittermanViewConfig, view)
        for view in config.baselines.optimized.views
    ]
    _validate_config(config)
    return config
