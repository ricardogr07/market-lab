from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

ALLOCATION_MODES = {"equal", "group_weights", "symbol_weights"}
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
class CostsConfig:
    bps_per_trade: float = 10.0


@dataclass(slots=True)
class PortfolioConfig:
    ranking: RankingConfig = field(default_factory=RankingConfig)
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
class BaselinesConfig:
    buy_hold: bool = True
    sma: SMAConfig = field(default_factory=SMAConfig)
    allocation: AllocationConfig = field(default_factory=AllocationConfig)


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


def _validate_weights(label: str, weights: dict[str, float]) -> None:
    if any(value < 0.0 for value in weights.values()):
        raise ValueError(f"{label} must contain non-negative weights.")

    if abs(sum(weights.values()) - 1.0) > WEIGHT_TOLERANCE:
        raise ValueError(f"{label} must sum to 1.0.")


def _validate_config(config: ExperimentConfig) -> None:
    symbols = list(config.data.symbols)
    symbol_set = set(symbols)
    group_symbol_keys = set(config.data.symbol_groups)

    unknown_group_symbols = sorted(group_symbol_keys - symbol_set)
    if unknown_group_symbols:
        joined = ", ".join(unknown_group_symbols)
        raise ValueError(f"data.symbol_groups contains unknown symbols: {joined}")

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
        ),
        models=[
            _section(ModelSpec, item)
            for item in payload.get("models", [{"name": "logistic_regression"}])
        ],
        evaluation=EvaluationConfig(
            walk_forward=_section(
                WalkForwardConfig,
                (payload.get("evaluation") or {}).get("walk_forward"),
            )
        ),
        artifacts=_section(ArtifactsConfig, payload.get("artifacts")),
        base_dir=_config_base_dir(config_path),
    )
    _normalize_mapping_sections(config)
    _validate_config(config)
    return config

