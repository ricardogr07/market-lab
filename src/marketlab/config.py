from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


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
class BaselinesConfig:
    buy_hold: bool = True
    sma: SMAConfig = field(default_factory=SMAConfig)


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
    return config

