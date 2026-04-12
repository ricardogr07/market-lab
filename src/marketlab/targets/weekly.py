from __future__ import annotations

import pandas as pd

from marketlab.config import ExperimentConfig
from marketlab.targets import timing as timing_targets

add_forward_targets = timing_targets.add_forward_targets


def build_weekly_snapshots(
    featured_panel: pd.DataFrame,
    feature_columns: list[str] | None = None,
    frequency: str = "W-FRI",
) -> pd.DataFrame:
    return timing_targets.build_rebalance_snapshots(
        featured_panel,
        feature_columns=feature_columns,
        frequency=frequency,
    )


def build_weekly_modeling_dataset(
    panel: pd.DataFrame,
    config: ExperimentConfig,
) -> pd.DataFrame:
    return timing_targets.build_modeling_dataset(panel, config)
