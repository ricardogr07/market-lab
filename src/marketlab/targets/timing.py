from __future__ import annotations

from dataclasses import asdict

import pandas as pd

from marketlab.config import ExperimentConfig
from marketlab.data.panel import PANEL_COLUMNS
from marketlab.features.engineering import add_feature_set
from marketlab.rebalance import next_effective_dates, rebalance_signal_dates


def _resolve_feature_columns(
    featured_panel: pd.DataFrame,
    feature_columns: list[str] | None,
) -> list[str]:
    if feature_columns is None:
        excluded = set(PANEL_COLUMNS)
        return [column for column in featured_panel.columns if column not in excluded]

    missing = [column for column in feature_columns if column not in featured_panel.columns]
    if missing:
        joined = ", ".join(missing)
        raise ValueError(f"Snapshots are missing feature columns: {joined}")
    return list(feature_columns)


def build_rebalance_snapshots(
    featured_panel: pd.DataFrame,
    feature_columns: list[str] | None = None,
    frequency: str = "W-FRI",
) -> pd.DataFrame:
    required = {"symbol", "timestamp"}
    missing = required - set(featured_panel.columns)
    if missing:
        joined = ", ".join(sorted(missing))
        raise ValueError(f"Featured panel is missing required columns: {joined}")

    working = featured_panel.sort_values(["symbol", "timestamp"]).copy()
    feature_names = _resolve_feature_columns(working, feature_columns)
    effective_dates = next_effective_dates(working, rebalance_signal_dates(working, frequency))

    if effective_dates.empty:
        columns = ["symbol", "signal_date", "effective_date", *feature_names]
        return pd.DataFrame(columns=columns)

    snapshots = working.loc[
        working["timestamp"].isin(effective_dates.index),
        ["symbol", "timestamp", *feature_names],
    ].copy()
    snapshots = snapshots.rename(columns={"timestamp": "signal_date"})
    snapshots["effective_date"] = snapshots["signal_date"].map(effective_dates)
    snapshots = snapshots[["symbol", "signal_date", "effective_date", *feature_names]]
    return snapshots.sort_values(["signal_date", "symbol"]).reset_index(drop=True)


def add_forward_targets(
    snapshots: pd.DataFrame,
    panel: pd.DataFrame,
    horizon_days: int,
    target_type: str = "direction",
) -> pd.DataFrame:
    if horizon_days < 1:
        raise ValueError("horizon_days must be at least 1.")

    required_snapshot_columns = {"symbol", "signal_date", "effective_date"}
    missing_snapshot_columns = required_snapshot_columns - set(snapshots.columns)
    if missing_snapshot_columns:
        joined = ", ".join(sorted(missing_snapshot_columns))
        raise ValueError(f"Snapshots are missing required columns: {joined}")

    required_panel_columns = {"symbol", "timestamp", "adj_open", "adj_close"}
    missing_panel_columns = required_panel_columns - set(panel.columns)
    if missing_panel_columns:
        joined = ", ".join(sorted(missing_panel_columns))
        raise ValueError(f"Panel is missing required columns: {joined}")

    if snapshots.empty:
        columns = [*snapshots.columns, "target_end_date", "forward_return", "target"]
        return pd.DataFrame(columns=columns)

    prices = panel.loc[:, ["symbol", "timestamp", "adj_open", "adj_close"]].copy()
    unique_dates = pd.Index(sorted(prices["timestamp"].drop_duplicates()))
    date_positions = pd.Series(range(len(unique_dates)), index=unique_dates)

    working = snapshots.copy()
    effective_positions = working["effective_date"].map(date_positions)
    working = working.loc[effective_positions.notna()].copy()
    effective_positions = effective_positions.loc[working.index].astype(int)

    horizon_positions = effective_positions + (horizon_days - 1)
    valid_horizon = horizon_positions < len(unique_dates)
    working = working.loc[valid_horizon].copy()
    horizon_positions = horizon_positions.loc[valid_horizon].astype(int)
    working["target_end_date"] = unique_dates.take(horizon_positions.to_numpy())

    entry_prices = prices.rename(
        columns={"timestamp": "effective_date", "adj_open": "entry_adj_open"}
    )[["symbol", "effective_date", "entry_adj_open"]]
    exit_prices = prices.rename(
        columns={"timestamp": "target_end_date", "adj_close": "exit_adj_close"}
    )[["symbol", "target_end_date", "exit_adj_close"]]

    working = working.merge(entry_prices, on=["symbol", "effective_date"], how="left")
    working = working.merge(exit_prices, on=["symbol", "target_end_date"], how="left")
    working = working.dropna(subset=["entry_adj_open", "exit_adj_close"]).copy()

    working["forward_return"] = (working["exit_adj_close"] / working["entry_adj_open"]) - 1.0

    if target_type == "direction":
        working["target"] = working["forward_return"].gt(0.0).astype(int)
    elif target_type == "return":
        working["target"] = working["forward_return"]
    else:
        raise ValueError(f"Unsupported target_type: {target_type}")

    working = working.drop(columns=["entry_adj_open", "exit_adj_close"])
    return working.sort_values(["signal_date", "symbol"]).reset_index(drop=True)


def build_modeling_dataset(
    panel: pd.DataFrame,
    config: ExperimentConfig,
) -> pd.DataFrame:
    featured_panel = add_feature_set(
        panel=panel,
        **asdict(config.features),
    )
    feature_columns = _resolve_feature_columns(featured_panel, feature_columns=None)
    snapshots = build_rebalance_snapshots(
        featured_panel,
        feature_columns=feature_columns,
        frequency=config.portfolio.ranking.rebalance_frequency,
    )
    dataset = add_forward_targets(
        snapshots,
        panel=featured_panel,
        horizon_days=config.target.horizon_days,
        target_type=config.target.type,
    )
    required_columns = [*feature_columns, "forward_return", "target"]
    dataset = dataset.dropna(subset=required_columns).reset_index(drop=True)
    return dataset
