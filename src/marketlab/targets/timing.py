from __future__ import annotations

from dataclasses import asdict

import pandas as pd

from marketlab.config import ExperimentConfig
from marketlab.data.panel import PANEL_COLUMNS
from marketlab.features.engineering import add_feature_set
from marketlab.rebalance import next_effective_dates, rebalance_signal_dates
from marketlab.strategies.indicator_stack import build_indicator_frame

ALLOCATION_UTILITY_TIERS = (0.0, 0.25, 0.50, 1.0)
ALLOCATION_UTILITY_LABELS = {
    0.0: 0,
    0.25: 1,
    0.50: 2,
    1.0: 3,
}
REGIME_STATE_LABELS = {
    0.0: 0,
    0.25: 1,
    0.50: 1,
    1.0: 2,
}


def apply_allocation_utility_profile(
    rows: pd.DataFrame,
    *,
    target_type: str = "allocation_utility",
    cost_bps: float = 0.0,
    drawdown_penalty: float = 0.50,
    volatility_penalty: float = 0.25,
    risk_penalty_power: float = 2.0,
) -> pd.DataFrame:
    required_columns = {
        "forward_return",
        "forward_drawdown",
        "forward_realized_volatility",
    }
    missing_columns = required_columns - set(rows.columns)
    if missing_columns:
        joined = ", ".join(sorted(missing_columns))
        raise ValueError(f"Allocation utility rows are missing required columns: {joined}")
    if target_type not in {"allocation_utility", "regime_state"}:
        raise ValueError(f"Unsupported allocation utility target_type: {target_type}")

    working = rows.copy()
    entry_cost = float(cost_bps) / 10_000.0
    utility_rows: dict[float, list[float]] = {tier: [] for tier in ALLOCATION_UTILITY_TIERS}
    target_weights: list[float] = []
    target_labels: list[int] = []
    label_map = (
        REGIME_STATE_LABELS if target_type == "regime_state" else ALLOCATION_UTILITY_LABELS
    )

    for _, row in working.iterrows():
        forward_return = float(row["forward_return"])
        path_drawdown = abs(float(row["forward_drawdown"]))
        path_volatility = float(row["forward_realized_volatility"])
        utilities = {
            tier: (
                tier * forward_return
                - drawdown_penalty * (tier**risk_penalty_power) * path_drawdown
                - volatility_penalty * (tier**risk_penalty_power) * path_volatility
                - entry_cost * tier
            )
            for tier in ALLOCATION_UTILITY_TIERS
        }
        selected_tier = max(
            ALLOCATION_UTILITY_TIERS,
            key=lambda tier: (utilities[tier], -tier),
        )
        target_weights.append(selected_tier)
        target_labels.append(label_map[selected_tier])
        for tier, utility in utilities.items():
            utility_rows[tier].append(float(utility))

    working["target_weight"] = target_weights
    working["target"] = target_labels
    for tier, values in utility_rows.items():
        suffix = str(int(tier * 100))
        working[f"allocation_utility_{suffix}"] = values
    return working


def _add_indicator_stack_ml_features(
    featured_panel: pd.DataFrame,
    config: ExperimentConfig,
) -> pd.DataFrame:
    indicator = config.baselines.indicator_stack
    indicator_frame = build_indicator_frame(
        featured_panel,
        ema_fast_window=indicator.ema_fast_window,
        ema_slow_window=indicator.ema_slow_window,
        rsi_window=indicator.rsi_window,
        rsi_min=indicator.rsi_min,
        rsi_max=indicator.rsi_max,
        macd_fast_window=indicator.macd_fast_window,
        macd_slow_window=indicator.macd_slow_window,
        macd_signal_window=indicator.macd_signal_window,
        bollinger_window=indicator.bollinger_window,
        bollinger_std=indicator.bollinger_std,
        bollinger_mode=indicator.bollinger_mode,
        volume_window=indicator.volume_window,
        volume_multiplier=indicator.volume_multiplier,
        vwap_window=indicator.vwap_window,
        use_vwap=indicator.use_vwap,
    )

    close = indicator_frame["adj_close"]
    bollinger_width = (
        indicator_frame["bollinger_upper"] - indicator_frame["bollinger_mid"]
    ).replace(0.0, pd.NA)
    bollinger_side = pd.Series(0, index=indicator_frame.index, dtype=int)
    bollinger_side = bollinger_side.mask(close.gt(indicator_frame["bollinger_upper"]), 1)
    bollinger_side = bollinger_side.mask(close.lt(indicator_frame["bollinger_lower"]), -1)

    features = pd.DataFrame(
        {
            "symbol": indicator_frame["symbol"],
            "timestamp": indicator_frame["timestamp"],
            "indicator_ema_spread": (
                indicator_frame["ema_fast"] / indicator_frame["ema_slow"].replace(0.0, pd.NA)
            )
            - 1.0,
            "indicator_rsi": indicator_frame["rsi"],
            "indicator_macd_hist": indicator_frame["macd"] - indicator_frame["macd_signal"],
            "indicator_bollinger_z": (close - indicator_frame["bollinger_mid"])
            / bollinger_width,
            "indicator_bollinger_side": bollinger_side,
            "indicator_volume_ratio": indicator_frame["volume"]
            / indicator_frame["volume_average"].replace(0.0, pd.NA),
            "indicator_volume_z": (
                indicator_frame["volume"] - indicator_frame["volume_average"]
            )
            / indicator_frame["volume_std"].replace(0.0, pd.NA),
            "indicator_ema_confirmed": indicator_frame["ema_confirmed"].fillna(False).astype(int),
            "indicator_rsi_confirmed": indicator_frame["rsi_confirmed"].fillna(False).astype(int),
            "indicator_macd_confirmed": indicator_frame["macd_confirmed"].fillna(False).astype(int),
            "indicator_bollinger_confirmed": indicator_frame["bollinger_confirmed"]
            .fillna(False)
            .astype(int),
            "indicator_volume_confirmed": indicator_frame["volume_confirmed"]
            .fillna(False)
            .astype(int),
            "indicator_confirmation_count": indicator_frame["confirmation_count"],
        }
    )
    if indicator.use_vwap:
        features["indicator_vwap_spread"] = (
            close / indicator_frame["vwap"].replace(0.0, pd.NA)
        ) - 1.0
        features["indicator_vwap_confirmed"] = (
            indicator_frame["vwap_confirmed"].fillna(False).astype(int)
        )

    return featured_panel.merge(features, on=["symbol", "timestamp"], how="left")


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
    *,
    cost_bps: float = 0.0,
    allocation_utility_drawdown_penalty: float = 0.50,
    allocation_utility_volatility_penalty: float = 0.25,
    allocation_utility_risk_penalty_power: float = 2.0,
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
    elif target_type in {"allocation_utility", "regime_state"}:
        price_paths = {
            symbol: frame.set_index("timestamp").sort_index()
            for symbol, frame in prices.groupby("symbol", sort=True)
        }

        drawdowns: list[float] = []
        realized_volatility: list[float] = []

        for index, row in working.iterrows():
            symbol = str(row["symbol"])
            start_position = int(effective_positions.loc[index])
            end_position = int(horizon_positions.loc[index])
            path_dates = unique_dates[start_position : end_position + 1]
            symbol_prices = price_paths[symbol].reindex(path_dates)
            closes = symbol_prices["adj_close"].dropna().astype(float)
            entry_price = float(row["entry_adj_open"])

            if closes.empty or entry_price == 0.0:
                path_drawdown = 0.0
                path_volatility = 0.0
            else:
                path_returns = (closes / entry_price) - 1.0
                path_drawdown = min(0.0, float(path_returns.min()))
                period_prices = pd.concat(
                    [pd.Series([entry_price]), closes.reset_index(drop=True)],
                    ignore_index=True,
                )
                path_volatility = float(period_prices.pct_change().dropna().std(ddof=0))
                if pd.isna(path_volatility):
                    path_volatility = 0.0

            drawdowns.append(path_drawdown)
            realized_volatility.append(path_volatility)

        working["forward_drawdown"] = drawdowns
        working["forward_realized_volatility"] = realized_volatility
        working = apply_allocation_utility_profile(
            working,
            target_type=target_type,
            cost_bps=cost_bps,
            drawdown_penalty=allocation_utility_drawdown_penalty,
            volatility_penalty=allocation_utility_volatility_penalty,
            risk_penalty_power=allocation_utility_risk_penalty_power,
        )
    else:
        raise ValueError(f"Unsupported target_type: {target_type}")

    working = working.drop(columns=["entry_adj_open", "exit_adj_close"])
    return working.sort_values(["signal_date", "symbol"]).reset_index(drop=True)


def build_modeling_dataset(
    panel: pd.DataFrame,
    config: ExperimentConfig,
) -> pd.DataFrame:
    snapshots = build_scoring_dataset(panel, config)
    dataset = add_forward_targets(
        snapshots,
        panel=panel,
        horizon_days=config.target.horizon_days,
        target_type=config.target.type,
        cost_bps=config.portfolio.costs.bps_per_trade,
        allocation_utility_drawdown_penalty=config.target.allocation_utility_drawdown_penalty,
        allocation_utility_volatility_penalty=config.target.allocation_utility_volatility_penalty,
        allocation_utility_risk_penalty_power=config.target.allocation_utility_risk_penalty_power,
    )
    feature_columns = [
        column
        for column in snapshots.columns
        if column not in {"symbol", "signal_date", "effective_date"}
    ]
    required_columns = [*feature_columns, "forward_return", "target"]
    dataset = dataset.dropna(subset=required_columns).reset_index(drop=True)
    return dataset


def build_scoring_dataset(
    panel: pd.DataFrame,
    config: ExperimentConfig,
) -> pd.DataFrame:
    feature_options = asdict(config.features)
    feature_options.pop("indicator_stack_ml_features_enabled", None)
    featured_panel = add_feature_set(
        panel=panel,
        **feature_options,
    )
    if config.features.indicator_stack_ml_features_enabled:
        featured_panel = _add_indicator_stack_ml_features(featured_panel, config)
    feature_columns = _resolve_feature_columns(featured_panel, feature_columns=None)
    snapshots = build_rebalance_snapshots(
        featured_panel,
        feature_columns=feature_columns,
        frequency=config.portfolio.ranking.rebalance_frequency,
    )
    return snapshots.dropna(subset=feature_columns).reset_index(drop=True)
