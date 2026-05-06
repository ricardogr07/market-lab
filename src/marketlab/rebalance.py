from __future__ import annotations

import pandas as pd


def rebalance_signal_dates(
    panel: pd.DataFrame,
    frequency: str = "W-FRI",
) -> list[pd.Timestamp]:
    calendar = pd.DataFrame({"timestamp": sorted(panel["timestamp"].drop_duplicates())})
    if frequency.lower() in {"bar", "1h"}:
        return calendar["timestamp"].sort_values().tolist()
    calendar["rebalance_period"] = calendar["timestamp"].dt.to_period(frequency)
    return calendar.groupby("rebalance_period")["timestamp"].max().sort_values().tolist()


def weekly_signal_dates(
    panel: pd.DataFrame,
    frequency: str = "W-FRI",
) -> list[pd.Timestamp]:
    return rebalance_signal_dates(panel, frequency)


def next_effective_dates(
    panel: pd.DataFrame,
    signal_dates: list[pd.Timestamp],
) -> pd.Series:
    unique_dates = pd.Index(sorted(panel["timestamp"].drop_duplicates()))
    if unique_dates.empty or not signal_dates:
        return pd.Series(dtype="datetime64[ns]", name="effective_date")

    signal_index = pd.Index(pd.to_datetime(signal_dates), name="signal_date")
    positions = unique_dates.searchsorted(signal_index, side="right")
    valid = positions < len(unique_dates)
    if not bool(valid.any()):
        return pd.Series(dtype="datetime64[ns]", name="effective_date")

    return pd.Series(
        data=unique_dates.take(positions[valid]),
        index=signal_index[valid],
        name="effective_date",
    )


def signal_effective_dates(
    panel: pd.DataFrame,
    frequency: str = "W-FRI",
) -> pd.Series:
    return next_effective_dates(panel, rebalance_signal_dates(panel, frequency))


def next_rebalance_effective_date(
    panel: pd.DataFrame,
    signal_date: pd.Timestamp,
    frequency: str = "W-FRI",
) -> pd.Timestamp | None:
    effective_dates = signal_effective_dates(panel, frequency)
    future_signal_dates = effective_dates.index[effective_dates.index > pd.Timestamp(signal_date)]
    if future_signal_dates.empty:
        return None
    return pd.Timestamp(effective_dates.loc[future_signal_dates.min()])
