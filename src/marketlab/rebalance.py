from __future__ import annotations

import pandas as pd


def rebalance_signal_dates(
    panel: pd.DataFrame,
    frequency: str = "W-FRI",
) -> list[pd.Timestamp]:
    calendar = pd.DataFrame({"timestamp": sorted(panel["timestamp"].drop_duplicates())})
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
    rows: list[tuple[pd.Timestamp, pd.Timestamp]] = []

    for signal_date in signal_dates:
        next_dates = unique_dates[unique_dates > signal_date]
        if next_dates.empty:
            continue
        rows.append((signal_date, next_dates.min()))

    if not rows:
        return pd.Series(dtype="datetime64[ns]", name="effective_date")

    return pd.Series(
        data=[effective_date for _, effective_date in rows],
        index=pd.Index([signal_date for signal_date, _ in rows], name="signal_date"),
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
