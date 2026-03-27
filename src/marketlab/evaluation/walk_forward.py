from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from marketlab.config import WalkForwardConfig

REQUIRED_COLUMNS = {"signal_date", "target_end_date"}


@dataclass(slots=True, frozen=True)
class WalkForwardFold:
    fold_id: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    label_cutoff: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    train_rows: int
    test_rows: int


def _prepare_modeling_dataset(modeling_dataset: pd.DataFrame) -> pd.DataFrame:
    missing = REQUIRED_COLUMNS - set(modeling_dataset.columns)
    if missing:
        joined = ", ".join(sorted(missing))
        raise ValueError(f"Modeling dataset is missing required columns: {joined}")

    working = modeling_dataset.copy()
    working["signal_date"] = pd.to_datetime(working["signal_date"])
    working["target_end_date"] = pd.to_datetime(working["target_end_date"])

    sort_columns = ["signal_date"]
    if "symbol" in working.columns:
        sort_columns.append("symbol")

    return working.sort_values(sort_columns).reset_index(drop=True)


def _first_date_on_or_after(
    available_dates: pd.Index,
    anchor: pd.Timestamp,
) -> pd.Timestamp | None:
    candidates = available_dates[available_dates >= anchor]
    if candidates.empty:
        return None
    return pd.Timestamp(candidates.min())


def _build_masks(
    modeling_dataset: pd.DataFrame,
    fold: WalkForwardFold,
) -> tuple[pd.Series, pd.Series]:
    train_mask = (
        modeling_dataset["signal_date"].ge(fold.train_start)
        & modeling_dataset["signal_date"].le(fold.train_end)
        & modeling_dataset["signal_date"].lt(fold.test_start)
        & modeling_dataset["target_end_date"].le(fold.label_cutoff)
    )
    test_mask = modeling_dataset["signal_date"].ge(fold.test_start) & modeling_dataset[
        "signal_date"
    ].le(fold.test_end)
    return train_mask, test_mask


def _has_full_test_window(
    test_rows: pd.DataFrame,
    exclusive_test_cutoff: pd.Timestamp,
    frequency: str,
) -> bool:
    if test_rows.empty:
        return False

    expected_last_period = exclusive_test_cutoff.to_period(frequency) - 1
    test_periods = set(test_rows["signal_date"].dt.to_period(frequency))
    return expected_last_period in test_periods


def build_walk_forward_folds(
    modeling_dataset: pd.DataFrame,
    walk_forward: WalkForwardConfig,
    frequency: str = "W-FRI",
) -> list[WalkForwardFold]:
    working = _prepare_modeling_dataset(modeling_dataset)
    available_dates = pd.Index(sorted(working["signal_date"].drop_duplicates()))
    if available_dates.empty:
        return []

    first_anchor = pd.Timestamp(available_dates.min()) + pd.DateOffset(
        years=walk_forward.train_years
    )
    test_start = _first_date_on_or_after(available_dates, first_anchor)
    if test_start is None:
        return []

    folds: list[WalkForwardFold] = []
    fold_id = 1

    while test_start is not None:
        train_start_anchor = test_start - pd.DateOffset(years=walk_forward.train_years)
        exclusive_test_cutoff = test_start + pd.DateOffset(months=walk_forward.test_months)

        train_rows = working.loc[
            working["signal_date"].ge(train_start_anchor)
            & working["signal_date"].lt(test_start)
            & working["target_end_date"].le(test_start)
        ].copy()
        test_rows = working.loc[
            working["signal_date"].ge(test_start)
            & working["signal_date"].lt(exclusive_test_cutoff)
        ].copy()

        if (
            not train_rows.empty
            and not test_rows.empty
            and _has_full_test_window(test_rows, exclusive_test_cutoff, frequency)
        ):
            folds.append(
                WalkForwardFold(
                    fold_id=fold_id,
                    train_start=pd.Timestamp(train_rows["signal_date"].min()),
                    train_end=pd.Timestamp(train_rows["signal_date"].max()),
                    label_cutoff=pd.Timestamp(test_start),
                    test_start=pd.Timestamp(test_rows["signal_date"].min()),
                    test_end=pd.Timestamp(test_rows["signal_date"].max()),
                    train_rows=len(train_rows),
                    test_rows=len(test_rows),
                )
            )
            fold_id += 1

        next_anchor = test_start + pd.DateOffset(months=walk_forward.step_months)
        next_test_start = _first_date_on_or_after(available_dates, next_anchor)
        if next_test_start is None or next_test_start <= test_start:
            break
        test_start = next_test_start

    return folds


def slice_fold_rows(
    modeling_dataset: pd.DataFrame,
    fold: WalkForwardFold,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    working = _prepare_modeling_dataset(modeling_dataset)
    train_mask, test_mask = _build_masks(working, fold)

    sort_columns = ["signal_date"]
    if "symbol" in working.columns:
        sort_columns.append("symbol")

    train_rows = working.loc[train_mask].sort_values(sort_columns).reset_index(drop=True)
    test_rows = working.loc[test_mask].sort_values(sort_columns).reset_index(drop=True)
    return train_rows, test_rows
