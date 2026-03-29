from __future__ import annotations

from dataclasses import asdict, dataclass

import pandas as pd
from pandas.tseries.frequencies import to_offset

from marketlab.config import WalkForwardConfig

REQUIRED_COLUMNS = {"signal_date", "target_end_date"}
DIAGNOSTIC_COLUMNS = [
    "candidate_id",
    "fold_id",
    "status",
    "skip_reasons",
    "train_start",
    "train_end",
    "label_cutoff",
    "test_start",
    "test_end",
    "train_rows",
    "test_rows",
    "train_positive_rate",
    "test_positive_rate",
]
SKIP_REASON_INCOMPLETE_TEST_WINDOW = "incomplete_test_window"
SKIP_REASON_INSUFFICIENT_TRAIN_ROWS = "insufficient_train_rows"
SKIP_REASON_INSUFFICIENT_TEST_ROWS = "insufficient_test_rows"
SKIP_REASON_INSUFFICIENT_TRAIN_POSITIVE_RATE = "insufficient_train_positive_rate"
SKIP_REASON_INSUFFICIENT_TEST_POSITIVE_RATE = "insufficient_test_positive_rate"


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


@dataclass(slots=True, frozen=True)
class _WalkForwardEvaluation:
    folds: list[WalkForwardFold]
    diagnostics: pd.DataFrame


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
        & modeling_dataset["signal_date"].lt(fold.label_cutoff)
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


def _embargo_cutoff(
    test_start: pd.Timestamp,
    embargo_periods: int,
    frequency: str,
) -> pd.Timestamp:
    cutoff = pd.Timestamp(test_start)
    for _ in range(max(0, embargo_periods)):
        cutoff = pd.Timestamp(cutoff - to_offset(frequency))
    return cutoff


def _positive_rate(rows: pd.DataFrame) -> float:
    if rows.empty or "target" not in rows.columns:
        return float("nan")
    return float(pd.to_numeric(rows["target"], errors="coerce").mean())


def _range_start(rows: pd.DataFrame) -> pd.Timestamp:
    if rows.empty:
        return pd.NaT
    return pd.Timestamp(rows["signal_date"].min())


def _range_end(rows: pd.DataFrame) -> pd.Timestamp:
    if rows.empty:
        return pd.NaT
    return pd.Timestamp(rows["signal_date"].max())


def _evaluate_walk_forward_candidates(
    modeling_dataset: pd.DataFrame,
    walk_forward: WalkForwardConfig,
    frequency: str,
) -> _WalkForwardEvaluation:
    working = _prepare_modeling_dataset(modeling_dataset)
    available_dates = pd.Index(sorted(working["signal_date"].drop_duplicates()))
    if available_dates.empty:
        return _WalkForwardEvaluation(
            folds=[],
            diagnostics=pd.DataFrame(columns=DIAGNOSTIC_COLUMNS),
        )

    first_anchor = pd.Timestamp(available_dates.min()) + pd.DateOffset(
        years=walk_forward.train_years
    )
    test_start = _first_date_on_or_after(available_dates, first_anchor)
    if test_start is None:
        return _WalkForwardEvaluation(
            folds=[],
            diagnostics=pd.DataFrame(columns=DIAGNOSTIC_COLUMNS),
        )

    folds: list[WalkForwardFold] = []
    diagnostics_rows: list[dict[str, object]] = []
    fold_id = 1
    candidate_id = 1
    min_train_rows = max(1, walk_forward.min_train_rows)
    min_test_rows = max(1, walk_forward.min_test_rows)

    while test_start is not None:
        train_start_anchor = test_start - pd.DateOffset(years=walk_forward.train_years)
        exclusive_test_cutoff = test_start + pd.DateOffset(months=walk_forward.test_months)
        label_cutoff = _embargo_cutoff(
            test_start=test_start,
            embargo_periods=walk_forward.embargo_periods,
            frequency=frequency,
        )

        train_rows = working.loc[
            working["signal_date"].ge(train_start_anchor)
            & working["signal_date"].lt(label_cutoff)
            & working["target_end_date"].le(label_cutoff)
        ].copy()
        test_rows = working.loc[
            working["signal_date"].ge(test_start)
            & working["signal_date"].lt(exclusive_test_cutoff)
        ].copy()

        train_positive_rate = _positive_rate(train_rows)
        test_positive_rate = _positive_rate(test_rows)
        skip_reasons: list[str] = []

        if not _has_full_test_window(test_rows, exclusive_test_cutoff, frequency):
            skip_reasons.append(SKIP_REASON_INCOMPLETE_TEST_WINDOW)
        if len(train_rows) < min_train_rows:
            skip_reasons.append(SKIP_REASON_INSUFFICIENT_TRAIN_ROWS)
        if len(test_rows) < min_test_rows:
            skip_reasons.append(SKIP_REASON_INSUFFICIENT_TEST_ROWS)
        if walk_forward.min_train_positive_rate > 0.0 and (
            pd.isna(train_positive_rate)
            or train_positive_rate < walk_forward.min_train_positive_rate
        ):
            skip_reasons.append(SKIP_REASON_INSUFFICIENT_TRAIN_POSITIVE_RATE)
        if walk_forward.min_test_positive_rate > 0.0 and (
            pd.isna(test_positive_rate)
            or test_positive_rate < walk_forward.min_test_positive_rate
        ):
            skip_reasons.append(SKIP_REASON_INSUFFICIENT_TEST_POSITIVE_RATE)

        assigned_fold_id: int | None = None
        if not skip_reasons:
            assigned_fold_id = fold_id
            folds.append(
                WalkForwardFold(
                    fold_id=fold_id,
                    train_start=_range_start(train_rows),
                    train_end=_range_end(train_rows),
                    label_cutoff=label_cutoff,
                    test_start=pd.Timestamp(test_start),
                    test_end=_range_end(test_rows),
                    train_rows=len(train_rows),
                    test_rows=len(test_rows),
                )
            )
            fold_id += 1

        diagnostics_rows.append(
            {
                "candidate_id": candidate_id,
                "fold_id": assigned_fold_id,
                "status": "used" if not skip_reasons else "skipped",
                "skip_reasons": ";".join(skip_reasons),
                "train_start": _range_start(train_rows),
                "train_end": _range_end(train_rows),
                "label_cutoff": label_cutoff,
                "test_start": pd.Timestamp(test_start),
                "test_end": _range_end(test_rows),
                "train_rows": len(train_rows),
                "test_rows": len(test_rows),
                "train_positive_rate": train_positive_rate,
                "test_positive_rate": test_positive_rate,
            }
        )
        candidate_id += 1

        next_anchor = test_start + pd.DateOffset(months=walk_forward.step_months)
        next_test_start = _first_date_on_or_after(available_dates, next_anchor)
        if next_test_start is None or next_test_start <= test_start:
            break
        test_start = next_test_start

    diagnostics = pd.DataFrame(diagnostics_rows)
    if diagnostics.empty:
        diagnostics = pd.DataFrame(columns=DIAGNOSTIC_COLUMNS)
    else:
        diagnostics = diagnostics.loc[:, DIAGNOSTIC_COLUMNS]

    return _WalkForwardEvaluation(folds=folds, diagnostics=diagnostics)


def build_walk_forward_folds(
    modeling_dataset: pd.DataFrame,
    walk_forward: WalkForwardConfig,
    frequency: str = "W-FRI",
) -> list[WalkForwardFold]:
    return _evaluate_walk_forward_candidates(
        modeling_dataset=modeling_dataset,
        walk_forward=walk_forward,
        frequency=frequency,
    ).folds


def build_walk_forward_diagnostics(
    modeling_dataset: pd.DataFrame,
    walk_forward: WalkForwardConfig,
    frequency: str = "W-FRI",
) -> pd.DataFrame:
    return _evaluate_walk_forward_candidates(
        modeling_dataset=modeling_dataset,
        walk_forward=walk_forward,
        frequency=frequency,
    ).diagnostics


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


def folds_to_frame(folds: list[WalkForwardFold]) -> pd.DataFrame:
    columns = [
        "fold_id",
        "train_start",
        "train_end",
        "label_cutoff",
        "test_start",
        "test_end",
        "train_rows",
        "test_rows",
    ]
    if not folds:
        return pd.DataFrame(columns=columns)

    frame = pd.DataFrame([asdict(fold) for fold in folds])
    return frame.loc[:, columns]
