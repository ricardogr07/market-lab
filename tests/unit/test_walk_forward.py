from __future__ import annotations

from pathlib import Path

import pandas as pd

from marketlab.config import (
    ExperimentConfig,
    FeaturesConfig,
    PortfolioConfig,
    RankingConfig,
    TargetConfig,
    WalkForwardConfig,
)
from marketlab.data.panel import load_panel_csv
from marketlab.evaluation.walk_forward import (
    build_walk_forward_folds,
    folds_to_frame,
    slice_fold_rows,
)
from marketlab.targets.weekly import build_weekly_modeling_dataset

DEFAULT_SIGNAL_DATES = pd.date_range("2020-01-03", "2021-09-24", freq="W-FRI")


def _build_modeling_dataset(signal_dates: pd.DatetimeIndex = DEFAULT_SIGNAL_DATES) -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    for symbol, feature_offset, target_value in (
        ("AAA", 0, 1),
        ("BBB", 1000, 0),
    ):
        for index, signal_date in enumerate(signal_dates):
            rows.append(
                {
                    "symbol": symbol,
                    "signal_date": signal_date,
                    "target_end_date": signal_date + pd.Timedelta(days=5),
                    "feature_a": feature_offset + index,
                    "forward_return": 0.01 if target_value == 1 else -0.01,
                    "target": target_value,
                }
            )

    return pd.DataFrame(rows)


def test_walk_forward_first_fold_uses_expected_boundaries() -> None:
    dataset = _build_modeling_dataset()
    folds = build_walk_forward_folds(
        dataset,
        WalkForwardConfig(train_years=1, test_months=3, step_months=3),
    )

    first_fold = folds[0]

    assert first_fold.train_start == pd.Timestamp("2020-01-10")
    assert first_fold.train_end == pd.Timestamp("2021-01-01")
    assert first_fold.label_cutoff == pd.Timestamp("2021-01-08")
    assert first_fold.test_start == pd.Timestamp("2021-01-08")
    assert first_fold.test_end == pd.Timestamp("2021-04-02")


def test_walk_forward_uses_label_cutoff_not_just_signal_date() -> None:
    dataset = _build_modeling_dataset()
    mask = (dataset["symbol"] == "AAA") & (
        dataset["signal_date"] == pd.Timestamp("2021-01-01")
    )
    dataset.loc[mask, "target_end_date"] = pd.Timestamp("2021-01-15")

    fold = build_walk_forward_folds(
        dataset,
        WalkForwardConfig(train_years=1, test_months=3, step_months=3),
    )[0]
    train_rows, _ = slice_fold_rows(dataset, fold)

    assert not (
        (train_rows["symbol"] == "AAA")
        & (train_rows["signal_date"] == pd.Timestamp("2021-01-01"))
    ).any()
    assert fold.train_rows == len(train_rows)


def test_walk_forward_test_windows_do_not_overlap() -> None:
    dataset = _build_modeling_dataset()
    folds = build_walk_forward_folds(
        dataset,
        WalkForwardConfig(train_years=1, test_months=3, step_months=3),
    )

    for current_fold, next_fold in zip(folds, folds[1:]):
        assert current_fold.test_end < next_fold.test_start


def test_walk_forward_step_months_uses_first_available_signal_after_anchor() -> None:
    signal_dates = DEFAULT_SIGNAL_DATES.drop(pd.Timestamp("2021-04-09"))
    dataset = _build_modeling_dataset(signal_dates)
    folds = build_walk_forward_folds(
        dataset,
        WalkForwardConfig(train_years=1, test_months=3, step_months=3),
    )

    assert folds[1].test_start == pd.Timestamp("2021-04-16")


def test_walk_forward_returns_no_folds_without_enough_history() -> None:
    signal_dates = pd.date_range("2020-01-03", "2020-06-26", freq="W-FRI")
    dataset = _build_modeling_dataset(signal_dates)

    folds = build_walk_forward_folds(
        dataset,
        WalkForwardConfig(train_years=1, test_months=3, step_months=3),
    )

    assert folds == []


def test_walk_forward_drops_trailing_partial_test_window() -> None:
    signal_dates = pd.date_range("2020-01-03", "2021-05-21", freq="W-FRI")
    dataset = _build_modeling_dataset(signal_dates)

    folds = build_walk_forward_folds(
        dataset,
        WalkForwardConfig(train_years=1, test_months=3, step_months=3),
    )

    assert len(folds) == 1
    assert folds[0].test_start == pd.Timestamp("2021-01-08")


def test_walk_forward_tolerates_unsorted_input_rows() -> None:
    sorted_dataset = _build_modeling_dataset()
    shuffled_dataset = sorted_dataset.sample(frac=1.0, random_state=7).reset_index(drop=True)

    sorted_folds = build_walk_forward_folds(
        sorted_dataset,
        WalkForwardConfig(train_years=1, test_months=3, step_months=3),
    )
    shuffled_folds = build_walk_forward_folds(
        shuffled_dataset,
        WalkForwardConfig(train_years=1, test_months=3, step_months=3),
    )

    pd.testing.assert_frame_equal(
        folds_to_frame(sorted_folds),
        folds_to_frame(shuffled_folds),
    )


def test_slice_fold_rows_and_metadata_frame_are_consistent() -> None:
    dataset = _build_modeling_dataset()
    fold = build_walk_forward_folds(
        dataset,
        WalkForwardConfig(train_years=1, test_months=3, step_months=3),
    )[0]

    train_rows, test_rows = slice_fold_rows(dataset, fold)
    frame = folds_to_frame([fold])

    assert fold.train_rows == len(train_rows)
    assert fold.test_rows == len(test_rows)
    assert frame.columns.tolist() == [
        "fold_id",
        "train_start",
        "train_end",
        "label_cutoff",
        "test_start",
        "test_end",
        "train_rows",
        "test_rows",
    ]
    assert frame.loc[0, "fold_id"] == fold.fold_id
    assert frame.loc[0, "train_rows"] == fold.train_rows
    assert frame.loc[0, "test_rows"] == fold.test_rows


def test_walk_forward_repo_fixture_gracefully_returns_zero_folds_for_short_history() -> None:
    fixture_path = Path(__file__).resolve().parents[1] / "fixtures" / "market_panel.csv"
    panel = load_panel_csv(fixture_path)
    config = ExperimentConfig(
        features=FeaturesConfig(
            return_windows=[2],
            ma_windows=[2, 3],
            vol_windows=[2],
            momentum_window=2,
        ),
        target=TargetConfig(horizon_days=2, type="direction"),
        portfolio=PortfolioConfig(
            ranking=RankingConfig(rebalance_frequency="W-FRI")
        ),
    )
    dataset = build_weekly_modeling_dataset(panel, config)

    folds = build_walk_forward_folds(
        dataset,
        WalkForwardConfig(train_years=1, test_months=1, step_months=1),
    )

    assert not dataset.empty
    assert folds == []
