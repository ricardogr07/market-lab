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
    load_config,
)
from marketlab.data.panel import load_panel_csv
from marketlab.evaluation.walk_forward import (
    DIAGNOSTIC_COLUMNS,
    SKIP_REASON_INCOMPLETE_TEST_WINDOW,
    SKIP_REASON_INSUFFICIENT_TEST_POSITIVE_RATE,
    SKIP_REASON_INSUFFICIENT_TEST_ROWS,
    SKIP_REASON_INSUFFICIENT_TRAIN_POSITIVE_RATE,
    SKIP_REASON_INSUFFICIENT_TRAIN_ROWS,
    build_walk_forward_diagnostics,
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


def test_walk_forward_embargo_adjusts_label_cutoff_and_training_range() -> None:
    dataset = _build_modeling_dataset()
    fold = build_walk_forward_folds(
        dataset,
        WalkForwardConfig(
            train_years=1,
            test_months=3,
            step_months=3,
            embargo_periods=1,
        ),
    )[0]
    train_rows, _ = slice_fold_rows(dataset, fold)

    assert fold.label_cutoff == pd.Timestamp("2021-01-01")
    assert fold.train_end == pd.Timestamp("2020-12-25")
    assert train_rows["signal_date"].max() < fold.label_cutoff
    assert train_rows["target_end_date"].max() <= fold.label_cutoff


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


def test_walk_forward_diagnostics_include_trailing_partial_candidate() -> None:
    signal_dates = pd.date_range("2020-01-03", "2021-05-21", freq="W-FRI")
    dataset = _build_modeling_dataset(signal_dates)

    folds = build_walk_forward_folds(
        dataset,
        WalkForwardConfig(train_years=1, test_months=3, step_months=3),
    )
    diagnostics = build_walk_forward_diagnostics(
        dataset,
        WalkForwardConfig(train_years=1, test_months=3, step_months=3),
    )

    assert len(folds) == 1
    assert folds[0].test_start == pd.Timestamp("2021-01-08")
    assert diagnostics.columns.tolist() == DIAGNOSTIC_COLUMNS
    assert diagnostics["candidate_id"].tolist() == [1, 2]
    assert diagnostics["status"].tolist() == ["used", "skipped"]
    assert diagnostics.loc[0, "fold_id"] == 1
    assert pd.isna(diagnostics.loc[1, "fold_id"])
    assert diagnostics.loc[1, "skip_reasons"] == SKIP_REASON_INCOMPLETE_TEST_WINDOW


def test_walk_forward_min_train_rows_guardrail_skips_fold() -> None:
    dataset = _build_modeling_dataset()
    diagnostics = build_walk_forward_diagnostics(
        dataset,
        WalkForwardConfig(
            train_years=1,
            test_months=3,
            step_months=3,
            min_train_rows=200,
        ),
    )

    assert build_walk_forward_folds(
        dataset,
        WalkForwardConfig(
            train_years=1,
            test_months=3,
            step_months=3,
            min_train_rows=200,
        ),
    ) == []
    assert SKIP_REASON_INSUFFICIENT_TRAIN_ROWS in diagnostics.loc[0, "skip_reasons"].split(";")


def test_walk_forward_min_test_rows_guardrail_skips_fold() -> None:
    dataset = _build_modeling_dataset()
    diagnostics = build_walk_forward_diagnostics(
        dataset,
        WalkForwardConfig(
            train_years=1,
            test_months=3,
            step_months=3,
            min_test_rows=30,
        ),
    )

    assert build_walk_forward_folds(
        dataset,
        WalkForwardConfig(
            train_years=1,
            test_months=3,
            step_months=3,
            min_test_rows=30,
        ),
    ) == []
    assert SKIP_REASON_INSUFFICIENT_TEST_ROWS in diagnostics.loc[0, "skip_reasons"].split(";")


def test_walk_forward_positive_rate_guardrails_skip_fold() -> None:
    dataset = _build_modeling_dataset()
    diagnostics = build_walk_forward_diagnostics(
        dataset,
        WalkForwardConfig(
            train_years=1,
            test_months=3,
            step_months=3,
            min_train_positive_rate=0.75,
            min_test_positive_rate=0.75,
        ),
    )

    reasons = diagnostics.loc[0, "skip_reasons"].split(";")
    assert SKIP_REASON_INSUFFICIENT_TRAIN_POSITIVE_RATE in reasons
    assert SKIP_REASON_INSUFFICIENT_TEST_POSITIVE_RATE in reasons


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


def test_walk_forward_config_loads_guardrail_keys(tmp_path: Path) -> None:
    config_path = tmp_path / "guardrails.yaml"
    config_path.write_text(
        "\n".join(
            [
                "experiment_name: guardrail_test",
                "evaluation:",
                "  walk_forward:",
                "    train_years: 2",
                "    test_months: 1",
                "    step_months: 1",
                "    min_train_rows: 123",
                "    min_test_rows: 45",
                "    min_train_positive_rate: 0.1",
                "    min_test_positive_rate: 0.2",
                "    embargo_periods: 1",
                "",
            ]
        ),
        encoding="utf-8",
    )

    config = load_config(config_path)

    assert config.evaluation.walk_forward == WalkForwardConfig(
        train_years=2,
        test_months=1,
        step_months=1,
        min_train_rows=123,
        min_test_rows=45,
        min_train_positive_rate=0.1,
        min_test_positive_rate=0.2,
        embargo_periods=1,
    )


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
