from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from marketlab.config import (
    ExperimentConfig,
    FeaturesConfig,
    PortfolioConfig,
    RankingConfig,
    TargetConfig,
)
from marketlab.data.panel import load_panel_csv
from marketlab.rebalance import rebalance_signal_dates, weekly_signal_dates
from marketlab.targets.timing import build_modeling_dataset, build_rebalance_snapshots
from marketlab.targets.weekly import (
    add_forward_targets,
    build_weekly_modeling_dataset,
    build_weekly_snapshots,
)

TRADING_DATES = pd.to_datetime(
    [
        "2024-01-02",
        "2024-01-03",
        "2024-01-04",
        "2024-01-05",
        "2024-01-08",
        "2024-01-09",
        "2024-01-10",
        "2024-01-11",
        "2024-01-12",
        "2024-01-16",
        "2024-01-17",
        "2024-01-18",
        "2024-01-22",
        "2024-01-23",
        "2024-01-24",
        "2024-01-25",
        "2024-01-26",
        "2024-01-29",
        "2024-01-30",
    ]
)


@pytest.fixture()
def featured_panel() -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    for symbol, base_price, step, feature_offset in (
        ("AAA", 100.0, 1.0, 0),
        ("BBB", 200.0, -1.0, 100),
    ):
        for index, timestamp in enumerate(TRADING_DATES):
            price = base_price + (index * step)
            rows.append(
                {
                    "symbol": symbol,
                    "timestamp": timestamp,
                    "adj_open": price,
                    "adj_close": price,
                    "feature_a": feature_offset + index,
                    "feature_b": feature_offset + (index * 10),
                }
            )

    return pd.DataFrame(rows).sort_values(["symbol", "timestamp"]).reset_index(drop=True)


def test_weekly_samples_use_last_available_date_per_wfri_period(
    featured_panel: pd.DataFrame,
) -> None:
    snapshots = build_weekly_snapshots(featured_panel, feature_columns=["feature_a", "feature_b"])

    assert snapshots["signal_date"].nunique() == 4
    assert set(snapshots["signal_date"]) == {
        pd.Timestamp("2024-01-05"),
        pd.Timestamp("2024-01-12"),
        pd.Timestamp("2024-01-18"),
        pd.Timestamp("2024-01-26"),
    }
    assert pd.Timestamp("2024-01-19") not in set(snapshots["signal_date"])

    shortened_week = snapshots.loc[snapshots["signal_date"] == pd.Timestamp("2024-01-18")]
    assert len(shortened_week) == 2
    assert set(shortened_week["effective_date"]) == {pd.Timestamp("2024-01-22")}


def test_weekly_samples_copy_features_from_signal_date_only(
    featured_panel: pd.DataFrame,
) -> None:
    snapshots = build_weekly_snapshots(featured_panel, feature_columns=["feature_a", "feature_b"])

    sample = snapshots.loc[
        (snapshots["symbol"] == "AAA")
        & (snapshots["signal_date"] == pd.Timestamp("2024-01-18"))
    ].iloc[0]

    assert sample["feature_a"] == 11
    assert sample["feature_b"] == 110


def test_future_data_change_does_not_change_prior_weekly_samples(
    featured_panel: pd.DataFrame,
) -> None:
    baseline = build_weekly_snapshots(featured_panel, feature_columns=["feature_a", "feature_b"])

    modified = featured_panel.copy()
    mask = (modified["symbol"] == "AAA") & (
        modified["timestamp"] == pd.Timestamp("2024-01-29")
    )
    modified.loc[mask, "feature_a"] = 9999

    updated = build_weekly_snapshots(modified, feature_columns=["feature_a", "feature_b"])

    pd.testing.assert_frame_equal(baseline, updated)


def test_targets_use_only_future_prices_after_signal_date(
    featured_panel: pd.DataFrame,
) -> None:
    snapshots = build_weekly_snapshots(featured_panel, feature_columns=["feature_a", "feature_b"])
    labeled = add_forward_targets(snapshots, featured_panel, horizon_days=3)

    aaa_row = labeled.loc[
        (labeled["symbol"] == "AAA")
        & (labeled["signal_date"] == pd.Timestamp("2024-01-12"))
    ].iloc[0]
    bbb_row = labeled.loc[
        (labeled["symbol"] == "BBB")
        & (labeled["signal_date"] == pd.Timestamp("2024-01-12"))
    ].iloc[0]

    assert aaa_row["effective_date"] == pd.Timestamp("2024-01-16")
    assert aaa_row["target_end_date"] == pd.Timestamp("2024-01-18")
    assert aaa_row["forward_return"] == pytest.approx((111.0 / 109.0) - 1.0)
    assert aaa_row["target"] == 1

    assert bbb_row["effective_date"] == pd.Timestamp("2024-01-16")
    assert bbb_row["target_end_date"] == pd.Timestamp("2024-01-18")
    assert bbb_row["forward_return"] == pytest.approx((189.0 / 191.0) - 1.0)
    assert bbb_row["target"] == 0


def test_trailing_week_without_full_horizon_is_dropped(
    featured_panel: pd.DataFrame,
) -> None:
    snapshots = build_weekly_snapshots(featured_panel, feature_columns=["feature_a", "feature_b"])
    labeled = add_forward_targets(snapshots, featured_panel, horizon_days=3)

    assert pd.Timestamp("2024-01-26") not in set(labeled["signal_date"])
    assert len(labeled) == 6


def test_missing_symbol_on_signal_date_is_dropped_for_global_rebalance_calendar(
    featured_panel: pd.DataFrame,
) -> None:
    partial_panel = featured_panel.loc[
        ~(
            (featured_panel["symbol"] == "BBB")
            & (featured_panel["timestamp"] == pd.Timestamp("2024-01-18"))
        )
    ].copy()

    snapshots = build_weekly_snapshots(partial_panel, feature_columns=["feature_a", "feature_b"])

    assert not (
        (snapshots["symbol"] == "BBB")
        & (snapshots["signal_date"] == pd.Timestamp("2024-01-18"))
    ).any()
    assert (
        (snapshots["symbol"] == "AAA")
        & (snapshots["signal_date"] == pd.Timestamp("2024-01-18"))
    ).any()


def test_missing_symbol_on_exit_date_is_dropped_explicitly(
    featured_panel: pd.DataFrame,
) -> None:
    partial_panel = featured_panel.loc[
        ~(
            (featured_panel["symbol"] == "BBB")
            & (featured_panel["timestamp"] == pd.Timestamp("2024-01-24"))
        )
    ].copy()

    snapshots = build_weekly_snapshots(partial_panel, feature_columns=["feature_a", "feature_b"])
    labeled = add_forward_targets(snapshots, partial_panel, horizon_days=3)

    assert not (
        (labeled["symbol"] == "BBB")
        & (labeled["signal_date"] == pd.Timestamp("2024-01-18"))
    ).any()
    assert (
        (labeled["symbol"] == "AAA")
        & (labeled["signal_date"] == pd.Timestamp("2024-01-18"))
    ).any()


def test_build_weekly_modeling_dataset_uses_fixture_panel_contract() -> None:
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

    assert not dataset.empty
    assert set(dataset["signal_date"]).issubset(set(weekly_signal_dates(panel)))
    assert dataset["effective_date"].gt(dataset["signal_date"]).all()
    assert dataset["target_end_date"].ge(dataset["effective_date"]).all()
    feature_columns = [
        column
        for column in dataset.columns
        if column
        not in {
            "symbol",
            "signal_date",
            "effective_date",
            "target_end_date",
            "forward_return",
            "target",
        }
    ]
    assert dataset[feature_columns].notna().all().all()


def test_daily_snapshots_use_each_trading_day_as_signal_date(
    featured_panel: pd.DataFrame,
) -> None:
    snapshots = build_rebalance_snapshots(
        featured_panel,
        feature_columns=["feature_a", "feature_b"],
        frequency="D",
    )

    assert snapshots["signal_date"].nunique() == len(TRADING_DATES) - 1
    assert set(snapshots["signal_date"]) == set(TRADING_DATES[:-1])

    aaa_row = snapshots.loc[
        (snapshots["symbol"] == "AAA")
        & (snapshots["signal_date"] == pd.Timestamp("2024-01-18"))
    ].iloc[0]
    assert aaa_row["effective_date"] == pd.Timestamp("2024-01-22")
    assert aaa_row["feature_a"] == 11


def test_daily_modeling_dataset_respects_daily_frequency_and_one_day_target() -> None:
    fixture_path = Path(__file__).resolve().parents[1] / "fixtures" / "market_panel.csv"
    panel = load_panel_csv(fixture_path)
    config = ExperimentConfig(
        features=FeaturesConfig(
            return_windows=[2],
            ma_windows=[2, 3],
            vol_windows=[2],
            momentum_window=2,
        ),
        target=TargetConfig(horizon_days=1, type="direction"),
        portfolio=PortfolioConfig(
            ranking=RankingConfig(rebalance_frequency="D", mode="long_only", long_n=1, short_n=1)
        ),
    )

    dataset = build_modeling_dataset(panel, config)

    assert not dataset.empty
    assert set(dataset["signal_date"]).issubset(set(rebalance_signal_dates(panel, "D")))
    assert dataset["effective_date"].gt(dataset["signal_date"]).all()
    assert dataset["target_end_date"].eq(dataset["effective_date"]).all()
    assert dataset["target"].isin({0, 1}).all()
