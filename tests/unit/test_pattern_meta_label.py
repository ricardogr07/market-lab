from __future__ import annotations

import pandas as pd
import pytest

from marketlab.config import WalkForwardConfig
from marketlab.strategies.chart_patterns import PATTERN_COLUMNS
from marketlab.strategies.pattern_meta_label import (
    build_labels,
    generate_meta_overlay_diagnostics,
    predict_exit_candidates,
)


def _panel(closes: list[float]) -> pd.DataFrame:
    rows = []
    for index, close in enumerate(closes):
        timestamp = pd.Timestamp("2024-01-01 00:00:00") + pd.Timedelta(days=index)
        rows.append(
            {
                "symbol": "BTC-USD",
                "timestamp": timestamp,
                "open": close,
                "high": close + 1,
                "low": close - 1,
                "close": close,
                "volume": 1000.0,
                "adj_close": close,
                "adj_factor": 1.0,
                "adj_open": close,
                "adj_high": close + 1,
                "adj_low": close - 1,
            }
        )
    return pd.DataFrame(rows)


def _overlay_diagnostics() -> pd.DataFrame:
    rows = []
    for index, close in enumerate([100.0, 100.0, 100.0]):
        timestamp = pd.Timestamp("2024-01-01 00:00:00") + pd.Timedelta(days=index)
        patterns = {column: False for column in PATTERN_COLUMNS}
        patterns["double_top_breakdown"] = True
        rows.append(
            {
                "strategy": "pattern_exit_overlay",
                "timestamp": timestamp,
                "effective_date": timestamp + pd.Timedelta(days=1),
                "symbol": "BTC-USD",
                "close": close,
                "trend_ema": close,
                "support_level": 95.0,
                "resistance_level": 105.0,
                "volume_average": 1000.0,
                "volume_confirmed": True,
                **patterns,
                "bullish_pattern_count": 0,
                "bearish_pattern_count": 1,
                "bearish_exit_candidate": True,
                "bullish_reentry_candidate": False,
                "bearish_clear": False,
                "reentry_clear_count": 0,
                "target_weight": 0.0,
            }
        )
    return pd.DataFrame(rows)


def test_meta_label_target_uses_next_bar_open_horizon_close_and_cost() -> None:
    labels = build_labels(
        _panel([100.0, 100.0, 90.0, 110.0, 110.0]),
        _overlay_diagnostics(),
        label_horizon_bars=2,
        cost_bps=10.0,
    )

    assert labels["effective_date"].iloc[0] == pd.Timestamp("2024-01-02 00:00:00")
    assert labels["target_end_date"].iloc[0] == pd.Timestamp("2024-01-03 00:00:00")
    assert labels["forward_return"].iloc[0] == pytest.approx(-0.10)
    assert labels["target"].iloc[0] == 1
    assert labels["target"].iloc[1] == 0


def test_meta_labels_drop_rows_with_incomplete_horizon() -> None:
    labels = build_labels(
        _panel([100.0, 100.0, 90.0]),
        _overlay_diagnostics(),
        label_horizon_bars=3,
        cost_bps=10.0,
    )

    assert len(labels) == 0


def test_meta_prediction_skips_one_class_training_folds() -> None:
    dates = pd.date_range("2024-01-01", periods=430, freq="D")
    labels = pd.DataFrame(
        {
            "symbol": "BTC-USD",
            "signal_date": dates,
            "effective_date": dates + pd.Timedelta(days=1),
            "target_end_date": dates + pd.Timedelta(days=2),
            "forward_return": 0.01,
            "turnover_cost": 0.001,
            "target": 0,
            "close": 100.0,
            "trend_ema": 100.0,
        }
    )

    predictions, diagnostics = predict_exit_candidates(
        labels,
        walk_forward=WalkForwardConfig(
            train_years=1,
            test_months=1,
            step_months=1,
            min_train_rows=1,
            min_test_rows=1,
        ),
        rebalance_frequency="D",
        model_names=["logistic_l1"],
        threshold=0.55,
    )

    assert predictions.empty
    assert "single_train_class" in ";".join(diagnostics["skip_reasons"].astype(str))


def test_meta_overlay_only_exits_when_oos_prediction_passes_threshold() -> None:
    overlay = _overlay_diagnostics()
    predictions = pd.DataFrame(
        [
            {
                "model_name": "logistic_l1",
                "fold_id": 1,
                "symbol": "BTC-USD",
                "signal_date": overlay.loc[0, "timestamp"],
                "effective_date": overlay.loc[0, "effective_date"],
                "target_end_date": overlay.loc[0, "effective_date"],
                "forward_return": -0.02,
                "target": 1,
                "score": 0.60,
                "predicted_target": 1,
            }
        ]
    )

    diagnostics = generate_meta_overlay_diagnostics(
        overlay,
        predictions,
        threshold=0.55,
        reentry_clear_bars=1,
    )

    assert diagnostics["target_weight"].tolist() == pytest.approx([0.0, 0.0, 0.0])
    assert bool(diagnostics.loc[0, "meta_exit_predicted"]) is True
    assert bool(diagnostics.loc[1, "meta_exit_predicted"]) is False
