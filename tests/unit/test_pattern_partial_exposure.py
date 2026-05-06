from __future__ import annotations

import pandas as pd
import pytest

from marketlab.strategies.chart_patterns import PATTERN_COLUMNS
from marketlab.strategies.pattern_partial_exposure import (
    generate_diagnostics,
    generate_weights,
)


def _overlay() -> pd.DataFrame:
    rows = []
    for index in range(4):
        timestamp = pd.Timestamp("2024-01-01") + pd.Timedelta(hours=index)
        patterns = {column: False for column in PATTERN_COLUMNS}
        patterns["double_top_breakdown"] = index in {0, 1}
        rows.append(
            {
                "strategy": "pattern_exit_overlay",
                "timestamp": timestamp,
                "effective_date": timestamp + pd.Timedelta(hours=1),
                "symbol": "BTC-USD",
                "close": 100.0,
                "trend_ema": 100.0,
                "support_level": 95.0,
                "resistance_level": 105.0,
                "volume_average": 1000.0,
                "volume_confirmed": True,
                **patterns,
                "bullish_pattern_count": 0,
                "bearish_pattern_count": 1 if index in {0, 1} else 0,
                "bearish_exit_candidate": index in {0, 1},
                "bullish_reentry_candidate": index in {2, 3},
                "bearish_clear": index in {2, 3},
                "bearish_confirmation_count": 0,
                "exit_blocked_by_trend": False,
                "exit_blocked_by_confirmation": False,
                "exit_blocked_by_cooldown": False,
                "reentry_blocked_by_min_cash": False,
                "bars_since_exit": pd.NA,
                "bars_since_reentry": index,
                "exit_reason": "",
                "reentry_reason": "",
                "reentry_clear_count": 0,
                "target_weight": 1.0,
            }
        )
    return pd.DataFrame(rows)


def test_partial_overlay_maps_score_bands_to_weights() -> None:
    overlay = _overlay()
    predictions = pd.DataFrame(
        [
            {
                "model_name": "logistic_l1",
                "fold_id": 1,
                "symbol": "BTC-USD",
                "signal_date": overlay.loc[0, "timestamp"],
                "effective_date": overlay.loc[0, "effective_date"],
                "target_end_date": overlay.loc[0, "effective_date"],
                "forward_return": -0.01,
                "target": 1,
                "score": 0.62,
                "predicted_target": 1,
            },
            {
                "model_name": "logistic_l1",
                "fold_id": 1,
                "symbol": "BTC-USD",
                "signal_date": overlay.loc[1, "timestamp"],
                "effective_date": overlay.loc[1, "effective_date"],
                "target_end_date": overlay.loc[1, "effective_date"],
                "forward_return": -0.02,
                "target": 1,
                "score": 0.82,
                "predicted_target": 1,
            },
        ]
    )

    diagnostics = generate_diagnostics(
        overlay,
        predictions,
        partial_threshold=0.6,
        full_threshold=0.8,
        partial_weight=0.5,
        reentry_clear_bars=1,
    )

    assert diagnostics["target_weight"].tolist() == pytest.approx([0.5, 0.0, 0.5, 1.0])
    assert diagnostics["risk_state"].tolist() == ["partial", "cash", "partial", "long"]
    weights = generate_weights(diagnostics)
    assert list(weights.columns) == ["strategy", "effective_date", "symbol", "weight"]
    assert weights["weight"].tolist() == pytest.approx([0.5, 0.0, 0.5, 1.0])
