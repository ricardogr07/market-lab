from __future__ import annotations

import pandas as pd
import pytest

from marketlab.strategies.tiered_allocation import (
    RegimeParticipationPolicy,
    generate_weights,
    nearest_tier,
    target_weight_for_score,
)


def _panel() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "symbol": ["BTC/USD"] * 4,
            "timestamp": pd.date_range("2026-01-01", periods=4, freq="4h"),
        }
    )


def _signal_weights(weights: pd.DataFrame, predictions: pd.DataFrame) -> list[float]:
    return weights.loc[
        weights["effective_date"].isin(predictions["effective_date"]),
        "weight",
    ].tolist()


def test_target_weight_for_score_uses_locked_tiers_and_risk_cap() -> None:
    thresholds = (0.50, 0.55, 0.62)

    assert target_weight_for_score(0.49, thresholds) == 0.0
    assert target_weight_for_score(0.52, thresholds) == 0.25
    assert target_weight_for_score(0.58, thresholds) == 0.50
    assert target_weight_for_score(0.70, thresholds) == 1.0
    assert target_weight_for_score(0.70, thresholds, risk_off=True) == 0.25
    assert nearest_tier(0.41) == 0.50


def test_generate_weights_maps_predictions_to_tiered_btc_exposure() -> None:
    predictions = pd.DataFrame(
        {
            "model_name": ["m"] * 3,
            "fold_id": [1, 1, 1],
            "signal_date": pd.date_range("2026-01-01", periods=3, freq="4h"),
            "effective_date": pd.date_range("2026-01-01 04:00", periods=3, freq="4h"),
            "symbol": ["BTC/USD"] * 3,
            "score": [0.51, 0.58, 0.80],
            "crypto_regime_risk_off": [0, 1, 0],
        }
    )

    weights = generate_weights(
        predictions=predictions,
        panel=_panel(),
        thresholds=(0.50, 0.55, 0.62),
        frequency="4h",
        strategy_name="btc_tiered",
    )

    non_boundary = weights.loc[weights["weight"].gt(0.0), "weight"].tolist()
    assert non_boundary == [0.25, 0.25, 1.0]


def test_generate_weights_model_only_policy_matches_default_behavior() -> None:
    predictions = pd.DataFrame(
        {
            "model_name": ["m"] * 3,
            "fold_id": [1, 1, 1],
            "signal_date": pd.date_range("2026-01-01", periods=3, freq="4h"),
            "effective_date": pd.date_range("2026-01-01 04:00", periods=3, freq="4h"),
            "symbol": ["BTC/USD"] * 3,
            "score": [0.51, 0.80, 0.80],
            "crypto_regime_risk_off": [0, 0, 1],
            "crypto_regime_trend_state": [0, 1, 1],
        }
    )

    default_weights = generate_weights(
        predictions=predictions,
        panel=_panel(),
        thresholds=(0.50, 0.55, 0.62),
        frequency="4h",
        strategy_name="btc_tiered",
    )
    policy_weights = generate_weights(
        predictions=predictions,
        panel=_panel(),
        thresholds=(0.50, 0.55, 0.62),
        frequency="4h",
        strategy_name="btc_tiered",
        regime_policy=RegimeParticipationPolicy(name="model_only"),
    )

    assert _signal_weights(policy_weights, predictions) == _signal_weights(
        default_weights,
        predictions,
    )


def test_generate_weights_applies_regime_participation_floors_and_risk_cap() -> None:
    predictions = pd.DataFrame(
        {
            "model_name": ["m"] * 4,
            "fold_id": [1] * 4,
            "signal_date": pd.date_range("2026-01-01", periods=4, freq="4h"),
            "effective_date": pd.date_range("2026-01-01 04:00", periods=4, freq="4h"),
            "symbol": ["BTC/USD"] * 4,
            "score": [0.10, 0.10, 0.10, 0.90],
            "crypto_regime_risk_off": [0, 0, 0, 1],
            "crypto_regime_trend_state": [1, 0, -1, 1],
        }
    )

    weights = generate_weights(
        predictions=predictions,
        panel=pd.DataFrame(
            {
                "symbol": ["BTC/USD"] * 5,
                "timestamp": pd.date_range("2026-01-01", periods=5, freq="4h"),
            }
        ),
        thresholds=(0.50, 0.55, 0.62),
        frequency="4h",
        strategy_name="btc_tiered",
        regime_policy=RegimeParticipationPolicy(
            name="bull100_sideways50_bear25",
            bull_floor=1.0,
            sideways_floor=0.50,
            bear_floor=0.25,
            risk_off_cap=0.25,
        ),
    )

    assert _signal_weights(weights, predictions) == [1.0, 0.50, 0.25, 0.25]


def test_generate_weights_applies_completed_bar_gate_bull_floor_after_runtime_policy() -> None:
    predictions = pd.DataFrame(
        {
            "model_name": ["m"] * 4,
            "fold_id": [1] * 4,
            "signal_date": pd.date_range("2026-01-01", periods=4, freq="4h"),
            "effective_date": pd.date_range("2026-01-01 04:00", periods=4, freq="4h"),
            "symbol": ["BTC/USD"] * 4,
            "score": [0.10, 0.10, 0.90, 0.10],
            "crypto_regime_risk_off": [1, 0, 1, 0],
            "crypto_regime_trend_state": [1, 0, 1, -1],
            "gate_bull": [True, True, False, False],
        }
    )

    weights = generate_weights(
        predictions=predictions,
        panel=pd.DataFrame(
            {
                "symbol": ["BTC/USD"] * 5,
                "timestamp": pd.date_range("2026-01-01", periods=5, freq="4h"),
            }
        ),
        thresholds=(0.50, 0.55, 0.62),
        frequency="4h",
        strategy_name="btc_tiered",
        regime_policy=RegimeParticipationPolicy(
            name="gate_bull_override",
            bull_floor=0.75,
            sideways_floor=0.25,
            bear_floor=0.0,
            risk_off_cap=0.25,
            gate_bull_floor=1.0,
        ),
    )

    assert _signal_weights(weights, predictions) == [1.0, 1.0, 0.25, 0.0]


def test_generate_weights_regime_policy_uses_only_current_signal_row() -> None:
    predictions = pd.DataFrame(
        {
            "model_name": ["m"] * 2,
            "fold_id": [1, 1],
            "signal_date": pd.date_range("2026-01-01", periods=2, freq="4h"),
            "effective_date": pd.date_range("2026-01-01 04:00", periods=2, freq="4h"),
            "symbol": ["BTC/USD"] * 2,
            "score": [0.10, 0.10],
            "crypto_regime_risk_off": [0, 0],
            "crypto_regime_trend_state": [-1, 1],
        }
    )

    weights = generate_weights(
        predictions=predictions,
        panel=_panel(),
        thresholds=(0.50, 0.55, 0.62),
        frequency="4h",
        strategy_name="btc_tiered",
        min_holding_period_bars=0,
        regime_policy=RegimeParticipationPolicy(
            name="bull100_sideways50_bear25",
            bull_floor=1.0,
            sideways_floor=0.50,
            bear_floor=0.25,
            risk_off_cap=0.25,
        ),
    )

    assert _signal_weights(weights, predictions) == [0.25, 1.0]


def test_generate_weights_enforces_minimum_holding_period() -> None:
    predictions = pd.DataFrame(
        {
            "model_name": ["m"] * 5,
            "fold_id": [1] * 5,
            "signal_date": pd.date_range("2026-01-01", periods=5, freq="4h"),
            "effective_date": pd.date_range("2026-01-01 04:00", periods=5, freq="4h"),
            "symbol": ["BTC/USD"] * 5,
            "score": [0.80, 0.40, 0.80, 0.80, 0.40],
            "crypto_regime_risk_off": [0] * 5,
        }
    )
    panel = pd.DataFrame(
        {
            "symbol": ["BTC/USD"] * 6,
            "timestamp": pd.date_range("2026-01-01", periods=6, freq="4h"),
        }
    )

    weights = generate_weights(
        predictions=predictions,
        panel=panel,
        thresholds=(0.50, 0.55, 0.62),
        frequency="4h",
        strategy_name="btc_tiered",
        min_holding_period_bars=3,
    )

    signal_weights = weights.loc[
        weights["effective_date"].isin(predictions["effective_date"]),
        "weight",
    ].tolist()
    assert signal_weights == [1.0, 1.0, 1.0, 1.0, 0.0]


def test_generate_weights_risk_off_de_risks_during_holding_period() -> None:
    predictions = pd.DataFrame(
        {
            "model_name": ["m"] * 2,
            "fold_id": [1, 1],
            "signal_date": pd.date_range("2026-01-01", periods=2, freq="4h"),
            "effective_date": pd.date_range("2026-01-01 04:00", periods=2, freq="4h"),
            "symbol": ["BTC/USD"] * 2,
            "score": [0.80, 0.80],
            "crypto_regime_risk_off": [0, 1],
        }
    )

    weights = generate_weights(
        predictions=predictions,
        panel=_panel(),
        thresholds=(0.50, 0.55, 0.62),
        frequency="4h",
        strategy_name="btc_tiered",
        min_holding_period_bars=12,
    )

    signal_weights = weights.loc[
        weights["effective_date"].isin(predictions["effective_date"]),
        "weight",
    ].tolist()
    assert signal_weights == [1.0, 0.25]


def test_generate_weights_hysteresis_requires_larger_move_to_change_tier() -> None:
    predictions = pd.DataFrame(
        {
            "model_name": ["m"] * 5,
            "fold_id": [1] * 5,
            "signal_date": pd.date_range("2026-01-01", periods=5, freq="4h"),
            "effective_date": pd.date_range("2026-01-01 04:00", periods=5, freq="4h"),
            "symbol": ["BTC/USD"] * 5,
            "score": [0.58, 0.63, 0.66, 0.60, 0.58],
            "crypto_regime_risk_off": [0] * 5,
        }
    )
    panel = pd.DataFrame(
        {
            "symbol": ["BTC/USD"] * 6,
            "timestamp": pd.date_range("2026-01-01", periods=6, freq="4h"),
        }
    )

    weights = generate_weights(
        predictions=predictions,
        panel=panel,
        thresholds=(0.50, 0.55, 0.62),
        frequency="4h",
        strategy_name="btc_tiered",
        hysteresis_margin=0.03,
    )

    signal_weights = weights.loc[
        weights["effective_date"].isin(predictions["effective_date"]),
        "weight",
    ].tolist()
    assert signal_weights == [0.50, 0.50, 1.0, 1.0, 0.50]


def test_generate_weights_hysteresis_keeps_risk_off_de_risk_immediate() -> None:
    predictions = pd.DataFrame(
        {
            "model_name": ["m"] * 2,
            "fold_id": [1, 1],
            "signal_date": pd.date_range("2026-01-01", periods=2, freq="4h"),
            "effective_date": pd.date_range("2026-01-01 04:00", periods=2, freq="4h"),
            "symbol": ["BTC/USD"] * 2,
            "score": [0.80, 0.80],
            "crypto_regime_risk_off": [0, 1],
        }
    )

    weights = generate_weights(
        predictions=predictions,
        panel=_panel(),
        thresholds=(0.50, 0.55, 0.62),
        frequency="4h",
        strategy_name="btc_tiered",
        hysteresis_margin=0.25,
    )

    signal_weights = weights.loc[
        weights["effective_date"].isin(predictions["effective_date"]),
        "weight",
    ].tolist()
    assert signal_weights == [1.0, 0.25]


def test_generate_weights_direct_tiered_uses_expected_allocation_score() -> None:
    predictions = pd.DataFrame(
        {
            "model_name": ["m"] * 4,
            "fold_id": [1] * 4,
            "signal_date": pd.date_range("2026-01-01", periods=4, freq="4h"),
            "effective_date": pd.date_range("2026-01-01 04:00", periods=4, freq="4h"),
            "symbol": ["BTC/USD"] * 4,
            "score": [0.10, 0.30, 0.62, 0.90],
            "crypto_regime_risk_off": [0] * 4,
        }
    )
    panel = pd.DataFrame(
        {
            "symbol": ["BTC/USD"] * 5,
            "timestamp": pd.date_range("2026-01-01", periods=5, freq="4h"),
        }
    )

    weights = generate_weights(
        predictions=predictions,
        panel=panel,
        thresholds=(0.25, 0.50, 0.75),
        frequency="4h",
        strategy_name="btc_direct_tiered",
        direct_scores=True,
    )

    signal_weights = weights.loc[
        weights["effective_date"].isin(predictions["effective_date"]),
        "weight",
    ].tolist()
    assert signal_weights == [0.0, 0.25, 0.50, 1.0]


def test_generate_weights_direct_tiered_applies_risk_off_and_hysteresis() -> None:
    predictions = pd.DataFrame(
        {
            "model_name": ["m"] * 3,
            "fold_id": [1] * 3,
            "signal_date": pd.date_range("2026-01-01", periods=3, freq="4h"),
            "effective_date": pd.date_range("2026-01-01 04:00", periods=3, freq="4h"),
            "symbol": ["BTC/USD"] * 3,
            "score": [0.52, 0.70, 0.95],
            "crypto_regime_risk_off": [0, 0, 1],
        }
    )
    panel = pd.DataFrame(
        {
            "symbol": ["BTC/USD"] * 4,
            "timestamp": pd.date_range("2026-01-01", periods=4, freq="4h"),
        }
    )

    weights = generate_weights(
        predictions=predictions,
        panel=panel,
        thresholds=(0.25, 0.50, 0.75),
        frequency="4h",
        strategy_name="btc_direct_tiered",
        hysteresis_margin=0.10,
        min_holding_period_bars=12,
        direct_scores=True,
    )

    signal_weights = weights.loc[
        weights["effective_date"].isin(predictions["effective_date"]),
        "weight",
    ].tolist()
    assert signal_weights == [0.50, 0.50, 0.25]


def test_generate_weights_rejects_multi_symbol_allocations() -> None:
    predictions = pd.DataFrame(
        {
            "model_name": ["m", "m"],
            "fold_id": [1, 1],
            "signal_date": [pd.Timestamp("2026-01-01")] * 2,
            "effective_date": [pd.Timestamp("2026-01-01 04:00")] * 2,
            "symbol": ["BTC/USD", "ETH/USD"],
            "score": [0.8, 0.8],
        }
    )
    panel = pd.DataFrame(
        {
            "symbol": ["BTC/USD", "ETH/USD"],
            "timestamp": [pd.Timestamp("2026-01-01")] * 2,
        }
    )

    with pytest.raises(ValueError, match="one BTC symbol"):
        generate_weights(predictions, panel, (0.5, 0.55, 0.62))
