from __future__ import annotations

import math
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pandas as pd

from marketlab.pipeline import ShadowCandidateEvaluation
from marketlab.shadow import (
    NativeShadowDecisionEvaluator,
    ShadowBar,
    ShadowDecisionContext,
    verify_shadow_contract,
)
from marketlab.targets import build_modeling_dataset, build_scoring_dataset

ROOT = Path(__file__).resolve().parents[2]
SHADOW_CONFIG = ROOT / "configs" / "experiment.btc_phase9_shadow_daily.yaml"


def _bars(count: int = 500) -> tuple[ShadowBar, ...]:
    start = datetime(2025, 1, 1, tzinfo=UTC)
    bars = []
    for index in range(count):
        close = 100.0 + (index * 0.05) + (5.0 * math.sin(index / 9.0))
        bars.append(
            ShadowBar(
            symbol="BTC-USD",
            timestamp=start + timedelta(days=index),
            open=close - 0.2,
            high=close + 0.8,
            low=close - 0.8,
            close=close,
            volume=1_000.0 + (50.0 * math.cos(index / 7.0)),
            adj_close=close,
            adj_open=close - 0.2,
            adj_high=close + 0.8,
            adj_low=close - 0.8,
            )
        )
    return tuple(bars)


def test_scoring_dataset_preserves_modeling_features() -> None:
    contract = verify_shadow_contract(SHADOW_CONFIG)
    bars = _bars()
    panel = pd.DataFrame([bar.as_fingerprint_payload() for bar in bars])
    panel["timestamp"] = pd.to_datetime(panel["timestamp"]).dt.tz_localize(None)

    scoring = build_scoring_dataset(panel, contract.config)
    modeling = build_modeling_dataset(panel, contract.config)
    scoring["signal_date"] = pd.to_datetime(scoring["signal_date"])
    scoring["effective_date"] = pd.to_datetime(scoring["effective_date"])
    modeling["signal_date"] = pd.to_datetime(modeling["signal_date"])
    modeling["effective_date"] = pd.to_datetime(modeling["effective_date"])

    shared = scoring.merge(
        modeling,
        on=["symbol", "signal_date", "effective_date"],
        suffixes=("_score", "_model"),
    )
    assert not shared.empty
    feature_columns = [
        column.removesuffix("_score")
        for column in shared.columns
        if column.endswith("_score")
    ]
    for column in feature_columns:
        assert shared[f"{column}_score"].equals(shared[f"{column}_model"])


def test_native_evaluator_returns_deterministic_phase8_diagnostics(monkeypatch) -> None:
    contract = verify_shadow_contract(SHADOW_CONFIG)
    bars = _bars()
    context = ShadowDecisionContext(
        contract=contract,
        as_of=bars[-1].timestamp + timedelta(days=1, hours=1, minutes=15),
        completed_bars=bars,
        signal_date=bars[-1].timestamp.date(),
        effective_date=bars[-1].timestamp.date() + timedelta(days=1),
        matured_label_cutoff=bars[-15].timestamp.date(),
    )

    def _evaluate(**kwargs):
        assert kwargs["signal_date"] == context.signal_date
        assert pd.Timestamp(kwargs["panel"]["timestamp"].max()).date() == context.signal_date
        return ShadowCandidateEvaluation(
            selection_source="strict",
            target_allocation=0.50,
            raw_score=0.61,
            selected_tier=0.50,
            regime_classification="bull",
            diagnostics={"selected_model": "hist_gradient_boosting"},
        )

    monkeypatch.setattr("marketlab.shadow.evaluator.evaluate_shadow_candidate", _evaluate)
    evaluator = NativeShadowDecisionEvaluator()

    first = evaluator(context)
    second = evaluator(context)

    assert first == second
    assert first.target_allocation == 0.50
    assert first.selection_source == "strict"
    assert first.input_payload["raw_score"] == 0.61
    assert first.input_payload["regime_classification"] == "bull"
    assert len(str(first.input_payload["diagnostic_fingerprint"])) == 64
