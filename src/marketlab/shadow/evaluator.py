from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from marketlab.pipeline import ShadowCandidateEvaluation, evaluate_shadow_candidate
from marketlab.shadow.decision import (
    ShadowDecisionContext,
    ShadowDecisionEvaluation,
)
from marketlab.shadow.journal import canonical_fingerprint


@dataclass(slots=True)
class NativeShadowDecisionEvaluator:
    """Adapter from the frozen Phase 8 candidate to the P9-04 decision contract."""

    last_candidate_evaluation: ShadowCandidateEvaluation | None = None

    def __call__(self, context: ShadowDecisionContext) -> ShadowDecisionEvaluation:
        panel = pd.DataFrame(
            [bar.as_fingerprint_payload() for bar in context.completed_bars]
        )
        panel["timestamp"] = pd.to_datetime(panel["timestamp"], utc=True).dt.tz_localize(
            None
        )
        candidate = evaluate_shadow_candidate(
            config=context.contract.config,
            panel=panel,
            signal_date=context.signal_date,
        )
        self.last_candidate_evaluation = candidate
        fallback_mode = (
            candidate.selection_source
            if candidate.selection_source
            in {"best_active_fallback", "regime_policy_fallback"}
            else "none"
        )
        diagnostics = {
            "raw_score": candidate.raw_score,
            "selected_tier": candidate.selected_tier,
            "selection_source": candidate.selection_source,
            "fallback_mode": fallback_mode,
            "regime_classification": candidate.regime_classification,
            "input_cutoff": context.signal_date.isoformat(),
            "pipeline": candidate.diagnostics,
        }
        diagnostics["diagnostic_fingerprint"] = canonical_fingerprint(diagnostics)
        return ShadowDecisionEvaluation(
            status="success",
            selection_source=candidate.selection_source,
            fallback_mode=fallback_mode,
            target_allocation=candidate.target_allocation,
            input_payload=diagnostics,
        )
