from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping


@dataclass(frozen=True, slots=True)
class Phase8GateResult:
    conditions: dict[str, bool]

    @property
    def passed(self) -> bool:
        return bool(self.conditions) and all(self.conditions.values())


def calculate_signal_validity_gate(
    metrics: Mapping[str, object],
) -> Phase8GateResult:
    return Phase8GateResult(
        conditions={
            "score_target_weight_correlation": _positive(
                metrics.get("score_target_weight_correlation")
            ),
            "score_forward_return_correlation": _positive(
                metrics.get("score_forward_return_correlation")
            ),
            "score_realized_utility_correlation": _positive(
                metrics.get("score_realized_utility_correlation")
            ),
            "predicted_tier_100_fraction": _positive(
                metrics.get("predicted_tier_100_fraction")
            ),
            "any_selected_oos_predicted_tier_100": _truthy(
                metrics.get("any_selected_oos_predicted_tier_100")
            ),
        }
    )


def calculate_bull_participation_gate(
    metrics: Mapping[str, object],
) -> Phase8GateResult:
    return Phase8GateResult(
        conditions={
            "gate_bull_average_long_exposure": _at_least(
                metrics.get("gate_bull_average_long_exposure"),
                0.50,
            ),
            "gate_bull_active_return_sum": _positive(
                metrics.get("gate_bull_active_return_sum")
            ),
            "gate_bull_underexposed_positive_benchmark_return_sum": _at_most(
                metrics.get(
                    "gate_bull_underexposed_positive_benchmark_return_sum"
                ),
                0.0,
            ),
            "selected_fold_fraction": _at_least(
                metrics.get("selected_fold_fraction"),
                0.75,
            ),
        }
    )


def _number(value: object) -> float | None:
    try:
        resolved = float(str(value))
    except (TypeError, ValueError):
        return None
    return resolved if math.isfinite(resolved) else None


def _positive(value: object) -> bool:
    resolved = _number(value)
    return resolved is not None and resolved > 0.0


def _at_least(value: object, minimum: float) -> bool:
    resolved = _number(value)
    return resolved is not None and resolved >= minimum


def _at_most(value: object, maximum: float) -> bool:
    resolved = _number(value)
    return resolved is not None and resolved <= maximum


def _truthy(value: object) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "t", "yes", "y"}
