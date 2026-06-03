from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import pandas as pd

from marketlab.config import ExperimentConfig
from marketlab.targets import apply_allocation_utility_profile

DEFAULT_OUTPUT_PATH = Path("artifacts/runs/phase8_btc_target_profile_sweep.csv")
ALLOCATION_TIERS = (0.0, 0.25, 0.50, 1.0)
PHASE8_TARGET_PROFILE_SWEEP_COLUMNS = [
    "profile_name",
    "drawdown_penalty",
    "volatility_penalty",
    "risk_penalty_power",
    "target_tier_0_fraction",
    "target_tier_25_fraction",
    "target_tier_50_fraction",
    "target_tier_100_fraction",
    "passes_partial_target_support",
    "selected",
]


def _profile_name(
    *,
    drawdown_penalty: float,
    volatility_penalty: float,
    risk_penalty_power: float,
) -> str:
    return (
        f"dd{drawdown_penalty:g}_vol{volatility_penalty:g}_"
        f"power{risk_penalty_power:g}"
    )


def build_phase8_target_profile_sweep(
    rows: pd.DataFrame,
    *,
    target_type: str = "allocation_utility",
    cost_bps: float = 0.0,
    drawdown_penalties: Sequence[float] = (0.50, 0.75, 1.00),
    volatility_penalties: Sequence[float] = (0.25,),
    risk_penalty_powers: Sequence[float] = (2.0, 2.5),
    min_partial_target_fraction: float = 0.05,
) -> pd.DataFrame:
    if target_type != "allocation_utility":
        raise ValueError("Phase 8 target-profile sweep requires allocation_utility targets.")

    sweep_rows: list[dict[str, object]] = []
    for drawdown_penalty in drawdown_penalties:
        for volatility_penalty in volatility_penalties:
            for risk_penalty_power in risk_penalty_powers:
                profiled = apply_allocation_utility_profile(
                    rows,
                    target_type=target_type,
                    cost_bps=cost_bps,
                    drawdown_penalty=float(drawdown_penalty),
                    volatility_penalty=float(volatility_penalty),
                    risk_penalty_power=float(risk_penalty_power),
                )
                fractions = {
                    tier: float(
                        profiled["target_weight"].sub(tier).abs().le(1e-9).mean()
                    )
                    for tier in ALLOCATION_TIERS
                }
                passes_support = (
                    fractions[0.0] > 0.0
                    and fractions[0.25] >= float(min_partial_target_fraction)
                    and fractions[0.50] >= float(min_partial_target_fraction)
                    and fractions[1.0] > 0.0
                )
                sweep_rows.append(
                    {
                        "profile_name": _profile_name(
                            drawdown_penalty=float(drawdown_penalty),
                            volatility_penalty=float(volatility_penalty),
                            risk_penalty_power=float(risk_penalty_power),
                        ),
                        "drawdown_penalty": float(drawdown_penalty),
                        "volatility_penalty": float(volatility_penalty),
                        "risk_penalty_power": float(risk_penalty_power),
                        "target_tier_0_fraction": fractions[0.0],
                        "target_tier_25_fraction": fractions[0.25],
                        "target_tier_50_fraction": fractions[0.50],
                        "target_tier_100_fraction": fractions[1.0],
                        "passes_partial_target_support": passes_support,
                        "selected": False,
                    }
                )

    sweep = pd.DataFrame(sweep_rows, columns=PHASE8_TARGET_PROFILE_SWEEP_COLUMNS)
    passing = sweep.index[sweep["passes_partial_target_support"].astype(bool)]
    if len(passing) > 0:
        sweep.loc[passing[0], "selected"] = True
    return sweep


def write_phase8_target_profile_sweep(
    rows: pd.DataFrame,
    *,
    config: ExperimentConfig,
    output_path: str | Path = DEFAULT_OUTPUT_PATH,
) -> Path:
    resolved_output_path = Path(output_path)
    resolved_output_path.parent.mkdir(parents=True, exist_ok=True)
    build_phase8_target_profile_sweep(
        rows,
        target_type=config.target.type,
        cost_bps=config.portfolio.costs.bps_per_trade,
        min_partial_target_fraction=(
            config.evaluation.strict_research_gate.min_partial_target_fraction
        ),
    ).to_csv(resolved_output_path, index=False)
    return resolved_output_path
