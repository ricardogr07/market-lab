from __future__ import annotations

import pandas as pd
import pytest

from marketlab.config import ExperimentConfig
from marketlab.reports.phase8_target_profile_sweep import (
    build_phase8_target_profile_sweep,
    write_phase8_target_profile_sweep,
)


def _rows() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "forward_return": [-0.10, 0.10],
            "forward_drawdown": [-0.10, 0.0],
            "forward_realized_volatility": [0.0, 0.0],
        }
    )


def test_target_profile_sweep_selects_first_passing_profile_deterministically() -> None:
    sweep = build_phase8_target_profile_sweep(
        _rows(),
        drawdown_penalties=[0.50, 0.75],
        volatility_penalties=[0.25],
        risk_penalty_powers=[2.0],
        min_partial_target_fraction=0.0,
    )

    assert sweep["profile_name"].tolist() == [
        "dd0.5_vol0.25_power2",
        "dd0.75_vol0.25_power2",
    ]
    assert sweep["passes_partial_target_support"].tolist() == [True, True]
    assert sweep["selected"].tolist() == [True, False]


def test_write_target_profile_sweep_uses_strict_gate_support_threshold(tmp_path) -> None:
    config = ExperimentConfig()
    config.target.type = "allocation_utility"
    config.portfolio.costs.bps_per_trade = 0.0
    config.evaluation.strict_research_gate.min_partial_target_fraction = 0.0
    output_path = tmp_path / "target-profile-sweep.csv"

    written_path = write_phase8_target_profile_sweep(
        _rows(),
        config=config,
        output_path=output_path,
    )

    written = pd.read_csv(written_path)
    assert written_path == output_path
    assert written["selected"].sum() == 1
    assert written.iloc[0]["profile_name"] == "dd0.5_vol0.25_power2"


def test_target_profile_sweep_rejects_non_allocation_target() -> None:
    with pytest.raises(ValueError, match="requires allocation_utility"):
        build_phase8_target_profile_sweep(_rows(), target_type="binary")
