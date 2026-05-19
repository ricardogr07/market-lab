from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from marketlab.cli import main
from marketlab.reports.phase8_selection_probe import (
    build_phase8_selection_probe,
    write_phase8_selection_probe,
)


def _candidate(
    *,
    fold_id: int,
    model_name: str,
    passed_gate: bool,
    failure_reasons: str,
    min_benchmark_excess: float,
    excess_return: float = 0.10,
    active_candidate: bool = True,
    predicted_support: float = 0.20,
    sharpe_delta: float = 0.05,
    drawdown_delta: float = 0.05,
    turnover: float = 4.0,
    average_exposure: float = 0.40,
) -> dict[str, object]:
    return {
        "fold_id": fold_id,
        "model_name": model_name,
        "allocation_mode": "direct_tiered",
        "utility_profile": "partial_p25",
        "rolling_train_bars": 730,
        "min_holding_period_bars": 18,
        "hysteresis_margin": 0.02,
        "regime_policy": "bull100_sideways25",
        "threshold": 0.0,
        "tier_min_threshold": pd.NA,
        "tier_half_threshold": pd.NA,
        "tier_full_threshold": pd.NA,
        "cumulative_return": 0.20,
        "excess_cumulative_return": excess_return,
        "min_benchmark_excess_cumulative_return": min_benchmark_excess,
        "average_exposure": average_exposure,
        "annualized_turnover": turnover,
        "min_validation_predicted_target_fraction": predicted_support,
        "sharpe_like_delta": sharpe_delta,
        "drawdown_delta": drawdown_delta,
        "selection_benchmark_excess_cumulative_returns": (
            f"buy_hold:{excess_return};btc_rebalanced_25:{min_benchmark_excess}"
        ),
        "active_candidate": active_candidate,
        "failure_reasons": failure_reasons,
        "passed_gate": passed_gate,
    }


def _write_candidates(run_dir: Path, rows: list[dict[str, object]]) -> None:
    run_dir.mkdir(parents=True)
    pd.DataFrame(rows).to_csv(run_dir / "ml_strategy_tuning_candidates.csv", index=False)


def test_build_phase8_selection_probe_classifies_profiles(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    _write_candidates(
        run_dir,
        [
            _candidate(
                fold_id=1,
                model_name="strict_low",
                passed_gate=True,
                failure_reasons="",
                min_benchmark_excess=0.05,
                excess_return=0.08,
            ),
            _candidate(
                fold_id=1,
                model_name="strict_high",
                passed_gate=True,
                failure_reasons="",
                min_benchmark_excess=0.10,
                excess_return=0.12,
            ),
            _candidate(
                fold_id=2,
                model_name="benchmark_shortfall",
                passed_gate=False,
                failure_reasons="non_positive_required_benchmark_excess",
                min_benchmark_excess=-0.20,
                excess_return=0.25,
            ),
            _candidate(
                fold_id=3,
                model_name="inactive",
                passed_gate=False,
                failure_reasons="inactive_candidate;non_positive_required_benchmark_excess",
                min_benchmark_excess=-0.05,
                active_candidate=False,
            ),
            _candidate(
                fold_id=4,
                model_name="high_turnover",
                passed_gate=False,
                failure_reasons="turnover_budget_exceeded",
                min_benchmark_excess=0.20,
                turnover=30.0,
            ),
        ],
    )

    selections, summary = build_phase8_selection_probe(run_dir)
    strict_summary = summary.loc[summary["probe_profile"].eq("strict")].iloc[0]
    tolerance_010 = summary.loc[
        summary["probe_profile"].eq("benchmark_tolerance")
        & summary["benchmark_tolerance"].eq(0.10)
    ].iloc[0]
    tolerance_025 = summary.loc[
        summary["probe_profile"].eq("benchmark_tolerance")
        & summary["benchmark_tolerance"].eq(0.25)
    ].iloc[0]
    fallback_summary = summary.loc[
        summary["probe_profile"].eq("best_active_fallback")
    ].iloc[0]
    control_summary = summary.loc[
        summary["probe_profile"].eq("turnover_only_control")
    ].iloc[0]

    assert strict_summary["selected_fold_fraction"] == pytest.approx(0.25)
    assert tolerance_010["selected_fold_ids"] == "1"
    assert tolerance_025["selected_fold_ids"] == "1,2"
    assert fallback_summary["selected_fold_ids"] == "1,2"
    assert bool(control_summary["diagnostic_only"]) is True
    assert control_summary["selected_fold_ids"] == "1,2,3"

    fallback_fold_2 = selections.loc[
        selections["probe_profile"].eq("best_active_fallback")
        & selections["fold_id"].eq(2)
    ].iloc[0]
    assert fallback_fold_2["selection_status"] == "fallback_selected"
    assert fallback_fold_2["selection_source"] == "best_active_fallback"
    assert fallback_fold_2["selected_model_name"] == "benchmark_shortfall"


def test_build_phase8_selection_probe_handles_missing_candidates(tmp_path: Path) -> None:
    selections, summary = build_phase8_selection_probe(tmp_path / "missing-run")

    assert selections.empty
    assert summary.iloc[0]["probe_profile"] == "missing_candidates"
    assert summary.iloc[0]["candidate_rows"] == 0
    assert "ml_strategy_tuning_candidates.csv" in summary.iloc[0]["selection_rule"]


def test_phase8_selection_probe_cli_writes_outputs(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    output_dir = tmp_path / "probe"
    _write_candidates(
        run_dir,
        [
            _candidate(
                fold_id=1,
                model_name="strict",
                passed_gate=True,
                failure_reasons="",
                min_benchmark_excess=0.10,
            )
        ],
    )

    assert (
        main(
            [
                "phase8-selection-probe",
                "--run-dir",
                str(run_dir),
                "--output-dir",
                str(output_dir),
            ]
        )
        == 0
    )

    assert (output_dir / "phase8_selection_probe.csv").exists()
    assert (output_dir / "phase8_selection_probe_summary.csv").exists()
    summary = pd.read_csv(output_dir / "phase8_selection_probe_summary.csv")
    assert "best_active_fallback" in set(summary["probe_profile"])


def test_write_phase8_selection_probe_defaults_to_run_directory(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    _write_candidates(
        run_dir,
        [
            _candidate(
                fold_id=1,
                model_name="strict",
                passed_gate=True,
                failure_reasons="",
                min_benchmark_excess=0.10,
            )
        ],
    )

    selections_path, summary_path = write_phase8_selection_probe(run_dir)

    assert selections_path == run_dir / "phase8_selection_probe.csv"
    assert summary_path == run_dir / "phase8_selection_probe_summary.csv"
    assert selections_path.exists()
    assert summary_path.exists()
