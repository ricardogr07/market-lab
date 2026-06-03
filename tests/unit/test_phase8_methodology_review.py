from __future__ import annotations

from pathlib import Path

import pandas as pd

from marketlab.cli import main
from marketlab.reports.phase8_methodology import (
    build_phase8_methodology_review,
    write_phase8_methodology_review,
)


def _review_passed(
    review: pd.DataFrame,
    *,
    methodology_gate: str,
    metric: str,
    section: str | None = None,
) -> bool:
    mask = review["methodology_gate"].eq(methodology_gate) & review["metric"].eq(metric)
    if section is not None:
        mask &= review["section"].eq(section)
    return bool(review.loc[mask, "passed"].iloc[0])


def _write_methodology_artifacts(run_dir: Path) -> None:
    run_dir.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "condition": "sharpe_like_matches_or_improves",
                "passed": True,
                "observed": 0.4,
                "required": ">= 0",
            },
            {
                "condition": "max_drawdown_matches_or_improves",
                "passed": True,
                "observed": 0.3,
                "required": ">= 0",
            },
            {
                "condition": "average_exposure_in_range",
                "passed": True,
                "observed": 0.44,
                "required": "0.2 to 0.85",
            },
            {
                "condition": "annualized_turnover_budget",
                "passed": True,
                "observed": 8.0,
                "required": "<= 24",
            },
            {
                "condition": "multiple_selected_walk_forward_folds",
                "passed": True,
                "observed": 10,
                "required": ">= 2",
            },
            {
                "condition": "selected_walk_forward_fold_fraction",
                "passed": False,
                "observed": 0.67,
                "required": ">= 0.75",
            },
            {
                "condition": "partial_target_25_global_fraction",
                "passed": False,
                "observed": 0.04,
                "required": ">= 0.05",
            },
            {
                "condition": "predicted_target_25_global_fraction",
                "passed": True,
                "observed": 0.30,
                "required": ">= 0.03",
            },
            {
                "condition": "net_cumulative_return_beats_buy_hold",
                "passed": False,
                "observed": -5.0,
                "required": "> 0",
            },
            {
                "condition": "overall",
                "passed": False,
                "observed": "",
                "required": "all conditions pass",
            },
        ]
    ).to_csv(run_dir / "strict_research_gate.csv", index=False)
    pd.DataFrame(
        [
            {"strategy": "buy_hold", "cumulative_return": 12.0},
            {"strategy": "btc_static_25", "cumulative_return": 10.0},
            {"strategy": "btc_rebalanced_25", "cumulative_return": 1.0},
            {
                "strategy": "ml_indicator_tuned__long_only__cash",
                "cumulative_return": 7.0,
            },
        ]
    ).to_csv(run_dir / "strategy_summary.csv", index=False)
    pd.DataFrame(
        [
            {
                "section": "score_deciles",
                "metric": "score_target_weight_correlation",
                "value": 0.1,
                "detail": "selected OOS prediction rows",
            },
            {
                "section": "score_deciles",
                "metric": "score_forward_return_correlation",
                "value": -0.2,
                "detail": "selected OOS prediction rows",
            },
            {
                "section": "score_deciles",
                "metric": "score_realized_utility_correlation",
                "value": 0.3,
                "detail": "selected OOS prediction rows",
            },
            {
                "section": "score_deciles",
                "metric": "predicted_tier_100_fraction",
                "value": 0.1,
                "detail": "selected OOS prediction rows",
            },
            {
                "section": "model_family_support",
                "metric": "any_selected_oos_predicted_tier_100",
                "value": True,
                "detail": "selected OOS prediction rows",
            },
        ]
    ).to_csv(run_dir / "phase8_score_diagnostic_summary.csv", index=False)
    pd.DataFrame(
        [
            {
                "section": "runtime_participation",
                "metric": "gate_bull_average_long_exposure",
                "value": 0.61,
                "detail": "rows=100",
            },
            {
                "section": "bull_active_return",
                "metric": "gate_bull_active_return_sum",
                "value": -1.5,
                "detail": "daily excess_return sum",
            },
            {
                "section": "bull_active_return",
                "metric": "gate_bull_underexposed_positive_benchmark_return_sum",
                "value": 8.0,
                "detail": "missed positive BTC return",
            },
            {
                "section": "selection_context",
                "metric": "selected_fold_fraction",
                "value": 0.73,
                "detail": "selected=11; total=15",
            },
        ]
    ).to_csv(run_dir / "phase8_bull_participation_summary.csv", index=False)
    pd.DataFrame(
        [
            {
                "scenario": "actual_runtime",
                "condition": "overall",
                "passed": False,
                "observed": "",
                "required": "all diagnostic conditions pass",
                "diagnostic_only": True,
                "detail": "",
            },
            {
                "scenario": "force_runtime_bull_100",
                "condition": "overall",
                "passed": True,
                "observed": "",
                "required": "all diagnostic conditions pass",
                "diagnostic_only": True,
                "detail": "",
            },
        ]
    ).to_csv(run_dir / "phase8_bull_counterfactual_gate.csv", index=False)


def test_build_phase8_methodology_review_classifies_current_research_gaps(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    _write_methodology_artifacts(run_dir)

    review = build_phase8_methodology_review(run_dir)

    assert _review_passed(
        review,
        methodology_gate="deployment_gate",
        section="strict_gate",
        metric="overall",
    ) is False
    assert _review_passed(
        review,
        methodology_gate="risk_allocation_gate",
        section="summary",
        metric="overall",
    ) is True
    assert _review_passed(
        review,
        methodology_gate="benchmark_family",
        metric="active_return_vs_buy_hold",
    ) is False
    assert _review_passed(
        review,
        methodology_gate="benchmark_family",
        metric="active_return_vs_btc_rebalanced_25",
    ) is True
    assert _review_passed(
        review,
        methodology_gate="signal_validity_gate",
        metric="score_forward_return_correlation",
    ) is False
    assert _review_passed(
        review,
        methodology_gate="bull_participation_gate",
        metric="gate_bull_active_return_sum",
    ) is False

    counterfactual = review.loc[
        review["methodology_gate"].eq("counterfactual_hypothesis")
        & review["section"].eq("force_runtime_bull_100")
        & review["metric"].eq("overall")
    ].iloc[0]
    assert bool(counterfactual["passed"]) is True
    assert bool(counterfactual["diagnostic_only"]) is True


def test_build_phase8_methodology_review_handles_missing_artifacts(tmp_path: Path) -> None:
    review = build_phase8_methodology_review(tmp_path / "missing-run")

    assert _review_passed(
        review,
        methodology_gate="artifact_presence",
        metric="strict_research_gate.csv",
    ) is False
    assert _review_passed(
        review,
        methodology_gate="deployment_gate",
        section="strict_gate",
        metric="overall",
    ) is False
    counterfactual = review.loc[
        review["methodology_gate"].eq("counterfactual_hypothesis")
        & review["metric"].eq("counterfactual_gate_present")
    ].iloc[0]
    assert bool(counterfactual["diagnostic_only"]) is True


def test_phase8_methodology_review_cli_writes_output(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    output_path = tmp_path / "methodology.csv"
    _write_methodology_artifacts(run_dir)

    assert (
        main(
            [
                "phase8-methodology-review",
                "--run-dir",
                str(run_dir),
                "--output",
                str(output_path),
            ]
        )
        == 0
    )

    assert output_path.exists()


def test_write_phase8_methodology_review_defaults_to_run_directory(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    _write_methodology_artifacts(run_dir)

    output_path = write_phase8_methodology_review(run_dir)

    assert output_path == run_dir / "phase8_methodology_review.csv"
    assert output_path.exists()
