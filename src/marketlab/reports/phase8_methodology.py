from __future__ import annotations

import math
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pandas as pd

from marketlab.reports.phase8_gates import (
    calculate_bull_participation_gate,
    calculate_signal_validity_gate,
)

STRATEGY_NAME = "ml_indicator_tuned__long_only__cash"

PHASE8_METHODOLOGY_REVIEW_COLUMNS = [
    "methodology_gate",
    "section",
    "metric",
    "passed",
    "value",
    "required",
    "diagnostic_only",
    "source_artifact",
    "detail",
]


def _read_csv(run_dir: Path, name: str) -> pd.DataFrame:
    path = run_dir / name
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if pd.isna(value):
        return False
    return str(value).strip().lower() in {"1", "true", "t", "yes", "y"}


def _numeric(value: Any) -> float:
    try:
        numeric_value = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return numeric_value if math.isfinite(numeric_value) else float("nan")


def _append(
    rows: list[dict[str, object]],
    *,
    methodology_gate: str,
    section: str,
    metric: str,
    passed: bool,
    value: object = pd.NA,
    required: object = "",
    diagnostic_only: bool = False,
    source_artifact: str = "",
    detail: str = "",
) -> None:
    rows.append(
        {
            "methodology_gate": methodology_gate,
            "section": section,
            "metric": metric,
            "passed": bool(passed),
            "value": value,
            "required": required,
            "diagnostic_only": bool(diagnostic_only),
            "source_artifact": source_artifact,
            "detail": detail,
        }
    )


def _artifact_presence_rows(
    rows: list[dict[str, object]],
    *,
    run_dir: Path,
) -> None:
    for name in (
        "strict_research_gate.csv",
        "phase8_run_summary.csv",
        "strategy_summary.csv",
        "phase8_score_diagnostic_summary.csv",
        "phase8_bull_participation_summary.csv",
        "phase8_bull_counterfactual_gate.csv",
    ):
        _append(
            rows,
            methodology_gate="artifact_presence",
            section="inputs",
            metric=name,
            passed=(run_dir / name).exists(),
            value="present" if (run_dir / name).exists() else "missing",
            required="present when this diagnostic is available",
            source_artifact=name,
        )


def _strict_condition(strict_gate: pd.DataFrame, condition: str) -> pd.Series | None:
    if strict_gate.empty or "condition" not in strict_gate.columns:
        return None
    matches = strict_gate.loc[strict_gate["condition"].astype(str).eq(condition)]
    if matches.empty:
        return None
    return matches.iloc[0]


def _append_strict_condition(
    rows: list[dict[str, object]],
    *,
    strict_gate: pd.DataFrame,
    methodology_gate: str,
    section: str,
    condition: str,
    source_artifact: str = "strict_research_gate.csv",
) -> bool:
    row = _strict_condition(strict_gate, condition)
    if row is None:
        _append(
            rows,
            methodology_gate=methodology_gate,
            section=section,
            metric=condition,
            passed=False,
            value="missing",
            required="strict-gate condition present",
            source_artifact=source_artifact,
        )
        return False
    passed = _truthy(row.get("passed"))
    _append(
        rows,
        methodology_gate=methodology_gate,
        section=section,
        metric=condition,
        passed=passed,
        value=row.get("observed", pd.NA),
        required=row.get("required", ""),
        source_artifact=source_artifact,
    )
    return passed


def _deployment_gate_rows(rows: list[dict[str, object]], strict_gate: pd.DataFrame) -> None:
    if strict_gate.empty:
        _append(
            rows,
            methodology_gate="deployment_gate",
            section="strict_gate",
            metric="overall",
            passed=False,
            value="missing",
            required="strict_research_gate.csv overall row",
            source_artifact="strict_research_gate.csv",
        )
        return

    overall = _strict_condition(strict_gate, "overall")
    if overall is not None:
        _append(
            rows,
            methodology_gate="deployment_gate",
            section="strict_gate",
            metric="overall",
            passed=_truthy(overall.get("passed")),
            value=overall.get("observed", ""),
            required=overall.get("required", ""),
            source_artifact="strict_research_gate.csv",
            detail="unchanged Phase 8 blocker for any Phase 9 BTC deployment",
        )

    if "passed" not in strict_gate.columns:
        return
    failed = strict_gate.loc[~strict_gate["passed"].map(_truthy)].copy()
    for _, row in failed.iterrows():
        condition = str(row.get("condition", "unknown"))
        if condition == "overall":
            continue
        _append(
            rows,
            methodology_gate="deployment_gate",
            section="failed_strict_condition",
            metric=condition,
            passed=False,
            value=row.get("observed", pd.NA),
            required=row.get("required", ""),
            source_artifact="strict_research_gate.csv",
        )


def _benchmark_family_rows(
    rows: list[dict[str, object]],
    strategy_summary: pd.DataFrame,
) -> dict[str, bool]:
    family_passes: dict[str, bool] = {
        "buy_hold": False,
        "static_partial": False,
        "rebalanced_partial": False,
    }
    if strategy_summary.empty or not {"strategy", "cumulative_return"}.issubset(
        strategy_summary.columns
    ):
        for family in family_passes:
            _append(
                rows,
                methodology_gate="benchmark_family",
                section=family,
                metric=f"{family}_passed",
                passed=False,
                value="missing",
                required="strategy_summary.csv with cumulative_return",
                source_artifact="strategy_summary.csv",
            )
        return family_passes

    strategy_rows = strategy_summary.loc[
        strategy_summary["strategy"].astype(str).eq(STRATEGY_NAME)
    ]
    if strategy_rows.empty:
        strategy_rows = strategy_summary.loc[
            strategy_summary["strategy"].astype(str).str.startswith("ml_")
        ]
    if strategy_rows.empty:
        for family in family_passes:
            _append(
                rows,
                methodology_gate="benchmark_family",
                section=family,
                metric=f"{family}_passed",
                passed=False,
                value="missing",
                required=STRATEGY_NAME,
                source_artifact="strategy_summary.csv",
            )
        return family_passes

    strategy_return = _numeric(strategy_rows.iloc[0].get("cumulative_return"))
    families = {
        "buy_hold": lambda name: name == "buy_hold",
        "static_partial": lambda name: name.startswith("btc_static_"),
        "rebalanced_partial": lambda name: name.startswith("btc_rebalanced_"),
    }
    for family, matcher in families.items():
        family_results: list[bool] = []
        for _, row in strategy_summary.iterrows():
            benchmark_name = str(row.get("strategy", ""))
            if benchmark_name == STRATEGY_NAME or not matcher(benchmark_name):
                continue
            benchmark_return = _numeric(row.get("cumulative_return"))
            delta = strategy_return - benchmark_return
            passed = math.isfinite(delta) and delta > 0.0
            family_results.append(passed)
            _append(
                rows,
                methodology_gate="benchmark_family",
                section=family,
                metric=f"active_return_vs_{benchmark_name}",
                passed=passed,
                value=delta if math.isfinite(delta) else "missing",
                required="> 0",
                source_artifact="strategy_summary.csv",
                detail=f"{STRATEGY_NAME} cumulative_return minus {benchmark_name}",
            )
        family_passed = bool(family_results) and all(family_results)
        family_passes[family] = family_passed
        _append(
            rows,
            methodology_gate="benchmark_family",
            section=family,
            metric=f"{family}_passed",
            passed=family_passed,
            value=sum(1 for value in family_results if value),
            required=f"{len(family_results)} benchmarks beat",
            source_artifact="strategy_summary.csv",
        )
    return family_passes


def _risk_allocation_gate_rows(
    rows: list[dict[str, object]],
    *,
    strict_gate: pd.DataFrame,
    family_passes: dict[str, bool],
) -> None:
    condition_passes = [
        _append_strict_condition(
            rows,
            strict_gate=strict_gate,
            methodology_gate="risk_allocation_gate",
            section="risk_profile",
            condition=condition,
        )
        for condition in (
            "sharpe_like_matches_or_improves",
            "max_drawdown_matches_or_improves",
            "average_exposure_in_range",
            "annualized_turnover_budget",
        )
    ]
    rebalanced_passed = bool(family_passes.get("rebalanced_partial", False))
    _append(
        rows,
        methodology_gate="risk_allocation_gate",
        section="benchmark_family",
        metric="rebalanced_partial_passed",
        passed=rebalanced_passed,
        value=rebalanced_passed,
        required="beat all configured rebalanced BTC/cash benchmarks",
        source_artifact="strategy_summary.csv",
    )
    overall = all(condition_passes) and rebalanced_passed
    _append(
        rows,
        methodology_gate="risk_allocation_gate",
        section="summary",
        metric="overall",
        passed=overall,
        value=overall,
        required="risk profile and rebalanced benchmark family pass",
        detail="diagnostic risk-allocation view; does not replace strict deployment gate",
    )


def _selection_gate_rows(rows: list[dict[str, object]], strict_gate: pd.DataFrame) -> None:
    passes = [
        _append_strict_condition(
            rows,
            strict_gate=strict_gate,
            methodology_gate="selection_coverage_gate",
            section="walk_forward",
            condition=condition,
        )
        for condition in (
            "multiple_selected_walk_forward_folds",
            "selected_walk_forward_fold_fraction",
        )
    ]
    overall = all(passes)
    _append(
        rows,
        methodology_gate="selection_coverage_gate",
        section="summary",
        metric="overall",
        passed=overall,
        value=overall,
        required="walk-forward selection coverage passes",
    )


def _target_support_gate_rows(rows: list[dict[str, object]], strict_gate: pd.DataFrame) -> None:
    if strict_gate.empty or "condition" not in strict_gate.columns:
        _append(
            rows,
            methodology_gate="target_support_gate",
            section="summary",
            metric="overall",
            passed=False,
            value="missing",
            required="partial and predicted target support rows",
            source_artifact="strict_research_gate.csv",
        )
        return

    support_rows = strict_gate.loc[
        strict_gate["condition"].astype(str).str.startswith(("partial_target_", "predicted_target_"))
    ].copy()
    if support_rows.empty:
        _append(
            rows,
            methodology_gate="target_support_gate",
            section="summary",
            metric="overall",
            passed=False,
            value="missing",
            required="partial and predicted target support rows",
            source_artifact="strict_research_gate.csv",
        )
        return

    passes: list[bool] = []
    for _, row in support_rows.iterrows():
        condition = str(row.get("condition", "unknown"))
        passed = _truthy(row.get("passed"))
        passes.append(passed)
        section = "predicted_support" if condition.startswith("predicted_") else "target_support"
        _append(
            rows,
            methodology_gate="target_support_gate",
            section=section,
            metric=condition,
            passed=passed,
            value=row.get("observed", pd.NA),
            required=row.get("required", ""),
            source_artifact="strict_research_gate.csv",
        )

    overall = all(passes)
    _append(
        rows,
        methodology_gate="target_support_gate",
        section="summary",
        metric="overall",
        passed=overall,
        value=overall,
        required="all configured target and predicted-tier support rows pass",
    )


def _summary_value(summary: pd.DataFrame, metric: str) -> pd.Series | None:
    if summary.empty or "metric" not in summary.columns:
        return None
    matches = summary.loc[summary["metric"].astype(str).eq(metric)]
    if matches.empty:
        return None
    return matches.iloc[0]


def _append_numeric_summary_condition(
    rows: list[dict[str, object]],
    *,
    summary: pd.DataFrame,
    methodology_gate: str,
    section: str,
    metric: str,
    required: str,
    predicate: Callable[[float], bool],
    source_artifact: str,
) -> bool:
    row = _summary_value(summary, metric)
    if row is None:
        _append(
            rows,
            methodology_gate=methodology_gate,
            section=section,
            metric=metric,
            passed=False,
            value="missing",
            required=required,
            source_artifact=source_artifact,
        )
        return False
    value = _numeric(row.get("value"))
    passed = math.isfinite(value) and bool(predicate(value))
    _append(
        rows,
        methodology_gate=methodology_gate,
        section=section,
        metric=metric,
        passed=passed,
        value=value if math.isfinite(value) else "missing",
        required=required,
        source_artifact=source_artifact,
        detail=str(row.get("detail", "")),
    )
    return passed


def _append_bool_summary_condition(
    rows: list[dict[str, object]],
    *,
    summary: pd.DataFrame,
    methodology_gate: str,
    section: str,
    metric: str,
    source_artifact: str,
) -> bool:
    row = _summary_value(summary, metric)
    if row is None:
        _append(
            rows,
            methodology_gate=methodology_gate,
            section=section,
            metric=metric,
            passed=False,
            value="missing",
            required=True,
            source_artifact=source_artifact,
        )
        return False
    passed = _truthy(row.get("value"))
    _append(
        rows,
        methodology_gate=methodology_gate,
        section=section,
        metric=metric,
        passed=passed,
        value=passed,
        required=True,
        source_artifact=source_artifact,
        detail=str(row.get("detail", "")),
    )
    return passed


def _signal_validity_gate_rows(rows: list[dict[str, object]], score_summary: pd.DataFrame) -> None:
    source = "phase8_score_diagnostic_summary.csv"
    if score_summary.empty:
        _append(
            rows,
            methodology_gate="signal_validity_gate",
            section="summary",
            metric="overall",
            passed=False,
            value="missing",
            required=source,
            source_artifact=source,
        )
        return

    values = {
        metric: (
            _summary_value(score_summary, metric).get("value")
            if _summary_value(score_summary, metric) is not None
            else pd.NA
        )
        for metric in (
            "score_target_weight_correlation",
            "score_forward_return_correlation",
            "score_realized_utility_correlation",
            "predicted_tier_100_fraction",
            "any_selected_oos_predicted_tier_100",
        )
    }
    gate = calculate_signal_validity_gate(values)
    for metric in (
        "score_target_weight_correlation",
        "score_forward_return_correlation",
        "score_realized_utility_correlation",
    ):
        _append_numeric_summary_condition(
            rows,
            summary=score_summary,
            methodology_gate="signal_validity_gate",
            section="score_relationship",
            metric=metric,
            required="> 0",
            predicate=lambda value: value > 0.0,
            source_artifact=source,
        )
    _append_numeric_summary_condition(
        rows,
        summary=score_summary,
        methodology_gate="signal_validity_gate",
        section="tier_support",
        metric="predicted_tier_100_fraction",
        required="> 0",
        predicate=lambda value: value > 0.0,
        source_artifact=source,
    )
    _append_bool_summary_condition(
        rows,
        summary=score_summary,
        methodology_gate="signal_validity_gate",
        section="tier_support",
        metric="any_selected_oos_predicted_tier_100",
        source_artifact=source,
    )
    _append(
        rows,
        methodology_gate="signal_validity_gate",
        section="summary",
        metric="overall",
        passed=gate.passed,
        value=gate.passed,
        required="score relationships and selected OOS 100% support pass",
        source_artifact=source,
    )


def _bull_participation_gate_rows(
    rows: list[dict[str, object]],
    bull_summary: pd.DataFrame,
) -> None:
    source = "phase8_bull_participation_summary.csv"
    if bull_summary.empty:
        _append(
            rows,
            methodology_gate="bull_participation_gate",
            section="summary",
            metric="overall",
            passed=False,
            value="missing",
            required=source,
            source_artifact=source,
        )
        return

    values = {
        metric: (
            _summary_value(bull_summary, metric).get("value")
            if _summary_value(bull_summary, metric) is not None
            else pd.NA
        )
        for metric in (
            "gate_bull_average_long_exposure",
            "gate_bull_active_return_sum",
            "gate_bull_underexposed_positive_benchmark_return_sum",
            "selected_fold_fraction",
        )
    }
    gate = calculate_bull_participation_gate(values)
    for section, metric, required, predicate in (
        (
            "exposure",
            "gate_bull_average_long_exposure",
            ">= 0.50",
            lambda value: value >= 0.50,
        ),
        (
            "active_return",
            "gate_bull_active_return_sum",
            "> 0",
            lambda value: value > 0.0,
        ),
        (
            "missed_upside",
            "gate_bull_underexposed_positive_benchmark_return_sum",
            "<= 0",
            lambda value: value <= 0.0,
        ),
        (
            "selection_context",
            "selected_fold_fraction",
            ">= 0.75",
            lambda value: value >= 0.75,
        ),
    ):
        _append_numeric_summary_condition(
            rows,
            summary=bull_summary,
            methodology_gate="bull_participation_gate",
            section=section,
            metric=metric,
            required=required,
            predicate=predicate,
            source_artifact=source,
        )
    _append(
        rows,
        methodology_gate="bull_participation_gate",
        section="summary",
        metric="overall",
        passed=gate.passed,
        value=gate.passed,
        required="bull exposure, bull active return, missed-upside, and fold coverage pass",
        source_artifact=source,
    )


def _counterfactual_rows(rows: list[dict[str, object]], counterfactual_gate: pd.DataFrame) -> None:
    source = "phase8_bull_counterfactual_gate.csv"
    if counterfactual_gate.empty:
        _append(
            rows,
            methodology_gate="counterfactual_hypothesis",
            section="artifact",
            metric="counterfactual_gate_present",
            passed=False,
            value="missing",
            required=source,
            diagnostic_only=True,
            source_artifact=source,
        )
        return
    if not {"scenario", "condition", "passed"}.issubset(counterfactual_gate.columns):
        _append(
            rows,
            methodology_gate="counterfactual_hypothesis",
            section="artifact",
            metric="counterfactual_gate_schema",
            passed=False,
            value="missing",
            required="scenario, condition, passed columns",
            diagnostic_only=True,
            source_artifact=source,
        )
        return

    overall_rows = counterfactual_gate.loc[
        counterfactual_gate["condition"].astype(str).eq("overall")
    ]
    any_passing = False
    for _, row in overall_rows.iterrows():
        scenario = str(row.get("scenario", "unknown"))
        passed = _truthy(row.get("passed"))
        any_passing = any_passing or passed
        _append(
            rows,
            methodology_gate="counterfactual_hypothesis",
            section=scenario,
            metric="overall",
            passed=passed,
            value=passed,
            required=row.get("required", "all diagnostic conditions pass"),
            diagnostic_only=True,
            source_artifact=source,
            detail="diagnostic-only; does not approve deployment",
        )
    _append(
        rows,
        methodology_gate="counterfactual_hypothesis",
        section="summary",
        metric="counterfactual_pass_available",
        passed=any_passing,
        value=any_passing,
        required=True,
        diagnostic_only=True,
        source_artifact=source,
        detail="passing counterfactuals are hypotheses for validation-selected rules only",
    )


def build_phase8_methodology_review(run_dir: str | Path) -> pd.DataFrame:
    resolved_run_dir = Path(run_dir)
    strict_gate = _read_csv(resolved_run_dir, "strict_research_gate.csv")
    strategy_summary = _read_csv(resolved_run_dir, "strategy_summary.csv")
    score_summary = _read_csv(resolved_run_dir, "phase8_score_diagnostic_summary.csv")
    bull_summary = _read_csv(resolved_run_dir, "phase8_bull_participation_summary.csv")
    counterfactual_gate = _read_csv(resolved_run_dir, "phase8_bull_counterfactual_gate.csv")

    rows: list[dict[str, object]] = []
    _artifact_presence_rows(rows, run_dir=resolved_run_dir)
    _deployment_gate_rows(rows, strict_gate)
    family_passes = _benchmark_family_rows(rows, strategy_summary)
    _risk_allocation_gate_rows(
        rows,
        strict_gate=strict_gate,
        family_passes=family_passes,
    )
    _selection_gate_rows(rows, strict_gate)
    _target_support_gate_rows(rows, strict_gate)
    _signal_validity_gate_rows(rows, score_summary)
    _bull_participation_gate_rows(rows, bull_summary)
    _counterfactual_rows(rows, counterfactual_gate)
    return pd.DataFrame(rows, columns=PHASE8_METHODOLOGY_REVIEW_COLUMNS)


def write_phase8_methodology_review(
    run_dir: str | Path,
    output_path: str | Path | None = None,
) -> Path:
    resolved_run_dir = Path(run_dir)
    resolved_output_path = (
        Path(output_path)
        if output_path is not None
        else resolved_run_dir / "phase8_methodology_review.csv"
    )
    resolved_output_path.parent.mkdir(parents=True, exist_ok=True)
    build_phase8_methodology_review(resolved_run_dir).to_csv(resolved_output_path, index=False)
    return resolved_output_path
