from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import pandas as pd

PHASE8_RUN_SUMMARY_COLUMNS = ["section", "metric", "value", "detail"]
STRATEGY_NAME = "ml_indicator_tuned__long_only__cash"
ALLOCATION_TIERS = (0.0, 0.25, 0.50, 1.0)


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


def _tier_label(weight: float) -> str:
    return f"{int(round(weight * 100))}"


def _nearest_tier(value: float) -> float:
    if not math.isfinite(value):
        return 0.0
    return min(ALLOCATION_TIERS, key=lambda tier: (abs(tier - value), tier))


def _append(
    rows: list[dict[str, object]],
    *,
    section: str,
    metric: str,
    value: object,
    detail: str = "",
) -> None:
    rows.append(
        {
            "section": section,
            "metric": metric,
            "value": value,
            "detail": detail,
        }
    )


def _strict_gate_rows(rows: list[dict[str, object]], strict_gate: pd.DataFrame) -> None:
    if strict_gate.empty:
        _append(
            rows,
            section="strict_gate",
            metric="strict_gate_present",
            value=False,
            detail="strict_research_gate.csv was not found",
        )
        return

    passed = (
        strict_gate["passed"].map(_truthy)
        if "passed" in strict_gate.columns
        else pd.Series([False] * len(strict_gate), index=strict_gate.index)
    )
    if "condition" in strict_gate.columns:
        overall = strict_gate.loc[strict_gate["condition"].astype(str).eq("overall")]
    else:
        overall = pd.DataFrame()
    overall_passed = bool(passed.loc[overall.index].iloc[0]) if not overall.empty else False
    _append(
        rows,
        section="strict_gate",
        metric="strict_gate_overall",
        value=overall_passed,
        detail="overall row in strict_research_gate.csv",
    )

    failed = strict_gate.loc[~passed].copy()
    _append(
        rows,
        section="strict_gate",
        metric="failed_strict_gate_rows",
        value=int(len(failed)),
        detail="failed rows including overall",
    )
    for _, row in failed.iterrows():
        condition = str(row.get("condition", "unknown"))
        observed = row.get("observed", "")
        required = row.get("required", "")
        _append(
            rows,
            section="strict_gate",
            metric="failed_strict_gate_row",
            value=condition,
            detail=f"observed={observed}; required={required}",
        )


def _selection_rows(rows: list[dict[str, object]], selections: pd.DataFrame) -> None:
    if selections.empty:
        _append(
            rows,
            section="fold_selection",
            metric="selected_fold_fraction",
            value=float("nan"),
            detail="ml_strategy_tuning_selections.csv was not found",
        )
        return

    statuses = selections["selection_status"].astype(str) if "selection_status" in selections.columns else pd.Series(dtype=str)
    total_folds = int(len(selections))
    selected_folds = int(statuses.eq("selected").sum())
    no_valid_folds = int(statuses.eq("no_valid_candidate").sum())
    selected_fraction = selected_folds / total_folds if total_folds else float("nan")
    _append(
        rows,
        section="fold_selection",
        metric="selected_fold_fraction",
        value=selected_fraction,
        detail=f"selected={selected_folds}; total={total_folds}",
    )
    _append(
        rows,
        section="fold_selection",
        metric="no_valid_candidate_folds",
        value=no_valid_folds,
        detail="folds that stayed cash because no candidate was selected",
    )
    if "selected_regime_gate_bull_floor" in selections.columns:
        values = (
            selections.loc[statuses.eq("selected"), "selected_regime_gate_bull_floor"]
            .map(_numeric)
            .dropna()
        )
        if not values.empty:
            _append(
                rows,
                section="fold_selection",
                metric="selected_regime_gate_bull_floor_mode",
                value=float(values.mode().iloc[0]),
                detail="mode among selected walk-forward folds",
            )
    if "selected_allocation_score_transform" in selections.columns:
        selected = selections.loc[statuses.eq("selected")].copy()
        selected_transforms = (
            selected["selected_allocation_score_transform"].dropna().astype(str)
        )
        if not selected_transforms.empty:
            mode = selected_transforms.mode().iloc[0]
            _append(
                rows,
                section="score_transform",
                metric="selected_score_transform_mode",
                value=mode,
                detail="mode among selected walk-forward folds",
            )
            for transform_name, count in selected_transforms.value_counts().sort_index().items():
                _append(
                    rows,
                    section="score_transform",
                    metric=f"selected_score_transform_{transform_name}_folds",
                    value=int(count),
                    detail="selected walk-forward fold count",
                )
    if "validation_score_policy_repair_authorized" in selections.columns:
        selected = selections.loc[statuses.eq("selected")]
        if not selected.empty:
            _append(
                rows,
                section="score_policy_repair",
                metric="selected_validation_score_policy_repair_authorized_fraction",
                value=float(
                    selected["validation_score_policy_repair_authorized"]
                    .map(_truthy)
                    .mean()
                ),
                detail="selected walk-forward folds authorized by raw validation score ordering",
            )
    if "validation_guarded_gate_bull_risk_off_override_authorized" in selections.columns:
        selected = selections.loc[statuses.eq("selected")]
        if not selected.empty:
            _append(
                rows,
                section="guarded_gate_bull_risk_off_override",
                metric="selected_validation_guarded_gate_bull_risk_off_override_authorized_fraction",
                value=float(
                    selected[
                        "validation_guarded_gate_bull_risk_off_override_authorized"
                    ]
                    .map(_truthy)
                    .mean()
                ),
                detail="selected walk-forward folds authorized by raw validation score ordering",
            )


def _fallback_candidate_reasons(row: pd.Series) -> list[str]:
    reasons: list[str] = []
    if "active_candidate" in row.index and not _truthy(row["active_candidate"]):
        reasons.append("inactive_candidate")
    if _numeric(row.get("excess_cumulative_return")) <= 0.0:
        reasons.append("non_positive_buy_hold_excess")
    if _numeric(row.get("min_benchmark_excess_cumulative_return")) <= 0.0:
        reasons.append("non_positive_required_benchmark_excess")
    if (
        _numeric(
            row.get(
                "min_selection_validation_cost_benchmark_excess_cumulative_return"
            )
        )
        <= 0.0
    ):
        reasons.append("non_positive_validation_cost_benchmark_excess")
    if _numeric(row.get("min_validation_predicted_target_fraction")) <= 0.0:
        reasons.append("missing_predicted_partial_tier_support")
    if _numeric(row.get("sharpe_like_delta")) <= 0.0 and _numeric(row.get("drawdown_delta")) < 0.0:
        reasons.append("risk_not_improved")
    return reasons or ["failed_other"]


def _candidate_reason_rows(rows: list[dict[str, object]], candidates: pd.DataFrame) -> None:
    if candidates.empty:
        _append(
            rows,
            section="candidate_rejections",
            metric="candidate_rows",
            value=0,
            detail="ml_strategy_tuning_candidates.csv was not found",
        )
        return

    passed = (
        candidates["passed_gate"].map(_truthy)
        if "passed_gate" in candidates.columns
        else pd.Series([False] * len(candidates), index=candidates.index)
    )
    failed = candidates.loc[~passed].copy()
    reason_counts: dict[str, int] = {}
    if "failure_reasons" in failed.columns:
        for value in failed["failure_reasons"].fillna("failed_other"):
            reasons = [reason.strip() for reason in str(value).split(";") if reason.strip()]
            for reason in reasons or ["failed_other"]:
                reason_counts[reason] = reason_counts.get(reason, 0) + 1
    else:
        for _, row in failed.iterrows():
            for reason in _fallback_candidate_reasons(row):
                reason_counts[reason] = reason_counts.get(reason, 0) + 1

    _append(
        rows,
        section="candidate_rejections",
        metric="failed_candidate_rows",
        value=int(len(failed)),
        detail="validation candidates with passed_gate=False",
    )
    for reason, count in sorted(reason_counts.items()):
        _append(
            rows,
            section="candidate_rejections",
            metric=f"candidate_rejection_reason_{reason}",
            value=int(count),
            detail="counted per failed candidate reason",
        )
    for column in (
        "validation_score_forward_return_correlation",
        "validation_raw_score_forward_return_correlation",
        "validation_score_target_correlation",
        "validation_gate_bull_average_exposure",
        "validation_gate_bull_underexposed_positive_benchmark_fraction",
        "validation_gate_bull_underexposed_positive_benchmark_return_sum",
        "min_selection_validation_cost_benchmark_excess_cumulative_return",
    ):
        if column not in candidates.columns:
            continue
        values = candidates[column].map(_numeric).dropna()
        if values.empty:
            continue
        _append(
            rows,
            section="score_validity",
            metric=f"{column}_mean",
            value=float(values.mean()),
            detail="validation candidate mean",
        )
        _append(
            rows,
            section="score_validity",
            metric=f"{column}_min",
            value=float(values.min()),
            detail="validation candidate minimum",
        )
    if "validation_score_forward_return_correlation" in candidates.columns:
        correlations = candidates["validation_score_forward_return_correlation"].map(
            _numeric
        )
        _append(
            rows,
            section="score_validity",
            metric="negative_validation_score_forward_return_correlation_candidates",
            value=int(correlations.lt(0.0).sum()),
            detail="validation candidates with score/forward-return correlation < 0",
        )
    if "validation_score_policy_repair_authorized" in candidates.columns:
        _append(
            rows,
            section="score_policy_repair",
            metric="validation_score_policy_repair_authorized_fraction",
            value=float(
                candidates["validation_score_policy_repair_authorized"].map(_truthy).mean()
            ),
            detail="validation candidates authorized by raw score/forward-return ordering",
        )
    if "score_policy_repair_denied_reason" in candidates.columns:
        denied_reasons = candidates["score_policy_repair_denied_reason"].dropna().astype(str)
        denied_reasons = denied_reasons.loc[denied_reasons.str.strip().ne("")]
        for reason, count in denied_reasons.value_counts().sort_index().items():
            _append(
                rows,
                section="score_policy_repair",
                metric=f"score_policy_repair_denied_reason_{reason}",
                value=int(count),
                detail="validation candidate denial count",
            )
    if "validation_guarded_gate_bull_risk_off_override_authorized" in candidates.columns:
        _append(
            rows,
            section="guarded_gate_bull_risk_off_override",
            metric="validation_guarded_gate_bull_risk_off_override_authorized_fraction",
            value=float(
                candidates[
                    "validation_guarded_gate_bull_risk_off_override_authorized"
                ]
                .map(_truthy)
                .mean()
            ),
            detail="validation candidates authorized by raw score/forward-return ordering",
        )
    if "guarded_gate_bull_risk_off_override_denied_reason" in candidates.columns:
        denied_reasons = (
            candidates["guarded_gate_bull_risk_off_override_denied_reason"]
            .dropna()
            .astype(str)
        )
        denied_reasons = denied_reasons.loc[denied_reasons.str.strip().ne("")]
        for reason, count in denied_reasons.value_counts().sort_index().items():
            _append(
                rows,
                section="guarded_gate_bull_risk_off_override",
                metric=f"guarded_gate_bull_risk_off_override_denied_reason_{reason}",
                value=int(count),
                detail="validation candidate denial count",
            )


def _target_support_rows(rows: list[dict[str, object]], target_diagnostics: pd.DataFrame) -> None:
    if target_diagnostics.empty or not {"scope", "target_weight", "row_fraction"}.issubset(target_diagnostics.columns):
        _append(
            rows,
            section="target_support",
            metric="target_support_present",
            value=False,
            detail="allocation_target_diagnostics.csv was not found or lacks support columns",
        )
        return

    global_rows = target_diagnostics.loc[target_diagnostics["scope"].astype(str).eq("global")]
    for weight in ALLOCATION_TIERS:
        support = global_rows.loc[
            global_rows["target_weight"].map(_numeric).sub(weight).abs().le(1e-9)
        ]
        value = float(support["row_fraction"].sum()) if not support.empty else 0.0
        _append(
            rows,
            section="target_support",
            metric=f"target_tier_{_tier_label(weight)}_global_fraction",
            value=value,
            detail="realized training target support",
        )


def _predicted_support_rows(rows: list[dict[str, object]], probability_diagnostics: pd.DataFrame) -> None:
    if probability_diagnostics.empty:
        _append(
            rows,
            section="predicted_support",
            metric="predicted_support_present",
            value=False,
            detail="allocation_probability_diagnostics.csv was not found",
        )
        return

    if "predicted_tier_weight" in probability_diagnostics.columns:
        tiers = probability_diagnostics["predicted_tier_weight"].map(_numeric)
    elif "score" in probability_diagnostics.columns:
        tiers = probability_diagnostics["score"].map(lambda value: _nearest_tier(_numeric(value)))
    else:
        _append(
            rows,
            section="predicted_support",
            metric="predicted_support_present",
            value=False,
            detail="no predicted_tier_weight or score column was found",
        )
        return

    total_rows = max(1, int(tiers.notna().sum()))
    for weight in ALLOCATION_TIERS:
        fraction = float(tiers.sub(weight).abs().le(1e-9).sum() / total_rows)
        _append(
            rows,
            section="predicted_support",
            metric=f"predicted_tier_{_tier_label(weight)}_fraction",
            value=fraction,
            detail="outer OOS predicted tier support",
        )
    if "score_policy_triggered_100" in probability_diagnostics.columns:
        _append(
            rows,
            section="score_policy_repair",
            metric="score_policy_triggered_100_fraction",
            value=float(probability_diagnostics["score_policy_triggered_100"].map(_truthy).mean()),
            detail="outer OOS rows promoted to the 100% tier by score policy",
        )
    if "score_policy_repair_authorized" in probability_diagnostics.columns:
        _append(
            rows,
            section="score_policy_repair",
            metric="score_policy_repair_authorized_fraction",
            value=float(
                probability_diagnostics["score_policy_repair_authorized"].map(_truthy).mean()
            ),
            detail="outer OOS rows carrying validation-authorized repair context",
        )
    if "guarded_gate_bull_risk_off_override_triggered" in probability_diagnostics.columns:
        _append(
            rows,
            section="guarded_gate_bull_risk_off_override",
            metric="guarded_gate_bull_risk_off_override_triggered_fraction",
            value=float(
                probability_diagnostics[
                    "guarded_gate_bull_risk_off_override_triggered"
                ]
                .map(_truthy)
                .mean()
            ),
            detail="outer OOS rows lifted to 100% after the risk-off cap",
        )
    if "guarded_gate_bull_risk_off_override_authorized" in probability_diagnostics.columns:
        _append(
            rows,
            section="guarded_gate_bull_risk_off_override",
            metric="guarded_gate_bull_risk_off_override_authorized_fraction",
            value=float(
                probability_diagnostics[
                    "guarded_gate_bull_risk_off_override_authorized"
                ]
                .map(_truthy)
                .mean()
            ),
            detail="outer OOS rows carrying validation-authorized override context",
        )


def _benchmark_delta_rows(rows: list[dict[str, object]], strategy_summary: pd.DataFrame) -> None:
    if strategy_summary.empty or not {"strategy", "cumulative_return"}.issubset(strategy_summary.columns):
        _append(
            rows,
            section="benchmark_deltas",
            metric="benchmark_deltas_present",
            value=False,
            detail="strategy_summary.csv was not found or lacks cumulative_return",
        )
        return

    strategy_rows = strategy_summary.loc[strategy_summary["strategy"].astype(str).eq(STRATEGY_NAME)]
    if strategy_rows.empty:
        strategy_rows = strategy_summary.loc[strategy_summary["strategy"].astype(str).str.startswith("ml_")]
    if strategy_rows.empty:
        return
    strategy_return = _numeric(strategy_rows.iloc[0]["cumulative_return"])
    for _, row in strategy_summary.iterrows():
        benchmark_name = str(row.get("strategy", ""))
        if benchmark_name == STRATEGY_NAME:
            continue
        if not (
            benchmark_name == "buy_hold"
            or benchmark_name.startswith("btc_static_")
            or benchmark_name.startswith("btc_rebalanced_")
        ):
            continue
        benchmark_return = _numeric(row.get("cumulative_return"))
        _append(
            rows,
            section="benchmark_deltas",
            metric=f"active_return_vs_{benchmark_name}",
            value=strategy_return - benchmark_return,
            detail=f"{STRATEGY_NAME} cumulative_return minus benchmark",
        )


def _regime_slice_rows(rows: list[dict[str, object]], regime_slices: pd.DataFrame) -> None:
    if regime_slices.empty or not {"slice_name", "active_return"}.issubset(regime_slices.columns):
        _append(
            rows,
            section="regime_slices",
            metric="regime_slices_present",
            value=False,
            detail="regime_slice_diagnostics.csv was not found or lacks active_return",
        )
        return

    for _, row in regime_slices.iterrows():
        slice_name = str(row.get("slice_name", "unknown"))
        _append(
            rows,
            section="regime_slices",
            metric=f"{slice_name}_active_return",
            value=_numeric(row.get("active_return")),
            detail="strategy cumulative_return minus BTC benchmark in regime slice",
        )


def build_phase8_run_summary(run_dir: str | Path) -> pd.DataFrame:
    resolved_run_dir = Path(run_dir)
    rows: list[dict[str, object]] = []
    _strict_gate_rows(rows, _read_csv(resolved_run_dir, "strict_research_gate.csv"))
    _selection_rows(rows, _read_csv(resolved_run_dir, "ml_strategy_tuning_selections.csv"))
    _candidate_reason_rows(rows, _read_csv(resolved_run_dir, "ml_strategy_tuning_candidates.csv"))
    _target_support_rows(rows, _read_csv(resolved_run_dir, "allocation_target_diagnostics.csv"))
    _predicted_support_rows(
        rows,
        _read_csv(resolved_run_dir, "allocation_probability_diagnostics.csv"),
    )
    _benchmark_delta_rows(rows, _read_csv(resolved_run_dir, "strategy_summary.csv"))
    _regime_slice_rows(rows, _read_csv(resolved_run_dir, "regime_slice_diagnostics.csv"))
    return pd.DataFrame(rows, columns=PHASE8_RUN_SUMMARY_COLUMNS)


def write_phase8_run_summary(
    run_dir: str | Path,
    output_path: str | Path | None = None,
) -> Path:
    resolved_run_dir = Path(run_dir)
    resolved_output_path = Path(output_path) if output_path is not None else resolved_run_dir / "phase8_run_summary.csv"
    resolved_output_path.parent.mkdir(parents=True, exist_ok=True)
    build_phase8_run_summary(resolved_run_dir).to_csv(resolved_output_path, index=False)
    return resolved_output_path
