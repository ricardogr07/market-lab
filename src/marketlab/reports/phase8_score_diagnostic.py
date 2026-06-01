from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import pandas as pd

ALLOCATION_TIERS = (0.0, 0.25, 0.50, 1.0)
PHASE8_SCORE_DIAGNOSTIC_COLUMNS = [
    "section",
    "metric",
    "group",
    "subgroup",
    "value",
    "row_count",
    "detail",
]
PHASE8_SCORE_DIAGNOSTIC_SUMMARY_COLUMNS = ["section", "metric", "value", "detail"]


def _read_csv(run_dir: Path, name: str) -> pd.DataFrame:
    path = run_dir / name
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def _numeric(value: Any) -> float:
    try:
        numeric_value = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return numeric_value if math.isfinite(numeric_value) else float("nan")


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if pd.isna(value):
        return False
    return str(value).strip().lower() in {"1", "true", "t", "yes", "y"}


def _tier_label(value: float) -> str:
    return f"{int(round(value * 100))}"


def _append_detail(
    rows: list[dict[str, object]],
    *,
    section: str,
    metric: str,
    value: object,
    group: object = "",
    subgroup: object = "",
    row_count: object = pd.NA,
    detail: str = "",
) -> None:
    rows.append(
        {
            "section": section,
            "metric": metric,
            "group": group,
            "subgroup": subgroup,
            "value": value,
            "row_count": row_count,
            "detail": detail,
        }
    )


def _append_summary(
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


def _probability_with_tiers(probability: pd.DataFrame) -> pd.DataFrame:
    working = probability.copy()
    for column in (
        "score",
        "target_weight",
        "predicted_tier_weight",
        "forward_return",
        "realized_utility",
        "prob_tier_100",
    ):
        if column in working.columns:
            working[column] = working[column].map(_numeric)
    if "runtime_regime" not in working.columns:
        working["runtime_regime"] = "missing"
    return working


def _score_decile_rows(
    *,
    detail_rows: list[dict[str, object]],
    summary_rows: list[dict[str, object]],
    probability: pd.DataFrame,
) -> None:
    if probability.empty:
        _append_summary(
            summary_rows,
            section="score_deciles",
            metric="allocation_probability_present",
            value=False,
            detail="allocation_probability_diagnostics.csv was not found",
        )
        return
    if "score" not in probability.columns:
        _append_summary(
            summary_rows,
            section="score_deciles",
            metric="score_present",
            value=False,
            detail="allocation_probability_diagnostics.csv lacks score",
        )
        return

    working = _probability_with_tiers(probability).dropna(subset=["score"]).copy()
    if working.empty:
        _append_summary(
            summary_rows,
            section="score_deciles",
            metric="score_rows",
            value=0,
            detail="score column has no numeric rows",
        )
        return

    ranked = working["score"].rank(method="first")
    decile_count = min(10, len(working))
    working["score_decile"] = pd.qcut(
        ranked,
        q=decile_count,
        labels=False,
        duplicates="drop",
    ).astype(int) + 1

    for decile, group in working.groupby("score_decile", sort=True):
        score = group["score"]
        _append_detail(
            detail_rows,
            section="score_deciles",
            metric="score_min",
            group=int(decile),
            value=float(score.min()),
            row_count=len(group),
        )
        _append_detail(
            detail_rows,
            section="score_deciles",
            metric="score_max",
            group=int(decile),
            value=float(score.max()),
            row_count=len(group),
        )
        _append_detail(
            detail_rows,
            section="score_deciles",
            metric="score_mean",
            group=int(decile),
            value=float(score.mean()),
            row_count=len(group),
        )
        for column in ("forward_return", "realized_utility", "prob_tier_100"):
            if column in group.columns:
                values = group[column].dropna()
                if not values.empty:
                    _append_detail(
                        detail_rows,
                        section="score_deciles",
                        metric=f"{column}_mean",
                        group=int(decile),
                        value=float(values.mean()),
                        row_count=len(group),
                    )
        if "target_weight" in group.columns:
            target_100 = group["target_weight"].sub(1.0).abs().le(1e-9)
            _append_detail(
                detail_rows,
                section="score_deciles",
                metric="target_tier_100_fraction",
                group=int(decile),
                value=float(target_100.mean()),
                row_count=len(group),
            )
        if "predicted_tier_weight" in group.columns:
            predicted_100 = group["predicted_tier_weight"].sub(1.0).abs().le(1e-9)
            _append_detail(
                detail_rows,
                section="score_deciles",
                metric="predicted_tier_100_fraction",
                group=int(decile),
                value=float(predicted_100.mean()),
                row_count=len(group),
            )

    for column in ("target_weight", "forward_return", "realized_utility"):
        if column in working.columns:
            value = float(working["score"].corr(working[column]))
            _append_summary(
                summary_rows,
                section="score_deciles",
                metric=f"score_{column}_correlation",
                value=value,
                detail="selected OOS prediction rows",
            )

    _append_summary(
        summary_rows,
        section="score_deciles",
        metric="score_max",
        value=float(working["score"].max()),
        detail="selected OOS prediction rows",
    )
    if "predicted_tier_weight" in working.columns:
        predicted_100 = working["predicted_tier_weight"].sub(1.0).abs().le(1e-9)
        _append_summary(
            summary_rows,
            section="score_deciles",
            metric="predicted_tier_100_fraction",
            value=float(predicted_100.mean()),
            detail="selected OOS prediction rows",
        )
    if "allocation_score_transform" in working.columns:
        transforms = working["allocation_score_transform"].dropna().astype(str)
        if not transforms.empty:
            _append_summary(
                summary_rows,
                section="score_transform",
                metric="selected_score_transform_mode",
                value=transforms.mode().iloc[0],
                detail="mode among selected OOS prediction rows",
            )
    if "score_transform_applied" in working.columns:
        _append_summary(
            summary_rows,
            section="score_transform",
            metric="score_transform_applied_fraction",
            value=float(working["score_transform_applied"].map(_truthy).mean()),
            detail="selected OOS prediction rows",
        )
    if "score_policy_triggered_100" in working.columns:
        _append_summary(
            summary_rows,
            section="score_policy_repair",
            metric="score_policy_triggered_100_fraction",
            value=float(working["score_policy_triggered_100"].map(_truthy).mean()),
            detail="selected OOS rows promoted to the 100% tier by score policy",
        )
    if "score_policy_repair_authorized" in working.columns:
        _append_summary(
            summary_rows,
            section="score_policy_repair",
            metric="score_policy_repair_authorized_fraction",
            value=float(working["score_policy_repair_authorized"].map(_truthy).mean()),
            detail="selected OOS rows carrying validation-authorized repair context",
        )
    if "guarded_gate_bull_risk_off_override_triggered" in working.columns:
        _append_summary(
            summary_rows,
            section="guarded_gate_bull_risk_off_override",
            metric="guarded_gate_bull_risk_off_override_triggered_fraction",
            value=float(
                working["guarded_gate_bull_risk_off_override_triggered"]
                .map(_truthy)
                .mean()
            ),
            detail="selected OOS rows lifted to 100% after the risk-off cap",
        )
    if "guarded_gate_bull_risk_off_override_authorized" in working.columns:
        _append_summary(
            summary_rows,
            section="guarded_gate_bull_risk_off_override",
            metric="guarded_gate_bull_risk_off_override_authorized_fraction",
            value=float(
                working["guarded_gate_bull_risk_off_override_authorized"]
                .map(_truthy)
                .mean()
            ),
            detail="selected OOS rows carrying validation-authorized override context",
        )


def _confusion_rows_for_group(
    *,
    detail_rows: list[dict[str, object]],
    group_name: str,
    group: pd.DataFrame,
) -> None:
    if not {"target_weight", "predicted_tier_weight"}.issubset(group.columns):
        return
    counts = (
        group.groupby(["target_weight", "predicted_tier_weight"], dropna=False)
        .size()
        .rename("row_count")
        .reset_index()
    )
    totals = group.groupby("target_weight", dropna=False).size().rename("total")
    counts = counts.merge(totals, on="target_weight", how="left")
    for _, row in counts.iterrows():
        target = _numeric(row["target_weight"])
        predicted = _numeric(row["predicted_tier_weight"])
        total = int(row["total"])
        count = int(row["row_count"])
        _append_detail(
            detail_rows,
            section="target_prediction_confusion",
            metric="target_vs_predicted_tier_fraction",
            group=group_name,
            subgroup=(
                f"{_tier_label(target)}_to_{_tier_label(predicted)}"
                if math.isfinite(target) and math.isfinite(predicted)
                else "missing"
            ),
            value=count / total if total else float("nan"),
            row_count=count,
            detail="fraction normalized within target tier for this group",
        )


def _confusion_rows(
    *,
    detail_rows: list[dict[str, object]],
    summary_rows: list[dict[str, object]],
    probability: pd.DataFrame,
) -> None:
    if probability.empty:
        return
    working = _probability_with_tiers(probability)
    if not {"target_weight", "predicted_tier_weight"}.issubset(working.columns):
        _append_summary(
            summary_rows,
            section="target_prediction_confusion",
            metric="confusion_present",
            value=False,
            detail="target_weight or predicted_tier_weight is missing",
        )
        return

    _confusion_rows_for_group(
        detail_rows=detail_rows,
        group_name="all",
        group=working,
    )
    if "fold_id" in working.columns:
        for fold_id, group in working.groupby("fold_id", sort=True):
            _confusion_rows_for_group(
                detail_rows=detail_rows,
                group_name=f"fold_{fold_id}",
                group=group,
            )
    if "runtime_regime" in working.columns:
        for regime, group in working.groupby("runtime_regime", sort=True):
            _confusion_rows_for_group(
                detail_rows=detail_rows,
                group_name=f"runtime_{regime}",
                group=group,
            )

    exact_match = working["target_weight"].sub(working["predicted_tier_weight"]).abs().le(1e-9)
    _append_summary(
        summary_rows,
        section="target_prediction_confusion",
        metric="exact_tier_match_fraction",
        value=float(exact_match.mean()),
        detail="selected OOS prediction rows",
    )


def _candidate_selected_mask(candidates: pd.DataFrame, selections: pd.DataFrame) -> pd.Series:
    if candidates.empty or selections.empty:
        return pd.Series([False] * len(candidates), index=candidates.index)

    selected = selections.loc[
        selections.get("selection_status", pd.Series(dtype=str)).astype(str).eq("selected")
    ].copy()
    if selected.empty:
        return pd.Series([False] * len(candidates), index=candidates.index)

    pairs = [
        ("fold_id", "fold_id"),
        ("model_name", "selected_model_name"),
        ("utility_profile", "selected_utility_profile"),
        ("rolling_train_bars", "selected_rolling_train_bars"),
        ("min_holding_period_bars", "selected_min_holding_period_bars"),
        ("hysteresis_margin", "selected_hysteresis_margin"),
        ("regime_policy", "selected_regime_policy"),
        ("regime_gate_bull_floor", "selected_regime_gate_bull_floor"),
        ("allocation_score_transform", "selected_allocation_score_transform"),
        (
            "score_transform_bull_multiplier",
            "selected_score_transform_bull_multiplier",
        ),
        ("score_transform_bull_addend", "selected_score_transform_bull_addend"),
        (
            "score_transform_risk_off_score_cap",
            "selected_score_transform_risk_off_score_cap",
        ),
        (
            "score_transform_non_bull_score_cap",
            "selected_score_transform_non_bull_score_cap",
        ),
        ("threshold", "selected_threshold"),
        ("tier_min_threshold", "selected_tier_min_threshold"),
        ("tier_half_threshold", "selected_tier_half_threshold"),
        ("tier_full_threshold", "selected_tier_full_threshold"),
    ]
    usable_pairs = [
        pair for pair in pairs if pair[0] in candidates.columns and pair[1] in selected.columns
    ]
    if len(usable_pairs) < 2:
        return pd.Series([False] * len(candidates), index=candidates.index)

    def key_from_row(row: pd.Series, mapping: list[tuple[str, str]], side: int) -> tuple[str, ...]:
        values: list[str] = []
        for left, right in mapping:
            value = row[left if side == 0 else right]
            if pd.isna(value):
                values.append("NA")
            else:
                numeric = _numeric(value)
                values.append(f"{numeric:.10g}" if math.isfinite(numeric) else str(value))
        return tuple(values)

    selected_keys = {
        key_from_row(row, usable_pairs, 1)
        for _, row in selected.iterrows()
    }
    return candidates.apply(
        lambda row: key_from_row(row, usable_pairs, 0) in selected_keys,
        axis=1,
    )


def _candidate_rows(
    *,
    detail_rows: list[dict[str, object]],
    summary_rows: list[dict[str, object]],
    candidates: pd.DataFrame,
    selections: pd.DataFrame,
) -> None:
    if candidates.empty:
        _append_summary(
            summary_rows,
            section="candidate_score_support",
            metric="candidate_artifact_present",
            value=False,
            detail="ml_strategy_tuning_candidates.csv was not found",
        )
        return

    working = candidates.copy()
    working["selected_runtime_candidate"] = _candidate_selected_mask(working, selections)
    if "passed_gate" in working.columns:
        working["passed_gate_bool"] = working["passed_gate"].map(_truthy)
    else:
        working["passed_gate_bool"] = False

    for column in (
        "validation_predicted_25_fraction",
        "validation_predicted_50_fraction",
        "validation_predicted_100_fraction",
        "min_validation_predicted_target_fraction",
        "min_benchmark_excess_cumulative_return",
        "validation_score_forward_return_correlation",
        "validation_raw_score_forward_return_correlation",
        "validation_score_target_correlation",
        "validation_gate_bull_average_exposure",
        "validation_gate_bull_underexposed_positive_benchmark_fraction",
        "validation_gate_bull_underexposed_positive_benchmark_return_sum",
        "min_selection_validation_cost_benchmark_excess_cumulative_return",
        "validation_score_transform_applied_fraction",
    ):
        if column in working.columns:
            working[column] = working[column].map(_numeric)

    for selected_flag, group in working.groupby("selected_runtime_candidate", sort=True):
        group_name = "selected" if bool(selected_flag) else "not_selected"
        _append_detail(
            detail_rows,
            section="candidate_score_support",
            metric="candidate_count",
            group=group_name,
            value=int(len(group)),
            row_count=int(len(group)),
        )
        for column in (
            "validation_predicted_25_fraction",
            "validation_predicted_50_fraction",
            "validation_predicted_100_fraction",
            "min_validation_predicted_target_fraction",
            "min_benchmark_excess_cumulative_return",
            "validation_score_forward_return_correlation",
            "validation_raw_score_forward_return_correlation",
            "validation_score_target_correlation",
            "validation_gate_bull_average_exposure",
            "validation_gate_bull_underexposed_positive_benchmark_fraction",
            "validation_gate_bull_underexposed_positive_benchmark_return_sum",
            "min_selection_validation_cost_benchmark_excess_cumulative_return",
            "validation_score_transform_applied_fraction",
        ):
            if column in group.columns:
                values = group[column].dropna()
                if not values.empty:
                    _append_detail(
                        detail_rows,
                        section="candidate_score_support",
                        metric=f"{column}_mean",
                        group=group_name,
                        value=float(values.mean()),
                        row_count=int(len(group)),
                    )
        _append_detail(
            detail_rows,
            section="candidate_score_support",
            metric="strict_pass_fraction",
            group=group_name,
            value=float(group["passed_gate_bool"].mean()),
            row_count=int(len(group)),
        )

    if "model_name" in working.columns:
        for model_name, group in working.groupby("model_name", sort=True):
            _append_detail(
                detail_rows,
                section="model_family_support",
                metric="candidate_count",
                group=model_name,
                value=int(len(group)),
                row_count=int(len(group)),
            )
            _append_detail(
                detail_rows,
                section="model_family_support",
                metric="strict_pass_count",
                group=model_name,
                value=int(group["passed_gate_bool"].sum()),
                row_count=int(len(group)),
            )
            if "validation_predicted_100_fraction" in group.columns:
                support = group["validation_predicted_100_fraction"].dropna()
                value = float(support.max()) if not support.empty else float("nan")
                _append_detail(
                    detail_rows,
                    section="model_family_support",
                    metric="max_validation_predicted_100_fraction",
                    group=model_name,
                    value=value,
                    row_count=int(len(group)),
                )

    _append_summary(
        summary_rows,
        section="candidate_score_support",
        metric="selected_candidate_rows",
        value=int(working["selected_runtime_candidate"].sum()),
        detail="matched by selected fold/model/tuning fields",
    )
    _append_summary(
        summary_rows,
        section="model_family_support",
        metric="candidate_validation_predicted_100_available",
        value="validation_predicted_100_fraction" in working.columns,
        detail="current candidate artifacts may only persist required partial-tier support",
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
        if column not in working.columns:
            continue
        values = working[column].dropna()
        if values.empty:
            continue
        _append_summary(
            summary_rows,
            section="candidate_score_validity",
            metric=f"{column}_mean",
            value=float(values.mean()),
            detail="validation candidate mean",
        )
        _append_summary(
            summary_rows,
            section="candidate_score_validity",
            metric=f"{column}_min",
            value=float(values.min()),
            detail="validation candidate minimum",
        )
    if "validation_score_forward_return_correlation" in working.columns:
        correlations = working["validation_score_forward_return_correlation"]
        _append_summary(
            summary_rows,
            section="candidate_score_validity",
            metric="negative_validation_score_forward_return_correlation_candidates",
            value=int(correlations.lt(0.0).sum()),
            detail="validation candidates with score/forward-return correlation < 0",
        )
    if "validation_score_policy_repair_authorized" in working.columns:
        _append_summary(
            summary_rows,
            section="score_policy_repair",
            metric="validation_score_policy_repair_authorized_fraction",
            value=float(
                working["validation_score_policy_repair_authorized"].map(_truthy).mean()
            ),
            detail="validation candidates authorized by raw score/forward-return ordering",
        )
    if "validation_guarded_gate_bull_risk_off_override_authorized" in working.columns:
        _append_summary(
            summary_rows,
            section="guarded_gate_bull_risk_off_override",
            metric="validation_guarded_gate_bull_risk_off_override_authorized_fraction",
            value=float(
                working[
                    "validation_guarded_gate_bull_risk_off_override_authorized"
                ]
                .map(_truthy)
                .mean()
            ),
            detail="validation candidates authorized by raw score/forward-return ordering",
        )


def _model_family_oos_rows(
    *,
    detail_rows: list[dict[str, object]],
    summary_rows: list[dict[str, object]],
    probability: pd.DataFrame,
) -> None:
    if probability.empty or "model_name" not in probability.columns:
        return
    working = _probability_with_tiers(probability)
    if "predicted_tier_weight" not in working.columns:
        return
    for model_name, group in working.groupby("model_name", sort=True):
        predicted_100 = group["predicted_tier_weight"].sub(1.0).abs().le(1e-9)
        _append_detail(
            detail_rows,
            section="model_family_support",
            metric="oos_predicted_tier_100_fraction",
            group=model_name,
            value=float(predicted_100.mean()),
            row_count=int(len(group)),
            detail="selected OOS prediction rows",
        )
        if "score" in group.columns:
            _append_detail(
                detail_rows,
                section="model_family_support",
                metric="oos_score_max",
                group=model_name,
                value=float(group["score"].max()),
                row_count=int(len(group)),
                detail="selected OOS prediction rows",
            )
    any_100 = bool(working["predicted_tier_weight"].sub(1.0).abs().le(1e-9).any())
    _append_summary(
        summary_rows,
        section="model_family_support",
        metric="any_selected_oos_predicted_tier_100",
        value=any_100,
        detail="selected OOS prediction rows",
    )


def _validation_oos_rows(
    *,
    detail_rows: list[dict[str, object]],
    summary_rows: list[dict[str, object]],
    probability: pd.DataFrame,
    selections: pd.DataFrame,
) -> None:
    if probability.empty or selections.empty or "fold_id" not in probability.columns:
        _append_summary(
            summary_rows,
            section="validation_oos_stability",
            metric="stability_present",
            value=False,
            detail="requires allocation_probability_diagnostics.csv and selections",
        )
        return
    if "predicted_tier_weight" not in probability.columns:
        _append_summary(
            summary_rows,
            section="validation_oos_stability",
            metric="stability_present",
            value=False,
            detail="predicted_tier_weight is missing",
        )
        return

    working = _probability_with_tiers(probability)
    fold_rows: list[dict[str, object]] = []
    for fold_id, group in working.groupby("fold_id", sort=True):
        row: dict[str, object] = {"fold_id": fold_id}
        for tier in ALLOCATION_TIERS:
            row[f"oos_predicted_{_tier_label(tier)}_fraction"] = float(
                group["predicted_tier_weight"].sub(tier).abs().le(1e-9).mean()
            )
        if "score" in group.columns:
            row["oos_score_mean"] = float(group["score"].mean())
        fold_rows.append(row)
    oos = pd.DataFrame(fold_rows)
    selected = selections.copy()
    selected = selected.loc[
        selected.get("selection_status", pd.Series(dtype=str)).astype(str).eq("selected")
    ]
    if selected.empty:
        return
    joined = selected.merge(oos, on="fold_id", how="left")
    abs_deltas: dict[str, list[float]] = {"25": [], "50": []}
    for _, row in joined.iterrows():
        fold_id = row.get("fold_id")
        for tier_label, validation_column in {
            "25": "validation_predicted_25_fraction",
            "50": "validation_predicted_50_fraction",
        }.items():
            if validation_column not in row.index:
                continue
            validation_value = _numeric(row.get(validation_column))
            oos_value = _numeric(row.get(f"oos_predicted_{tier_label}_fraction"))
            if not math.isfinite(validation_value) or not math.isfinite(oos_value):
                continue
            delta = oos_value - validation_value
            abs_deltas[tier_label].append(abs(delta))
            _append_detail(
                detail_rows,
                section="validation_oos_stability",
                metric="predicted_tier_fraction_delta",
                group=f"fold_{fold_id}",
                subgroup=tier_label,
                value=delta,
                row_count=1,
                detail="OOS selected prediction support minus validation support",
            )
        if "oos_score_mean" in row.index:
            _append_detail(
                detail_rows,
                section="validation_oos_stability",
                metric="oos_score_mean",
                group=f"fold_{fold_id}",
                value=_numeric(row.get("oos_score_mean")),
                row_count=1,
                detail="validation score mean is not persisted in current artifacts",
            )

    for tier_label, values in abs_deltas.items():
        if values:
            _append_summary(
                summary_rows,
                section="validation_oos_stability",
                metric=f"avg_abs_predicted_{tier_label}_fraction_delta",
                value=float(pd.Series(values).mean()),
                detail="absolute OOS-vs-validation support delta among selected folds",
            )
    _append_summary(
        summary_rows,
        section="validation_oos_stability",
        metric="validation_score_mean_available",
        value=False,
        detail="current artifacts persist validation tier support but not validation score mean",
    )


def _context_rows(
    *,
    summary_rows: list[dict[str, object]],
    phase8_summary: pd.DataFrame,
    bull_summary: pd.DataFrame,
) -> None:
    for source_name, frame in {
        "phase8_run_summary": phase8_summary,
        "phase8_bull_participation_summary": bull_summary,
    }.items():
        _append_summary(
            summary_rows,
            section="artifact_context",
            metric=f"{source_name}_present",
            value=not frame.empty,
            detail=f"{source_name}.csv was {'found' if not frame.empty else 'not found'}",
        )


def build_phase8_score_diagnostic(run_dir: str | Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    resolved_run_dir = Path(run_dir)
    probability = _read_csv(resolved_run_dir, "allocation_probability_diagnostics.csv")
    candidates = _read_csv(resolved_run_dir, "ml_strategy_tuning_candidates.csv")
    selections = _read_csv(resolved_run_dir, "ml_strategy_tuning_selections.csv")
    detail_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []

    _score_decile_rows(
        detail_rows=detail_rows,
        summary_rows=summary_rows,
        probability=probability,
    )
    _confusion_rows(
        detail_rows=detail_rows,
        summary_rows=summary_rows,
        probability=probability,
    )
    _candidate_rows(
        detail_rows=detail_rows,
        summary_rows=summary_rows,
        candidates=candidates,
        selections=selections,
    )
    _model_family_oos_rows(
        detail_rows=detail_rows,
        summary_rows=summary_rows,
        probability=probability,
    )
    _validation_oos_rows(
        detail_rows=detail_rows,
        summary_rows=summary_rows,
        probability=probability,
        selections=selections,
    )
    _context_rows(
        summary_rows=summary_rows,
        phase8_summary=_read_csv(resolved_run_dir, "phase8_run_summary.csv"),
        bull_summary=_read_csv(resolved_run_dir, "phase8_bull_participation_summary.csv"),
    )

    return (
        pd.DataFrame(detail_rows, columns=PHASE8_SCORE_DIAGNOSTIC_COLUMNS),
        pd.DataFrame(summary_rows, columns=PHASE8_SCORE_DIAGNOSTIC_SUMMARY_COLUMNS),
    )


def write_phase8_score_diagnostic(
    run_dir: str | Path,
    *,
    output_dir: str | Path | None = None,
) -> tuple[Path, Path]:
    resolved_run_dir = Path(run_dir)
    resolved_output_dir = Path(output_dir) if output_dir is not None else resolved_run_dir
    resolved_output_dir.mkdir(parents=True, exist_ok=True)
    detail, summary = build_phase8_score_diagnostic(resolved_run_dir)
    detail_path = resolved_output_dir / "phase8_score_diagnostic.csv"
    summary_path = resolved_output_dir / "phase8_score_diagnostic_summary.csv"
    detail.to_csv(detail_path, index=False)
    summary.to_csv(summary_path, index=False)
    return detail_path, summary_path
