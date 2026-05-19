from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import pandas as pd

TOLERANCE_LADDER = (0.0, 0.001, 0.005, 0.01, 0.025, 0.05, 0.10, 0.25, 0.50, 1.0)
BENCHMARK_FAILURE_REASONS = {
    "non_positive_buy_hold_excess",
    "non_positive_required_benchmark_excess",
}
TURNOVER_FAILURE_REASON = "turnover_budget_exceeded"

PHASE8_SELECTION_PROBE_COLUMNS = [
    "probe_profile",
    "benchmark_tolerance",
    "fold_id",
    "selection_status",
    "selection_source",
    "selected_model_name",
    "selected_utility_profile",
    "selected_regime_policy",
    "selected_rolling_train_bars",
    "selected_min_holding_period_bars",
    "selected_hysteresis_margin",
    "selected_threshold",
    "selected_tier_min_threshold",
    "selected_tier_half_threshold",
    "selected_tier_full_threshold",
    "selected_passed_gate",
    "selected_active_candidate",
    "selected_failure_reasons",
    "selected_cumulative_return",
    "selected_excess_cumulative_return",
    "selected_min_benchmark_excess_cumulative_return",
    "selected_average_exposure",
    "selected_annualized_turnover",
    "selected_min_validation_predicted_target_fraction",
    "selected_sharpe_like_delta",
    "selected_drawdown_delta",
    "selected_selection_benchmark_excess_cumulative_returns",
]

PHASE8_SELECTION_PROBE_SUMMARY_COLUMNS = [
    "probe_profile",
    "benchmark_tolerance",
    "diagnostic_only",
    "candidate_rows",
    "eligible_candidate_rows",
    "total_folds",
    "selected_folds",
    "no_valid_candidate_folds",
    "selected_fold_fraction",
    "selected_fold_ids",
    "no_valid_fold_ids",
    "min_selected_min_benchmark_excess",
    "avg_selected_min_benchmark_excess",
    "avg_selected_exposure",
    "avg_selected_annualized_turnover",
    "min_selected_predicted_target_fraction",
    "selection_rule",
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


def _numeric_or_na(value: Any) -> object:
    numeric_value = _numeric(value)
    return numeric_value if math.isfinite(numeric_value) else pd.NA


def _failure_reasons(row: pd.Series) -> list[str]:
    if "failure_reasons" in row.index and pd.notna(row.get("failure_reasons")):
        return [
            reason.strip()
            for reason in str(row.get("failure_reasons")).split(";")
            if reason.strip()
        ]
    if _truthy(row.get("passed_gate")):
        return []

    reasons: list[str] = []
    if "active_candidate" in row.index and not _truthy(row.get("active_candidate")):
        reasons.append("inactive_candidate")
    if _numeric(row.get("min_benchmark_excess_cumulative_return")) <= 0.0:
        reasons.append("non_positive_required_benchmark_excess")
    elif _numeric(row.get("excess_cumulative_return")) <= 0.0:
        reasons.append("non_positive_buy_hold_excess")
    if _numeric(row.get("min_validation_predicted_target_fraction")) <= 0.0:
        reasons.append("insufficient_predicted_tier_support")
    if (
        _numeric(row.get("sharpe_like_delta")) <= 0.0
        and _numeric(row.get("drawdown_delta")) < 0.0
    ):
        reasons.append("risk_not_improved")
    return reasons or ["failed_other"]


def _benchmark_value_for_reason(row: pd.Series, reason: str) -> float:
    if reason == "non_positive_buy_hold_excess":
        return _numeric(row.get("excess_cumulative_return"))
    return _numeric(row.get("min_benchmark_excess_cumulative_return"))


def _benchmark_failure_is_tolerated(
    row: pd.Series,
    *,
    reason: str,
    tolerance: float,
) -> bool:
    value = _benchmark_value_for_reason(row, reason)
    if not math.isfinite(value):
        return False
    if tolerance <= 0.0:
        return value > 0.0
    return value >= -float(tolerance)


def _reasons_after_benchmark_tolerance(row: pd.Series, *, tolerance: float) -> list[str]:
    remaining: list[str] = []
    for reason in _failure_reasons(row):
        if reason in BENCHMARK_FAILURE_REASONS and _benchmark_failure_is_tolerated(
            row,
            reason=reason,
            tolerance=tolerance,
        ):
            continue
        remaining.append(reason)
    return remaining


def _eligible_strict(candidates: pd.DataFrame) -> pd.DataFrame:
    if candidates.empty or "passed_gate" not in candidates.columns:
        return candidates.iloc[0:0].copy()
    passed = candidates["passed_gate"].map(_truthy)
    return candidates.loc[passed].copy()


def _eligible_benchmark_tolerance(candidates: pd.DataFrame, tolerance: float) -> pd.DataFrame:
    if candidates.empty:
        return candidates.copy()
    eligible = candidates.apply(
        lambda row: not _reasons_after_benchmark_tolerance(row, tolerance=tolerance),
        axis=1,
    )
    return candidates.loc[eligible].copy()


def _eligible_best_active_fallback(candidates: pd.DataFrame) -> pd.DataFrame:
    if candidates.empty:
        return candidates.copy()
    eligible = candidates.apply(
        lambda row: not [
            reason
            for reason in _failure_reasons(row)
            if reason not in BENCHMARK_FAILURE_REASONS
        ],
        axis=1,
    )
    return candidates.loc[eligible].copy()


def _eligible_turnover_only(candidates: pd.DataFrame) -> pd.DataFrame:
    if candidates.empty:
        return candidates.copy()
    eligible = candidates.apply(
        lambda row: TURNOVER_FAILURE_REASON not in set(_failure_reasons(row)),
        axis=1,
    )
    return candidates.loc[eligible].copy()


def _selection_score(row: pd.Series) -> tuple[float, float, float, float, float]:
    min_benchmark_excess = _numeric(row.get("min_benchmark_excess_cumulative_return"))
    if not math.isfinite(min_benchmark_excess):
        min_benchmark_excess = _numeric(row.get("excess_cumulative_return"))
    if not math.isfinite(min_benchmark_excess):
        min_benchmark_excess = float("-inf")

    excess_return = _numeric(row.get("excess_cumulative_return"))
    drawdown_delta = _numeric(row.get("drawdown_delta"))
    sharpe_delta = _numeric(row.get("sharpe_like_delta"))
    turnover = _numeric(row.get("annualized_turnover"))
    return (
        min_benchmark_excess,
        excess_return if math.isfinite(excess_return) else float("-inf"),
        drawdown_delta if math.isfinite(drawdown_delta) else float("-inf"),
        sharpe_delta if math.isfinite(sharpe_delta) else float("-inf"),
        -turnover if math.isfinite(turnover) else float("-inf"),
    )


def _select_candidate(candidates: pd.DataFrame) -> pd.Series | None:
    if candidates.empty:
        return None
    scored = candidates.copy()
    scored["_selection_score"] = scored.apply(_selection_score, axis=1)
    selected = scored.sort_values("_selection_score", ascending=False).iloc[0]
    return selected.drop(labels=["_selection_score"])


def _fold_sort_key(value: Any) -> tuple[int, float | str]:
    try:
        return (0, float(value))
    except (TypeError, ValueError):
        return (1, str(value))


def _fold_ids(candidates: pd.DataFrame) -> list[object]:
    if candidates.empty or "fold_id" not in candidates.columns:
        return []
    return sorted(candidates["fold_id"].dropna().unique().tolist(), key=_fold_sort_key)


def _selected_row(
    *,
    profile: str,
    tolerance: float | None,
    fold_id: object,
    selected: pd.Series | None,
    status: str,
    source: str,
) -> dict[str, object]:
    row: dict[str, object] = {
        "probe_profile": profile,
        "benchmark_tolerance": tolerance if tolerance is not None else pd.NA,
        "fold_id": fold_id,
        "selection_status": status,
        "selection_source": source,
    }
    if selected is None:
        for column in PHASE8_SELECTION_PROBE_COLUMNS:
            row.setdefault(column, pd.NA)
        row["selection_status"] = "no_valid_candidate"
        row["selection_source"] = "none"
        return row

    mappings = {
        "selected_model_name": "model_name",
        "selected_utility_profile": "utility_profile",
        "selected_regime_policy": "regime_policy",
        "selected_rolling_train_bars": "rolling_train_bars",
        "selected_min_holding_period_bars": "min_holding_period_bars",
        "selected_hysteresis_margin": "hysteresis_margin",
        "selected_threshold": "threshold",
        "selected_tier_min_threshold": "tier_min_threshold",
        "selected_tier_half_threshold": "tier_half_threshold",
        "selected_tier_full_threshold": "tier_full_threshold",
        "selected_passed_gate": "passed_gate",
        "selected_active_candidate": "active_candidate",
        "selected_failure_reasons": "failure_reasons",
        "selected_cumulative_return": "cumulative_return",
        "selected_excess_cumulative_return": "excess_cumulative_return",
        "selected_min_benchmark_excess_cumulative_return": (
            "min_benchmark_excess_cumulative_return"
        ),
        "selected_average_exposure": "average_exposure",
        "selected_annualized_turnover": "annualized_turnover",
        "selected_min_validation_predicted_target_fraction": (
            "min_validation_predicted_target_fraction"
        ),
        "selected_sharpe_like_delta": "sharpe_like_delta",
        "selected_drawdown_delta": "drawdown_delta",
        "selected_selection_benchmark_excess_cumulative_returns": (
            "selection_benchmark_excess_cumulative_returns"
        ),
    }
    for output_column, input_column in mappings.items():
        value = selected.get(input_column, pd.NA)
        if output_column.startswith("selected_") and output_column.endswith(
            (
                "return",
                "exposure",
                "turnover",
                "fraction",
                "delta",
                "threshold",
                "bars",
            )
        ):
            row[output_column] = _numeric_or_na(value)
        else:
            row[output_column] = value
    return row


def _strict_selection_row(
    *,
    profile: str,
    tolerance: float | None,
    fold_id: object,
    selected: pd.Series,
) -> dict[str, object]:
    row: dict[str, object] = {
        "probe_profile": profile,
        "benchmark_tolerance": tolerance if tolerance is not None else pd.NA,
        "fold_id": fold_id,
        "selection_status": "selected",
        "selection_source": "strict",
        "selected_model_name": selected.get("selected_model_name", pd.NA),
        "selected_utility_profile": selected.get("selected_utility_profile", pd.NA),
        "selected_regime_policy": selected.get("selected_regime_policy", pd.NA),
        "selected_rolling_train_bars": _numeric_or_na(
            selected.get("selected_rolling_train_bars")
        ),
        "selected_min_holding_period_bars": _numeric_or_na(
            selected.get("selected_min_holding_period_bars")
        ),
        "selected_hysteresis_margin": _numeric_or_na(
            selected.get("selected_hysteresis_margin")
        ),
        "selected_threshold": _numeric_or_na(selected.get("selected_threshold")),
        "selected_tier_min_threshold": _numeric_or_na(
            selected.get("selected_tier_min_threshold")
        ),
        "selected_tier_half_threshold": _numeric_or_na(
            selected.get("selected_tier_half_threshold")
        ),
        "selected_tier_full_threshold": _numeric_or_na(
            selected.get("selected_tier_full_threshold")
        ),
        "selected_passed_gate": selected.get("passed_gate", True),
        "selected_active_candidate": True,
        "selected_failure_reasons": "",
        "selected_cumulative_return": pd.NA,
        "selected_excess_cumulative_return": _numeric_or_na(
            selected.get("excess_cumulative_return")
        ),
        "selected_min_benchmark_excess_cumulative_return": _numeric_or_na(
            selected.get("min_benchmark_excess_cumulative_return")
        ),
        "selected_average_exposure": _numeric_or_na(selected.get("average_exposure")),
        "selected_annualized_turnover": _numeric_or_na(
            selected.get("annualized_turnover")
        ),
        "selected_min_validation_predicted_target_fraction": _numeric_or_na(
            selected.get("min_validation_predicted_target_fraction")
        ),
        "selected_sharpe_like_delta": _numeric_or_na(
            selected.get("sharpe_like_delta")
        ),
        "selected_drawdown_delta": _numeric_or_na(selected.get("drawdown_delta")),
        "selected_selection_benchmark_excess_cumulative_returns": selected.get(
            "selection_benchmark_excess_cumulative_returns",
            pd.NA,
        ),
    }
    return row


def _strict_selections_by_fold(selections: pd.DataFrame) -> dict[object, pd.Series]:
    if selections.empty or "fold_id" not in selections.columns:
        return {}
    if "selection_status" in selections.columns:
        selected_rows = selections.loc[
            selections["selection_status"].astype(str).eq("selected")
        ]
    else:
        selected_rows = selections
    if "passed_gate" in selected_rows.columns:
        selected_rows = selected_rows.loc[selected_rows["passed_gate"].map(_truthy)]
    return {
        row["fold_id"]: row
        for _, row in selected_rows.sort_values("fold_id").iterrows()
    }


def _profile_rows(
    *,
    candidates: pd.DataFrame,
    profile: str,
    tolerance: float | None,
    fold_ids: list[object],
    strict_selection_by_fold: dict[object, pd.Series],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    strict_candidates = _eligible_strict(candidates)
    for fold_id in fold_ids:
        fold_candidates = candidates.loc[candidates["fold_id"].eq(fold_id)].copy()
        fold_strict = strict_candidates.loc[strict_candidates["fold_id"].eq(fold_id)]
        strict_selection = strict_selection_by_fold.get(fold_id)

        if profile == "strict":
            if strict_selection is not None:
                rows.append(
                    _strict_selection_row(
                        profile=profile,
                        tolerance=tolerance,
                        fold_id=fold_id,
                        selected=strict_selection,
                    )
                )
                continue
            eligible = fold_strict
            status = "selected"
            source = "strict"
        elif profile == "benchmark_tolerance":
            if strict_selection is not None:
                rows.append(
                    _strict_selection_row(
                        profile=profile,
                        tolerance=tolerance,
                        fold_id=fold_id,
                        selected=strict_selection,
                    )
                )
                continue
            eligible = _eligible_benchmark_tolerance(
                fold_candidates,
                tolerance=float(tolerance or 0.0),
            )
            status = "selected"
            source = "benchmark_tolerance"
        elif profile == "best_active_fallback":
            if strict_selection is not None:
                rows.append(
                    _strict_selection_row(
                        profile=profile,
                        tolerance=tolerance,
                        fold_id=fold_id,
                        selected=strict_selection,
                    )
                )
                continue
            eligible = _eligible_best_active_fallback(fold_candidates)
            status = "fallback_selected"
            source = "best_active_fallback"
        elif profile == "turnover_only_control":
            eligible = _eligible_turnover_only(fold_candidates)
            status = "control_selected"
            source = "turnover_only_control"
        else:
            raise ValueError(f"Unknown Phase 8 selection probe profile: {profile}")

        selected = _select_candidate(eligible)
        rows.append(
            _selected_row(
                profile=profile,
                tolerance=tolerance,
                fold_id=fold_id,
                selected=selected,
                status=status if selected is not None else "no_valid_candidate",
                source=source if selected is not None else "none",
            )
        )
    return rows


def _selection_rule(profile: str, tolerance: float | None) -> str:
    if profile == "strict":
        return "passed_gate=True"
    if profile == "benchmark_tolerance":
        return f"strict gates with benchmark excess tolerance {float(tolerance or 0.0):g}"
    if profile == "best_active_fallback":
        return "strict candidate first, otherwise ignore benchmark-only validation failures"
    return "diagnostic control: require only no turnover_budget_exceeded reason"


def _selected_values(rows: pd.DataFrame, column: str) -> pd.Series:
    selected = rows.loc[rows["selection_status"].ne("no_valid_candidate")]
    return selected[column].map(_numeric).dropna()


def _summary_row(
    *,
    candidates: pd.DataFrame,
    selections: pd.DataFrame,
    profile: str,
    tolerance: float | None,
    eligible_candidate_rows: int,
) -> dict[str, object]:
    total_folds = int(len(selections))
    selected = selections.loc[selections["selection_status"].ne("no_valid_candidate")]
    no_valid = selections.loc[selections["selection_status"].eq("no_valid_candidate")]
    selected_fold_ids = [str(value) for value in selected["fold_id"].tolist()]
    no_valid_fold_ids = [str(value) for value in no_valid["fold_id"].tolist()]
    selected_folds = int(len(selected))
    selected_fraction = selected_folds / total_folds if total_folds else float("nan")
    min_benchmark = _selected_values(
        selections,
        "selected_min_benchmark_excess_cumulative_return",
    )
    exposure = _selected_values(selections, "selected_average_exposure")
    turnover = _selected_values(selections, "selected_annualized_turnover")
    predicted = _selected_values(
        selections,
        "selected_min_validation_predicted_target_fraction",
    )
    return {
        "probe_profile": profile,
        "benchmark_tolerance": tolerance if tolerance is not None else pd.NA,
        "diagnostic_only": profile == "turnover_only_control",
        "candidate_rows": int(len(candidates)),
        "eligible_candidate_rows": int(eligible_candidate_rows),
        "total_folds": total_folds,
        "selected_folds": selected_folds,
        "no_valid_candidate_folds": int(len(no_valid)),
        "selected_fold_fraction": selected_fraction,
        "selected_fold_ids": ",".join(selected_fold_ids),
        "no_valid_fold_ids": ",".join(no_valid_fold_ids),
        "min_selected_min_benchmark_excess": (
            float(min_benchmark.min()) if not min_benchmark.empty else pd.NA
        ),
        "avg_selected_min_benchmark_excess": (
            float(min_benchmark.mean()) if not min_benchmark.empty else pd.NA
        ),
        "avg_selected_exposure": float(exposure.mean()) if not exposure.empty else pd.NA,
        "avg_selected_annualized_turnover": (
            float(turnover.mean()) if not turnover.empty else pd.NA
        ),
        "min_selected_predicted_target_fraction": (
            float(predicted.min()) if not predicted.empty else pd.NA
        ),
        "selection_rule": _selection_rule(profile, tolerance),
    }


def build_phase8_selection_probe(
    run_dir: str | Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    resolved_run_dir = Path(run_dir)
    candidates = _read_csv(resolved_run_dir, "ml_strategy_tuning_candidates.csv")
    persisted_selections = _read_csv(resolved_run_dir, "ml_strategy_tuning_selections.csv")
    if candidates.empty or "fold_id" not in candidates.columns:
        return (
            pd.DataFrame(columns=PHASE8_SELECTION_PROBE_COLUMNS),
            pd.DataFrame(
                [
                    {
                        "probe_profile": "missing_candidates",
                        "benchmark_tolerance": pd.NA,
                        "diagnostic_only": True,
                        "candidate_rows": 0,
                        "eligible_candidate_rows": 0,
                        "total_folds": 0,
                        "selected_folds": 0,
                        "no_valid_candidate_folds": 0,
                        "selected_fold_fraction": float("nan"),
                        "selected_fold_ids": "",
                        "no_valid_fold_ids": "",
                        "min_selected_min_benchmark_excess": pd.NA,
                        "avg_selected_min_benchmark_excess": pd.NA,
                        "avg_selected_exposure": pd.NA,
                        "avg_selected_annualized_turnover": pd.NA,
                        "min_selected_predicted_target_fraction": pd.NA,
                        "selection_rule": "ml_strategy_tuning_candidates.csv was not found or has no fold_id",
                    }
                ],
                columns=PHASE8_SELECTION_PROBE_SUMMARY_COLUMNS,
            ),
        )

    fold_ids = _fold_ids(candidates)
    strict_selection_by_fold = _strict_selections_by_fold(persisted_selections)
    all_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []

    profile_specs: list[tuple[str, float | None]] = [
        ("strict", None),
        *[("benchmark_tolerance", tolerance) for tolerance in TOLERANCE_LADDER],
        ("best_active_fallback", None),
        ("turnover_only_control", None),
    ]
    for profile, tolerance in profile_specs:
        rows = _profile_rows(
            candidates=candidates,
            profile=profile,
            tolerance=tolerance,
            fold_ids=fold_ids,
            strict_selection_by_fold=strict_selection_by_fold,
        )
        selections = pd.DataFrame(rows, columns=PHASE8_SELECTION_PROBE_COLUMNS)
        all_rows.extend(rows)
        if profile == "strict":
            eligible_count = len(_eligible_strict(candidates))
        elif profile == "benchmark_tolerance":
            eligible_count = len(
                _eligible_benchmark_tolerance(candidates, float(tolerance or 0.0))
            )
        elif profile == "best_active_fallback":
            eligible_count = len(_eligible_best_active_fallback(candidates))
        else:
            eligible_count = len(_eligible_turnover_only(candidates))
        summary_rows.append(
            _summary_row(
                candidates=candidates,
                selections=selections,
                profile=profile,
                tolerance=tolerance,
                eligible_candidate_rows=eligible_count,
            )
        )

    return (
        pd.DataFrame(all_rows, columns=PHASE8_SELECTION_PROBE_COLUMNS),
        pd.DataFrame(summary_rows, columns=PHASE8_SELECTION_PROBE_SUMMARY_COLUMNS),
    )


def write_phase8_selection_probe(
    run_dir: str | Path,
    output_dir: str | Path | None = None,
) -> tuple[Path, Path]:
    resolved_run_dir = Path(run_dir)
    resolved_output_dir = Path(output_dir) if output_dir is not None else resolved_run_dir
    resolved_output_dir.mkdir(parents=True, exist_ok=True)
    selections, summary = build_phase8_selection_probe(resolved_run_dir)
    selections_path = resolved_output_dir / "phase8_selection_probe.csv"
    summary_path = resolved_output_dir / "phase8_selection_probe_summary.csv"
    selections.to_csv(selections_path, index=False)
    summary.to_csv(summary_path, index=False)
    return selections_path, summary_path
