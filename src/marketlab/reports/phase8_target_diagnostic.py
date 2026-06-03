from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import pandas as pd

from marketlab.reports.phase8_bull_participation import _resolve_date_regime

ALLOCATION_TIERS = (0.0, 0.25, 0.50, 1.0)
DEFENSE_DRAWDOWN_THRESHOLD = -0.10

PHASE8_TARGET_DIAGNOSTIC_COLUMNS = [
    "section",
    "metric",
    "group",
    "subgroup",
    "value",
    "row_count",
    "detail",
]
PHASE8_TARGET_DIAGNOSTIC_SUMMARY_COLUMNS = ["section", "metric", "value", "detail"]


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


def _tier_label(value: Any) -> str:
    numeric_value = _numeric(value)
    if not math.isfinite(numeric_value):
        return "missing"
    return f"{int(round(numeric_value * 100))}"


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


def _merge_date_regime(
    probability: pd.DataFrame,
    *,
    config_path: str | Path | None,
) -> tuple[pd.DataFrame, str]:
    if probability.empty:
        return probability.copy(), "missing"

    date_regime, date_regime_source = _resolve_date_regime(
        probability=probability,
        config_path=config_path,
    )
    working = probability.copy()
    date_column = "effective_date" if "effective_date" in working.columns else "signal_date"
    if date_regime.empty or date_column not in working.columns:
        if "runtime_regime" not in working.columns:
            working["runtime_regime"] = "missing"
        return working, date_regime_source

    working[date_column] = pd.to_datetime(working[date_column], errors="coerce")
    regime = date_regime.copy()
    regime["date"] = pd.to_datetime(regime["date"], errors="coerce")
    regime_columns = [
        column
        for column in regime.columns
        if column == "runtime_regime" or str(column).startswith("gate_")
    ]
    joined = working.merge(
        regime.loc[:, ["date", *regime_columns]],
        left_on=date_column,
        right_on="date",
        how="left",
        suffixes=("", "_regime"),
    )
    for column in regime_columns:
        regime_column = f"{column}_regime"
        if regime_column not in joined.columns:
            continue
        if date_regime_source == "config" or column not in working.columns:
            joined[column] = joined[regime_column].combine_first(joined.get(column))
        else:
            joined[column] = joined.get(column).combine_first(joined[regime_column])
        joined = joined.drop(columns=[regime_column])
    return joined.drop(columns=["date"]), date_regime_source


def _prepare_probability(
    probability: pd.DataFrame,
    *,
    config_path: str | Path | None,
) -> tuple[pd.DataFrame, str]:
    working, date_regime_source = _merge_date_regime(
        probability,
        config_path=config_path,
    )
    if working.empty:
        return working, date_regime_source

    for column in (
        "target_weight",
        "predicted_tier_weight",
        "score",
        "forward_return",
        "forward_drawdown",
        "realized_utility",
    ):
        if column in working.columns:
            working[column] = working[column].map(_numeric)

    if "runtime_regime" not in working.columns:
        working["runtime_regime"] = "missing"
    working["runtime_regime"] = working["runtime_regime"].fillna("missing").astype(str)
    if "forward_return" in working.columns:
        forward_return = working["forward_return"]
        working["forward_return_sign"] = "missing"
        working.loc[forward_return.gt(0.0), "forward_return_sign"] = "positive"
        working.loc[forward_return.le(0.0), "forward_return_sign"] = "zero_or_negative"
    else:
        working["forward_return_sign"] = "missing"

    if "forward_drawdown" in working.columns:
        forward_drawdown = working["forward_drawdown"]
        working["forward_drawdown_bucket"] = "missing"
        working.loc[forward_drawdown.gt(DEFENSE_DRAWDOWN_THRESHOLD), "forward_drawdown_bucket"] = (
            "mild"
        )
        working.loc[forward_drawdown.le(DEFENSE_DRAWDOWN_THRESHOLD), "forward_drawdown_bucket"] = (
            "defense"
        )
    else:
        working["forward_drawdown_bucket"] = "missing"

    working["target_tier"] = (
        working["target_weight"].map(_tier_label)
        if "target_weight" in working.columns
        else "missing"
    )
    working["predicted_tier"] = (
        working["predicted_tier_weight"].map(_tier_label)
        if "predicted_tier_weight" in working.columns
        else "missing"
    )
    working["bull_continuation"] = (
        working["runtime_regime"].eq("bull")
        & working.get("forward_return", pd.Series(index=working.index, dtype=float)).gt(0.0)
    )
    defense_return = (
        working.get("forward_return", pd.Series(index=working.index, dtype=float)).le(0.0)
    )
    defense_drawdown = (
        working.get("forward_drawdown", pd.Series(index=working.index, dtype=float)).le(
            DEFENSE_DRAWDOWN_THRESHOLD
        )
    )
    working["drawdown_defense"] = defense_return | defense_drawdown
    return working, date_regime_source


def _fraction(rows: pd.DataFrame, column: str, tier: float, *, below: bool = False) -> object:
    if rows.empty or column not in rows.columns:
        return pd.NA
    values = rows[column].dropna()
    if values.empty:
        return pd.NA
    if below:
        return float(values.lt(tier - 1e-9).mean())
    return float(values.sub(tier).abs().le(1e-9).mean())


def _mean(rows: pd.DataFrame, column: str) -> object:
    if rows.empty or column not in rows.columns:
        return pd.NA
    values = rows[column].dropna()
    if values.empty:
        return pd.NA
    return float(values.mean())


def _append_group_metrics(
    rows: list[dict[str, object]],
    *,
    section: str,
    group: object,
    frame: pd.DataFrame,
    detail: str,
) -> None:
    _append_detail(
        rows,
        section=section,
        metric="row_count",
        group=group,
        value=int(len(frame)),
        row_count=int(len(frame)),
        detail=detail,
    )
    for metric, value in {
        "full_target_fraction": _fraction(frame, "target_weight", 1.0),
        "full_prediction_fraction": _fraction(frame, "predicted_tier_weight", 1.0),
        "below_full_target_fraction": _fraction(
            frame,
            "target_weight",
            1.0,
            below=True,
        ),
        "below_full_prediction_fraction": _fraction(
            frame,
            "predicted_tier_weight",
            1.0,
            below=True,
        ),
        "avg_target_weight": _mean(frame, "target_weight"),
        "avg_predicted_tier_weight": _mean(frame, "predicted_tier_weight"),
        "avg_score": _mean(frame, "score"),
        "avg_forward_return": _mean(frame, "forward_return"),
        "avg_forward_drawdown": _mean(frame, "forward_drawdown"),
        "avg_realized_utility": _mean(frame, "realized_utility"),
    }.items():
        _append_detail(
            rows,
            section=section,
            metric=metric,
            group=group,
            value=value,
            row_count=int(len(frame)),
            detail=detail,
        )


def _grouped_detail_rows(
    *,
    detail_rows: list[dict[str, object]],
    working: pd.DataFrame,
) -> None:
    group_columns = {
        "runtime_regime": "runtime regime labels on selected OOS rows",
        "forward_return_sign": "realized forward-return sign",
        "forward_drawdown_bucket": f"forward_drawdown <= {DEFENSE_DRAWDOWN_THRESHOLD:g}",
        "target_tier": "actual target tier",
        "predicted_tier": "nearest traded prediction tier",
    }
    if "allocation_score_transform" in working.columns:
        group_columns["allocation_score_transform"] = "selected score-transform profile"
    if "gate_bull" in working.columns:
        working["gate_bull_label"] = working["gate_bull"].map(_truthy).map(str)
        group_columns["gate_bull_label"] = "strict-gate bull labels from config or artifacts"

    for column, detail in group_columns.items():
        for group_name, group in working.groupby(column, dropna=False, sort=True):
            _append_group_metrics(
                detail_rows,
                section=column,
                group=group_name,
                frame=group,
                detail=detail,
            )

    if {"target_tier", "predicted_tier"}.issubset(working.columns):
        counts = (
            working.groupby(["target_tier", "predicted_tier"], dropna=False)
            .size()
            .rename("row_count")
            .reset_index()
        )
        totals = working.groupby("target_tier", dropna=False).size().rename("total")
        counts = counts.merge(totals, on="target_tier", how="left")
        for _, row in counts.iterrows():
            total = int(row["total"])
            count = int(row["row_count"])
            _append_detail(
                detail_rows,
                section="target_prediction_matrix",
                metric="predicted_tier_fraction_within_target",
                group=str(row["target_tier"]),
                subgroup=str(row["predicted_tier"]),
                value=count / total if total else float("nan"),
                row_count=count,
                detail="fraction normalized within target tier",
            )


def _correlation(rows: pd.DataFrame, left: str, right: str) -> object:
    if rows.empty or not {left, right}.issubset(rows.columns):
        return pd.NA
    valid = rows[[left, right]].dropna()
    if len(valid) < 2:
        return pd.NA
    value = valid[left].corr(valid[right])
    return float(value) if math.isfinite(value) else pd.NA


def _append_summary_rows(
    *,
    summary_rows: list[dict[str, object]],
    working: pd.DataFrame,
    date_regime_source: str,
) -> None:
    _append_summary(
        summary_rows,
        section="artifact_context",
        metric="allocation_probability_present",
        value=True,
        detail=f"rows={len(working)}",
    )
    _append_summary(
        summary_rows,
        section="artifact_context",
        metric="runtime_regime_source",
        value=date_regime_source,
        detail="config is preferred when supplied",
    )

    for column in (
        "score",
        "target_weight",
        "predicted_tier_weight",
        "forward_return",
        "forward_drawdown",
        "realized_utility",
        "allocation_score_transform",
        "score_transform_applied",
    ):
        _append_summary(
            summary_rows,
            section="score_target_context",
            metric=f"{column}_present",
            value=column in working.columns,
            detail="allocation_probability_diagnostics.csv column availability",
        )

    for tier in ALLOCATION_TIERS:
        _append_summary(
            summary_rows,
            section="target_support",
            metric=f"target_tier_{_tier_label(tier)}_fraction",
            value=_fraction(working, "target_weight", tier),
            detail="selected OOS prediction rows",
        )
        _append_summary(
            summary_rows,
            section="target_support",
            metric=f"predicted_tier_{_tier_label(tier)}_fraction",
            value=_fraction(working, "predicted_tier_weight", tier),
            detail="selected OOS prediction rows",
        )

    bull_continuation = working.loc[working["bull_continuation"]].copy()
    positive_return = working.loc[working["forward_return_sign"].eq("positive")].copy()
    drawdown_defense = working.loc[working["drawdown_defense"]].copy()
    for name, frame in {
        "bull_continuation": bull_continuation,
        "positive_return": positive_return,
        "drawdown_defense": drawdown_defense,
    }.items():
        _append_summary(
            summary_rows,
            section="target_methodology",
            metric=f"{name}_rows",
            value=int(len(frame)),
            detail="selected OOS prediction rows",
        )

    _append_summary(
        summary_rows,
        section="target_methodology",
        metric="bull_continuation_full_target_fraction",
        value=_fraction(bull_continuation, "target_weight", 1.0),
        detail="runtime_regime=bull and forward_return > 0",
    )
    _append_summary(
        summary_rows,
        section="target_methodology",
        metric="bull_continuation_full_prediction_fraction",
        value=_fraction(bull_continuation, "predicted_tier_weight", 1.0),
        detail="runtime_regime=bull and forward_return > 0",
    )
    _append_summary(
        summary_rows,
        section="target_methodology",
        metric="positive_return_rows_assigned_below_100_fraction",
        value=_fraction(positive_return, "target_weight", 1.0, below=True),
        detail="target_weight < 1.0 among rows with forward_return > 0",
    )
    _append_summary(
        summary_rows,
        section="target_methodology",
        metric="positive_return_rows_predicted_below_100_fraction",
        value=_fraction(positive_return, "predicted_tier_weight", 1.0, below=True),
        detail="predicted_tier_weight < 1.0 among rows with forward_return > 0",
    )
    _append_summary(
        summary_rows,
        section="target_methodology",
        metric="drawdown_defense_rows_assigned_below_100_fraction",
        value=_fraction(drawdown_defense, "target_weight", 1.0, below=True),
        detail=(
            "target_weight < 1.0 among rows with forward_return <= 0 "
            f"or forward_drawdown <= {DEFENSE_DRAWDOWN_THRESHOLD:g}"
        ),
    )
    _append_summary(
        summary_rows,
        section="target_methodology",
        metric="drawdown_defense_rows_predicted_below_100_fraction",
        value=_fraction(drawdown_defense, "predicted_tier_weight", 1.0, below=True),
        detail=(
            "predicted_tier_weight < 1.0 among rows with forward_return <= 0 "
            f"or forward_drawdown <= {DEFENSE_DRAWDOWN_THRESHOLD:g}"
        ),
    )

    if "gate_bull" in working.columns:
        gate_bull = working.loc[working["gate_bull"].map(_truthy)].copy()
        _append_summary(
            summary_rows,
            section="target_methodology",
            metric="gate_bull_rows",
            value=int(len(gate_bull)),
            detail="strict-gate bull rows joined from config or artifacts",
        )
        _append_summary(
            summary_rows,
            section="target_methodology",
            metric="gate_bull_full_target_fraction",
            value=_fraction(gate_bull, "target_weight", 1.0),
            detail="target_weight == 1.0 among strict-gate bull rows",
        )
        _append_summary(
            summary_rows,
            section="target_methodology",
            metric="gate_bull_full_prediction_fraction",
            value=_fraction(gate_bull, "predicted_tier_weight", 1.0),
            detail="predicted_tier_weight == 1.0 among strict-gate bull rows",
        )

    for right in ("target_weight", "forward_return", "realized_utility"):
        _append_summary(
            summary_rows,
            section="score_target_context",
            metric=f"score_{right}_correlation",
            value=_correlation(working, "score", right),
            detail="selected OOS prediction rows",
        )


def build_phase8_target_diagnostic(
    run_dir: str | Path,
    *,
    config_path: str | Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    resolved_run_dir = Path(run_dir)
    probability = _read_csv(resolved_run_dir, "allocation_probability_diagnostics.csv")
    detail_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []

    if probability.empty:
        _append_summary(
            summary_rows,
            section="artifact_context",
            metric="allocation_probability_present",
            value=False,
            detail="allocation_probability_diagnostics.csv was not found",
        )
        return (
            pd.DataFrame(detail_rows, columns=PHASE8_TARGET_DIAGNOSTIC_COLUMNS),
            pd.DataFrame(summary_rows, columns=PHASE8_TARGET_DIAGNOSTIC_SUMMARY_COLUMNS),
        )

    working, date_regime_source = _prepare_probability(
        probability,
        config_path=config_path,
    )
    _grouped_detail_rows(detail_rows=detail_rows, working=working)
    _append_summary_rows(
        summary_rows=summary_rows,
        working=working,
        date_regime_source=date_regime_source,
    )
    return (
        pd.DataFrame(detail_rows, columns=PHASE8_TARGET_DIAGNOSTIC_COLUMNS),
        pd.DataFrame(summary_rows, columns=PHASE8_TARGET_DIAGNOSTIC_SUMMARY_COLUMNS),
    )


def write_phase8_target_diagnostic(
    run_dir: str | Path,
    *,
    config_path: str | Path | None = None,
    output_dir: str | Path | None = None,
) -> tuple[Path, Path]:
    resolved_run_dir = Path(run_dir)
    resolved_output_dir = Path(output_dir) if output_dir is not None else resolved_run_dir
    resolved_output_dir.mkdir(parents=True, exist_ok=True)
    detail, summary = build_phase8_target_diagnostic(
        resolved_run_dir,
        config_path=config_path,
    )
    detail_path = resolved_output_dir / "phase8_target_diagnostic.csv"
    summary_path = resolved_output_dir / "phase8_target_diagnostic_summary.csv"
    detail.to_csv(detail_path, index=False)
    summary.to_csv(summary_path, index=False)
    return detail_path, summary_path
