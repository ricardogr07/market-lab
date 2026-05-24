from __future__ import annotations

import math
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pandas as pd

STRATEGY_NAME = "ml_indicator_tuned__long_only__cash"
DEFAULT_RUNS_ROOT = Path("artifacts/runs")
DEFAULT_EXPERIMENT_PREFIX = "btc_phase8"

REQUIRED_COMPLETE_ARTIFACTS = (
    "strategy_summary.csv",
    "strict_research_gate.csv",
    "ml_strategy_tuning_selections.csv",
)
DIAGNOSTIC_ARTIFACTS = (
    "phase8_run_summary.csv",
    "phase8_methodology_review.csv",
    "phase8_bull_participation_summary.csv",
    "phase8_score_diagnostic_summary.csv",
    "phase8_bull_counterfactual_summary.csv",
    "phase8_bull_counterfactual_gate.csv",
)

PHASE8_GRID_COMPARISON_COLUMNS = [
    "experiment",
    "run_id",
    "run_dir",
    "artifact_status",
    "recommended_artifact_action",
    "latest_complete_for_experiment",
    "file_count",
    "size_mb",
    "strict_gate_present",
    "strategy_summary_present",
    "methodology_review_present",
    "bull_participation_present",
    "score_diagnostic_present",
    "counterfactual_present",
    "strict_gate_passed",
    "deployment_gate_passed",
    "risk_allocation_gate_passed",
    "selection_coverage_gate_passed",
    "target_support_gate_passed",
    "signal_validity_gate_passed",
    "bull_participation_gate_passed",
    "counterfactual_pass_available",
    "strategy_cumulative_return",
    "buy_hold_cumulative_return",
    "active_return_vs_buy_hold",
    "active_return_vs_btc_static_25",
    "active_return_vs_btc_static_50",
    "active_return_vs_btc_static_75",
    "active_return_vs_btc_rebalanced_25",
    "active_return_vs_btc_rebalanced_50",
    "active_return_vs_btc_rebalanced_75",
    "strategy_sharpe_like",
    "buy_hold_sharpe_like",
    "strategy_max_drawdown",
    "buy_hold_max_drawdown",
    "strategy_avg_gross_exposure",
    "strategy_avg_turnover",
    "bull_upside_capture_ratio",
    "downside_capture_ratio",
    "selected_fold_fraction",
    "no_valid_candidate_folds",
    "target_tier_25_global_fraction",
    "target_tier_50_global_fraction",
    "predicted_tier_25_fraction",
    "predicted_tier_50_fraction",
    "predicted_tier_100_fraction",
    "score_target_weight_correlation",
    "score_forward_return_correlation",
    "score_realized_utility_correlation",
    "any_selected_oos_predicted_tier_100",
    "gate_bull_average_long_exposure",
    "gate_bull_active_return_sum",
    "gate_bull_underexposed_positive_benchmark_fraction",
    "gate_bull_underexposed_positive_benchmark_return_sum",
    "force_runtime_bull_100_cumulative_return",
    "force_runtime_bull_100_avg_long_exposure",
    "buy_hold_gate_bull_model_elsewhere_cumulative_return",
    "buy_hold_gate_bull_model_elsewhere_avg_long_exposure",
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


def _numeric_or_na(value: Any) -> float | object:
    numeric_value = _numeric(value)
    return numeric_value if math.isfinite(numeric_value) else pd.NA


def _summary_value(
    summary: pd.DataFrame,
    metric: str,
    *,
    section: str | None = None,
) -> object:
    if summary.empty or "metric" not in summary.columns:
        return pd.NA
    matches = summary.loc[summary["metric"].astype(str).eq(metric)]
    if section is not None and "section" in matches.columns:
        matches = matches.loc[matches["section"].astype(str).eq(section)]
    if matches.empty or "value" not in matches.columns:
        return pd.NA
    return matches.iloc[0]["value"]


def _counterfactual_summary_value(
    summary: pd.DataFrame,
    *,
    scenario: str,
    metric: str,
) -> object:
    if summary.empty or not {"scenario", "metric", "value"}.issubset(summary.columns):
        return pd.NA
    matches = summary.loc[
        summary["scenario"].astype(str).eq(scenario)
        & summary["metric"].astype(str).eq(metric)
    ]
    if matches.empty:
        return pd.NA
    return matches.iloc[0]["value"]


def _methodology_passed(
    review: pd.DataFrame,
    *,
    methodology_gate: str,
    metric: str = "overall",
    section: str | None = "summary",
) -> object:
    if review.empty or not {"methodology_gate", "metric", "passed"}.issubset(review.columns):
        return pd.NA
    matches = review.loc[
        review["methodology_gate"].astype(str).eq(methodology_gate)
        & review["metric"].astype(str).eq(metric)
    ]
    if section is not None and "section" in matches.columns:
        matches = matches.loc[matches["section"].astype(str).eq(section)]
    if matches.empty:
        return pd.NA
    return _truthy(matches.iloc[0]["passed"])


def _strict_condition_passed(strict_gate: pd.DataFrame, condition: str) -> object:
    if strict_gate.empty or not {"condition", "passed"}.issubset(strict_gate.columns):
        return pd.NA
    matches = strict_gate.loc[strict_gate["condition"].astype(str).eq(condition)]
    if matches.empty:
        return pd.NA
    return _truthy(matches.iloc[0]["passed"])


def _strategy_row(strategy_summary: pd.DataFrame, strategy_name: str) -> pd.Series:
    if strategy_summary.empty or "strategy" not in strategy_summary.columns:
        return pd.Series(dtype=object)
    matches = strategy_summary.loc[strategy_summary["strategy"].astype(str).eq(strategy_name)]
    if matches.empty and strategy_name == STRATEGY_NAME:
        matches = strategy_summary.loc[
            strategy_summary["strategy"].astype(str).str.startswith("ml_")
        ]
    if matches.empty:
        return pd.Series(dtype=object)
    return matches.iloc[0]


def _strategy_metric(strategy_summary: pd.DataFrame, strategy_name: str, metric: str) -> object:
    row = _strategy_row(strategy_summary, strategy_name)
    if row.empty:
        return pd.NA
    return _numeric_or_na(row.get(metric))


def _active_return_vs(strategy_summary: pd.DataFrame, benchmark_name: str) -> object:
    strategy_return = _numeric(_strategy_metric(strategy_summary, STRATEGY_NAME, "cumulative_return"))
    benchmark_return = _numeric(
        _strategy_metric(strategy_summary, benchmark_name, "cumulative_return")
    )
    if not math.isfinite(strategy_return) or not math.isfinite(benchmark_return):
        return pd.NA
    return strategy_return - benchmark_return


def _artifact_inventory(run_dir: Path) -> tuple[int, int]:
    if not run_dir.exists():
        return 0, 0
    file_count = 0
    total_bytes = 0
    for path in run_dir.rglob("*"):
        if not path.is_file():
            continue
        file_count += 1
        total_bytes += path.stat().st_size
    return file_count, total_bytes


def _artifact_status(run_dir: Path, file_count: int) -> str:
    if not run_dir.exists():
        return "missing"
    present = {name for name in REQUIRED_COMPLETE_ARTIFACTS if (run_dir / name).exists()}
    if present == set(REQUIRED_COMPLETE_ARTIFACTS):
        return "complete"
    if file_count <= 2:
        return "incomplete"
    if any((run_dir / name).exists() for name in DIAGNOSTIC_ARTIFACTS):
        return "summary_only"
    return "incomplete"


def discover_phase8_run_dirs(
    runs_root: str | Path = DEFAULT_RUNS_ROOT,
    *,
    experiment_prefix: str = DEFAULT_EXPERIMENT_PREFIX,
) -> list[Path]:
    resolved_root = Path(runs_root)
    if not resolved_root.exists():
        return []

    run_dirs: list[Path] = []
    for experiment_dir in sorted(resolved_root.iterdir()):
        if not experiment_dir.is_dir():
            continue
        if not experiment_dir.name.startswith(experiment_prefix):
            continue
        run_dirs.extend(sorted(path for path in experiment_dir.iterdir() if path.is_dir()))
    return run_dirs


def _run_row(run_dir: Path) -> dict[str, object]:
    file_count, total_bytes = _artifact_inventory(run_dir)
    strict_gate = _read_csv(run_dir, "strict_research_gate.csv")
    strategy_summary = _read_csv(run_dir, "strategy_summary.csv")
    phase8_summary = _read_csv(run_dir, "phase8_run_summary.csv")
    methodology_review = _read_csv(run_dir, "phase8_methodology_review.csv")
    bull_summary = _read_csv(run_dir, "phase8_bull_participation_summary.csv")
    score_summary = _read_csv(run_dir, "phase8_score_diagnostic_summary.csv")
    counterfactual_summary = _read_csv(run_dir, "phase8_bull_counterfactual_summary.csv")

    status = _artifact_status(run_dir, file_count)
    row: dict[str, object] = {
        "experiment": run_dir.parent.name,
        "run_id": run_dir.name,
        "run_dir": str(run_dir),
        "artifact_status": status,
        "recommended_artifact_action": "review_before_pruning",
        "latest_complete_for_experiment": False,
        "file_count": file_count,
        "size_mb": round(total_bytes / (1024 * 1024), 6),
        "strict_gate_present": (run_dir / "strict_research_gate.csv").exists(),
        "strategy_summary_present": (run_dir / "strategy_summary.csv").exists(),
        "methodology_review_present": (run_dir / "phase8_methodology_review.csv").exists(),
        "bull_participation_present": (
            run_dir / "phase8_bull_participation_summary.csv"
        ).exists(),
        "score_diagnostic_present": (run_dir / "phase8_score_diagnostic_summary.csv").exists(),
        "counterfactual_present": (
            run_dir / "phase8_bull_counterfactual_summary.csv"
        ).exists(),
        "strict_gate_passed": _strict_condition_passed(strict_gate, "overall"),
        "deployment_gate_passed": _methodology_passed(
            methodology_review,
            methodology_gate="deployment_gate",
            section="strict_gate",
        ),
        "risk_allocation_gate_passed": _methodology_passed(
            methodology_review,
            methodology_gate="risk_allocation_gate",
        ),
        "selection_coverage_gate_passed": _methodology_passed(
            methodology_review,
            methodology_gate="selection_coverage_gate",
        ),
        "target_support_gate_passed": _methodology_passed(
            methodology_review,
            methodology_gate="target_support_gate",
        ),
        "signal_validity_gate_passed": _methodology_passed(
            methodology_review,
            methodology_gate="signal_validity_gate",
        ),
        "bull_participation_gate_passed": _methodology_passed(
            methodology_review,
            methodology_gate="bull_participation_gate",
        ),
        "counterfactual_pass_available": _methodology_passed(
            methodology_review,
            methodology_gate="counterfactual_hypothesis",
            metric="counterfactual_pass_available",
        ),
        "strategy_cumulative_return": _strategy_metric(
            strategy_summary,
            STRATEGY_NAME,
            "cumulative_return",
        ),
        "buy_hold_cumulative_return": _strategy_metric(
            strategy_summary,
            "buy_hold",
            "cumulative_return",
        ),
        "active_return_vs_buy_hold": _active_return_vs(strategy_summary, "buy_hold"),
        "active_return_vs_btc_static_25": _active_return_vs(
            strategy_summary,
            "btc_static_25",
        ),
        "active_return_vs_btc_static_50": _active_return_vs(
            strategy_summary,
            "btc_static_50",
        ),
        "active_return_vs_btc_static_75": _active_return_vs(
            strategy_summary,
            "btc_static_75",
        ),
        "active_return_vs_btc_rebalanced_25": _active_return_vs(
            strategy_summary,
            "btc_rebalanced_25",
        ),
        "active_return_vs_btc_rebalanced_50": _active_return_vs(
            strategy_summary,
            "btc_rebalanced_50",
        ),
        "active_return_vs_btc_rebalanced_75": _active_return_vs(
            strategy_summary,
            "btc_rebalanced_75",
        ),
        "strategy_sharpe_like": _strategy_metric(strategy_summary, STRATEGY_NAME, "sharpe_like"),
        "buy_hold_sharpe_like": _strategy_metric(strategy_summary, "buy_hold", "sharpe_like"),
        "strategy_max_drawdown": _strategy_metric(
            strategy_summary,
            STRATEGY_NAME,
            "max_drawdown",
        ),
        "buy_hold_max_drawdown": _strategy_metric(
            strategy_summary,
            "buy_hold",
            "max_drawdown",
        ),
        "strategy_avg_gross_exposure": _strategy_metric(
            strategy_summary,
            STRATEGY_NAME,
            "avg_gross_exposure",
        ),
        "strategy_avg_turnover": _strategy_metric(
            strategy_summary,
            STRATEGY_NAME,
            "avg_turnover",
        ),
        "bull_upside_capture_ratio": _strategy_metric(
            strategy_summary,
            STRATEGY_NAME,
            "up_capture",
        ),
        "downside_capture_ratio": _strategy_metric(
            strategy_summary,
            STRATEGY_NAME,
            "down_capture",
        ),
        "selected_fold_fraction": _numeric_or_na(
            _summary_value(phase8_summary, "selected_fold_fraction")
        ),
        "no_valid_candidate_folds": _numeric_or_na(
            _summary_value(phase8_summary, "no_valid_candidate_folds")
        ),
        "target_tier_25_global_fraction": _numeric_or_na(
            _summary_value(phase8_summary, "target_tier_25_global_fraction")
        ),
        "target_tier_50_global_fraction": _numeric_or_na(
            _summary_value(phase8_summary, "target_tier_50_global_fraction")
        ),
        "predicted_tier_25_fraction": _numeric_or_na(
            _summary_value(phase8_summary, "predicted_tier_25_fraction")
        ),
        "predicted_tier_50_fraction": _numeric_or_na(
            _summary_value(phase8_summary, "predicted_tier_50_fraction")
        ),
        "predicted_tier_100_fraction": _numeric_or_na(
            _summary_value(score_summary, "predicted_tier_100_fraction")
        ),
        "score_target_weight_correlation": _numeric_or_na(
            _summary_value(score_summary, "score_target_weight_correlation")
        ),
        "score_forward_return_correlation": _numeric_or_na(
            _summary_value(score_summary, "score_forward_return_correlation")
        ),
        "score_realized_utility_correlation": _numeric_or_na(
            _summary_value(score_summary, "score_realized_utility_correlation")
        ),
        "any_selected_oos_predicted_tier_100": _summary_value(
            score_summary,
            "any_selected_oos_predicted_tier_100",
        ),
        "gate_bull_average_long_exposure": _numeric_or_na(
            _summary_value(bull_summary, "gate_bull_average_long_exposure")
        ),
        "gate_bull_active_return_sum": _numeric_or_na(
            _summary_value(bull_summary, "gate_bull_active_return_sum")
        ),
        "gate_bull_underexposed_positive_benchmark_fraction": _numeric_or_na(
            _summary_value(
                bull_summary,
                "gate_bull_underexposed_positive_benchmark_fraction",
            )
        ),
        "gate_bull_underexposed_positive_benchmark_return_sum": _numeric_or_na(
            _summary_value(
                bull_summary,
                "gate_bull_underexposed_positive_benchmark_return_sum",
            )
        ),
        "force_runtime_bull_100_cumulative_return": _numeric_or_na(
            _counterfactual_summary_value(
                counterfactual_summary,
                scenario="force_runtime_bull_100",
                metric="cumulative_return",
            )
        ),
        "force_runtime_bull_100_avg_long_exposure": _numeric_or_na(
            _counterfactual_summary_value(
                counterfactual_summary,
                scenario="force_runtime_bull_100",
                metric="avg_long_exposure",
            )
        ),
        "buy_hold_gate_bull_model_elsewhere_cumulative_return": _numeric_or_na(
            _counterfactual_summary_value(
                counterfactual_summary,
                scenario="buy_hold_gate_bull_model_elsewhere",
                metric="cumulative_return",
            )
        ),
        "buy_hold_gate_bull_model_elsewhere_avg_long_exposure": _numeric_or_na(
            _counterfactual_summary_value(
                counterfactual_summary,
                scenario="buy_hold_gate_bull_model_elsewhere",
                metric="avg_long_exposure",
            )
        ),
    }
    if status in {"missing", "incomplete"}:
        row["recommended_artifact_action"] = "archive_or_prune_after_manifest"
    elif status == "summary_only":
        row["recommended_artifact_action"] = "review_before_pruning"
    return row


def _mark_latest_complete(rows: list[dict[str, object]]) -> None:
    latest_by_experiment: dict[str, dict[str, object]] = {}
    for row in rows:
        if row["artifact_status"] != "complete":
            continue
        experiment = str(row["experiment"])
        current = latest_by_experiment.get(experiment)
        if current is None or str(row["run_id"]) > str(current["run_id"]):
            latest_by_experiment[experiment] = row

    for row in rows:
        if row["artifact_status"] != "complete":
            continue
        latest = latest_by_experiment.get(str(row["experiment"]))
        is_latest = latest is row
        row["latest_complete_for_experiment"] = is_latest
        row["recommended_artifact_action"] = (
            "keep_latest_complete" if is_latest else "keep_for_grid_comparison"
        )


def build_phase8_grid_comparison(
    *,
    runs_root: str | Path = DEFAULT_RUNS_ROOT,
    run_dirs: Sequence[str | Path] | None = None,
    experiment_prefix: str = DEFAULT_EXPERIMENT_PREFIX,
) -> pd.DataFrame:
    resolved_run_dirs = (
        [Path(path) for path in run_dirs]
        if run_dirs
        else discover_phase8_run_dirs(runs_root, experiment_prefix=experiment_prefix)
    )
    rows = [_run_row(path) for path in resolved_run_dirs]
    _mark_latest_complete(rows)
    return pd.DataFrame(rows, columns=PHASE8_GRID_COMPARISON_COLUMNS)


def write_phase8_grid_comparison(
    *,
    runs_root: str | Path = DEFAULT_RUNS_ROOT,
    run_dirs: Sequence[str | Path] | None = None,
    output_path: str | Path | None = None,
    experiment_prefix: str = DEFAULT_EXPERIMENT_PREFIX,
) -> Path:
    resolved_output_path = (
        Path(output_path)
        if output_path is not None
        else Path(runs_root) / "phase8_btc_grid_comparison.csv"
    )
    resolved_output_path.parent.mkdir(parents=True, exist_ok=True)
    build_phase8_grid_comparison(
        runs_root=runs_root,
        run_dirs=run_dirs,
        experiment_prefix=experiment_prefix,
    ).to_csv(resolved_output_path, index=False)
    return resolved_output_path
