from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

from marketlab.config import ExperimentConfig
from marketlab.reports.risk_diagnostics import (
    build_covariance_summary,
    build_factor_summary,
)

EXPOSURE_SUMMARY_COLUMNS = [
    "strategy",
    "avg_long_exposure",
    "avg_short_exposure",
    "avg_gross_exposure",
    "avg_net_exposure",
    "avg_cash_weight",
    "avg_engine_cash_weight",
    "avg_active_positions",
    "max_position_weight",
    "max_group_weight",
]

BENCHMARK_SUMMARY_COLUMNS = [
    "strategy",
    "benchmark_strategy",
    "excess_cumulative_return",
    "annualized_excess_return",
    "tracking_error",
    "information_ratio",
    "correlation_to_benchmark",
    "up_capture",
    "down_capture",
]

COST_SENSITIVITY_SUMMARY_COLUMNS = [
    "strategy",
    "bps_per_trade",
    "cumulative_return",
    "annualized_return",
    "max_drawdown",
    "cost_drag",
]
FACTOR_DIAGNOSTICS_DISPLAY_COLUMNS = [
    "strategy",
    "factor",
    "beta_like_exposure",
    "mean_factor_return",
    "mean_factor_contribution",
    "alpha_like_intercept",
    "mean_strategy_return",
    "modeled_mean_return",
    "r_squared",
]


def _markdown_table(frame: pd.DataFrame) -> str:
    columns = list(frame.columns)
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join(["---"] * len(columns)) + " |"
    rows = [
        "| " + " | ".join(str(value) for value in row) + " |"
        for row in frame.itertuples(index=False, name=None)
    ]
    return "\n".join([header, separator, *rows])


def _scope_lines(performance: pd.DataFrame) -> list[str]:
    strategy_names = performance["strategy"].drop_duplicates().tolist()
    has_ml_strategy = any(strategy.startswith("ml_") for strategy in strategy_names)
    if has_ml_strategy:
        return [
            "- Phase 2 baseline plus ML experiment",
            "- Performance is sliced to the shared walk-forward OOS window",
        ]
    return ["- Baseline-only experiment"]


def _trend_signal_gate_lines(strategy_summary: pd.DataFrame) -> list[str]:
    strategy_names = set(strategy_summary["strategy"].astype(str))
    gated_strategies = [
        strategy
        for strategy in (
            "indicator_stack",
            "chart_patterns",
            "pattern_exit_overlay",
            "pattern_meta_label_exit_overlay",
        )
        if strategy in strategy_names
    ]
    if not gated_strategies:
        return []
    if "buy_hold" not in strategy_names:
        return ["- Research gate: unavailable because `buy_hold` is missing."]

    buy_hold_row = strategy_summary.loc[
        strategy_summary["strategy"].astype(str) == "buy_hold"
    ].iloc[0]
    lines: list[str] = []
    for strategy in gated_strategies:
        strategy_row = strategy_summary.loc[
            strategy_summary["strategy"].astype(str) == strategy
        ].iloc[0]
        net_return_delta = float(strategy_row["cumulative_return"]) - float(
            buy_hold_row["cumulative_return"]
        )
        sharpe_delta = float(strategy_row["sharpe_like"]) - float(buy_hold_row["sharpe_like"])
        max_drawdown_delta = float(strategy_row["max_drawdown"]) - float(
            buy_hold_row["max_drawdown"]
        )
        passed = net_return_delta > 0.0 and (sharpe_delta > 0.0 or max_drawdown_delta > 0.0)
        verdict = "pass" if passed else "fail"
        lines.extend(
            [
                f"- `{strategy}` gate: `{verdict}` versus `buy_hold`.",
                f"- `{strategy}` net cumulative return delta: {net_return_delta:.6f}",
                f"- `{strategy}` Sharpe-like delta: {sharpe_delta:.6f}",
                f"- `{strategy}` max drawdown delta: {max_drawdown_delta:.6f}",
            ]
        )
    lines.append(
        "- Paper-shadow work remains blocked until a research strategy passes on reviewed runs."
    )
    return lines


def _headline_lines(
    metrics: pd.DataFrame,
    model_summary: pd.DataFrame | None,
) -> list[str]:
    if metrics.empty:
        return []

    lines: list[str] = []
    best_strategy = metrics.sort_values(
        ["cumulative_return", "strategy"],
        ascending=[False, True],
    ).iloc[0]
    lines.append(
        "- Best overall strategy by cumulative return: "
        f"`{best_strategy['strategy']}` ({best_strategy['cumulative_return']:.6f})"
    )

    ml_metrics = metrics.loc[metrics["strategy"].astype(str).str.startswith("ml_")]
    if not ml_metrics.empty:
        best_ml = ml_metrics.sort_values(
            ["cumulative_return", "strategy"],
            ascending=[False, True],
        ).iloc[0]
        lines.append(
            "- Best ML strategy by cumulative return: "
            f"`{best_ml['strategy']}` ({best_ml['cumulative_return']:.6f})"
        )

    if model_summary is not None and not model_summary.empty:
        if {"model_name", "mean_roc_auc"}.issubset(model_summary.columns):
            ranked_models = model_summary.dropna(subset=["mean_roc_auc"]).sort_values(
                ["mean_roc_auc", "model_name"],
                ascending=[False, True],
            )
            if ranked_models.empty:
                lines.append("- Best model by mean ROC AUC: n/a")
            else:
                best_model = ranked_models.iloc[0]
                lines.append(
                    "- Best model by mean ROC AUC: "
                    f"`{best_model['model_name']}` ({best_model['mean_roc_auc']:.6f})"
                )

        if {"model_name", "mean_top_bucket_return"}.issubset(model_summary.columns):
            ranked_top_bucket = model_summary.dropna(
                subset=["mean_top_bucket_return"]
            ).sort_values(
                ["mean_top_bucket_return", "model_name"],
                ascending=[False, True],
            )
            if ranked_top_bucket.empty:
                lines.append("- Best model by mean top-bucket return: n/a")
            else:
                best_top_bucket = ranked_top_bucket.iloc[0]
                lines.append(
                    "- Best model by mean top-bucket return: "
                    f"`{best_top_bucket['model_name']}` ({best_top_bucket['mean_top_bucket_return']:.6f})"
                )

        if {"model_name", "mean_top_bottom_spread"}.issubset(model_summary.columns):
            ranked_spread = model_summary.dropna(
                subset=["mean_top_bottom_spread"]
            ).sort_values(
                ["mean_top_bottom_spread", "model_name"],
                ascending=[False, True],
            )
            if ranked_spread.empty:
                lines.append("- Best model by mean top-bottom spread: n/a")
            else:
                best_spread = ranked_spread.iloc[0]
                lines.append(
                    "- Best model by mean top-bottom spread: "
                    f"`{best_spread['model_name']}` ({best_spread['mean_top_bottom_spread']:.6f})"
                )

    return lines


def _monthly_returns_table(monthly_returns: pd.DataFrame) -> str:
    if monthly_returns.empty:
        return "No monthly return rows were generated."

    pivot = (
        monthly_returns.loc[:, ["month", "strategy", "net_return"]]
        .pivot(index="month", columns="strategy", values="net_return")
        .reset_index()
    )
    pivot.columns.name = None
    return _markdown_table(pivot.round(6))


def _turnover_costs_table(turnover_costs: pd.DataFrame) -> str:
    summary = (
        turnover_costs.groupby("strategy", as_index=False)
        .agg(
            avg_turnover=("turnover", "mean"),
            total_turnover=("turnover", "sum"),
            avg_cost_return=("cost_return", "mean"),
            total_cost_return=("cost_return", "sum"),
        )
        .sort_values("strategy")
        .reset_index(drop=True)
    )
    return _markdown_table(summary.round(6))


def _walk_forward_diagnostics_lines(fold_diagnostics: pd.DataFrame) -> list[str]:
    used_count = int((fold_diagnostics["status"] == "used").sum())
    skipped = fold_diagnostics.loc[fold_diagnostics["status"] == "skipped"].copy()
    lines = [
        f"- Used candidates: {used_count}",
        f"- Skipped candidates: {len(skipped)}",
    ]
    if skipped.empty:
        lines.append("- No candidate folds were skipped.")
        return lines

    skipped = skipped.loc[
        :,
        [
            "candidate_id",
            "test_start",
            "test_end",
            "skip_reasons",
            "train_rows",
            "test_rows",
            "train_positive_rate",
            "test_positive_rate",
        ],
    ].copy()
    for column in ["train_positive_rate", "test_positive_rate"]:
        skipped[column] = skipped[column].round(6)
    lines.extend(["", _markdown_table(skipped)])
    return lines


def _calibration_summary_table(model_summary: pd.DataFrame) -> str:
    summary = model_summary.loc[:, ["model_name", "mean_ece", "mean_max_calibration_gap"]].copy()
    return _markdown_table(_display_frame(summary))


def _threshold_highlights(threshold_diagnostics: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for model_name, model_rows in threshold_diagnostics.groupby("model_name", sort=True):
        aggregated = (
            model_rows.groupby("threshold", as_index=False)
            .agg(
                mean_f1=("f1", "mean"),
                mean_balanced_accuracy=("balanced_accuracy", "mean"),
            )
            .sort_values("threshold")
            .reset_index(drop=True)
        )
        best_f1 = aggregated.sort_values(
            ["mean_f1", "threshold"],
            ascending=[False, True],
        ).iloc[0]
        best_balanced = aggregated.sort_values(
            ["mean_balanced_accuracy", "threshold"],
            ascending=[False, True],
        ).iloc[0]
        rows.append(
            {
                "model_name": model_name,
                "threshold_max_f1": float(best_f1["threshold"]),
                "max_f1": float(best_f1["mean_f1"]),
                "threshold_max_balanced_accuracy": float(best_balanced["threshold"]),
                "max_balanced_accuracy": float(best_balanced["mean_balanced_accuracy"]),
            }
        )
    return pd.DataFrame(rows)


def _relative_image_line(report_path: Path, image_path: Path, alt_text: str) -> str:
    relative_path = os.path.relpath(image_path, start=report_path.parent)
    return f"![{alt_text}]({relative_path})"


def _relative_artifact_line(report_path: Path, artifact_path: Path, label: str) -> str:
    relative_path = os.path.relpath(artifact_path, start=report_path.parent)
    return f"- {label}: [{artifact_path.name}]({relative_path})"


def _display_frame(frame: pd.DataFrame) -> pd.DataFrame:
    display = frame.copy()
    numeric_columns = display.select_dtypes(include="number").columns
    if len(numeric_columns) > 0:
        display.loc[:, numeric_columns] = display.loc[:, numeric_columns].round(6)
    return display


def _exposure_summary_lines(strategy_summary: pd.DataFrame) -> list[str]:
    if not set(EXPOSURE_SUMMARY_COLUMNS).issubset(strategy_summary.columns):
        return []

    lines = [
        _markdown_table(_display_frame(strategy_summary.loc[:, EXPOSURE_SUMMARY_COLUMNS]))
    ]
    lines.extend(
        [
            "",
            "- Lower drawdown can reflect lower gross exposure or more cash, not necessarily better selection.",
            "- `avg_cash_weight` is exposure-style slack; `avg_engine_cash_weight` is the engine's carried cash or collateral weight.",
        ]
    )
    if strategy_summary["max_group_weight"].notna().any():
        lines.append("- Group concentration details are also persisted in `group_exposure.csv`.")
    return lines


def _benchmark_summary_lines(strategy_summary: pd.DataFrame) -> list[str]:
    if not set(BENCHMARK_SUMMARY_COLUMNS).issubset(strategy_summary.columns):
        return []
    if not strategy_summary["benchmark_strategy"].astype(str).str.len().gt(0).any():
        return []

    lines = [
        _markdown_table(_display_frame(strategy_summary.loc[:, BENCHMARK_SUMMARY_COLUMNS]))
    ]
    lines.extend(
        [
            "",
            "- Benchmark-relative metrics separate absolute return from active return and active risk.",
            "- Lower tracking error does not imply outperformance; it only means the strategy stayed closer to the benchmark path.",
            "- Daily active return and relative equity are also persisted in `benchmark_relative.csv`.",
        ]
    )
    return lines


def _factor_diagnostics_lines(
    report_path: Path,
    factor_diagnostics: pd.DataFrame | None,
    factor_diagnostics_path: Path | None,
) -> list[str]:
    if (
        factor_diagnostics is None
        or factor_diagnostics.empty
        or factor_diagnostics_path is None
    ):
        return []

    factor_summary = build_factor_summary(factor_diagnostics)
    detail_frame = factor_diagnostics.loc[:, FACTOR_DIAGNOSTICS_DISPLAY_COLUMNS]
    relative_path = os.path.relpath(factor_diagnostics_path, start=report_path.parent)
    return [
        _markdown_table(_display_frame(factor_summary)),
        "",
        _markdown_table(_display_frame(detail_frame)),
        "",
        "- Factor attribution is descriptive and uses realized net returns plus local factor inputs only.",
        "- These diagnostics do not feed optimizer weights, scenario selection, or model ranking.",
        f"- Factor diagnostics artifact: [{factor_diagnostics_path.name}]({relative_path})",
    ]


def _covariance_diagnostics_lines(
    report_path: Path,
    covariance_diagnostics: pd.DataFrame | None,
    covariance_diagnostics_path: Path | None,
) -> list[str]:
    if (
        covariance_diagnostics is None
        or covariance_diagnostics.empty
        or covariance_diagnostics_path is None
    ):
        return []

    covariance_summary = build_covariance_summary(covariance_diagnostics)
    relative_path = os.path.relpath(covariance_diagnostics_path, start=report_path.parent)
    return [
        _markdown_table(_display_frame(covariance_summary)),
        "",
        "- Covariance diagnostics reflect the regularized matrix used by the optimizer at each rebalance window.",
        "- Pairwise correlation summaries exclude the diagonal and aggregate across optimizer windows.",
        f"- Covariance diagnostics artifact: [{covariance_diagnostics_path.name}]({relative_path})",
    ]


def _black_litterman_view_lines(
    config: ExperimentConfig,
    report_path: Path,
    black_litterman_assumptions_path: Path | None,
) -> list[str]:
    optimized = config.baselines.optimized
    if (
        optimized.method != "black_litterman"
        or not optimized.views
        or black_litterman_assumptions_path is None
    ):
        return []

    rows = []
    for view in optimized.views:
        weights = ", ".join(
            f"{symbol}:{view.weights[symbol]:+g}"
            for symbol in config.data.symbols
            if symbol in view.weights and abs(view.weights[symbol]) > 0.0
        )
        rows.append(
            {
                "name": view.name,
                "weights": weights,
                "view_return": view.view_return,
            }
        )

    lines = [
        _markdown_table(_display_frame(pd.DataFrame(rows))),
        "",
        "- View weights are signed basket coefficients and are used as written.",
        "- The default view uncertainty rule is `Omega = diag(P * tau * Sigma * P^T)`.",
    ]
    relative_path = os.path.relpath(black_litterman_assumptions_path, start=report_path.parent)
    lines.append(
        f"- Assumptions artifact: [{black_litterman_assumptions_path.name}]({relative_path})"
    )
    return lines


def _cost_sensitivity_lines(cost_sensitivity: pd.DataFrame) -> list[str]:
    if not set(COST_SENSITIVITY_SUMMARY_COLUMNS).issubset(cost_sensitivity.columns):
        return []

    lines = [
        _markdown_table(_display_frame(cost_sensitivity.loc[:, COST_SENSITIVITY_SUMMARY_COLUMNS]))
    ]
    lines.extend(
        [
            "",
            "- Zero-cost rows are theoretical gross-return baselines, not executable outcomes.",
            "- Higher implementation cost can worsen return and drawdown without changing signal quality.",
        ]
    )
    return lines


def _pattern_meta_threshold_sweep_lines(
    threshold_sweep: pd.DataFrame | None,
    report_path: Path,
    threshold_sweep_path: Path | None,
) -> list[str]:
    if threshold_sweep is None or threshold_sweep.empty:
        return []

    display_columns = [
        "threshold",
        "cumulative_return",
        "annualized_return",
        "max_drawdown",
        "sharpe_like",
        "total_turnover",
        "cost_drag",
        "exit_count",
        "cash_bar_count",
        "average_exposure",
        "excess_cumulative_return",
    ]
    lines = [_markdown_table(_display_frame(threshold_sweep.loc[:, display_columns]))]
    lines.extend(
        [
            "",
            "- Threshold sweeps are research diagnostics and must not be treated as an automatically selected production setting.",
            "- Better net return can reflect threshold overfit; compare turnover and drawdown before changing configs.",
            "- High-threshold rows with very low exit counts are close to buy-and-hold abstention, not stronger pattern evidence.",
        ]
    )
    if threshold_sweep_path is not None:
        lines.append(
            _relative_artifact_line(
                report_path,
                threshold_sweep_path,
                "Pattern meta threshold sweep",
            )
        )
    return lines


def _pattern_meta_tuning_lines(
    tuning_candidates: pd.DataFrame | None,
    tuning_selections: pd.DataFrame | None,
    partial_sweep: pd.DataFrame | None,
    report_path: Path,
    tuning_candidates_path: Path | None,
    tuning_selections_path: Path | None,
    partial_sweep_path: Path | None,
) -> list[str]:
    lines: list[str] = []
    if tuning_selections is not None and not tuning_selections.empty:
        display_columns = [
            "fold_id",
            "selected_threshold",
            "passed_gate",
            "excess_cumulative_return",
            "drawdown_delta",
            "exit_count",
            "average_exposure",
        ]
        lines.extend(
            [
                "Nested tuning selections:",
                "",
                _markdown_table(_display_frame(tuning_selections.loc[:, display_columns])),
            ]
        )
    if partial_sweep is not None and not partial_sweep.empty:
        display_columns = [
            "partial_threshold",
            "full_threshold",
            "cumulative_return",
            "max_drawdown",
            "total_turnover",
            "exit_count",
            "partial_bar_count",
            "average_exposure",
            "excess_cumulative_return",
        ]
        if lines:
            lines.append("")
        lines.extend(
            [
                "Partial-exposure threshold sweep:",
                "",
                _markdown_table(_display_frame(partial_sweep.loc[:, display_columns])),
            ]
        )
    if not lines:
        return []
    lines.extend(
        [
            "",
            "- Tuning rows are research diagnostics; a pass requires out-of-sample excess return and no worse drawdown after costs.",
            "- Near-1.0 average exposure with few exits is treated as abstention, not active edge.",
        ]
    )
    for artifact_path, label in [
        (tuning_candidates_path, "Pattern meta tuning candidates"),
        (tuning_selections_path, "Pattern meta tuning selections"),
        (partial_sweep_path, "Pattern partial threshold sweep"),
    ]:
        if artifact_path is not None:
            lines.append(_relative_artifact_line(report_path, artifact_path, label))
    return lines


def _ml_strategy_threshold_sweep_lines(
    threshold_sweep: pd.DataFrame | None,
    report_path: Path,
    threshold_sweep_path: Path | None,
) -> list[str]:
    if threshold_sweep is None or threshold_sweep.empty:
        return []

    display_columns = [
        "model_name",
        "threshold",
        "cumulative_return",
        "max_drawdown",
        "sharpe_like",
        "total_turnover",
        "exposure_changes",
        "average_exposure",
        "buy_hold_cumulative_return",
        "excess_cumulative_return",
        "best_comparison_strategy",
        "passed_gate",
    ]
    lines = [_markdown_table(_display_frame(threshold_sweep.loc[:, display_columns]))]
    lines.extend(
        [
            "",
            "- The pass gate requires positive net excess return versus `buy_hold` after costs.",
            "- The activity guardrail rejects near-buy-and-hold abstention by requiring enough exposure changes and average exposure below the configured maximum.",
            "- Pattern and rule baselines remain in the strategy summary so ML thresholds can be compared against the full Phase 8 research set.",
        ]
    )
    if threshold_sweep_path is not None:
        lines.append(
            _relative_artifact_line(
                report_path,
                threshold_sweep_path,
                "ML strategy threshold sweep",
            )
        )
    return lines


def _ml_strategy_tuning_lines(
    tuning_candidates: pd.DataFrame | None,
    tuning_selections: pd.DataFrame | None,
    report_path: Path,
    tuning_candidates_path: Path | None,
    tuning_selections_path: Path | None,
) -> list[str]:
    if (
        tuning_candidates is None
        and tuning_selections is None
        and tuning_candidates_path is None
        and tuning_selections_path is None
    ):
        return []

    lines: list[str] = []
    if tuning_selections is not None and not tuning_selections.empty:
        selection_columns = [
            "fold_id",
            "selection_status",
            "selected_model_name",
            "selected_threshold",
            "passed_gate",
            "excess_cumulative_return",
            "sharpe_like_delta",
            "drawdown_delta",
            "exposure_changes",
            "average_exposure",
        ]
        lines.append(_markdown_table(_display_frame(tuning_selections.loc[:, selection_columns])))

    if tuning_candidates is not None and not tuning_candidates.empty:
        candidate_columns = [
            "fold_id",
            "model_name",
            "threshold",
            "cumulative_return",
            "max_drawdown",
            "sharpe_like",
            "excess_cumulative_return",
            "sharpe_like_delta",
            "drawdown_delta",
            "active_candidate",
            "passed_gate",
        ]
        lines.extend(
            [
                "",
                "Top validation candidates:",
                _markdown_table(
                    _display_frame(tuning_candidates.loc[:, candidate_columns].head(20))
                ),
            ]
        )

    lines.extend(
        [
            "",
            "- Candidate selection uses only the validation tail inside each outer training fold.",
            "- Selected models are refit on the full outer training fold before scoring the outer test fold.",
            "- The pass gate requires net excess return and either Sharpe-like or drawdown improvement versus `buy_hold` after costs.",
        ]
    )
    for artifact_path, label in [
        (tuning_candidates_path, "ML strategy tuning candidates"),
        (tuning_selections_path, "ML strategy tuning selections"),
    ]:
        if artifact_path is not None:
            lines.append(_relative_artifact_line(report_path, artifact_path, label))
    return lines


def _signal_inspection_lines(
    config: ExperimentConfig,
    report_path: Path,
    indicator_diagnostics_path: Path | None,
    signal_price_overlay_plot_path: Path | None,
    signal_confirmations_plot_path: Path | None,
    signal_performance_focus_plot_path: Path | None,
    pattern_diagnostics_path: Path | None,
    pattern_exit_overlay_diagnostics_path: Path | None,
    pattern_meta_labels_path: Path | None,
    pattern_meta_predictions_path: Path | None,
    pattern_meta_fold_diagnostics_path: Path | None,
    pattern_meta_threshold_sweep_path: Path | None,
    pattern_meta_tuning_candidates_path: Path | None,
    pattern_meta_tuning_selections_path: Path | None,
    pattern_partial_exposure_diagnostics_path: Path | None,
    pattern_partial_threshold_sweep_path: Path | None,
    pattern_price_overlay_plot_path: Path | None,
    pattern_detections_plot_path: Path | None,
    pattern_detection_windows_plot_path: Path | None,
    pattern_performance_focus_plot_path: Path | None,
) -> list[str]:
    paths = [
        signal_price_overlay_plot_path,
        signal_confirmations_plot_path,
        signal_performance_focus_plot_path,
        pattern_price_overlay_plot_path,
        pattern_detections_plot_path,
        pattern_detection_windows_plot_path,
        pattern_performance_focus_plot_path,
    ]
    if (
        indicator_diagnostics_path is None
        and pattern_diagnostics_path is None
        and pattern_exit_overlay_diagnostics_path is None
        and pattern_meta_labels_path is None
        and pattern_meta_predictions_path is None
        and pattern_meta_fold_diagnostics_path is None
        and pattern_meta_threshold_sweep_path is None
        and pattern_meta_tuning_candidates_path is None
        and pattern_meta_tuning_selections_path is None
        and pattern_partial_exposure_diagnostics_path is None
        and pattern_partial_threshold_sweep_path is None
        and not any(path is not None for path in paths)
    ):
        return []

    focus_start = config.evaluation.focus_start or "run start"
    focus_end = config.evaluation.focus_end or "run end"
    lines = [
        f"- Focus window: `{focus_start}` to `{focus_end}`",
        "- Signals are evaluated at completed bar close and become target weights on the next available bar.",
    ]
    if indicator_diagnostics_path is not None:
        lines.append(
            _relative_artifact_line(report_path, indicator_diagnostics_path, "Indicator diagnostics")
        )
    if pattern_diagnostics_path is not None:
        lines.append(
            _relative_artifact_line(report_path, pattern_diagnostics_path, "Chart-pattern diagnostics")
        )
    if pattern_exit_overlay_diagnostics_path is not None:
        lines.append(
            _relative_artifact_line(
                report_path,
                pattern_exit_overlay_diagnostics_path,
                "Pattern exit overlay diagnostics",
            )
        )
    if pattern_meta_labels_path is not None:
        lines.append(
            _relative_artifact_line(report_path, pattern_meta_labels_path, "Pattern meta labels")
        )
    if pattern_meta_predictions_path is not None:
        lines.append(
            _relative_artifact_line(
                report_path,
                pattern_meta_predictions_path,
                "Pattern meta predictions",
            )
        )
    if pattern_meta_fold_diagnostics_path is not None:
        lines.append(
            _relative_artifact_line(
                report_path,
                pattern_meta_fold_diagnostics_path,
                "Pattern meta fold diagnostics",
            )
        )
    if pattern_meta_threshold_sweep_path is not None:
        lines.append(
            _relative_artifact_line(
                report_path,
                pattern_meta_threshold_sweep_path,
                "Pattern meta threshold sweep",
            )
        )
    if pattern_meta_tuning_candidates_path is not None:
        lines.append(
            _relative_artifact_line(
                report_path,
                pattern_meta_tuning_candidates_path,
                "Pattern meta tuning candidates",
            )
        )
    if pattern_meta_tuning_selections_path is not None:
        lines.append(
            _relative_artifact_line(
                report_path,
                pattern_meta_tuning_selections_path,
                "Pattern meta tuning selections",
            )
        )
    if pattern_partial_exposure_diagnostics_path is not None:
        lines.append(
            _relative_artifact_line(
                report_path,
                pattern_partial_exposure_diagnostics_path,
                "Pattern partial exposure diagnostics",
            )
        )
    if pattern_partial_threshold_sweep_path is not None:
        lines.append(
            _relative_artifact_line(
                report_path,
                pattern_partial_threshold_sweep_path,
                "Pattern partial threshold sweep",
            )
        )

    image_specs = [
        ("Signal Price Overlay", signal_price_overlay_plot_path),
        ("Signal Confirmations", signal_confirmations_plot_path),
        ("Focused Signal Performance", signal_performance_focus_plot_path),
        ("Chart Pattern Price Overlay", pattern_price_overlay_plot_path),
        ("Chart Pattern Detections", pattern_detections_plot_path),
        ("Chart Pattern Detection Windows", pattern_detection_windows_plot_path),
        ("Focused Pattern Performance", pattern_performance_focus_plot_path),
    ]
    for alt_text, plot_path in image_specs:
        if plot_path is not None and plot_path.exists():
            lines.extend(["", _relative_image_line(report_path, plot_path, alt_text)])
    return lines


def _section(title: str, body_lines: list[str]) -> list[str]:
    return [f"## {title}", "", *body_lines, ""]


def write_markdown_report(
    config: ExperimentConfig,
    metrics: pd.DataFrame,
    performance: pd.DataFrame,
    path: str | Path,
    model_summary: pd.DataFrame | None = None,
    fold_summary: pd.DataFrame | None = None,
    strategy_summary: pd.DataFrame | None = None,
    monthly_returns: pd.DataFrame | None = None,
    turnover_costs: pd.DataFrame | None = None,
    cost_sensitivity: pd.DataFrame | None = None,
    pattern_meta_threshold_sweep: pd.DataFrame | None = None,
    pattern_meta_tuning_candidates: pd.DataFrame | None = None,
    pattern_meta_tuning_selections: pd.DataFrame | None = None,
    pattern_partial_threshold_sweep: pd.DataFrame | None = None,
    ml_strategy_threshold_sweep: pd.DataFrame | None = None,
    ml_strategy_tuning_candidates: pd.DataFrame | None = None,
    ml_strategy_tuning_selections: pd.DataFrame | None = None,
    fold_diagnostics: pd.DataFrame | None = None,
    threshold_diagnostics: pd.DataFrame | None = None,
    calibration_curves_plot_path: Path | None = None,
    score_histograms_plot_path: Path | None = None,
    threshold_sweeps_plot_path: Path | None = None,
    factor_diagnostics: pd.DataFrame | None = None,
    factor_diagnostics_path: Path | None = None,
    covariance_diagnostics: pd.DataFrame | None = None,
    covariance_diagnostics_path: Path | None = None,
    black_litterman_assumptions_path: Path | None = None,
    indicator_diagnostics_path: Path | None = None,
    signal_price_overlay_plot_path: Path | None = None,
    signal_confirmations_plot_path: Path | None = None,
    signal_performance_focus_plot_path: Path | None = None,
    pattern_diagnostics_path: Path | None = None,
    pattern_exit_overlay_diagnostics_path: Path | None = None,
    pattern_meta_labels_path: Path | None = None,
    pattern_meta_predictions_path: Path | None = None,
    pattern_meta_fold_diagnostics_path: Path | None = None,
    pattern_meta_threshold_sweep_path: Path | None = None,
    pattern_meta_tuning_candidates_path: Path | None = None,
    pattern_meta_tuning_selections_path: Path | None = None,
    pattern_partial_exposure_diagnostics_path: Path | None = None,
    pattern_partial_threshold_sweep_path: Path | None = None,
    ml_strategy_threshold_sweep_path: Path | None = None,
    ml_strategy_tuning_candidates_path: Path | None = None,
    ml_strategy_tuning_selections_path: Path | None = None,
    pattern_price_overlay_plot_path: Path | None = None,
    pattern_detections_plot_path: Path | None = None,
    pattern_detection_windows_plot_path: Path | None = None,
    pattern_performance_focus_plot_path: Path | None = None,
) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    date_min = performance["date"].min().date().isoformat()
    date_max = performance["date"].max().date().isoformat()
    metrics_table = _markdown_table(_display_frame(metrics))
    strategy_lines = [
        f"- `{strategy}`"
        for strategy in performance["strategy"].drop_duplicates().tolist()
    ]

    content_lines = [f"# {config.experiment_name}", ""]
    content_lines.extend(
        _section(
            "Scope",
            [
                *_scope_lines(performance),
                f"- Symbols: {', '.join(config.data.symbols)}",
                f"- Window: {date_min} to {date_max}",
                f"- Cost model: {config.portfolio.costs.bps_per_trade} bps per unit turnover",
            ],
        )
    )
    content_lines.extend(_section("Strategies", strategy_lines))

    headline_lines = _headline_lines(metrics, model_summary)
    if headline_lines:
        content_lines.extend(_section("Headline Outcomes", headline_lines))

    content_lines.extend(_section("Strategy Metrics", [metrics_table]))

    if strategy_summary is not None and not strategy_summary.empty:
        content_lines.extend(
            _section("Strategy Summary", [_markdown_table(_display_frame(strategy_summary))])
        )
        trend_signal_gate_lines = _trend_signal_gate_lines(strategy_summary)
        if trend_signal_gate_lines:
            content_lines.extend(
                _section("Trend-Signal Acceptance Gate", trend_signal_gate_lines)
            )
        exposure_lines = _exposure_summary_lines(strategy_summary)
        if exposure_lines:
            content_lines.extend(_section("Exposure Summary", exposure_lines))
        benchmark_lines = _benchmark_summary_lines(strategy_summary)
        if benchmark_lines:
            content_lines.extend(_section("Benchmark-Relative Summary", benchmark_lines))

    factor_lines = _factor_diagnostics_lines(
        output_path,
        factor_diagnostics,
        factor_diagnostics_path,
    )
    if factor_lines:
        content_lines.extend(_section("Factor Attribution Diagnostics", factor_lines))

    covariance_lines = _covariance_diagnostics_lines(
        output_path,
        covariance_diagnostics,
        covariance_diagnostics_path,
    )
    if covariance_lines:
        content_lines.extend(_section("Covariance Diagnostics", covariance_lines))

    black_litterman_lines = _black_litterman_view_lines(
        config,
        output_path,
        black_litterman_assumptions_path,
    )
    if black_litterman_lines:
        content_lines.extend(_section("Black-Litterman Assumptions", black_litterman_lines))

    if monthly_returns is not None and not monthly_returns.empty:
        content_lines.extend(
            _section("Monthly Net Returns", [_monthly_returns_table(monthly_returns)])
        )

    if turnover_costs is not None and not turnover_costs.empty:
        content_lines.extend(
            _section("Turnover And Costs", [_turnover_costs_table(turnover_costs)])
        )

    if cost_sensitivity is not None and not cost_sensitivity.empty:
        content_lines.extend(
            _section("Cost Sensitivity", _cost_sensitivity_lines(cost_sensitivity))
        )

    threshold_sweep_lines = _pattern_meta_threshold_sweep_lines(
        pattern_meta_threshold_sweep,
        output_path,
        pattern_meta_threshold_sweep_path,
    )
    if threshold_sweep_lines:
        content_lines.extend(
            _section("Pattern Meta Threshold Sweep", threshold_sweep_lines)
        )

    tuning_lines = _pattern_meta_tuning_lines(
        pattern_meta_tuning_candidates,
        pattern_meta_tuning_selections,
        pattern_partial_threshold_sweep,
        output_path,
        pattern_meta_tuning_candidates_path,
        pattern_meta_tuning_selections_path,
        pattern_partial_threshold_sweep_path,
    )
    if tuning_lines:
        content_lines.extend(_section("Pattern Meta Tuning", tuning_lines))

    ml_sweep_lines = _ml_strategy_threshold_sweep_lines(
        ml_strategy_threshold_sweep,
        output_path,
        ml_strategy_threshold_sweep_path,
    )
    if ml_sweep_lines:
        content_lines.extend(_section("ML Strategy Threshold Sweep", ml_sweep_lines))

    ml_tuning_lines = _ml_strategy_tuning_lines(
        ml_strategy_tuning_candidates,
        ml_strategy_tuning_selections,
        output_path,
        ml_strategy_tuning_candidates_path,
        ml_strategy_tuning_selections_path,
    )
    if ml_tuning_lines:
        content_lines.extend(_section("ML Strategy Tuning", ml_tuning_lines))

    signal_inspection_lines = _signal_inspection_lines(
        config,
        output_path,
        indicator_diagnostics_path,
        signal_price_overlay_plot_path,
        signal_confirmations_plot_path,
        signal_performance_focus_plot_path,
        pattern_diagnostics_path,
        pattern_exit_overlay_diagnostics_path,
        pattern_meta_labels_path,
        pattern_meta_predictions_path,
        pattern_meta_fold_diagnostics_path,
        pattern_meta_threshold_sweep_path,
        pattern_meta_tuning_candidates_path,
        pattern_meta_tuning_selections_path,
        pattern_partial_exposure_diagnostics_path,
        pattern_partial_threshold_sweep_path,
        pattern_price_overlay_plot_path,
        pattern_detections_plot_path,
        pattern_detection_windows_plot_path,
        pattern_performance_focus_plot_path,
    )
    if signal_inspection_lines:
        content_lines.extend(_section("Signal Inspection", signal_inspection_lines))

    if fold_diagnostics is not None and not fold_diagnostics.empty:
        content_lines.extend(
            _section(
                "Walk-Forward Diagnostics",
                _walk_forward_diagnostics_lines(fold_diagnostics),
            )
        )

    if model_summary is not None and not model_summary.empty:
        content_lines.extend(
            _section("Model Summary", [_markdown_table(_display_frame(model_summary))])
        )

    if fold_summary is not None and not fold_summary.empty:
        content_lines.extend(
            _section("Fold Summary", [_markdown_table(_display_frame(fold_summary))])
        )

    show_calibration_section = (
        model_summary is not None
        and not model_summary.empty
        and {"mean_ece", "mean_max_calibration_gap"}.issubset(model_summary.columns)
    )
    if show_calibration_section:
        calibration_lines = [_calibration_summary_table(model_summary)]
        if threshold_diagnostics is not None and not threshold_diagnostics.empty:
            calibration_lines.extend(
                [
                    "",
                    _markdown_table(_display_frame(_threshold_highlights(threshold_diagnostics))),
                ]
            )
        for alt_text, plot_path in [
            ("Calibration Curves", calibration_curves_plot_path),
            ("Score Histograms", score_histograms_plot_path),
            ("Threshold Sweeps", threshold_sweeps_plot_path),
        ]:
            if plot_path is not None and plot_path.exists():
                calibration_lines.extend(
                    ["", _relative_image_line(output_path, plot_path, alt_text)]
                )
        content_lines.extend(
            _section("Calibration And Threshold Diagnostics", calibration_lines)
        )

    output_path.write_text("\n".join(content_lines).rstrip() + "\n", encoding="utf-8")
    return output_path
