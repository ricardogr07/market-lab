from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

from marketlab.config import ExperimentConfig

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
    fold_diagnostics: pd.DataFrame | None = None,
    threshold_diagnostics: pd.DataFrame | None = None,
    calibration_curves_plot_path: Path | None = None,
    score_histograms_plot_path: Path | None = None,
    threshold_sweeps_plot_path: Path | None = None,
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
        exposure_lines = _exposure_summary_lines(strategy_summary)
        if exposure_lines:
            content_lines.extend(_section("Exposure Summary", exposure_lines))
        benchmark_lines = _benchmark_summary_lines(strategy_summary)
        if benchmark_lines:
            content_lines.extend(_section("Benchmark-Relative Summary", benchmark_lines))

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
