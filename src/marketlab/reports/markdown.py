from __future__ import annotations

from pathlib import Path

import pandas as pd

from marketlab.config import ExperimentConfig


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
    return ["- Sprint 1 baseline pipeline"]


def _headline_lines(
    metrics: pd.DataFrame,
    model_summary: pd.DataFrame | None,
) -> list[str]:
    if metrics.empty:
        return []

    lines: list[str] = []
    best_strategy = metrics.sort_values(["cumulative_return", "strategy"], ascending=[False, True]).iloc[0]
    lines.append(
        f"- Best overall strategy by cumulative return: `{best_strategy['strategy']}` ({best_strategy['cumulative_return']:.6f})"
    )

    ml_metrics = metrics.loc[metrics["strategy"].astype(str).str.startswith("ml_")]
    if not ml_metrics.empty:
        best_ml = ml_metrics.sort_values(["cumulative_return", "strategy"], ascending=[False, True]).iloc[0]
        lines.append(
            f"- Best ML strategy by cumulative return: `{best_ml['strategy']}` ({best_ml['cumulative_return']:.6f})"
        )

    if model_summary is not None and not model_summary.empty:
        ranked_models = model_summary.dropna(subset=["mean_roc_auc"]).sort_values(
            ["mean_roc_auc", "model_name"],
            ascending=[False, True],
        )
        if ranked_models.empty:
            lines.append("- Best model by mean ROC AUC: n/a")
        else:
            best_model = ranked_models.iloc[0]
            lines.append(
                f"- Best model by mean ROC AUC: `{best_model['model_name']}` ({best_model['mean_roc_auc']:.6f})"
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
) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    date_min = performance["date"].min().date().isoformat()
    date_max = performance["date"].max().date().isoformat()
    metrics_table = _markdown_table(metrics.round(6))
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

    if model_summary is not None and not model_summary.empty:
        content_lines.extend(_section("Model Summary", [_markdown_table(model_summary.round(6))]))

    if fold_summary is not None and not fold_summary.empty:
        content_lines.extend(_section("Fold Summary", [_markdown_table(fold_summary.round(6))]))

    output_path.write_text("\n".join(content_lines).rstrip() + "\n", encoding="utf-8")
    return output_path
