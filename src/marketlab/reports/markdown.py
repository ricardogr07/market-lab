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


def write_markdown_report(
    config: ExperimentConfig,
    metrics: pd.DataFrame,
    performance: pd.DataFrame,
    path: str | Path,
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

    content = "\n".join(
        [
            f"# {config.experiment_name}",
            "",
            "## Scope",
            "",
            *_scope_lines(performance),
            f"- Symbols: {', '.join(config.data.symbols)}",
            f"- Window: {date_min} to {date_max}",
            f"- Cost model: {config.portfolio.costs.bps_per_trade} bps per unit turnover",
            "",
            "## Strategies",
            "",
            *strategy_lines,
            "",
            "## Metrics",
            "",
            metrics_table,
        ]
    )
    output_path.write_text(content, encoding="utf-8")
    return output_path
