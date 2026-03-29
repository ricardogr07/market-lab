from __future__ import annotations

from pathlib import Path

import pandas as pd

from marketlab.config import ExperimentConfig
from marketlab.reports.markdown import write_markdown_report


def _base_metrics() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "strategy": ["alpha"],
            "cumulative_return": [0.10],
            "annualized_return": [0.12],
            "annualized_volatility": [0.20],
            "sharpe_like": [0.60],
            "max_drawdown": [-0.05],
            "hit_rate": [0.55],
            "avg_turnover": [99.0],
            "total_turnover": [999.0],
        }
    )


def _base_performance() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-02", "2024-01-03"]),
            "strategy": ["alpha", "alpha"],
            "gross_return": [0.01, 0.02],
            "net_return": [0.009, 0.018],
            "turnover": [1.0, 3.0],
            "equity": [1.009, 1.027162],
        }
    )


def test_write_markdown_report_turnover_section_uses_turnover_costs_input(tmp_path: Path) -> None:
    config = ExperimentConfig(experiment_name="markdown_fixture")
    strategy_summary = pd.DataFrame(
        {
            "strategy": ["alpha"],
            "start_date": pd.to_datetime(["2024-01-02"]),
            "end_date": pd.to_datetime(["2024-01-03"]),
            "trading_days": [2],
            "final_equity": [1.027162],
            "gross_final_equity": [1.0302],
            "gross_cumulative_return": [0.0302],
            "cumulative_return": [0.027162],
            "cost_drag": [0.003038],
            "annualized_return": [0.12],
            "annualized_volatility": [0.20],
            "sharpe_like": [0.60],
            "max_drawdown": [-0.05],
            "hit_rate": [0.55],
            "avg_turnover": [99.0],
            "total_turnover": [999.0],
            "avg_cost_return": [0.50],
            "total_cost_return": [1.00],
        }
    )
    turnover_costs = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-02", "2024-01-03"]),
            "strategy": ["alpha", "alpha"],
            "turnover": [1.0, 3.0],
            "gross_return": [0.01, 0.02],
            "net_return": [0.009, 0.018],
            "cost_return": [0.001, 0.002],
        }
    )

    report_path = write_markdown_report(
        config=config,
        metrics=_base_metrics(),
        performance=_base_performance(),
        path=tmp_path / 'report.md',
        strategy_summary=strategy_summary,
        turnover_costs=turnover_costs,
    )

    report_text = report_path.read_text(encoding="utf-8")
    turnover_section = report_text.split("## Turnover And Costs", maxsplit=1)[1].split("## ", maxsplit=1)[0]

    assert "## Turnover And Costs" in report_text
    assert "| strategy | avg_turnover | total_turnover | avg_cost_return | total_cost_return |" in turnover_section
    assert "| alpha | 2.0 | 4.0 | 0.0015 | 0.003 |" in turnover_section
    assert "999.0" not in turnover_section
    assert "0.5" not in turnover_section


def test_write_markdown_report_walk_forward_diagnostics_section_uses_fold_diagnostics_input(
    tmp_path: Path,
) -> None:
    config = ExperimentConfig(experiment_name="markdown_fixture")
    fold_diagnostics = pd.DataFrame(
        {
            "candidate_id": [1, 2],
            "fold_id": [1, pd.NA],
            "status": ["used", "skipped"],
            "skip_reasons": ["", "insufficient_train_rows;insufficient_test_positive_rate"],
            "train_start": pd.to_datetime(["2023-01-06", "2023-03-31"]),
            "train_end": pd.to_datetime(["2024-01-05", "2024-05-31"]),
            "label_cutoff": pd.to_datetime(["2024-01-12", "2024-06-07"]),
            "test_start": pd.to_datetime(["2024-01-12", "2024-06-07"]),
            "test_end": pd.to_datetime(["2024-04-05", "2024-07-12"]),
            "train_rows": [110, 80],
            "test_rows": [26, 8],
            "train_positive_rate": [0.5, 0.03],
            "test_positive_rate": [0.5, 0.01],
        }
    )

    report_path = write_markdown_report(
        config=config,
        metrics=_base_metrics(),
        performance=_base_performance(),
        path=tmp_path / 'report.md',
        fold_diagnostics=fold_diagnostics,
    )

    report_text = report_path.read_text(encoding="utf-8")
    diagnostics_section = report_text.split("## Walk-Forward Diagnostics", maxsplit=1)[1]

    assert "## Walk-Forward Diagnostics" in report_text
    assert "- Used candidates: 1" in diagnostics_section
    assert "- Skipped candidates: 1" in diagnostics_section
    assert "| candidate_id | test_start | test_end | skip_reasons | train_rows | test_rows | train_positive_rate | test_positive_rate |" in diagnostics_section
    assert "insufficient_train_rows;insufficient_test_positive_rate" in diagnostics_section
