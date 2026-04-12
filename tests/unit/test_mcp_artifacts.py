from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from marketlab.mcp.server import create_server


async def _call_tool(server, name: str, arguments: dict | None = None) -> dict:
    _, structured = await server.app.call_tool(name, arguments or {})
    assert structured is not None
    return structured


@pytest.mark.anyio
async def test_compare_runs_reads_summary_tables_and_report_sections(tmp_path: Path) -> None:
    server = create_server(
        workspace_root=tmp_path / "workspace",
        artifact_root=tmp_path / "artifacts",
        repo_root=Path.cwd(),
        allow_network=False,
    )
    left_run = tmp_path / "artifacts" / "left"
    right_run = tmp_path / "artifacts" / "right"
    left_run.mkdir(parents=True)
    right_run.mkdir(parents=True)

    pd.DataFrame(
        [
            {"strategy": "buy_hold", "cagr": 0.10},
            {"strategy": "sma", "cagr": 0.08},
        ]
    ).to_csv(left_run / "metrics.csv", index=False)
    pd.DataFrame(
        [
            {"strategy": "buy_hold", "cagr": 0.11},
            {"strategy": "sma", "cagr": 0.09},
        ]
    ).to_csv(right_run / "metrics.csv", index=False)
    pd.DataFrame([{"strategy": "buy_hold", "avg_gross_exposure": 1.0}]).to_csv(
        left_run / "strategy_summary.csv",
        index=False,
    )
    pd.DataFrame([{"strategy": "buy_hold", "avg_gross_exposure": 0.95}]).to_csv(
        right_run / "strategy_summary.csv",
        index=False,
    )
    pd.DataFrame([{"model_name": "logistic_regression", "mean_roc_auc": 0.6}]).to_csv(
        left_run / "model_summary.csv",
        index=False,
    )
    pd.DataFrame([{"model_name": "logistic_regression", "mean_roc_auc": 0.62}]).to_csv(
        right_run / "model_summary.csv",
        index=False,
    )
    (left_run / "report.md").write_text(
        "# Report\n\n## Strategy Summary\n\nLeft\n\n## Cost Sensitivity\n\nBody\n",
        encoding="utf-8",
    )
    (right_run / "report.md").write_text(
        "# Report\n\n## Strategy Summary\n\nRight\n\n## Benchmark-Relative Summary\n\nBody\n",
        encoding="utf-8",
    )

    try:
        listed = await _call_tool(server, "marketlab_list_runs", {"limit": 10})
        run_paths = {Path(run["run_path"]).name for run in listed["runs"]}
        assert {"left", "right"} <= run_paths

        comparison = await _call_tool(
            server,
            "marketlab_compare_runs",
            {
                "left_run_path": str(left_run),
                "right_run_path": str(right_run),
            },
        )
        assert "metrics.csv" in comparison["tables"]
        assert comparison["tables"]["metrics.csv"]["key"] == "strategy"
        assert "Strategy Summary" in comparison["report_sections"]["left"]
        assert "Benchmark-Relative Summary" in comparison["report_sections"]["right"]
    finally:
        server.close()
