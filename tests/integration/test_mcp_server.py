from __future__ import annotations

import os
import sys
from pathlib import Path

import anyio
import pytest
from mcp.client.session import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client
from tests.integration._cli_harness import build_synthetic_panel, write_raw_symbol_cache

REPO_ROOT = Path(__file__).resolve().parents[2]
MCP_LAUNCHER = REPO_ROOT / "scripts" / "run_marketlab_mcp.py"


async def _call_tool(
    session: ClientSession,
    name: str,
    arguments: dict | None = None,
) -> dict:
    result = await session.call_tool(name, arguments or {})
    assert not result.isError, f"{name} returned an MCP error: {result}"
    assert result.structuredContent is not None, f"{name} did not return structured content"
    return result.structuredContent


@pytest.mark.anyio
async def test_mcp_server_supports_config_generation_run_execution_and_artifact_reads(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    artifact_root = tmp_path / "artifacts"
    workspace.mkdir(parents=True)
    artifact_root.mkdir(parents=True)

    synthetic_panel = build_synthetic_panel(
        (
            ("VOO", 100.0, 0.9),
            ("QQQ", 120.0, 1.0),
            ("SMH", 80.0, 1.2),
            ("XLV", 70.0, 0.6),
            ("IEMG", 60.0, 0.7),
        ),
        start_date="2024-01-02",
        end_date="2024-05-31",
    )
    cache_dir = workspace / "cache"
    write_raw_symbol_cache(cache_dir, panel=synthetic_panel)

    env = os.environ.copy()
    existing_pythonpath = env.get("PYTHONPATH")
    repo_src = str(REPO_ROOT / "src")
    env["PYTHONPATH"] = repo_src if not existing_pythonpath else os.pathsep.join(
        [repo_src, existing_pythonpath]
    )

    server = StdioServerParameters(
        command=sys.executable,
        args=[
            str(MCP_LAUNCHER),
            "--workspace-root",
            str(workspace),
            "--artifact-root",
            str(artifact_root),
            "--repo-root",
            str(REPO_ROOT),
        ],
        cwd=REPO_ROOT,
        env=env,
    )

    async with stdio_client(server) as (read_stream, write_stream):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()
            tools = await session.list_tools()
            tool_names = sorted(tool.name for tool in tools.tools)
            assert "marketlab_create_config_from_template" in tool_names
            assert "marketlab_start_job" in tool_names
            assert "marketlab_read_table_artifact" in tool_names

            server_info = await _call_tool(session, "marketlab_server_info")
            assert server_info["transport"] == "stdio"
            assert not server_info["allow_network"]

            created = await _call_tool(
                session,
                "marketlab_create_config_from_template",
                {
                    "template_name": "weekly_rank_smoke",
                    "destination": "configs/mcp_smoke.yaml",
                },
            )
            config_path = created["config_path"]

            await _call_tool(
                session,
                "marketlab_patch_config",
                {
                    "config_path": str(Path(config_path).relative_to(workspace)),
                    "patch": {
                        "data": {
                            "cache_dir": str(cache_dir),
                            "start_date": "2024-01-02",
                            "end_date": "2024-05-31",
                        },
                        "baselines": {
                            "sma": {"enabled": False},
                        },
                        "artifacts": {
                            "save_plots": False,
                            "save_report_md": True,
                        },
                    },
                },
            )

            validated = await _call_tool(
                session,
                "marketlab_validate_config",
                {"config_path": str(Path(config_path).relative_to(workspace))},
            )
            assert validated["summary"]["cache_dir"] == str(cache_dir.resolve())
            assert validated["summary"]["output_dir"] == str(artifact_root.resolve())

            plan = await _call_tool(
                session,
                "marketlab_plan_run",
                {
                    "command": "backtest",
                    "config_path": str(Path(config_path).relative_to(workspace)),
                },
            )
            assert not plan["network_required"]

            job = await _call_tool(
                session,
                "marketlab_start_job",
                {"plan_id": plan["id"]},
            )
            job_id = job["id"]

            for _ in range(240):
                status = await _call_tool(
                    session,
                    "marketlab_get_job_status",
                    {"job_id": job_id},
                )
                if status["status"] in {"succeeded", "failed", "cancelled"}:
                    break
                await anyio.sleep(0.1)
            else:
                raise AssertionError("Timed out waiting for MCP job completion.")

            assert status["status"] == "succeeded", status
            assert status["result_kind"] == "run_dir"
            run_dir = Path(status["result_path"])

            runs = await _call_tool(session, "marketlab_list_runs", {"limit": 5})
            assert any(Path(run["run_path"]) == run_dir for run in runs["runs"])

            summary = await _call_tool(
                session,
                "marketlab_get_run_summary",
                {"run_path": str(run_dir)},
            )
            assert "metrics_preview" in summary
            assert "strategy_summary_preview" in summary
            assert "Strategy Summary" in summary["report_sections"]

            artifacts = await _call_tool(
                session,
                "marketlab_list_artifacts",
                {"run_path": str(run_dir)},
            )
            artifact_names = {artifact["name"] for artifact in artifacts["artifacts"]}
            assert {"metrics.csv", "strategy_summary.csv", "report.md"} <= artifact_names

            metrics_preview = await _call_tool(
                session,
                "marketlab_read_table_artifact",
                {"path": str(run_dir / "metrics.csv"), "max_rows": 10},
            )
            assert metrics_preview["row_count"] >= 1
            assert "strategy" in metrics_preview["columns"]

            report_text = await _call_tool(
                session,
                "marketlab_read_text_artifact",
                {"path": str(run_dir / "report.md"), "max_chars": 4000},
            )
            assert "Strategy Summary" in report_text["content"]
