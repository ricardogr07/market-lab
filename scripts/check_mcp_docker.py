from __future__ import annotations

import importlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path, PurePosixPath

import anyio
from mcp.client.session import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client

ROOT = Path(__file__).resolve().parents[1]
SCRATCH_DIR = ROOT / ".mcp-docker-smoke"
IMAGE_TAG = "marketlab-mcp:smoke"
CONTAINER_NAME = "marketlab-mcp-smoke"
VSCODE_SAMPLE_PATH = ROOT / ".vscode" / "mcp.json.example"


def _fixture_helpers():
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    module = importlib.import_module("tests.integration._cli_harness")

    return module.build_synthetic_panel, module.write_raw_symbol_cache


def _run(
    command: list[str],
    *,
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        cwd=cwd,
        env=env,
        check=check,
        text=True,
        encoding="utf-8",
        errors="replace",
        capture_output=True,
    )


def _call_docker(*args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    return _run(["docker", *args], cwd=ROOT, check=check)


def _load_vscode_server_config(server_name: str) -> dict[str, object]:
    document = json.loads(VSCODE_SAMPLE_PATH.read_text(encoding="utf-8"))
    return document["servers"][server_name]


async def _call_tool(
    session: ClientSession,
    name: str,
    arguments: dict | None = None,
) -> dict:
    result = await session.call_tool(name, arguments or {})
    if result.isError:
        raise RuntimeError(f"{name} returned an MCP error: {result}")
    if result.structuredContent is None:
        raise RuntimeError(f"{name} did not return structured content.")
    return result.structuredContent


async def _exercise_container_mcp() -> None:
    env = os.environ.copy()
    server_config = _load_vscode_server_config("marketlab-docker-offline")
    server_args = list(server_config["args"])
    server_args[2] = CONTAINER_NAME
    server = StdioServerParameters(
        command=server_config["command"],
        args=server_args,
        cwd=ROOT,
        env=env,
    )

    async with stdio_client(server) as (read_stream, write_stream):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()

            created = await _call_tool(
                session,
                "marketlab_create_config_from_template",
                {
                    "template_name": "weekly_rank_smoke",
                    "destination": "configs/docker_mcp_smoke.yaml",
                },
            )
            config_path = "configs/docker_mcp_smoke.yaml"
            assert Path(created["config_path"]).name == "docker_mcp_smoke.yaml"

            await _call_tool(
                session,
                "marketlab_patch_config",
                {
                    "config_path": config_path,
                    "patch": {
                        "data": {
                            "cache_dir": "/app/workspace/cache",
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
                {"config_path": config_path},
            )
            if validated["summary"]["output_dir"] != "/app/artifacts":
                raise RuntimeError("Validated MCP config did not normalize to /app/artifacts.")

            plan = await _call_tool(
                session,
                "marketlab_plan_run",
                {
                    "command": "backtest",
                    "config_path": config_path,
                },
            )
            if plan["network_required"]:
                raise RuntimeError("Docker MCP smoke unexpectedly requires network access.")

            job = await _call_tool(
                session,
                "marketlab_start_job",
                {"plan_id": plan["id"]},
            )

            for _ in range(300):
                status = await _call_tool(
                    session,
                    "marketlab_get_job_status",
                    {"job_id": job["id"]},
                )
                if status["status"] in {"succeeded", "failed", "cancelled"}:
                    break
                await anyio.sleep(0.1)
            else:
                raise RuntimeError("Timed out waiting for Docker MCP smoke job completion.")

            if status["status"] != "succeeded":
                logs = await _call_tool(
                    session,
                    "marketlab_tail_job_logs",
                    {"job_id": job["id"], "lines": 80},
                )
                raise RuntimeError(
                    f"Docker MCP smoke job failed with status={status['status']}.\n{logs['log_tail']}"
                )

            run_summary = await _call_tool(
                session,
                "marketlab_get_run_summary",
                {"run_path": status["result_path"]},
            )
            run_path = PurePosixPath(status["result_path"])

            metrics_preview = await _call_tool(
                session,
                "marketlab_read_table_artifact",
                {
                    "path": str(run_path / "metrics.csv"),
                    "max_rows": 10,
                },
            )
            report_preview = await _call_tool(
                session,
                "marketlab_read_text_artifact",
                {
                    "path": str(run_path / "report.md"),
                    "max_chars": 2000,
                },
            )

            if "metrics.csv" not in run_summary["available_artifacts"]:
                raise RuntimeError("Docker MCP smoke run did not produce metrics.csv.")
            if metrics_preview["row_count"] < 1:
                raise RuntimeError("Docker MCP smoke metrics preview was empty.")
            if "Strategy Summary" not in report_preview["content"]:
                raise RuntimeError("Docker MCP smoke report preview did not include Strategy Summary.")

            print(
                "Verified Docker MCP smoke run:",
                status["result_path"],
                f"artifacts={len(run_summary['available_artifacts'])}",
            )


def main() -> int:
    if shutil.which("docker") is None:
        raise RuntimeError("Docker is required for the mcp-docker validation lane.")
    build_synthetic_panel, write_raw_symbol_cache = _fixture_helpers()

    workspace = SCRATCH_DIR / "workspace"
    artifacts = SCRATCH_DIR / "artifacts"

    shutil.rmtree(SCRATCH_DIR, ignore_errors=True)
    workspace.mkdir(parents=True, exist_ok=True)
    artifacts.mkdir(parents=True, exist_ok=True)

    panel = build_synthetic_panel(
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
    write_raw_symbol_cache(workspace / "cache", panel=panel)

    try:
        _call_docker("version")
        _call_docker("build", "-t", IMAGE_TAG, ".")
        _call_docker("rm", "-f", CONTAINER_NAME, check=False)
        _call_docker(
            "run",
            "-d",
            "--rm",
            "--name",
            CONTAINER_NAME,
            "-v",
            f"{workspace.resolve()}:/app/workspace",
            "-v",
            f"{artifacts.resolve()}:/app/artifacts",
            "-v",
            f"{ROOT.resolve()}:/app/repo:ro",
            "--entrypoint",
            "sleep",
            IMAGE_TAG,
            "infinity",
        )
        anyio.run(_exercise_container_mcp)
        return 0
    finally:
        _call_docker("rm", "-f", CONTAINER_NAME, check=False)
        shutil.rmtree(SCRATCH_DIR, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
