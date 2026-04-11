from __future__ import annotations

import sys
import time
from pathlib import Path

import pytest

from marketlab.mcp.jobs import MarketLabJobManager
from marketlab.mcp.workspace import WorkspaceSandbox


def _write_minimal_config(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(
            [
                "experiment_name: mcp_job_test",
                "data:",
                "  symbols: [VOO, QQQ, SMH, XLV, IEMG]",
                "  start_date: '2024-01-02'",
                "  end_date: '2024-03-29'",
                f"  cache_dir: '{(path.parent.parent / 'cache').as_posix()}'",
                "artifacts:",
                f"  output_dir: '{(path.parent.parent.parent / 'artifacts').as_posix()}'",
            ]
        ),
        encoding="utf-8",
    )
    return path


def _wait_for_status(manager: MarketLabJobManager, job_id: str, statuses: set[str]) -> dict:
    deadline = time.time() + 10.0
    while time.time() < deadline:
        job = manager.get_job(job_id)
        if job["status"] in statuses:
            return job
        time.sleep(0.05)
    raise AssertionError(f"Job {job_id} never reached one of {sorted(statuses)}")


def test_job_manager_queues_jobs_and_allows_cancelling_queued_work(tmp_path: Path) -> None:
    sandbox = WorkspaceSandbox(
        workspace_root=tmp_path / "workspace",
        artifact_root=tmp_path / "artifacts",
    )
    config_path = _write_minimal_config(sandbox.workspace_root / "configs" / "job.yaml")

    def command_builder(_: str, __: Path) -> list[str]:
        return [
            sys.executable,
            "-c",
            (
                "import time; "
                "print('job-start'); "
                "time.sleep(1.0); "
                "print(r'" + str(sandbox.artifact_root / 'runs' / 'run-a') + "')"
            ),
        ]

    manager = MarketLabJobManager(
        sandbox=sandbox,
        allow_network=True,
        command_builder=command_builder,
    )
    try:
        first_plan = manager.create_plan(command="backtest", config_path=config_path)
        second_plan = manager.create_plan(command="backtest", config_path=config_path)
        first_job = manager.start_job(first_plan["id"])
        second_job = manager.start_job(second_plan["id"])

        _wait_for_status(manager, first_job["id"], {"running"})
        queued = manager.get_job(second_job["id"])
        assert queued["status"] == "queued"

        cancelled = manager.cancel_job(second_job["id"])
        assert cancelled["status"] == "cancelled"

        finished = _wait_for_status(manager, first_job["id"], {"succeeded"})
        assert finished["result_kind"] == "run_dir"
        assert "run-a" in finished["result_path"]
    finally:
        manager.close()


def test_plan_creation_rejects_network_when_cache_is_missing(tmp_path: Path) -> None:
    sandbox = WorkspaceSandbox(
        workspace_root=tmp_path / "workspace",
        artifact_root=tmp_path / "artifacts",
    )
    config_path = _write_minimal_config(sandbox.workspace_root / "configs" / "network.yaml")

    manager = MarketLabJobManager(
        sandbox=sandbox,
        allow_network=False,
    )
    try:
        with pytest.raises(ValueError, match="requires network access"):
            manager.create_plan(command="backtest", config_path=config_path)
    finally:
        manager.close()
