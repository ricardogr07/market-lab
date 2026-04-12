from __future__ import annotations

from typing import Any

from marketlab.mcp.jobs import MarketLabJobManager


def register_job_tools(
    mcp: Any,
    *,
    jobs: MarketLabJobManager,
) -> None:
    @mcp.tool(
        name="marketlab_plan_run",
        description="Create an execution plan for a MarketLab workflow. A successful plan is required before starting a job.",
        structured_output=True,
    )
    def marketlab_plan_run(command: str, config_path: str) -> dict[str, Any]:
        return jobs.create_plan(command=command, config_path=config_path)

    @mcp.tool(
        name="marketlab_start_job",
        description="Start a queued MarketLab job from a previously created plan.",
        structured_output=True,
    )
    def marketlab_start_job(plan_id: str) -> dict[str, Any]:
        return jobs.start_job(plan_id)

    @mcp.tool(
        name="marketlab_list_jobs",
        description="List queued, active, and completed MarketLab jobs for the current session.",
        structured_output=True,
    )
    def marketlab_list_jobs() -> dict[str, Any]:
        return jobs.list_jobs()

    @mcp.tool(
        name="marketlab_get_job_status",
        description="Inspect the current status and metadata for one MarketLab job.",
        structured_output=True,
    )
    def marketlab_get_job_status(job_id: str) -> dict[str, Any]:
        return jobs.get_job(job_id)

    @mcp.tool(
        name="marketlab_tail_job_logs",
        description="Read the latest log lines for a queued or running MarketLab job.",
        structured_output=True,
    )
    def marketlab_tail_job_logs(job_id: str, lines: int = 40) -> dict[str, Any]:
        return jobs.tail_logs(job_id, lines=lines)

    @mcp.tool(
        name="marketlab_cancel_job",
        description="Cancel a queued or active MarketLab job for the current session.",
        structured_output=True,
    )
    def marketlab_cancel_job(job_id: str) -> dict[str, Any]:
        return jobs.cancel_job(job_id)
