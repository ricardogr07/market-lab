from __future__ import annotations

from pathlib import Path
from typing import Any

from marketlab._version import get_version
from marketlab.mcp.workspace import WorkspaceSandbox

CONFIG_TOOLS = [
    "marketlab_list_templates",
    "marketlab_create_config_from_template",
    "marketlab_copy_repo_config",
    "marketlab_read_config",
    "marketlab_patch_config",
    "marketlab_validate_config",
]
JOB_TOOLS = [
    "marketlab_plan_run",
    "marketlab_start_job",
    "marketlab_list_jobs",
    "marketlab_get_job_status",
    "marketlab_tail_job_logs",
    "marketlab_cancel_job",
]
ARTIFACT_TOOLS = [
    "marketlab_list_runs",
    "marketlab_get_run_summary",
    "marketlab_list_artifacts",
    "marketlab_read_table_artifact",
    "marketlab_read_text_artifact",
    "marketlab_get_plot_artifact",
    "marketlab_compare_runs",
]
PAPER_TOOLS = [
    "marketlab_list_paper_proposals",
    "marketlab_read_paper_proposal",
    "marketlab_get_paper_status",
    "marketlab_decide_paper_proposal",
]


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def register_admin_tools(
    mcp: Any,
    *,
    sandbox: WorkspaceSandbox,
    allow_network: bool,
) -> None:
    @mcp.tool(
        name="marketlab_server_info",
        description="Describe the MarketLab MCP server, installed version, and runtime defaults.",
        structured_output=True,
    )
    def marketlab_server_info() -> dict[str, Any]:
        return {
            "server_name": "marketlab",
            "version": get_version(),
            "allow_network": allow_network,
            "transport": "stdio",
            "tool_groups": {
                "admin": ["marketlab_server_info", "marketlab_workspace_info"],
                "configs": CONFIG_TOOLS,
                "jobs": JOB_TOOLS,
                "artifacts": ARTIFACT_TOOLS,
                "paper": PAPER_TOOLS,
            },
        }

    @mcp.tool(
        name="marketlab_workspace_info",
        description="Describe writable and readable roots managed by the MarketLab MCP server.",
        structured_output=True,
    )
    def marketlab_workspace_info() -> dict[str, Any]:
        workspace_config_files = sorted(
            str(path.relative_to(sandbox.workspace_root))
            for path in sandbox.workspace_root.rglob("*.yaml")
            if _is_relative_to(path, sandbox.workspace_root)
        )
        return {
            **sandbox.describe(),
            "workspace_yaml_files": workspace_config_files,
        }
