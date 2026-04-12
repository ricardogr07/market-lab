from __future__ import annotations

from typing import Any

from marketlab.config import load_config
from marketlab.mcp.workspace import WorkspaceSandbox
from marketlab.paper import (
    decide_paper_proposal,
    get_paper_status,
    list_paper_proposals,
    read_paper_proposal,
)


def _resolve_config_path(
    sandbox: WorkspaceSandbox,
    config_path: str,
):
    candidates = []
    try:
        candidates.append(sandbox.resolve_workspace_path(config_path))
    except ValueError:
        pass
    if sandbox.repo_root is not None:
        try:
            candidates.append(sandbox.resolve_repo_path(config_path))
        except ValueError:
            pass

    for candidate in candidates:
        if candidate.exists():
            return candidate
    if candidates:
        return candidates[0]
    raise ValueError(f"Config path {config_path!r} is outside the workspace and repo roots.")


def register_paper_tools(
    mcp: Any,
    *,
    sandbox: WorkspaceSandbox,
) -> None:
    @mcp.tool(
        name="marketlab_list_paper_proposals",
        description="List persisted paper-trading proposals from the file-backed approval inbox.",
        structured_output=True,
    )
    def marketlab_list_paper_proposals(config_path: str) -> dict[str, Any]:
        resolved = _resolve_config_path(sandbox, config_path)
        config = load_config(resolved)
        sandbox.validate_execution_paths(config)
        return {
            "config_path": str(resolved),
            "proposals": list_paper_proposals(config),
        }

    @mcp.tool(
        name="marketlab_read_paper_proposal",
        description="Read one persisted paper-trading proposal from the file-backed approval inbox.",
        structured_output=True,
    )
    def marketlab_read_paper_proposal(
        config_path: str,
        proposal_id: str,
    ) -> dict[str, Any]:
        resolved = _resolve_config_path(sandbox, config_path)
        config = load_config(resolved)
        sandbox.validate_execution_paths(config)
        return {
            "config_path": str(resolved),
            "proposal": read_paper_proposal(config, proposal_id=proposal_id),
        }

    @mcp.tool(
        name="marketlab_get_paper_status",
        description="Read the latest persisted paper-trading status and latest proposal summary.",
        structured_output=True,
    )
    def marketlab_get_paper_status(config_path: str) -> dict[str, Any]:
        resolved = _resolve_config_path(sandbox, config_path)
        config = load_config(resolved)
        sandbox.validate_execution_paths(config)
        status = get_paper_status(config)
        return {
            "config_path": str(resolved),
            **status,
        }

    @mcp.tool(
        name="marketlab_decide_paper_proposal",
        description="Approve or reject one persisted paper-trading proposal through the shared approval inbox.",
        structured_output=True,
    )
    def marketlab_decide_paper_proposal(
        config_path: str,
        proposal_id: str,
        decision: str,
        actor: str,
    ) -> dict[str, Any]:
        resolved = _resolve_config_path(sandbox, config_path)
        config = load_config(resolved)
        sandbox.validate_execution_paths(config)
        decision_result = decide_paper_proposal(
            config,
            proposal_id=proposal_id,
            decision=decision,
            actor=actor,
        )
        return {
            "config_path": str(resolved),
            **decision_result,
        }
