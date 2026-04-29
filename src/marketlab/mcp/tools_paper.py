from __future__ import annotations

import logging
from time import perf_counter
from typing import Any

from marketlab.config import load_config
from marketlab.log import (
    bind_execution_context,
    duration_ms_since,
    emit_structured_log,
)
from marketlab.mcp.workspace import WorkspaceSandbox
from marketlab.paper import (
    decide_paper_proposal,
    get_paper_status,
    list_paper_proposals,
    read_paper_proposal,
)
from marketlab.paper.observability import (
    paper_execution_context,
    root_execution_context,
)

LOGGER = logging.getLogger(__name__)


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
        root_context = root_execution_context(
            deployment="paper_mcp",
            phase="paper-approve",
            proposal_id=proposal_id,
            details={
                "tool_name": "marketlab_decide_paper_proposal",
                "config_path": config_path,
            },
        )
        emit_structured_log(
            LOGGER,
            logging.INFO,
            "Starting MCP paper approval tool.",
            event="paper.mcp.approval.start",
            execution_context=root_context,
        )
        start_time = perf_counter()
        resolved_path = config_path
        try:
            resolved = _resolve_config_path(sandbox, config_path)
            resolved_path = str(resolved)
            config = load_config(resolved)
            sandbox.validate_execution_paths(config)
            with bind_execution_context(root_context):
                decision_result = decide_paper_proposal(
                    config,
                    proposal_id=proposal_id,
                    decision=decision,
                    actor=actor,
                    execution_context=root_context,
                )
        except Exception as exc:
            emit_structured_log(
                LOGGER,
                logging.ERROR,
                "MCP paper approval tool failed.",
                event="paper.mcp.approval.error",
                execution_context=paper_execution_context(
                    root_context,
                    phase="paper-approve",
                    deployment="paper_mcp",
                    outcome="error",
                    duration_ms=duration_ms_since(start_time),
                    details={
                        "tool_name": "marketlab_decide_paper_proposal",
                        "config_path": resolved_path,
                    },
                ),
                exc_info=exc,
            )
            raise
        emit_structured_log(
            LOGGER,
            logging.INFO,
            "Finished MCP paper approval tool.",
            event="paper.mcp.approval.finish",
            execution_context=paper_execution_context(
                root_context,
                phase="paper-approve",
                deployment="paper_mcp",
                status=decision_result.get("status", {}),
                proposal={"proposal_id": proposal_id},
                duration_ms=duration_ms_since(start_time),
                details={
                    "tool_name": "marketlab_decide_paper_proposal",
                    "config_path": resolved_path,
                },
            ),
        )
        return {
            "config_path": resolved_path,
            **decision_result,
        }
