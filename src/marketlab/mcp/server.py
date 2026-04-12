from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from marketlab.mcp.jobs import MarketLabJobManager
from marketlab.mcp.tools_admin import register_admin_tools
from marketlab.mcp.tools_artifacts import register_artifact_tools
from marketlab.mcp.tools_configs import register_config_tools
from marketlab.mcp.tools_jobs import register_job_tools
from marketlab.mcp.tools_paper import register_paper_tools
from marketlab.mcp.workspace import WorkspaceSandbox


def _require_mcp() -> Any:
    try:
        from mcp.server.fastmcp import FastMCP
    except ModuleNotFoundError as exc:  # pragma: no cover - exercised by CLI error path
        if exc.name == "mcp":
            raise RuntimeError(
                "MarketLab MCP support requires the optional dependency group. Install it with "
                "`pip install marketlab[mcp]`."
            ) from exc
        raise
    return FastMCP


class MarketLabMCPServer:
    def __init__(
        self,
        *,
        workspace_root: Path,
        artifact_root: Path,
        repo_root: Path | None,
        allow_network: bool,
        log_level: str = "INFO",
    ) -> None:
        FastMCP = _require_mcp()
        self._sandbox = WorkspaceSandbox(
            workspace_root=workspace_root,
            artifact_root=artifact_root,
            repo_root=repo_root,
        )
        self._jobs = MarketLabJobManager(
            sandbox=self._sandbox,
            allow_network=allow_network,
        )

        @asynccontextmanager
        async def lifespan(_: Any):
            try:
                yield
            finally:
                self._jobs.close()

        self._mcp = FastMCP(
            name="marketlab",
            instructions=(
                "MarketLab MCP exposes sandboxed config authoring, async workflow execution, "
                "artifact inspection, and run comparison for the installed MarketLab package."
            ),
            log_level=log_level,
            dependencies=["marketlab[mcp]"],
            lifespan=lifespan,
        )
        register_admin_tools(
            self._mcp,
            sandbox=self._sandbox,
            allow_network=allow_network,
        )
        register_config_tools(self._mcp, sandbox=self._sandbox)
        register_job_tools(self._mcp, jobs=self._jobs)
        register_artifact_tools(self._mcp, sandbox=self._sandbox)
        register_paper_tools(self._mcp, sandbox=self._sandbox)

    @property
    def app(self) -> Any:
        return self._mcp

    def close(self) -> None:
        self._jobs.close()

    def run(self, transport: str = "stdio") -> None:
        self._mcp.run(transport=transport)


def create_server(
    *,
    workspace_root: Path,
    artifact_root: Path,
    repo_root: Path | None,
    allow_network: bool,
    log_level: str = "INFO",
) -> MarketLabMCPServer:
    return MarketLabMCPServer(
        workspace_root=workspace_root,
        artifact_root=artifact_root,
        repo_root=repo_root,
        allow_network=allow_network,
        log_level=log_level,
    )
