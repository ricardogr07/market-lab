from __future__ import annotations

from typing import Any

from marketlab.mcp.workspace import WorkspaceSandbox


def register_config_tools(
    mcp: Any,
    *,
    sandbox: WorkspaceSandbox,
) -> None:
    del mcp
    del sandbox
