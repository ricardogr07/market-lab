from __future__ import annotations

from typing import Any

from marketlab.mcp.workspace import WorkspaceSandbox
from marketlab.resources.templates import iter_config_template_names


def register_config_tools(
    mcp: Any,
    *,
    sandbox: WorkspaceSandbox,
) -> None:
    @mcp.tool(
        name="marketlab_list_templates",
        description="List the bundled MarketLab config templates available for workspace config generation.",
        structured_output=True,
    )
    def marketlab_list_templates() -> dict[str, Any]:
        return {"templates": list(iter_config_template_names())}

    @mcp.tool(
        name="marketlab_create_config_from_template",
        description="Create a workspace YAML config from a bundled MarketLab template and normalize its artifact root.",
        structured_output=True,
    )
    def marketlab_create_config_from_template(
        template_name: str,
        destination: str,
        force: bool = False,
    ) -> dict[str, Any]:
        created_path = sandbox.create_config_from_template(
            template_name=template_name,
            destination=destination,
            force=force,
        )
        return {
            "config_path": str(created_path),
            "template_name": template_name,
        }

    @mcp.tool(
        name="marketlab_copy_repo_config",
        description="Copy a repo-tracked config into the writable workspace and normalize its artifact root.",
        structured_output=True,
    )
    def marketlab_copy_repo_config(
        repo_config_path: str,
        destination: str,
        force: bool = False,
    ) -> dict[str, Any]:
        copied_path = sandbox.copy_repo_config(
            repo_config_path=repo_config_path,
            destination=destination,
            force=force,
        )
        return {
            "config_path": str(copied_path),
            "repo_config_path": repo_config_path,
        }

    @mcp.tool(
        name="marketlab_read_config",
        description="Read a workspace YAML config as structured data.",
        structured_output=True,
    )
    def marketlab_read_config(config_path: str) -> dict[str, Any]:
        resolved = sandbox.resolve_workspace_path(config_path)
        return {
            "config_path": str(resolved),
            "document": sandbox.read_config(resolved),
        }

    @mcp.tool(
        name="marketlab_patch_config",
        description="Apply a recursive structured patch to a workspace YAML config.",
        structured_output=True,
    )
    def marketlab_patch_config(config_path: str, patch: dict[str, Any]) -> dict[str, Any]:
        patched_path = sandbox.patch_config(config_path=config_path, patch=patch)
        return {
            "config_path": str(patched_path),
            "document": sandbox.read_config(patched_path),
        }

    @mcp.tool(
        name="marketlab_validate_config",
        description="Validate a workspace config through the existing MarketLab loader and path guardrails.",
        structured_output=True,
    )
    def marketlab_validate_config(config_path: str) -> dict[str, Any]:
        return sandbox.validate_config(config_path)
