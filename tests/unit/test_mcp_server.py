from __future__ import annotations

from pathlib import Path

import pytest

from marketlab.mcp.server import create_server


async def _call_tool(server, name: str, arguments: dict | None = None) -> dict:
    _, structured = await server.app.call_tool(name, arguments or {})
    assert structured is not None
    return structured


@pytest.mark.anyio
async def test_server_info_and_config_tools_work_in_process(tmp_path: Path) -> None:
    server = create_server(
        workspace_root=tmp_path / "workspace",
        artifact_root=tmp_path / "artifacts",
        repo_root=Path.cwd(),
        allow_network=False,
    )
    try:
        info = await _call_tool(server, "marketlab_server_info")
        assert info["server_name"] == "marketlab"
        assert "marketlab_create_config_from_template" in info["tool_groups"]["configs"]

        created = await _call_tool(
            server,
            "marketlab_create_config_from_template",
            {
                "template_name": "weekly_rank_smoke",
                "destination": "configs/test.yaml",
            },
        )
        config_path = Path(created["config_path"]).relative_to(tmp_path / "workspace")

        read_result = await _call_tool(
            server,
            "marketlab_read_config",
            {"config_path": str(config_path)},
        )
        assert read_result["document"]["artifacts"]["output_dir"] == str(
            (tmp_path / "artifacts").resolve()
        )

        patched = await _call_tool(
            server,
            "marketlab_patch_config",
            {
                "config_path": str(config_path),
                "patch": {
                    "data": {
                        "cache_dir": str((tmp_path / "workspace" / "cache").resolve()),
                        "start_date": "2024-01-02",
                        "end_date": "2024-03-29",
                    }
                },
            },
        )
        assert patched["document"]["data"]["start_date"] == "2024-01-02"

        validated = await _call_tool(
            server,
            "marketlab_validate_config",
            {"config_path": str(config_path)},
        )
        assert validated["summary"]["output_dir"] == str((tmp_path / "artifacts").resolve())
    finally:
        server.close()
