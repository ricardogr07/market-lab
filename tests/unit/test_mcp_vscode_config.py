from __future__ import annotations

import json
from pathlib import Path


def test_vscode_mcp_sample_exposes_offline_and_online_docker_servers() -> None:
    document = json.loads(Path(".vscode/mcp.json.example").read_text(encoding="utf-8"))

    assert set(document["servers"]) == {
        "marketlab-docker-offline",
        "marketlab-docker-online",
    }

    offline = document["servers"]["marketlab-docker-offline"]
    online = document["servers"]["marketlab-docker-online"]

    assert offline["type"] == "stdio"
    assert offline["command"] == "docker"
    assert offline["args"] == [
        "exec",
        "-i",
        "marketlab-mcp",
        "marketlab-mcp",
        "--workspace-root",
        "/app/workspace",
        "--artifact-root",
        "/app/artifacts",
        "--repo-root",
        "/app/repo",
    ]
    assert online["args"] == [*offline["args"], "--allow-network"]


def test_vscode_mcp_sample_matches_compose_container_contract() -> None:
    compose_text = Path("docker/compose.mcp.yml").read_text(encoding="utf-8")
    sample_text = Path(".vscode/mcp.json.example").read_text(encoding="utf-8")

    assert 'container_name: marketlab-mcp' in compose_text
    assert '/app/workspace' in compose_text
    assert '/app/artifacts' in compose_text
    assert '/app/repo:ro' in compose_text
    assert '"marketlab-mcp"' in sample_text
    assert '"/app/workspace"' in sample_text
    assert '"/app/artifacts"' in sample_text
    assert '"/app/repo"' in sample_text
