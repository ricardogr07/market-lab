from __future__ import annotations

import tomllib
from pathlib import Path


def test_codex_mcp_sample_exposes_offline_and_online_docker_servers() -> None:
    document = tomllib.loads(Path("docs/codex.config.toml.example").read_text(encoding="utf-8"))

    assert set(document["mcp_servers"]) == {
        "marketlab",
        "marketlab_online",
    }

    offline = document["mcp_servers"]["marketlab"]
    online = document["mcp_servers"]["marketlab_online"]

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
    assert offline["startup_timeout_sec"] == 20
    assert offline["tool_timeout_sec"] == 120
    assert online["args"] == [*offline["args"], "--allow-network"]
    assert online["startup_timeout_sec"] == offline["startup_timeout_sec"]
    assert online["tool_timeout_sec"] == offline["tool_timeout_sec"]


def test_codex_mcp_sample_matches_compose_container_contract() -> None:
    compose_text = Path("docker/compose.mcp.yml").read_text(encoding="utf-8")
    sample_text = Path("docs/codex.config.toml.example").read_text(encoding="utf-8")

    assert 'container_name: marketlab-mcp' in compose_text
    assert '/app/workspace' in compose_text
    assert '/app/artifacts' in compose_text
    assert '/app/repo:ro' in compose_text
    assert '"marketlab-mcp"' in sample_text
    assert '"/app/workspace"' in sample_text
    assert '"/app/artifacts"' in sample_text
    assert '"/app/repo"' in sample_text
