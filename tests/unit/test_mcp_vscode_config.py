from __future__ import annotations

import json
from pathlib import Path


def test_vscode_mcp_sample_exposes_research_and_paper_docker_servers() -> None:
    document = json.loads(Path(".vscode/mcp.json.example").read_text(encoding="utf-8"))

    assert set(document["servers"]) == {
        "marketlab-docker-offline",
        "marketlab-docker-online",
        "marketlab-paper-docker-offline",
        "marketlab-paper-docker-online",
    }

    research_offline = document["servers"]["marketlab-docker-offline"]
    research_online = document["servers"]["marketlab-docker-online"]
    paper_offline = document["servers"]["marketlab-paper-docker-offline"]
    paper_online = document["servers"]["marketlab-paper-docker-online"]

    assert research_offline["type"] == "stdio"
    assert research_offline["command"] == "docker"
    assert research_offline["args"] == [
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
    assert paper_offline["type"] == "stdio"
    assert paper_offline["command"] == "docker"
    assert paper_offline["args"] == [
        "exec",
        "-i",
        "marketlab-paper-mcp",
        "marketlab-mcp",
        "--workspace-root",
        "/app/workspace",
        "--artifact-root",
        "/app/repo/artifacts",
        "--repo-root",
        "/app/repo",
    ]
    assert research_online["args"] == [*research_offline["args"], "--allow-network"]
    assert paper_online["args"] == [*paper_offline["args"], "--allow-network"]


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


def test_vscode_paper_mcp_sample_matches_compose_paper_container_contract() -> None:
    compose_text = Path("docker/compose.paper.yml").read_text(encoding="utf-8")
    sample_text = Path(".vscode/mcp.json.example").read_text(encoding="utf-8")

    assert 'container_name: marketlab-paper-mcp' in compose_text
    assert '../artifacts:/app/repo/artifacts' in compose_text
    assert '/app/workspace' in compose_text
    assert '/app/repo:ro' in compose_text
    assert '"marketlab-paper-mcp"' in sample_text
    assert '"/app/workspace"' in sample_text
    assert '"/app/repo/artifacts"' in sample_text
    assert '"/app/repo"' in sample_text
