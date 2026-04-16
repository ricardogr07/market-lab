from __future__ import annotations

import tomllib
from pathlib import Path


def test_codex_mcp_sample_exposes_research_and_paper_docker_servers() -> None:
    document = tomllib.loads(Path("docs/codex.config.toml.example").read_text(encoding="utf-8"))

    assert set(document["mcp_servers"]) == {
        "marketlab",
        "marketlab_online",
        "marketlab_paper",
        "marketlab_paper_online",
    }

    research_offline = document["mcp_servers"]["marketlab"]
    research_online = document["mcp_servers"]["marketlab_online"]
    paper_offline = document["mcp_servers"]["marketlab_paper"]
    paper_online = document["mcp_servers"]["marketlab_paper_online"]

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
    assert research_offline["startup_timeout_sec"] == 20
    assert research_offline["tool_timeout_sec"] == 120
    assert research_online["args"] == [*research_offline["args"], "--allow-network"]
    assert research_online["startup_timeout_sec"] == research_offline["startup_timeout_sec"]
    assert research_online["tool_timeout_sec"] == research_offline["tool_timeout_sec"]
    assert paper_offline["startup_timeout_sec"] == 20
    assert paper_offline["tool_timeout_sec"] == 120
    assert paper_online["args"] == [*paper_offline["args"], "--allow-network"]
    assert paper_online["startup_timeout_sec"] == paper_offline["startup_timeout_sec"]
    assert paper_online["tool_timeout_sec"] == paper_offline["tool_timeout_sec"]


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


def test_codex_paper_mcp_sample_matches_compose_paper_container_contract() -> None:
    compose_text = Path("docker/compose.paper.yml").read_text(encoding="utf-8")
    sample_text = Path("docs/codex.config.toml.example").read_text(encoding="utf-8")

    assert 'container_name: marketlab-paper-mcp' in compose_text
    assert '../artifacts:/app/repo/artifacts' in compose_text
    assert '/app/workspace' in compose_text
    assert '/app/repo:ro' in compose_text
    assert '"marketlab-paper-mcp"' in sample_text
    assert '"/app/workspace"' in sample_text
    assert '"/app/repo/artifacts"' in sample_text
    assert '"/app/repo"' in sample_text
