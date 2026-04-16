from __future__ import annotations

from pathlib import Path


def test_paper_compose_includes_scheduler_agent_and_mcp_sidecar() -> None:
    compose_text = Path("docker/compose.paper.yml").read_text(encoding="utf-8")

    assert "marketlab-paper-scheduler:" in compose_text
    assert "marketlab-paper-agent:" in compose_text
    assert "marketlab-paper-mcp:" in compose_text
    assert "paper-scheduler" in compose_text
    assert "paper-agent-approve" in compose_text
    assert "/app/repo/configs/experiment.qqq_paper_daily.yaml" in compose_text
    assert "../artifacts:/app/repo/artifacts" in compose_text
    assert "..:/app/repo:ro" in compose_text
    assert "OPENAI_API_KEY" in compose_text
    assert "ANTHROPIC_API_KEY" in compose_text
    assert "TELEGRAM_BOT_TOKEN" in compose_text
    assert "TELEGRAM_CHAT_ID" in compose_text
    assert 'container_name: marketlab-paper-mcp' in compose_text
    assert compose_text.count("TELEGRAM_BOT_TOKEN") == 6
    assert compose_text.count("TELEGRAM_CHAT_ID") == 6
