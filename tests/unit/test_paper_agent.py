from __future__ import annotations

import sys
import types
from datetime import UTC, datetime
from pathlib import Path

from tests._paper_fakes import (
    FakeAlpacaBroker,
    FakeAlpacaProvider,
    build_phase7_paper_config,
)

from marketlab.paper.agent import run_agent_approval_iteration
from marketlab.paper.service import PaperStateStore, run_paper_decision


def test_agent_worker_approves_with_openai_backend(monkeypatch, tmp_path: Path) -> None:
    config = build_phase7_paper_config(
        tmp_path,
        execution_mode="agent_approval",
        agent_backend="openai",
        agent_model="gpt-4o-mini",
    )
    run_paper_decision(
        config,
        now=datetime(2026, 4, 10, 20, 10, tzinfo=UTC),
        provider=FakeAlpacaProvider(symbol="VOO"),
        broker=FakeAlpacaBroker(symbol="VOO"),
    )

    class FakeResponses:
        def create(self, **kwargs):
            return types.SimpleNamespace(
                output_text='{"decision":"approve","rationale":"Approve the consensus proposal."}'
            )

    class FakeOpenAIClient:
        def __init__(self, api_key: str, timeout: int) -> None:
            self.responses = FakeResponses()

    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setitem(sys.modules, "openai", types.SimpleNamespace(OpenAI=FakeOpenAIClient))

    result = run_agent_approval_iteration(
        config,
        now=datetime(2026, 4, 10, 20, 30, tzinfo=UTC),
        broker=FakeAlpacaBroker(symbol="VOO"),
    )

    proposal = PaperStateStore(config).latest_proposal()
    assert result["processed_count"] == 1
    assert proposal is not None
    assert proposal["approval_status"] == "approved"
    assert proposal["approval_backend"] == "openai"
    assert proposal["approval_model"] == "gpt-4o-mini"
    assert proposal["approval_fallback_used"] is False


def test_agent_worker_approves_with_claude_backend(monkeypatch, tmp_path: Path) -> None:
    config = build_phase7_paper_config(
        tmp_path,
        execution_mode="agent_approval",
        agent_backend="claude",
        agent_model="claude-sonnet-4-5",
    )
    run_paper_decision(
        config,
        now=datetime(2026, 4, 10, 20, 10, tzinfo=UTC),
        provider=FakeAlpacaProvider(symbol="VOO"),
        broker=FakeAlpacaBroker(symbol="VOO"),
    )

    class FakeMessages:
        def create(self, **kwargs):
            return types.SimpleNamespace(
                content=[
                    types.SimpleNamespace(
                        text='{"decision":"approve","rationale":"Claude approves the proposal."}'
                    )
                ]
            )

    class FakeAnthropicClient:
        def __init__(self, api_key: str, timeout: int) -> None:
            self.messages = FakeMessages()

    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.setitem(
        sys.modules,
        "anthropic",
        types.SimpleNamespace(Anthropic=FakeAnthropicClient),
    )

    result = run_agent_approval_iteration(
        config,
        now=datetime(2026, 4, 10, 20, 30, tzinfo=UTC),
        broker=FakeAlpacaBroker(symbol="VOO"),
    )

    proposal = PaperStateStore(config).latest_proposal()
    assert result["processed_count"] == 1
    assert proposal is not None
    assert proposal["approval_status"] == "approved"
    assert proposal["approval_backend"] == "claude"
    assert proposal["approval_model"] == "claude-sonnet-4-5"
    assert proposal["approval_fallback_used"] is False


def test_agent_worker_falls_back_to_deterministic_consensus(monkeypatch, tmp_path: Path) -> None:
    config = build_phase7_paper_config(
        tmp_path,
        execution_mode="agent_approval",
        agent_backend="openai",
        agent_model="gpt-4o-mini",
    )
    run_paper_decision(
        config,
        now=datetime(2026, 4, 10, 20, 10, tzinfo=UTC),
        provider=FakeAlpacaProvider(symbol="VOO"),
        broker=FakeAlpacaBroker(symbol="VOO"),
    )

    class FakeResponses:
        def create(self, **kwargs):
            return types.SimpleNamespace(output_text="not-json")

    class FakeOpenAIClient:
        def __init__(self, api_key: str, timeout: int) -> None:
            self.responses = FakeResponses()

    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setitem(sys.modules, "openai", types.SimpleNamespace(OpenAI=FakeOpenAIClient))

    run_agent_approval_iteration(
        config,
        now=datetime(2026, 4, 10, 20, 30, tzinfo=UTC),
        broker=FakeAlpacaBroker(symbol="VOO"),
    )

    proposal = PaperStateStore(config).latest_proposal()
    assert proposal is not None
    assert proposal["approval_status"] == "approved"
    assert proposal["approval_backend"] == "deterministic_consensus"
    assert proposal["approval_fallback_used"] is True
    assert "openai backend failed" in proposal["approval_fallback_reason"]


def test_agent_worker_is_idempotent_after_first_approval(monkeypatch, tmp_path: Path) -> None:
    config = build_phase7_paper_config(tmp_path, execution_mode="agent_approval")
    run_paper_decision(
        config,
        now=datetime(2026, 4, 10, 20, 10, tzinfo=UTC),
        provider=FakeAlpacaProvider(symbol="VOO"),
        broker=FakeAlpacaBroker(symbol="VOO"),
    )

    first = run_agent_approval_iteration(
        config,
        now=datetime(2026, 4, 10, 20, 30, tzinfo=UTC),
        broker=FakeAlpacaBroker(symbol="VOO"),
    )
    second = run_agent_approval_iteration(
        config,
        now=datetime(2026, 4, 10, 20, 31, tzinfo=UTC),
        broker=FakeAlpacaBroker(symbol="VOO"),
    )

    assert first["processed_count"] == 1
    assert second["processed_count"] == 0
