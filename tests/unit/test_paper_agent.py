from __future__ import annotations

import json
import sys
import types
from datetime import UTC, datetime
from pathlib import Path

import pytest
from tests._paper_fakes import (
    FakeAlpacaBroker,
    FakeAlpacaProvider,
    build_phase7_paper_config,
)

import marketlab.paper.agent as agent_module
from marketlab.paper.agent import run_agent_approval_iteration, run_agent_approval_loop
from marketlab.paper.service import PaperStateStore, run_paper_decision


def _capture_transport(calls: list[dict[str, object]]):
    def _transport(url: str, payload: dict[str, object], timeout_seconds: int) -> tuple[int, str]:
        calls.append(
            {
                "url": url,
                "payload": payload,
                "timeout_seconds": timeout_seconds,
            }
        )
        return 200, '{"ok": true, "result": {"message_id": 1}}'

    return _transport


def _configure_notification_env(monkeypatch) -> None:
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "bot-token")
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "chat-id")


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
        telegram_enabled=True,
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
    _configure_notification_env(monkeypatch)
    calls: list[dict[str, object]] = []

    run_agent_approval_iteration(
        config,
        now=datetime(2026, 4, 10, 20, 30, tzinfo=UTC),
        broker=FakeAlpacaBroker(symbol="VOO"),
        notification_transport=_capture_transport(calls),
    )

    proposal = PaperStateStore(config).latest_proposal()
    assert proposal is not None
    assert proposal["approval_status"] == "approved"
    assert proposal["approval_backend"] == "deterministic_consensus"
    assert proposal["approval_fallback_used"] is True
    assert "openai backend failed" in proposal["approval_fallback_reason"]
    assert len(calls) == 1
    assert "provider: deterministic_consensus" in calls[0]["payload"]["text"]
    assert "fallback_used: true" in calls[0]["payload"]["text"]
    assert "fallback_reason: openai backend failed" in calls[0]["payload"]["text"]


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


def test_agent_worker_rejects_proposal_when_evidence_is_missing(tmp_path: Path) -> None:
    config = build_phase7_paper_config(tmp_path, execution_mode="agent_approval")
    store = PaperStateStore(config)
    proposal_id = "2026-04-13-VOO-2026-04-10"
    store.save_proposal(
        {
            "proposal_id": proposal_id,
            "experiment_name": config.experiment_name,
            "symbol": "VOO",
            "signal_date": "2026-04-10",
            "effective_date": "2026-04-13",
            "execution_mode": config.paper.execution_mode,
            "approval_status": "pending",
            "submission_status": "pending",
            "created_at": datetime(2026, 4, 10, 20, 10, tzinfo=UTC).isoformat(),
        }
    )

    result = run_agent_approval_iteration(
        config,
        now=datetime(2026, 4, 10, 20, 30, tzinfo=UTC),
        broker=FakeAlpacaBroker(symbol="VOO"),
    )

    proposal = PaperStateStore(config).latest_proposal()
    assert result["processed_count"] == 1
    assert proposal is not None
    assert proposal["approval_status"] == "rejected"
    assert "could not read the persisted proposal evidence" in proposal["approval_rationale"]


def test_agent_worker_loop_deduplicates_repeated_error_alerts_until_recovery(
    monkeypatch,
    tmp_path: Path,
) -> None:
    config = build_phase7_paper_config(tmp_path, telegram_enabled=True)
    calls: list[dict[str, object]] = []
    _configure_notification_env(monkeypatch)

    def _failing_iteration(*args, **kwargs):
        raise RuntimeError("approval backend exploded")

    def _successful_iteration(*args, **kwargs):
        state = agent_module._load_worker_state(config)
        agent_module._clear_worker_error_state(state)
        state["last_checked_at"] = datetime(2026, 4, 10, 20, 40, tzinfo=UTC).isoformat()
        state["last_result"] = "no_pending_proposals"
        state_path = agent_module._save_worker_state(config, state)
        return {
            "agent_state_path": str(state_path),
            "events": [],
            "processed_count": 0,
        }

    monkeypatch.setattr(agent_module, "run_agent_approval_iteration", _failing_iteration)
    with pytest.raises(RuntimeError, match="approval backend exploded"):
        run_agent_approval_loop(
            config,
            once=True,
            notification_transport=_capture_transport(calls),
        )
    with pytest.raises(RuntimeError, match="approval backend exploded"):
        run_agent_approval_loop(
            config,
            once=True,
            notification_transport=_capture_transport(calls),
        )

    monkeypatch.setattr(agent_module, "run_agent_approval_iteration", _successful_iteration)
    run_agent_approval_loop(
        config,
        once=True,
        notification_transport=_capture_transport(calls),
    )

    monkeypatch.setattr(agent_module, "run_agent_approval_iteration", _failing_iteration)
    with pytest.raises(RuntimeError, match="approval backend exploded"):
        run_agent_approval_loop(
            config,
            once=True,
            notification_transport=_capture_transport(calls),
        )

    records = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted(PaperStateStore(config).notifications_root.glob("*.json"))
    ]
    assert len(calls) == 2
    assert len(records) == 2
    assert all(record["stage"] == "paper-error" for record in records)


def test_agent_worker_loop_once_propagates_iteration_failures_after_notifying(
    monkeypatch,
    tmp_path: Path,
) -> None:
    config = build_phase7_paper_config(tmp_path, telegram_enabled=True)
    calls: list[dict[str, object]] = []
    _configure_notification_env(monkeypatch)

    def _failing_iteration(*args, **kwargs):
        raise RuntimeError("agent approval failed")

    monkeypatch.setattr(agent_module, "run_agent_approval_iteration", _failing_iteration)

    with pytest.raises(RuntimeError, match="agent approval failed"):
        run_agent_approval_loop(
            config,
            once=True,
            notification_transport=_capture_transport(calls),
        )

    records = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted(PaperStateStore(config).notifications_root.glob("*.json"))
    ]
    assert len(calls) == 1
    assert len(records) == 1
    assert records[0]["stage"] == "paper-error"
