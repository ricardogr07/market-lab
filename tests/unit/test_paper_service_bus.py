from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
from tests._paper_fakes import (
    FakeAlpacaBroker,
    FakeAlpacaProvider,
    FakePaperApprovalClient,
    build_phase7_paper_config,
)

from marketlab.paper.application import DecisionService
from marketlab.paper.contracts import (
    PaperApprovalClientDecision,
    PaperDecisionRequest,
    PaperHostedExecutionContext,
)
from marketlab.paper.persistence import build_filesystem_paper_uow_factory
from marketlab.paper.service import read_paper_proposal
from marketlab.paper.service_bus import (
    consume_paper_approval_request,
    process_paper_approval_messages,
)


@dataclass
class _FakeServiceBusMessage:
    message_id: str
    body: str | bytes | tuple[bytes, ...]


@dataclass
class _RecordingServiceBusReceiver:
    completed: list[_FakeServiceBusMessage] = field(default_factory=list)
    abandoned: list[_FakeServiceBusMessage] = field(default_factory=list)

    def complete_message(self, message: _FakeServiceBusMessage) -> None:
        self.completed.append(message)

    def abandon_message(self, message: _FakeServiceBusMessage) -> None:
        self.abandoned.append(message)


def _decision_context() -> PaperHostedExecutionContext:
    return PaperHostedExecutionContext(
        deployment_id="qqq-paper-uat",
        environment="uat",
        phase="decision",
        execution_id="decision-run-1",
        correlation_id="correlation-1",
        idempotency_key="decision:2026-04-10",
        trigger_source="scheduler",
        requested_at="2026-04-10T20:20:00+00:00",
        config_version="config-sha-1",
        image_digest="sha256:image-1",
    )


def _approval_envelope(config, *, proposal_id: str) -> dict[str, object]:
    factory = build_filesystem_paper_uow_factory(config)
    with factory() as uow:
        records = [
            record
            for record in uow.outbox.list_pending()
            if record.event_type == "paper.approval.requested"
        ]
    assert len(records) == 1
    record = records[0]
    assert record.payload["proposal_id"] == proposal_id
    return {
        "message_id": record.message_id,
        "event_type": record.event_type,
        "created_at": record.created_at,
        "payload": record.payload,
    }


def test_service_bus_approval_consumer_processes_one_message_and_ignores_duplicate(
    tmp_path: Path,
) -> None:
    config = build_phase7_paper_config(tmp_path, symbol="QQQ", execution_mode="agent_approval")
    factory = build_filesystem_paper_uow_factory(config)
    decision = DecisionService(config, uow_factory=factory).run(
        PaperDecisionRequest(
            now=datetime(2026, 4, 10, 20, 20, tzinfo=UTC),
            provider=FakeAlpacaProvider(symbol="QQQ"),
            broker=FakeAlpacaBroker(symbol="QQQ"),
            hosted_context=_decision_context(),
        )
    )
    approval_client = FakePaperApprovalClient(
        decision=PaperApprovalClientDecision(
            decision="approve",
            rationale="Approved by the message consumer fake.",
            provider="fake",
            model="fake-model",
        )
    )
    broker = FakeAlpacaBroker(symbol="QQQ")
    envelope = _approval_envelope(config, proposal_id=decision.proposal_id)

    first = consume_paper_approval_request(
        config,
        envelope=envelope,
        now=datetime(2026, 4, 10, 20, 21, tzinfo=UTC),
        broker=broker,
        approval_client=approval_client,
    )
    second = consume_paper_approval_request(
        config,
        envelope=envelope,
        now=datetime(2026, 4, 10, 20, 22, tzinfo=UTC),
        broker=broker,
        approval_client=approval_client,
    )

    assert first["outcome"] == "processed"
    assert second == {
        "message_id": envelope["message_id"],
        "proposal_id": decision.proposal_id,
        "outcome": "already_processed",
    }
    assert len(approval_client.requests) == 1
    assert read_paper_proposal(config, proposal_id=decision.proposal_id)["approval_status"] == "approved"


def test_service_bus_approval_consumer_rejects_mismatched_message_id(tmp_path: Path) -> None:
    config = build_phase7_paper_config(tmp_path, symbol="QQQ", execution_mode="agent_approval")
    context = _decision_context().derive(phase="agent_approve", suffix="proposal-1")

    with pytest.raises(ValueError, match="message_id must match"):
        consume_paper_approval_request(
            config,
            envelope={
                "message_id": "different-id",
                "event_type": "paper.approval.requested",
                "payload": {
                    "proposal_id": "proposal-1",
                    "trade_date": "2026-04-13",
                    "hosted_context": context.as_metadata(),
                },
            },
        )


def test_service_bus_receiver_completes_only_after_the_handler_succeeds(tmp_path: Path) -> None:
    config = build_phase7_paper_config(tmp_path, symbol="QQQ", execution_mode="agent_approval")
    message = _FakeServiceBusMessage(
        message_id="approval-request-1",
        body=json.dumps({"message_id": "approval-request-1"}).encode("utf-8"),
    )
    receiver = _RecordingServiceBusReceiver()
    events: list[str] = []

    def _handler(envelope: dict[str, Any]) -> dict[str, Any]:
        events.append(f"handled:{envelope['message_id']}")
        return {"outcome": "processed"}

    result = process_paper_approval_messages(
        receiver,
        [message],
        config=config,
        handler=_handler,
    )

    assert events == ["handled:approval-request-1"]
    assert receiver.completed == [message]
    assert receiver.abandoned == []
    assert result.completed_message_ids == ("approval-request-1",)
    assert result.abandoned_message_ids == ()
    assert result.failure_messages == ()


def test_service_bus_receiver_abandons_a_failed_message_without_completion(tmp_path: Path) -> None:
    config = build_phase7_paper_config(tmp_path, symbol="QQQ", execution_mode="agent_approval")
    message = _FakeServiceBusMessage(
        message_id="approval-request-2",
        body=(b'{"message_id":"approval-request-2"}',),
    )
    receiver = _RecordingServiceBusReceiver()

    def _handler(envelope: dict[str, Any]) -> dict[str, Any]:
        del envelope
        raise RuntimeError("approval provider unavailable")

    result = process_paper_approval_messages(
        receiver,
        [message],
        config=config,
        handler=_handler,
    )

    assert receiver.completed == []
    assert receiver.abandoned == [message]
    assert result.completed_message_ids == ()
    assert result.abandoned_message_ids == ("approval-request-2",)
    assert result.failure_messages == (
        "approval-request-2: RuntimeError: approval provider unavailable",
    )
