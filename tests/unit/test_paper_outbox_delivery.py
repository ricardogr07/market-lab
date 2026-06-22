from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest
from tests._paper_fakes import (
    FakeAlpacaBroker,
    FakeAlpacaProvider,
    FakePaperNotificationSink,
    build_phase7_paper_config,
)

from marketlab.config import PaperAzureConfig
from marketlab.paper.application import DecisionService
from marketlab.paper.contracts import (
    PaperDecisionRequest,
    PaperHostedExecutionContext,
    PaperOutboxRecord,
)
from marketlab.paper.notifications import build_telegram_paper_notification_sink
from marketlab.paper.outbox import (
    PAPER_NOTIFICATION_EVENT_TYPE,
    AzureServiceBusPaperOutboxPublisher,
    InMemoryPaperOutboxPublisher,
    deliver_pending_paper_notifications,
    deliver_pending_paper_outbox,
)
from marketlab.paper.persistence import build_filesystem_paper_uow_factory
from marketlab.paper.service import run_paper_decision
from marketlab.paper.state import PaperStateStore, _json_load


def _hosted_context() -> PaperHostedExecutionContext:
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


def test_hosted_agent_approval_decision_enqueues_service_bus_intent_atomically(tmp_path: Path) -> None:
    config = build_phase7_paper_config(tmp_path, symbol="QQQ", execution_mode="agent_approval")
    factory = build_filesystem_paper_uow_factory(config)
    context = _hosted_context()

    result = DecisionService(config, uow_factory=factory).run(
        PaperDecisionRequest(
            now=datetime(2026, 4, 10, 20, 20, tzinfo=UTC),
            provider=FakeAlpacaProvider(symbol="QQQ"),
            broker=FakeAlpacaBroker(symbol="QQQ"),
            hosted_context=context,
        )
    )

    expected_context = context.derive(
        phase="agent_approve",
        suffix=result.proposal_id,
    )
    with factory() as uow:
        record = uow.outbox.get(expected_context.idempotency_key)
        assert record is not None
        assert record.event_type == "paper.approval.requested"
        assert record.payload == {
            "proposal_id": result.proposal_id,
            "trade_date": "2026-04-13",
            "hosted_context": expected_context.as_metadata(),
        }
        assert uow.trades.get_proposal(result.proposal_id) == result.proposal

    sink = FakePaperNotificationSink()
    delivered = deliver_pending_paper_notifications(
        uow_factory=factory,
        sink=sink,
        now=datetime(2026, 4, 10, 20, 21, tzinfo=UTC),
    )

    assert len(delivered.delivered_message_ids) == 1
    assert len(sink.decision_calls) == 1
    with factory() as uow:
        approval_request = uow.outbox.get(expected_context.idempotency_key)
        assert approval_request is not None
        assert approval_request.delivery_status == "pending"


def test_outbox_dispatcher_retries_failed_delivery_and_deduplicates_after_success(
    tmp_path: Path,
) -> None:
    config = build_phase7_paper_config(tmp_path, symbol="QQQ")
    factory = build_filesystem_paper_uow_factory(config)
    message_id = "approval-request:proposal-1"
    with factory() as uow:
        uow.outbox.enqueue(
            message_id=message_id,
            event_type="paper.approval.requested",
            payload={"proposal_id": "proposal-1"},
            created_at="2026-04-10T20:20:00+00:00",
        )
        uow.commit()

    class _FailingPublisher:
        def publish(self, record: PaperOutboxRecord) -> None:
            raise RuntimeError(f"queue unavailable for {record.message_id}")

    first = deliver_pending_paper_outbox(
        uow_factory=factory,
        publisher=_FailingPublisher(),
        now=datetime(2026, 4, 10, 20, 21, tzinfo=UTC),
    )

    assert first.delivered_message_ids == ()
    assert first.failed_message_ids == (message_id,)
    with factory() as uow:
        failed = uow.outbox.get(message_id)
        assert failed is not None
        assert failed.delivery_status == "failed"
        assert failed.delivery_attempts == 1

    publisher = InMemoryPaperOutboxPublisher()
    second = deliver_pending_paper_outbox(
        uow_factory=factory,
        publisher=publisher,
        now=datetime(2026, 4, 10, 20, 22, tzinfo=UTC),
    )
    third = deliver_pending_paper_outbox(
        uow_factory=factory,
        publisher=publisher,
        now=datetime(2026, 4, 10, 20, 23, tzinfo=UTC),
    )

    assert second.delivered_message_ids == (message_id,)
    assert second.failed_message_ids == ()
    assert third.delivered_message_ids == ()
    assert publisher.records[0].message_id == message_id
    with factory() as uow:
        delivered = uow.outbox.get(message_id)
        assert delivered is not None
        assert delivered.delivery_status == "delivered"
        assert delivered.delivery_attempts == 2


def test_azure_service_bus_publisher_preserves_outbox_message_id(tmp_path: Path) -> None:
    config = build_phase7_paper_config(tmp_path, symbol="QQQ")
    config.paper.azure = PaperAzureConfig(
        service_bus_backend="azure_service_bus",
        service_bus_namespace="marketlab-uat.servicebus.windows.net",
        service_bus_queue_name="qqq-paper-events",
    )
    envelopes: list[dict[str, object]] = []
    publisher = AzureServiceBusPaperOutboxPublisher(config, send=envelopes.append)
    record = PaperOutboxRecord(
        message_id="approval-request:proposal-1",
        event_type="paper.approval.requested",
        payload={"proposal_id": "proposal-1"},
        created_at="2026-04-10T20:20:00+00:00",
    )

    publisher.publish(record)

    assert envelopes == [
        {
            "message_id": "approval-request:proposal-1",
            "event_type": "paper.approval.requested",
            "created_at": "2026-04-10T20:20:00+00:00",
            "payload": {"proposal_id": "proposal-1"},
        }
    ]


def test_decision_persists_notification_intent_before_delivery(tmp_path: Path) -> None:
    config = build_phase7_paper_config(tmp_path, symbol="QQQ")
    factory = build_filesystem_paper_uow_factory(config)
    result = DecisionService(config, uow_factory=factory).run(
        PaperDecisionRequest(
            now=datetime(2026, 4, 10, 20, 20, tzinfo=UTC),
            provider=FakeAlpacaProvider(symbol="QQQ"),
            broker=FakeAlpacaBroker(symbol="QQQ"),
        )
    )

    assert list(PaperStateStore(config).notifications_root.glob("*.json")) == []
    with factory() as uow:
        pending = uow.outbox.list_pending()
        assert len(pending) == 1
        assert pending[0].event_type == PAPER_NOTIFICATION_EVENT_TYPE
        assert pending[0].payload["stage"] == "decision"
        assert pending[0].payload["outcome"] == "proposal_created"
        assert pending[0].payload["proposal"]["proposal_id"] == result.proposal_id

    class _RecordingSink:
        def __init__(self) -> None:
            self.calls: list[dict[str, object]] = []

        def notify_decision(self, *, outcome, status, proposal=None, now=None) -> Path:
            self.calls.append(
                {
                    "outcome": outcome,
                    "status": dict(status),
                    "proposal": dict(proposal) if proposal is not None else None,
                    "now": now,
                }
            )
            return Path("notification.json")

        def notify_approval(self, **kwargs) -> Path:  # pragma: no cover - stage filter is asserted above.
            raise AssertionError(f"Unexpected approval notification: {kwargs}")

        def notify_submission(self, **kwargs) -> Path:  # pragma: no cover - stage filter is asserted above.
            raise AssertionError(f"Unexpected submission notification: {kwargs}")

        def notify_error(self, **kwargs) -> Path:  # pragma: no cover - unused protocol member.
            raise AssertionError(f"Unexpected error notification: {kwargs}")

    sink = _RecordingSink()
    delivery = deliver_pending_paper_notifications(
        uow_factory=factory,
        sink=sink,
        now=datetime(2026, 4, 10, 20, 21, tzinfo=UTC),
    )

    assert len(delivery.delivered_message_ids) == 1
    assert len(sink.calls) == 1
    assert sink.calls[0]["outcome"] == "proposal_created"
    with factory() as uow:
        assert uow.outbox.list_pending() == []


def test_notification_delivery_failure_is_retryable_without_changing_decision(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    config = build_phase7_paper_config(tmp_path, symbol="QQQ", telegram_enabled=True)
    factory = build_filesystem_paper_uow_factory(config)
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "bot-token")
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "chat-id")
    monkeypatch.delenv("MARKETLAB_PAPER_TELEGRAM_ENABLED", raising=False)
    monkeypatch.delenv("MARKETLAB_PAPER_TELEGRAM_ALLOWED_EXPERIMENTS", raising=False)

    result = run_paper_decision(
        config,
        now=datetime(2026, 4, 10, 20, 20, tzinfo=UTC),
        provider=FakeAlpacaProvider(symbol="QQQ"),
        broker=FakeAlpacaBroker(symbol="QQQ"),
        notification_sink=build_telegram_paper_notification_sink(
            config,
            transport=lambda _url, _payload, _timeout: (_ for _ in ()).throw(
                RuntimeError("telegram unavailable")
            ),
        ),
    )

    assert result["status"]["status"] == "proposal_created"
    with factory() as uow:
        failed = uow.outbox.list_pending()
        assert len(failed) == 1
        assert failed[0].delivery_status == "failed"
        assert failed[0].delivery_attempts == 1

    successful_calls: list[dict[str, object]] = []
    retry = deliver_pending_paper_notifications(
        uow_factory=factory,
        sink=build_telegram_paper_notification_sink(
            config,
            transport=lambda url, payload, timeout: successful_calls.append(
                {"url": url, "payload": payload, "timeout": timeout}
            )
            or (200, '{"ok": true}'),
        ),
        now=datetime(2026, 4, 10, 20, 21, tzinfo=UTC),
    )

    assert len(retry.delivered_message_ids) == 1
    assert len(successful_calls) == 1
    records = [
        _json_load(path)
        for path in sorted(PaperStateStore(config).notifications_root.glob("*.json"))
    ]
    assert records[0]["delivery_status"] == "failed_delivery"
    assert records[-1]["delivery_status"] == "delivered"
    with factory() as uow:
        record = uow.outbox.get(retry.delivered_message_ids[0])
        assert record is not None
        assert record.delivery_status == "delivered"
        assert record.delivery_attempts == 2
