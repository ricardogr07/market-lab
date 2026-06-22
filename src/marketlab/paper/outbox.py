from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Protocol, runtime_checkable

from marketlab.config import ExperimentConfig
from marketlab.paper.contracts import (
    PaperNotificationSink,
    PaperOutboxRecord,
    PaperUnitOfWork,
    PaperUnitOfWorkFactory,
)
from marketlab.paper.core import _now_utc
from marketlab.paper.notifications import DELIVERY_FAILED
from marketlab.paper.state import _json_load

PAPER_NOTIFICATION_EVENT_TYPE = "paper.notification.requested"
_NOTIFICATION_STAGES = frozenset(("decision", "approval", "submission"))


@runtime_checkable
class PaperOutboxPublisher(Protocol):
    def publish(self, record: PaperOutboxRecord) -> None: ...


@dataclass(slots=True, frozen=True)
class PaperOutboxDeliveryResult:
    delivered_message_ids: tuple[str, ...]
    failed_message_ids: tuple[str, ...]


class InMemoryPaperOutboxPublisher(PaperOutboxPublisher):
    """Deterministic publisher for unit tests and local adapter verification."""

    def __init__(self) -> None:
        self.records: list[PaperOutboxRecord] = []

    def publish(self, record: PaperOutboxRecord) -> None:
        self.records.append(record)


def _project_fields(payload: Mapping[str, Any] | None, fields: tuple[str, ...]) -> dict[str, Any]:
    if payload is None:
        return {}
    return {field: payload[field] for field in fields if field in payload}


def enqueue_paper_notification(
    uow: PaperUnitOfWork,
    *,
    stage: str,
    outcome: str,
    status: Mapping[str, Any],
    proposal: Mapping[str, Any] | None = None,
    approval_record: Mapping[str, Any] | None = None,
    submission: Mapping[str, Any] | None = None,
) -> PaperOutboxRecord:
    """Persist one notification intent with the phase state change.

    The payload is intentionally limited to fields rendered in Telegram
    messages. This makes a replay of the same business outcome idempotent even
    when a status artifact has a new timestamp or path.
    """

    if stage not in _NOTIFICATION_STAGES:
        allowed = ", ".join(sorted(_NOTIFICATION_STAGES))
        raise ValueError(f"Unsupported paper notification stage: {stage}. Expected one of: {allowed}")
    normalized_outcome = outcome.strip()
    if normalized_outcome == "":
        raise ValueError("Paper notification outcome must not be empty.")
    payload: dict[str, Any] = {
        "stage": stage,
        "outcome": normalized_outcome,
        "status": _project_fields(
            status,
            ("event", "status", "reason", "market_date", "latest_signal_date", "order_status"),
        ),
        "proposal": _project_fields(
            proposal,
            (
                "proposal_id",
                "symbol",
                "signal_date",
                "effective_date",
                "decision",
                "target_weight",
                "long_vote_count",
                "cash_vote_count",
                "reference_price",
            ),
        ),
        "approval_record": _project_fields(
            approval_record,
            (
                "approval_status",
                "actor",
                "provider",
                "model",
                "fallback_used",
                "fallback_reason",
                "rationale",
            ),
        ),
        "submission": _project_fields(
            submission,
            (
                "proposal_id",
                "trade_date",
                "reason",
                "side",
                "qty",
                "notional",
                "order_id",
                "order_status",
            ),
        ),
    }
    fingerprint = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    created_at = str(status.get("updated_at", "")).strip() or _now_utc().isoformat()
    return uow.outbox.enqueue(
        message_id=f"paper.notification:{stage}:{fingerprint}",
        event_type=PAPER_NOTIFICATION_EVENT_TYPE,
        payload=payload,
        created_at=created_at,
    )


def _notification_now(record: PaperOutboxRecord) -> datetime:
    try:
        return datetime.fromisoformat(record.created_at.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError("Paper notification outbox created_at must be ISO-8601.") from exc


class PaperNotificationOutboxPublisher(PaperOutboxPublisher):
    """Deliver persisted notification intents through the existing sink port."""

    def __init__(self, sink: PaperNotificationSink) -> None:
        self._sink = sink

    def publish(self, record: PaperOutboxRecord) -> None:
        if record.event_type != PAPER_NOTIFICATION_EVENT_TYPE:
            raise ValueError(f"Unsupported notification outbox event: {record.event_type}")
        payload = record.payload
        stage = str(payload.get("stage", ""))
        outcome = str(payload.get("outcome", ""))
        status = _mapping_payload(payload, "status")
        proposal = _optional_mapping_payload(payload, "proposal")
        now = _notification_now(record)
        if stage == "decision":
            path = self._sink.notify_decision(
                outcome=outcome,
                status=status,
                proposal=proposal,
                now=now,
            )
        elif stage == "approval":
            if proposal is None:
                raise ValueError("Paper approval notification requires a proposal payload.")
            path = self._sink.notify_approval(
                proposal=proposal,
                approval_record=_mapping_payload(payload, "approval_record"),
                now=now,
            )
        elif stage == "submission":
            path = self._sink.notify_submission(
                outcome=outcome,
                status=status,
                proposal=proposal,
                submission=_optional_mapping_payload(payload, "submission"),
                now=now,
            )
        else:
            raise ValueError(f"Unsupported paper notification stage: {stage}")
        self._raise_on_failed_telegram_delivery(path)

    @staticmethod
    def _raise_on_failed_telegram_delivery(path: Path) -> None:
        if not path.exists():
            return
        payload = _json_load(path)
        if payload.get("delivery_status") == DELIVERY_FAILED:
            raise RuntimeError(str(payload.get("error", "Telegram delivery failed.")))


def _mapping_payload(payload: Mapping[str, Any], key: str) -> dict[str, Any]:
    value = payload.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"Paper outbox {key} payload must be an object.")
    return dict(value)


def _optional_mapping_payload(payload: Mapping[str, Any], key: str) -> dict[str, Any] | None:
    value = payload.get(key)
    if value in (None, {}):
        return None
    if not isinstance(value, dict):
        raise ValueError(f"Paper outbox {key} payload must be an object when present.")
    return dict(value)


def _service_bus_envelope(record: PaperOutboxRecord) -> dict[str, Any]:
    return {
        "message_id": record.message_id,
        "event_type": record.event_type,
        "created_at": record.created_at,
        "payload": dict(record.payload),
    }


def _send_azure_service_bus_message(
    *,
    namespace: str,
    queue_name: str,
    envelope: dict[str, Any],
) -> None:
    try:
        from azure.identity import DefaultAzureCredential
        from azure.servicebus import ServiceBusClient, ServiceBusMessage
    except ImportError as exc:  # pragma: no cover - depends on the optional Azure extra.
        raise RuntimeError(
            "Azure Service Bus publishing requires the 'azure' optional dependency. "
            "Install MarketLab with `.[azure]`."
        ) from exc

    credential = DefaultAzureCredential(exclude_interactive_browser_credential=True)
    message = ServiceBusMessage(
        json.dumps(envelope, sort_keys=True),
        message_id=str(envelope["message_id"]),
        subject=str(envelope["event_type"]),
    )
    with ServiceBusClient(
        fully_qualified_namespace=namespace,
        credential=credential,
    ) as client:
        with client.get_queue_sender(queue_name=queue_name) as sender:
            sender.send_messages(message)


class AzureServiceBusPaperOutboxPublisher(PaperOutboxPublisher):
    """Publish durable outbox records with a stable Service Bus message ID."""

    def __init__(
        self,
        config: ExperimentConfig,
        *,
        send: Callable[[dict[str, Any]], None] | None = None,
    ) -> None:
        azure = config.paper.azure
        if azure.service_bus_backend != "azure_service_bus":
            raise ValueError(
                "AzureServiceBusPaperOutboxPublisher requires "
                "paper.azure.service_bus_backend='azure_service_bus'."
            )
        namespace = azure.service_bus_namespace.strip()
        queue_name = azure.service_bus_queue_name.strip()
        if namespace == "" or queue_name == "":
            raise ValueError(
                "paper.azure.service_bus_namespace and paper.azure.service_bus_queue_name are required."
            )
        self._send = send or (
            lambda envelope: _send_azure_service_bus_message(
                namespace=namespace,
                queue_name=queue_name,
                envelope=envelope,
            )
        )

    def publish(self, record: PaperOutboxRecord) -> None:
        self._send(_service_bus_envelope(record))


def build_paper_outbox_publisher(config: ExperimentConfig) -> PaperOutboxPublisher | None:
    backend = config.paper.azure.service_bus_backend
    if backend == "disabled":
        return None
    if backend == "in_memory":
        return InMemoryPaperOutboxPublisher()
    if backend == "azure_service_bus":
        return AzureServiceBusPaperOutboxPublisher(config)
    raise ValueError(f"Unsupported paper Service Bus backend: {backend}")


def deliver_pending_paper_outbox(
    *,
    uow_factory: PaperUnitOfWorkFactory,
    publisher: PaperOutboxPublisher,
    now: datetime | None = None,
    limit: int = 100,
    event_types: frozenset[str] | None = None,
) -> PaperOutboxDeliveryResult:
    """Deliver queued records without holding a transaction across publication.

    A process crash after ``publish`` and before ``mark_delivered`` deliberately
    causes a duplicate message on retry. Consumers must use the message ID as
    their idempotency key.
    """

    if limit < 1:
        raise ValueError("Paper outbox limit must be at least 1.")
    with uow_factory() as uow:
        pending = uow.outbox.list_pending(limit=limit, event_types=event_types)

    delivered: list[str] = []
    failed: list[str] = []
    for record in pending:
        try:
            publisher.publish(record)
        except Exception as exc:
            with uow_factory() as uow:
                uow.outbox.mark_failed(
                    message_id=record.message_id,
                    error=f"{type(exc).__name__}: {exc}",
                )
                uow.commit()
            failed.append(record.message_id)
            continue
        with uow_factory() as uow:
            uow.outbox.mark_delivered(
                message_id=record.message_id,
                delivered_at=_now_utc(now).isoformat(),
            )
            uow.commit()
        delivered.append(record.message_id)

    return PaperOutboxDeliveryResult(
        delivered_message_ids=tuple(delivered),
        failed_message_ids=tuple(failed),
    )


def deliver_pending_paper_notifications(
    *,
    uow_factory: PaperUnitOfWorkFactory,
    sink: PaperNotificationSink,
    now: datetime | None = None,
    limit: int = 100,
) -> PaperOutboxDeliveryResult:
    return deliver_pending_paper_outbox(
        uow_factory=uow_factory,
        publisher=PaperNotificationOutboxPublisher(sink),
        now=now,
        limit=limit,
        event_types=frozenset((PAPER_NOTIFICATION_EVENT_TYPE,)),
    )
