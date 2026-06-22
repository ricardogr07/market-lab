from __future__ import annotations

import json
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Mapping, Protocol

from marketlab.config import ExperimentConfig
from marketlab.paper.approval_clients import build_default_paper_approval_client
from marketlab.paper.contracts import (
    PaperApprovalClient,
    PaperApprovalClientDecision,
    PaperApprovalEvaluationRequest,
    PaperBroker,
    PaperHostedExecutionContext,
)
from marketlab.paper.core import APPROVAL_PENDING, validate_paper_trading_config
from marketlab.paper.service import (
    _paper_broker_factory,
    decide_paper_proposal,
    get_paper_status,
    read_paper_evidence,
    read_paper_proposal,
)

PAPER_APPROVAL_REQUEST_EVENT_TYPE = "paper.approval.requested"


class PaperServiceBusReceiver(Protocol):
    """Subset of a peek-lock Service Bus receiver used by the worker."""

    def receive_messages(
        self,
        *,
        max_message_count: int,
        max_wait_time: float,
    ) -> Iterable[Any]: ...

    def complete_message(self, message: Any) -> None: ...

    def abandon_message(self, message: Any) -> None: ...


@dataclass(slots=True, frozen=True)
class PaperServiceBusReceiveResult:
    """One bounded receiver pass and its explicit settlement outcomes."""

    completed_message_ids: tuple[str, ...]
    abandoned_message_ids: tuple[str, ...]
    failure_messages: tuple[str, ...]
    processed: tuple[dict[str, Any], ...]


@dataclass(slots=True, frozen=True)
class PaperApprovalRequestMessage:
    message_id: str
    proposal_id: str
    trade_date: str
    hosted_context: PaperHostedExecutionContext

    @classmethod
    def from_envelope(cls, envelope: Mapping[str, Any]) -> PaperApprovalRequestMessage:
        event_type = str(envelope.get("event_type", "")).strip()
        if event_type != PAPER_APPROVAL_REQUEST_EVENT_TYPE:
            raise ValueError(f"Unsupported paper Service Bus event_type: {event_type}")
        message_id = str(envelope.get("message_id", "")).strip()
        payload = envelope.get("payload")
        if message_id == "" or not isinstance(payload, Mapping):
            raise ValueError("Paper approval request requires message_id and an object payload.")
        proposal_id = str(payload.get("proposal_id", "")).strip()
        trade_date = str(payload.get("trade_date", "")).strip()
        raw_context = payload.get("hosted_context")
        if proposal_id == "" or trade_date == "" or not isinstance(raw_context, Mapping):
            raise ValueError(
                "Paper approval request requires proposal_id, trade_date, and hosted_context."
            )
        hosted_context = PaperHostedExecutionContext.from_metadata(raw_context)
        if hosted_context.phase != "agent_approve":
            raise ValueError("Paper approval request hosted_context.phase must be 'agent_approve'.")
        if hosted_context.idempotency_key != message_id:
            raise ValueError("Paper approval request message_id must match hosted_context.idempotency_key.")
        return cls(
            message_id=message_id,
            proposal_id=proposal_id,
            trade_date=trade_date,
            hosted_context=hosted_context,
        )


def _account_context(
    config: ExperimentConfig,
    *,
    broker: PaperBroker | None,
) -> dict[str, Any]:
    client = broker if broker is not None else _paper_broker_factory(config)()
    symbol = str(config.data.symbols[0])
    return {
        "account": client.get_account(),
        "position": client.get_position(symbol),
    }


def _reject_missing_evidence(
    config: ExperimentConfig,
    *,
    message: PaperApprovalRequestMessage,
    error: FileNotFoundError,
    now: datetime | None,
) -> dict[str, Any]:
    return decide_paper_proposal(
        config,
        proposal_id=message.proposal_id,
        decision="reject",
        actor="agent",
        rationale=(
            "Rejected because the approval worker could not read the persisted "
            f"proposal evidence: {error}."
        ),
        fallback_reason=str(error),
        now=now,
        hosted_context=message.hosted_context,
    )


def consume_paper_approval_request(
    config: ExperimentConfig,
    *,
    envelope: Mapping[str, Any],
    now: datetime | None = None,
    broker: PaperBroker | None = None,
    approval_client: PaperApprovalClient | None = None,
) -> dict[str, Any]:
    """Process one Service Bus approval request without acknowledging transport.

    The caller completes the Service Bus message only after this function
    returns. A duplicate delivered after a committed approval is a no-op and
    does not call the broker or approval provider again.
    """

    message = PaperApprovalRequestMessage.from_envelope(envelope)
    validate_paper_trading_config(config)
    if config.paper.execution_mode != "agent_approval":
        raise RuntimeError("Paper approval requests require execution_mode='agent_approval'.")

    proposal = read_paper_proposal(config, proposal_id=message.proposal_id)
    if str(proposal.get("effective_date", "")) != message.trade_date:
        raise ValueError("Paper approval request trade_date does not match the persisted proposal.")
    if proposal.get("approval_status") != APPROVAL_PENDING:
        return {
            "message_id": message.message_id,
            "proposal_id": message.proposal_id,
            "outcome": "already_processed",
        }

    try:
        evidence = read_paper_evidence(config, proposal_id=message.proposal_id)
    except FileNotFoundError as exc:
        approval = _reject_missing_evidence(
            config,
            message=message,
            error=exc,
            now=now,
        )
        return {
            "message_id": message.message_id,
            "proposal_id": message.proposal_id,
            "outcome": "processed",
            "approval": approval,
        }

    decision: PaperApprovalClientDecision = (
        approval_client or build_default_paper_approval_client(config)
    ).evaluate(
        PaperApprovalEvaluationRequest(
            proposal=proposal,
            evidence=evidence,
            status=get_paper_status(config).get("status"),
            account_context=_account_context(config, broker=broker),
        )
    )
    approval = decide_paper_proposal(
        config,
        proposal_id=message.proposal_id,
        decision=decision.decision,
        actor="agent",
        rationale=decision.rationale,
        provider=decision.provider,
        model=decision.model,
        fallback_used=decision.fallback_used,
        fallback_reason=decision.fallback_reason,
        now=now,
        hosted_context=message.hosted_context,
    )
    return {
        "message_id": message.message_id,
        "proposal_id": message.proposal_id,
        "outcome": "processed",
        "approval": approval,
    }


def _decode_service_bus_envelope(message: Any) -> dict[str, Any]:
    """Decode an Azure SDK message body without depending on the SDK at import time."""

    raw_body = getattr(message, "body", None)
    if isinstance(raw_body, str):
        body_text = raw_body
    elif isinstance(raw_body, (bytes, bytearray)):
        body_text = bytes(raw_body).decode("utf-8")
    else:
        if raw_body is None:
            raise ValueError("Paper Service Bus message body must be text or bytes.")
        try:
            chunks = tuple(raw_body)
        except TypeError as exc:
            raise ValueError("Paper Service Bus message body must be text or bytes.") from exc
        if not chunks or not all(isinstance(chunk, (bytes, bytearray)) for chunk in chunks):
            raise ValueError("Paper Service Bus message body must contain UTF-8 bytes.")
        body_text = b"".join(bytes(chunk) for chunk in chunks).decode("utf-8")
    try:
        envelope = json.loads(body_text)
    except json.JSONDecodeError as exc:
        raise ValueError("Paper Service Bus message body must be valid JSON.") from exc
    if not isinstance(envelope, dict):
        raise ValueError("Paper Service Bus message body must be a JSON object.")
    return envelope


def _transport_message_id(message: Any) -> str:
    return str(getattr(message, "message_id", "") or "").strip() or "<unknown>"


def process_paper_approval_messages(
    receiver: PaperServiceBusReceiver,
    messages: Iterable[Any],
    *,
    config: ExperimentConfig,
    now: datetime | None = None,
    broker: PaperBroker | None = None,
    approval_client: PaperApprovalClient | None = None,
    handler: Callable[[Mapping[str, Any]], dict[str, Any]] | None = None,
) -> PaperServiceBusReceiveResult:
    """Process a bounded batch and settle each peek-lock message exactly once.

    Completion occurs only after the domain consumer returns. Domain or payload
    failures are abandoned so Service Bus can retry or dead-letter them under
    the queue policy. A settlement failure is allowed to raise because the
    broker's eventual lock outcome is then unknown.
    """

    consume = handler or (
        lambda envelope: consume_paper_approval_request(
            config,
            envelope=envelope,
            now=now,
            broker=broker,
            approval_client=approval_client,
        )
    )
    completed: list[str] = []
    abandoned: list[str] = []
    failures: list[str] = []
    processed: list[dict[str, Any]] = []
    for transport_message in messages:
        message_id = _transport_message_id(transport_message)
        try:
            envelope = _decode_service_bus_envelope(transport_message)
            message_id = str(envelope.get("message_id", message_id))
            result = consume(envelope)
        except Exception as exc:
            receiver.abandon_message(transport_message)
            abandoned.append(message_id)
            failures.append(f"{message_id}: {type(exc).__name__}: {exc}")
            continue
        receiver.complete_message(transport_message)
        completed.append(message_id)
        processed.append(result)
    return PaperServiceBusReceiveResult(
        completed_message_ids=tuple(completed),
        abandoned_message_ids=tuple(abandoned),
        failure_messages=tuple(failures),
        processed=tuple(processed),
    )


def receive_paper_approval_requests(
    config: ExperimentConfig,
    *,
    max_messages: int = 10,
    max_wait_seconds: float = 5.0,
    now: datetime | None = None,
    broker: PaperBroker | None = None,
    approval_client: PaperApprovalClient | None = None,
    receiver: PaperServiceBusReceiver | None = None,
) -> PaperServiceBusReceiveResult:
    """Receive and settle one bounded approval-worker batch.

    Passing a receiver keeps this deterministic for tests. The Azure client is
    constructed lazily only for the real worker runtime, using workload
    identity through ``DefaultAzureCredential``.
    """

    if max_messages < 1:
        raise ValueError("Paper Service Bus max_messages must be at least 1.")
    if max_wait_seconds < 0:
        raise ValueError("Paper Service Bus max_wait_seconds must not be negative.")
    validate_paper_trading_config(config)
    if config.paper.execution_mode != "agent_approval":
        raise ValueError("Paper Service Bus approval worker requires execution_mode='agent_approval'.")
    azure = config.paper.azure
    if azure.service_bus_backend != "azure_service_bus":
        raise ValueError(
            "Paper Service Bus approval worker requires "
            "paper.azure.service_bus_backend='azure_service_bus'."
        )

    def _process(active_receiver: PaperServiceBusReceiver) -> PaperServiceBusReceiveResult:
        messages = active_receiver.receive_messages(
            max_message_count=max_messages,
            max_wait_time=max_wait_seconds,
        )
        return process_paper_approval_messages(
            active_receiver,
            messages,
            config=config,
            now=now,
            broker=broker,
            approval_client=approval_client,
        )

    if receiver is not None:
        return _process(receiver)

    try:
        from azure.identity import DefaultAzureCredential
        from azure.servicebus import ServiceBusClient
    except ImportError as exc:  # pragma: no cover - depends on the optional Azure extra.
        raise RuntimeError(
            "Azure Service Bus receiving requires the 'azure' optional dependency. "
            "Install MarketLab with `.[azure]`."
        ) from exc

    credential = DefaultAzureCredential(exclude_interactive_browser_credential=True)
    with ServiceBusClient(
        fully_qualified_namespace=azure.service_bus_namespace,
        credential=credential,
    ) as client:
        with client.get_queue_receiver(queue_name=azure.service_bus_queue_name) as active_receiver:
            return _process(active_receiver)
