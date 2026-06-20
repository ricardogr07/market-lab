from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Any, Literal, cast, runtime_checkable

import pandas as pd
from typing_extensions import Protocol

PaperHostedEnvironment = Literal["dev", "uat", "paper-prod"]
PaperHostedPhase = Literal["decision", "agent_approve", "submit", "reconcile"]

PAPER_HOSTED_ENVIRONMENTS: frozenset[PaperHostedEnvironment] = frozenset(
    ("dev", "uat", "paper-prod")
)
PAPER_HOSTED_PHASES: frozenset[PaperHostedPhase] = frozenset(
    ("decision", "agent_approve", "submit", "reconcile")
)
PAPER_HOSTED_METADATA_FIELDS = (
    "deployment_id",
    "environment",
    "phase",
    "execution_id",
    "correlation_id",
    "idempotency_key",
    "trigger_source",
    "requested_at",
    "config_version",
    "image_digest",
)


class PaperDeploymentRegistryConflictError(RuntimeError):
    """Raised when a hosted idempotency key is replayed with different metadata."""


def _validated_hosted_metadata(payload: Mapping[str, Any]) -> dict[str, str]:
    extra_keys = sorted(set(payload) - set(PAPER_HOSTED_METADATA_FIELDS))
    if extra_keys:
        joined = ", ".join(extra_keys)
        raise ValueError(f"Paper hosted execution metadata contains unsupported fields: {joined}")
    missing = [
        key
        for key in PAPER_HOSTED_METADATA_FIELDS
        if str(payload.get(key, "")).strip() == ""
    ]
    if missing:
        joined = ", ".join(missing)
        raise ValueError(f"Paper hosted execution metadata is missing required fields: {joined}")
    metadata = {key: str(payload[key]).strip() for key in PAPER_HOSTED_METADATA_FIELDS}
    environment = metadata["environment"]
    if environment not in PAPER_HOSTED_ENVIRONMENTS:
        allowed = ", ".join(PAPER_HOSTED_ENVIRONMENTS)
        raise ValueError(f"Unsupported paper hosted environment: {environment}. Expected one of: {allowed}")
    phase = metadata["phase"]
    if phase not in PAPER_HOSTED_PHASES:
        allowed = ", ".join(PAPER_HOSTED_PHASES)
        raise ValueError(f"Unsupported paper hosted phase: {phase}. Expected one of: {allowed}")
    try:
        datetime.fromisoformat(metadata["requested_at"].replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError("Paper hosted requested_at must be an ISO-8601 datetime.") from exc
    return metadata


def _derive_hosted_value(value: str, *, suffix: str) -> str:
    suffix_value = suffix.strip()
    if suffix_value == "":
        return value
    return f"{value}:{suffix_value}"


@dataclass(slots=True, frozen=True)
class PaperHostedExecutionContext:
    deployment_id: str
    environment: PaperHostedEnvironment
    phase: PaperHostedPhase
    execution_id: str
    correlation_id: str
    idempotency_key: str
    trigger_source: str
    requested_at: str
    config_version: str
    image_digest: str

    def __post_init__(self) -> None:
        _validated_hosted_metadata(self.as_metadata())

    @classmethod
    def from_metadata(
        cls,
        payload: Mapping[str, Any],
    ) -> PaperHostedExecutionContext:
        metadata = _validated_hosted_metadata(payload)
        return cls(
            deployment_id=metadata["deployment_id"],
            environment=cast(PaperHostedEnvironment, metadata["environment"]),
            phase=cast(PaperHostedPhase, metadata["phase"]),
            execution_id=metadata["execution_id"],
            correlation_id=metadata["correlation_id"],
            idempotency_key=metadata["idempotency_key"],
            trigger_source=metadata["trigger_source"],
            requested_at=metadata["requested_at"],
            config_version=metadata["config_version"],
            image_digest=metadata["image_digest"],
        )

    def as_metadata(self) -> dict[str, str]:
        return {
            "deployment_id": self.deployment_id,
            "environment": self.environment,
            "phase": self.phase,
            "execution_id": self.execution_id,
            "correlation_id": self.correlation_id,
            "idempotency_key": self.idempotency_key,
            "trigger_source": self.trigger_source,
            "requested_at": self.requested_at,
            "config_version": self.config_version,
            "image_digest": self.image_digest,
        }

    def derive(
        self,
        *,
        phase: PaperHostedPhase,
        suffix: str = "",
    ) -> PaperHostedExecutionContext:
        if self.phase == phase and suffix.strip() == "":
            return self
        return PaperHostedExecutionContext(
            deployment_id=self.deployment_id,
            environment=self.environment,
            phase=phase,
            execution_id=_derive_hosted_value(self.execution_id, suffix=phase if suffix == "" else suffix),
            correlation_id=self.correlation_id,
            idempotency_key=_derive_hosted_value(
                self.idempotency_key,
                suffix=phase if suffix == "" else suffix,
            ),
            trigger_source=self.trigger_source,
            requested_at=self.requested_at,
            config_version=self.config_version,
            image_digest=self.image_digest,
        )


@dataclass(slots=True, frozen=True)
class PaperDeploymentRecord:
    deployment_id: str
    environment: PaperHostedEnvironment
    phase: PaperHostedPhase
    execution_id: str
    correlation_id: str
    idempotency_key: str
    trigger_source: str
    requested_at: str
    config_version: str
    image_digest: str

    @classmethod
    def from_context(
        cls,
        context: PaperHostedExecutionContext,
    ) -> PaperDeploymentRecord:
        return cls(
            deployment_id=context.deployment_id,
            environment=context.environment,
            phase=context.phase,
            execution_id=context.execution_id,
            correlation_id=context.correlation_id,
            idempotency_key=context.idempotency_key,
            trigger_source=context.trigger_source,
            requested_at=context.requested_at,
            config_version=context.config_version,
            image_digest=context.image_digest,
        )

    @classmethod
    def from_metadata(
        cls,
        payload: Mapping[str, Any],
    ) -> PaperDeploymentRecord:
        context = PaperHostedExecutionContext.from_metadata(payload)
        return cls.from_context(context)

    def as_metadata(self) -> dict[str, str]:
        return {
            "deployment_id": self.deployment_id,
            "environment": self.environment,
            "phase": self.phase,
            "execution_id": self.execution_id,
            "correlation_id": self.correlation_id,
            "idempotency_key": self.idempotency_key,
            "trigger_source": self.trigger_source,
            "requested_at": self.requested_at,
            "config_version": self.config_version,
            "image_digest": self.image_digest,
        }


@dataclass(slots=True, frozen=True)
class PaperPhaseRunRecord:
    deployment_id: str
    environment: PaperHostedEnvironment
    phase: PaperHostedPhase
    execution_id: str
    correlation_id: str
    idempotency_key: str
    trigger_source: str
    requested_at: str
    config_version: str
    image_digest: str

    @classmethod
    def from_context(
        cls,
        context: PaperHostedExecutionContext,
    ) -> PaperPhaseRunRecord:
        return cls(
            deployment_id=context.deployment_id,
            environment=context.environment,
            phase=context.phase,
            execution_id=context.execution_id,
            correlation_id=context.correlation_id,
            idempotency_key=context.idempotency_key,
            trigger_source=context.trigger_source,
            requested_at=context.requested_at,
            config_version=context.config_version,
            image_digest=context.image_digest,
        )

    @classmethod
    def from_metadata(
        cls,
        payload: Mapping[str, Any],
    ) -> PaperPhaseRunRecord:
        context = PaperHostedExecutionContext.from_metadata(payload)
        return cls.from_context(context)

    def as_metadata(self) -> dict[str, str]:
        return {
            "deployment_id": self.deployment_id,
            "environment": self.environment,
            "phase": self.phase,
            "execution_id": self.execution_id,
            "correlation_id": self.correlation_id,
            "idempotency_key": self.idempotency_key,
            "trigger_source": self.trigger_source,
            "requested_at": self.requested_at,
            "config_version": self.config_version,
            "image_digest": self.image_digest,
        }


def _string_field(payload: dict[str, Any], key: str) -> str:
    value = payload.get(key, "")
    if value is None:
        return ""
    return str(value)


def _status_field(payload: dict[str, Any]) -> dict[str, Any]:
    status = payload.get("status", {})
    if not isinstance(status, dict):
        return {}
    return dict(status)


def _mapping_field(payload: dict[str, Any], key: str) -> dict[str, Any] | None:
    value = payload.get(key)
    if not isinstance(value, dict):
        return None
    return dict(value)


@runtime_checkable
class PaperHistoryProvider(Protocol):
    def download_symbol_history(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
        interval: str,
    ) -> pd.DataFrame: ...


@runtime_checkable
class PaperBroker(Protocol):
    def get_calendar(
        self,
        *,
        start_date: date,
        end_date: date,
    ) -> list[dict[str, Any]]: ...

    def get_account(self) -> dict[str, Any]: ...

    def get_position(self, symbol: str) -> dict[str, Any] | None: ...

    def submit_fractional_day_market_order(
        self,
        *,
        symbol: str,
        qty: float,
        side: str,
        client_order_id: str,
        time_in_force: str = "day",
    ) -> dict[str, Any]: ...

    def submit_notional_day_market_order(
        self,
        *,
        symbol: str,
        notional: float,
        side: str,
        client_order_id: str,
        time_in_force: str = "day",
    ) -> dict[str, Any]: ...

    def get_order(self, order_id: str) -> dict[str, Any]: ...


@runtime_checkable
class PaperHistoryProviderFactory(Protocol):
    def __call__(self) -> PaperHistoryProvider: ...


@runtime_checkable
class PaperBrokerFactory(Protocol):
    def __call__(self) -> PaperBroker: ...


@runtime_checkable
class PaperTradeRepository(Protocol):
    def list_proposals(self) -> list[dict[str, Any]]: ...

    def get_latest_proposal(self) -> dict[str, Any] | None: ...

    def get_proposal(self, proposal_id: str) -> dict[str, Any] | None: ...

    def get_evidence(self, trade_date: str) -> dict[str, Any] | None: ...

    def get_submission(self, trade_date: str) -> dict[str, Any] | None: ...

    def save_evidence(self, evidence: dict[str, Any]) -> Path: ...

    def save_proposal(self, proposal: dict[str, Any]) -> Path: ...

    def save_approval(self, *, trade_date: str, approval: dict[str, Any]) -> Path: ...

    def save_submission(self, *, trade_date: str, submission: dict[str, Any]) -> Path: ...

    def save_order_status(self, *, trade_date: str, order_status: dict[str, Any]) -> Path: ...

    def proposal_path(self, proposal_id: str) -> Path: ...

    def trade_evidence_path(self, trade_date: str) -> Path: ...

    def trade_submission_path(self, trade_date: str) -> Path: ...

    def trade_order_status_path(self, trade_date: str) -> Path: ...

    def order_status_path_exists(self, trade_date: str) -> bool: ...

    def backup_submission_attempt_artifacts(
        self,
        *,
        trade_date: str,
        now: datetime | None = None,
    ) -> None: ...


@runtime_checkable
class PaperStatusRepository(Protocol):
    @property
    def status_path(self) -> Path: ...

    def read_status(self) -> dict[str, Any] | None: ...

    def write_status(self, payload: dict[str, Any]) -> Path: ...


@runtime_checkable
class PaperUnitOfWork(Protocol):
    @property
    def trades(self) -> PaperTradeRepository: ...

    @property
    def status(self) -> PaperStatusRepository: ...

    def commit(self) -> None: ...

    def rollback(self) -> None: ...

    def __enter__(self) -> PaperUnitOfWork: ...

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None: ...


@runtime_checkable
class PaperUnitOfWorkFactory(Protocol):
    def __call__(self) -> PaperUnitOfWork: ...


@runtime_checkable
class PaperDeploymentRegistry(Protocol):
    def record_deployment(
        self,
        context: PaperHostedExecutionContext,
    ) -> PaperDeploymentRecord: ...

    def record_phase_run(
        self,
        context: PaperHostedExecutionContext,
    ) -> PaperPhaseRunRecord: ...


@runtime_checkable
class PaperArtifactStore(Protocol):
    def write_trade_account_snapshot(
        self,
        *,
        trade_date: str,
        payload: dict[str, Any],
    ) -> Path: ...

    def write_trade_order_preview(
        self,
        *,
        trade_date: str,
        payload: dict[str, Any],
    ) -> Path: ...


@dataclass(slots=True, frozen=True)
class PaperDecisionRequest:
    now: datetime | None = None
    provider: PaperHistoryProvider | None = None
    broker: PaperBroker | None = None


@dataclass(slots=True, frozen=True)
class PaperApprovalRequest:
    proposal_id: str
    decision: str
    actor: str
    rationale: str | None = None
    provider: str | None = None
    model: str | None = None
    fallback_used: bool = False
    fallback_reason: str | None = None
    now: datetime | None = None


@dataclass(slots=True, frozen=True)
class PaperSubmissionRequest:
    now: datetime | None = None
    broker: PaperBroker | None = None
    retry_failed_submission: bool = False


@dataclass(slots=True, frozen=True)
class PaperReconciliationRequest:
    now: datetime | None = None
    broker: PaperBroker | None = None


@dataclass(slots=True, frozen=True)
class PaperApprovalEvaluationRequest:
    proposal: Mapping[str, Any]
    evidence: Mapping[str, Any]
    status: Mapping[str, Any] | None = None
    account_context: Mapping[str, Any] = field(default_factory=dict)


@dataclass(slots=True, frozen=True)
class PaperApprovalClientDecision:
    decision: str
    rationale: str
    provider: str
    model: str
    fallback_used: bool = False
    fallback_reason: str = ""


@runtime_checkable
class PaperApprovalClient(Protocol):
    def evaluate(
        self,
        request: PaperApprovalEvaluationRequest,
    ) -> PaperApprovalClientDecision: ...


@runtime_checkable
class PaperApprovalClientFactory(Protocol):
    def __call__(self) -> PaperApprovalClient: ...


@runtime_checkable
class PaperNotificationSink(Protocol):
    def notify_decision(
        self,
        *,
        outcome: str,
        status: Mapping[str, Any],
        proposal: Mapping[str, Any] | None = None,
        now: datetime | None = None,
    ) -> Path: ...

    def notify_approval(
        self,
        *,
        proposal: Mapping[str, Any],
        approval_record: Mapping[str, Any],
        now: datetime | None = None,
    ) -> Path: ...

    def notify_submission(
        self,
        *,
        outcome: str,
        status: Mapping[str, Any],
        proposal: Mapping[str, Any] | None = None,
        submission: Mapping[str, Any] | None = None,
        now: datetime | None = None,
    ) -> Path: ...

    def notify_error(
        self,
        *,
        loop_name: str,
        stage: str,
        exc: Exception,
        proposal_id: str = "",
        trade_date: str = "",
        now: datetime | None = None,
    ) -> Path: ...


@runtime_checkable
class PaperNotificationSinkFactory(Protocol):
    def __call__(self) -> PaperNotificationSink: ...


@dataclass(slots=True, frozen=True)
class PaperDecisionResult:
    proposal_id: str = ""
    proposal_path: str = ""
    evidence_path: str = ""
    status_path: str = ""
    status: dict[str, Any] = field(default_factory=dict)
    proposal: dict[str, Any] | None = None
    evidence: dict[str, Any] | None = None

    @classmethod
    def from_legacy(cls, payload: dict[str, Any]) -> PaperDecisionResult:
        return cls(
            proposal_id=_string_field(payload, "proposal_id"),
            proposal_path=_string_field(payload, "proposal_path"),
            evidence_path=_string_field(payload, "evidence_path"),
            status_path=_string_field(payload, "status_path"),
            status=_status_field(payload),
            proposal=_mapping_field(payload, "proposal"),
            evidence=_mapping_field(payload, "evidence"),
        )

    def as_legacy_payload(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "status_path": self.status_path,
            "status": dict(self.status),
        }
        if self.proposal_id:
            payload["proposal_id"] = self.proposal_id
        if self.proposal_path:
            payload["proposal_path"] = self.proposal_path
        if self.evidence_path:
            payload["evidence_path"] = self.evidence_path
        return payload


@dataclass(slots=True, frozen=True)
class PaperApprovalResult:
    proposal_id: str = ""
    proposal_path: str = ""
    approval_path: str = ""
    status_path: str = ""
    status: dict[str, Any] = field(default_factory=dict)
    proposal: dict[str, Any] | None = None
    approval: dict[str, Any] | None = None

    @classmethod
    def from_legacy(cls, payload: dict[str, Any]) -> PaperApprovalResult:
        return cls(
            proposal_id=_string_field(payload, "proposal_id"),
            proposal_path=_string_field(payload, "proposal_path"),
            approval_path=_string_field(payload, "approval_path"),
            status_path=_string_field(payload, "status_path"),
            status=_status_field(payload),
            proposal=_mapping_field(payload, "proposal"),
            approval=_mapping_field(payload, "approval"),
        )

    def as_legacy_payload(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "status_path": self.status_path,
            "status": dict(self.status),
        }
        if self.proposal_id:
            payload["proposal_id"] = self.proposal_id
        if self.proposal_path:
            payload["proposal_path"] = self.proposal_path
        if self.approval_path:
            payload["approval_path"] = self.approval_path
        return payload


@dataclass(slots=True, frozen=True)
class PaperSubmissionResult:
    proposal_id: str = ""
    submission_path: str = ""
    status_path: str = ""
    status: dict[str, Any] = field(default_factory=dict)
    submission: dict[str, Any] | None = None
    proposal: dict[str, Any] | None = None

    @classmethod
    def from_legacy(cls, payload: dict[str, Any]) -> PaperSubmissionResult:
        return cls(
            proposal_id=_string_field(payload, "proposal_id"),
            submission_path=_string_field(payload, "submission_path"),
            status_path=_string_field(payload, "status_path"),
            status=_status_field(payload),
            submission=_mapping_field(payload, "submission"),
            proposal=_mapping_field(payload, "proposal"),
        )

    def as_legacy_payload(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "status_path": self.status_path,
            "status": dict(self.status),
        }
        if self.proposal_id:
            payload["proposal_id"] = self.proposal_id
        if self.submission_path:
            payload["submission_path"] = self.submission_path
        return payload


@dataclass(slots=True, frozen=True)
class PaperReconciliationResult:
    proposal_id: str
    submission_path: str
    order_status_path: str
    order_status: str
    poll_status: str
    submission: dict[str, Any] | None = None

    @classmethod
    def from_legacy(cls, payload: dict[str, Any]) -> PaperReconciliationResult:
        return cls(
            proposal_id=_string_field(payload, "proposal_id"),
            submission_path=_string_field(payload, "submission_path"),
            order_status_path=_string_field(payload, "order_status_path"),
            order_status=_string_field(payload, "order_status"),
            poll_status=_string_field(payload, "poll_status"),
            submission=_mapping_field(payload, "submission"),
        )

    def as_legacy_payload(self) -> dict[str, Any]:
        return {
            "proposal_id": self.proposal_id,
            "submission_path": self.submission_path,
            "order_status_path": self.order_status_path,
            "order_status": self.order_status,
            "poll_status": self.poll_status,
        }
