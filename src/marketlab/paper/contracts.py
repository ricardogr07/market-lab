from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Any, runtime_checkable

import pandas as pd
from typing_extensions import Protocol


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
