from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
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
    ) -> dict[str, Any]: ...

    def submit_notional_day_market_order(
        self,
        *,
        symbol: str,
        notional: float,
        side: str,
        client_order_id: str,
    ) -> dict[str, Any]: ...

    def get_order(self, order_id: str) -> dict[str, Any]: ...


@dataclass(slots=True, frozen=True)
class PaperDecisionResult:
    proposal_id: str = ""
    proposal_path: str = ""
    evidence_path: str = ""
    status_path: str = ""
    status: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_legacy(cls, payload: dict[str, Any]) -> PaperDecisionResult:
        return cls(
            proposal_id=_string_field(payload, "proposal_id"),
            proposal_path=_string_field(payload, "proposal_path"),
            evidence_path=_string_field(payload, "evidence_path"),
            status_path=_string_field(payload, "status_path"),
            status=_status_field(payload),
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

    @classmethod
    def from_legacy(cls, payload: dict[str, Any]) -> PaperApprovalResult:
        return cls(
            proposal_id=_string_field(payload, "proposal_id"),
            proposal_path=_string_field(payload, "proposal_path"),
            approval_path=_string_field(payload, "approval_path"),
            status_path=_string_field(payload, "status_path"),
            status=_status_field(payload),
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

    @classmethod
    def from_legacy(cls, payload: dict[str, Any]) -> PaperSubmissionResult:
        return cls(
            proposal_id=_string_field(payload, "proposal_id"),
            submission_path=_string_field(payload, "submission_path"),
            status_path=_string_field(payload, "status_path"),
            status=_status_field(payload),
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

    @classmethod
    def from_legacy(cls, payload: dict[str, Any]) -> PaperReconciliationResult:
        return cls(
            proposal_id=_string_field(payload, "proposal_id"),
            submission_path=_string_field(payload, "submission_path"),
            order_status_path=_string_field(payload, "order_status_path"),
            order_status=_string_field(payload, "order_status"),
            poll_status=_string_field(payload, "poll_status"),
        )

    def as_legacy_payload(self) -> dict[str, Any]:
        return {
            "proposal_id": self.proposal_id,
            "submission_path": self.submission_path,
            "order_status_path": self.order_status_path,
            "order_status": self.order_status,
            "poll_status": self.poll_status,
        }
