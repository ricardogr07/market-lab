from __future__ import annotations

from datetime import datetime
from typing import Any

from marketlab.config import ExperimentConfig
from marketlab.paper.application import (
    ApprovalService,
    DecisionService,
    ReconciliationService,
    SubmissionService,
)
from marketlab.paper.contracts import (
    PaperApprovalRequest,
    PaperBroker,
    PaperDecisionRequest,
    PaperHistoryProvider,
    PaperReconciliationRequest,
    PaperSubmissionRequest,
)
from marketlab.paper.core import (
    APPROVAL_PENDING as _APPROVAL_PENDING,
)
from marketlab.paper.core import (
    _clock_value as _core_clock_value,
)
from marketlab.paper.core import (
    _local_now as _core_local_now,
)
from marketlab.paper.core import (
    _now_utc as _core_now_utc,
)
from marketlab.paper.core import (
    _paper_symbol as _core_paper_symbol,
)
from marketlab.paper.core import (
    validate_paper_trading_config,
)
from marketlab.paper.notifications import TelegramTransport, write_notification_record
from marketlab.paper.state import PaperStateStore

APPROVAL_PENDING = _APPROVAL_PENDING
_clock_value = _core_clock_value
_local_now = _core_local_now
_now_utc = _core_now_utc
_paper_symbol = _core_paper_symbol
_write_notification_record = write_notification_record


def run_paper_decision(
    config: ExperimentConfig,
    *,
    now: datetime | None = None,
    provider: PaperHistoryProvider | None = None,
    broker: PaperBroker | None = None,
    notification_transport: TelegramTransport | None = None,
) -> dict[str, Any]:
    result = DecisionService(config).run(
        PaperDecisionRequest(
            now=now,
            provider=provider,
            broker=broker,
            notification_transport=notification_transport,
        )
    )
    return result.as_legacy_payload()


def list_paper_proposals(config: ExperimentConfig) -> list[dict[str, Any]]:
    validate_paper_trading_config(config)
    return PaperStateStore(config).list_proposals()


def read_paper_proposal(
    config: ExperimentConfig,
    *,
    proposal_id: str,
) -> dict[str, Any]:
    validate_paper_trading_config(config)
    return PaperStateStore(config).load_proposal(proposal_id)


def read_paper_evidence(
    config: ExperimentConfig,
    *,
    proposal_id: str,
) -> dict[str, Any]:
    validate_paper_trading_config(config)
    store = PaperStateStore(config)
    proposal = store.load_proposal(proposal_id)
    return store.load_evidence(proposal["effective_date"])


def decide_paper_proposal(
    config: ExperimentConfig,
    *,
    proposal_id: str,
    decision: str,
    actor: str,
    rationale: str | None = None,
    provider: str | None = None,
    model: str | None = None,
    fallback_used: bool = False,
    fallback_reason: str | None = None,
    now: datetime | None = None,
    notification_transport: TelegramTransport | None = None,
) -> dict[str, Any]:
    result = ApprovalService(config).run(
        PaperApprovalRequest(
            proposal_id=proposal_id,
            decision=decision,
            actor=actor,
            rationale=rationale,
            provider=provider,
            model=model,
            fallback_used=fallback_used,
            fallback_reason=fallback_reason,
            now=now,
            notification_transport=notification_transport,
        )
    )
    return result.as_legacy_payload()


def reconcile_latest_submission_status(
    config: ExperimentConfig,
    *,
    now: datetime | None = None,
    broker: PaperBroker | None = None,
) -> dict[str, Any] | None:
    result = ReconciliationService(config).run(
        PaperReconciliationRequest(
            now=now,
            broker=broker,
        )
    )
    if result is None:
        return None
    return result.as_legacy_payload()


def run_paper_submit(
    config: ExperimentConfig,
    *,
    now: datetime | None = None,
    broker: PaperBroker | None = None,
    notification_transport: TelegramTransport | None = None,
    retry_failed_submission: bool = False,
) -> dict[str, Any]:
    result = SubmissionService(config).run(
        PaperSubmissionRequest(
            now=now,
            broker=broker,
            notification_transport=notification_transport,
            retry_failed_submission=retry_failed_submission,
        )
    )
    legacy = result.as_legacy_payload()
    legacy.pop("proposal_id", None)
    if result.submission is not None:
        legacy["submission"] = result.submission
    return legacy


def get_paper_status(config: ExperimentConfig) -> dict[str, Any]:
    validate_paper_trading_config(config)
    store = PaperStateStore(config)
    latest_proposal = store.latest_proposal()
    proposals = store.list_proposals()
    pending_proposals = [
        proposal
        for proposal in proposals
        if proposal.get("approval_status", APPROVAL_PENDING) == APPROVAL_PENDING
    ]
    return {
        "status_path": str(store.status_path),
        "status": store.read_status(),
        "latest_proposal": latest_proposal,
        "pending_proposal_count": len(pending_proposals),
    }
