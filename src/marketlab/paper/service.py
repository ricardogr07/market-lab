from __future__ import annotations

from datetime import datetime
from typing import Any

from marketlab.config import ExperimentConfig
from marketlab.paper.alpaca import AlpacaMarketDataProvider, AlpacaPaperBrokerClient
from marketlab.paper.application import (
    ApprovalService,
    DecisionService,
    ReconciliationService,
    SubmissionService,
)
from marketlab.paper.contracts import (
    PaperApprovalRequest,
    PaperArtifactStore,
    PaperBroker,
    PaperBrokerFactory,
    PaperDecisionRequest,
    PaperHistoryProvider,
    PaperHistoryProviderFactory,
    PaperNotificationSink,
    PaperReconciliationRequest,
    PaperSubmissionRequest,
    PaperUnitOfWorkFactory,
)
from marketlab.paper.core import (
    APPROVAL_PENDING as _APPROVAL_PENDING,
)
from marketlab.paper.core import (
    SUBMISSION_SKIPPED as _SUBMISSION_SKIPPED,
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
from marketlab.paper.notifications import build_telegram_paper_notification_sink
from marketlab.paper.persistence import (
    build_filesystem_paper_artifact_store,
    build_filesystem_paper_uow_factory,
    build_sqlite_paper_uow_factory,
)
from marketlab.paper.state import PaperStateStore

APPROVAL_PENDING = _APPROVAL_PENDING
SUBMISSION_SKIPPED = _SUBMISSION_SKIPPED
_clock_value = _core_clock_value
_local_now = _core_local_now
_now_utc = _core_now_utc
_paper_symbol = _core_paper_symbol


def _paper_uow_factory(config: ExperimentConfig) -> PaperUnitOfWorkFactory:
    if config.paper.persistence_backend == "filesystem":
        return build_filesystem_paper_uow_factory(config)
    if config.paper.persistence_backend == "sqlite":
        return build_sqlite_paper_uow_factory(config)
    raise ValueError(f"Unsupported paper persistence backend: {config.paper.persistence_backend}")


def _paper_history_provider_factory(config: ExperimentConfig) -> PaperHistoryProviderFactory:
    if config.paper.data_provider != "alpaca":
        raise ValueError(f"Unsupported paper data provider: {config.paper.data_provider}")
    return AlpacaMarketDataProvider


def _paper_broker_factory(config: ExperimentConfig) -> PaperBrokerFactory:
    if config.paper.broker != "alpaca":
        raise ValueError(f"Unsupported paper broker: {config.paper.broker}")
    return AlpacaPaperBrokerClient


def _paper_artifact_store(config: ExperimentConfig) -> PaperArtifactStore:
    return build_filesystem_paper_artifact_store(config)


def _paper_notification_sink(config: ExperimentConfig) -> PaperNotificationSink:
    return build_telegram_paper_notification_sink(config)


def _paper_state_store(config: ExperimentConfig) -> PaperStateStore:
    return PaperStateStore(config)


def _decision_notification_outcome(status: dict[str, Any]) -> str:
    outcome = str(status.get("status", ""))
    if outcome == SUBMISSION_SKIPPED:
        reason = str(status.get("reason", "")).strip()
        if reason != "":
            return reason
    return outcome


def run_paper_decision(
    config: ExperimentConfig,
    *,
    now: datetime | None = None,
    provider: PaperHistoryProvider | None = None,
    broker: PaperBroker | None = None,
    notification_sink: PaperNotificationSink | None = None,
) -> dict[str, Any]:
    result = DecisionService(
        config,
        uow_factory=_paper_uow_factory(config),
        history_provider_factory=_paper_history_provider_factory(config),
        broker_factory=_paper_broker_factory(config),
    ).run(
        PaperDecisionRequest(
            now=now,
            provider=provider,
            broker=broker,
        )
    )
    sink = notification_sink or _paper_notification_sink(config)
    sink.notify_decision(
        outcome=_decision_notification_outcome(result.status),
        status=result.status,
        proposal=result.proposal,
        now=now,
    )
    return result.as_legacy_payload()


def list_paper_proposals(config: ExperimentConfig) -> list[dict[str, Any]]:
    validate_paper_trading_config(config)
    with _paper_uow_factory(config)() as uow:
        return uow.trades.list_proposals()


def read_paper_proposal(
    config: ExperimentConfig,
    *,
    proposal_id: str,
) -> dict[str, Any]:
    validate_paper_trading_config(config)
    with _paper_uow_factory(config)() as uow:
        proposal = uow.trades.get_proposal(proposal_id)
    if proposal is None:
        raise FileNotFoundError(f"Unknown proposal_id: {proposal_id}")
    return proposal


def read_paper_evidence(
    config: ExperimentConfig,
    *,
    proposal_id: str,
) -> dict[str, Any]:
    validate_paper_trading_config(config)
    with _paper_uow_factory(config)() as uow:
        proposal = uow.trades.get_proposal(proposal_id)
        if proposal is None:
            raise FileNotFoundError(f"Unknown proposal_id: {proposal_id}")
        evidence = uow.trades.get_evidence(str(proposal["effective_date"]))
    if evidence is None:
        raise FileNotFoundError(f"Missing evidence for proposal_id: {proposal_id}")
    return evidence


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
    notification_sink: PaperNotificationSink | None = None,
) -> dict[str, Any]:
    result = ApprovalService(config, uow_factory=_paper_uow_factory(config)).run(
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
        )
    )
    if result.proposal is not None and result.approval is not None:
        sink = notification_sink or _paper_notification_sink(config)
        sink.notify_approval(
            proposal=result.proposal,
            approval_record=result.approval,
            now=now,
        )
    return result.as_legacy_payload()


def reconcile_latest_submission_status(
    config: ExperimentConfig,
    *,
    now: datetime | None = None,
    broker: PaperBroker | None = None,
) -> dict[str, Any] | None:
    result = ReconciliationService(
        config,
        uow_factory=_paper_uow_factory(config),
        broker_factory=_paper_broker_factory(config),
    ).run(
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
    notification_sink: PaperNotificationSink | None = None,
    retry_failed_submission: bool = False,
) -> dict[str, Any]:
    result = SubmissionService(
        config,
        uow_factory=_paper_uow_factory(config),
        artifact_store=_paper_artifact_store(config),
        broker_factory=_paper_broker_factory(config),
    ).run(
        PaperSubmissionRequest(
            now=now,
            broker=broker,
            retry_failed_submission=retry_failed_submission,
        )
    )
    sink = notification_sink or _paper_notification_sink(config)
    sink.notify_submission(
        outcome=str(result.status.get("status", "")),
        status=result.status,
        proposal=result.proposal,
        submission=result.submission,
        now=now,
    )
    legacy = result.as_legacy_payload()
    legacy.pop("proposal_id", None)
    if result.submission is not None:
        legacy["submission"] = result.submission
    return legacy


def get_paper_status(config: ExperimentConfig) -> dict[str, Any]:
    validate_paper_trading_config(config)
    with _paper_uow_factory(config)() as uow:
        latest_proposal = uow.trades.get_latest_proposal()
        proposals = uow.trades.list_proposals()
        status = uow.status.read_status()
        status_path = uow.status.status_path
    pending_proposals = [
        proposal
        for proposal in proposals
        if proposal.get("approval_status", APPROVAL_PENDING) == APPROVAL_PENDING
    ]
    return {
        "status_path": str(status_path),
        "status": status,
        "latest_proposal": latest_proposal,
        "pending_proposal_count": len(pending_proposals),
    }
