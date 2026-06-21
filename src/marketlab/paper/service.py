from __future__ import annotations

import logging
from datetime import datetime
from time import perf_counter
from typing import Any

from marketlab.config import ExperimentConfig
from marketlab.log import (
    ExecutionContext,
    bind_execution_context,
    duration_ms_since,
    emit_structured_log,
)
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
    PaperDeploymentRegistry,
    PaperHistoryProvider,
    PaperHistoryProviderFactory,
    PaperHostedExecutionContext,
    PaperHostedPhase,
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
from marketlab.paper.observability import (
    hosted_execution_details,
    hosted_root_execution_context,
    paper_execution_context,
)
from marketlab.paper.persistence import (
    build_filesystem_paper_artifact_store,
    build_filesystem_paper_deployment_registry,
    build_filesystem_paper_uow_factory,
    build_sqlite_paper_deployment_registry,
    build_sqlite_paper_uow_factory,
)
from marketlab.paper.state import PaperStateStore

APPROVAL_PENDING = _APPROVAL_PENDING
SUBMISSION_SKIPPED = _SUBMISSION_SKIPPED
_clock_value = _core_clock_value
_local_now = _core_local_now
_now_utc = _core_now_utc
_paper_symbol = _core_paper_symbol
LOGGER = logging.getLogger(__name__)


def _paper_uow_factory(config: ExperimentConfig) -> PaperUnitOfWorkFactory:
    if config.paper.persistence_backend == "filesystem":
        return build_filesystem_paper_uow_factory(config)
    if config.paper.persistence_backend == "sqlite":
        return build_sqlite_paper_uow_factory(config)
    raise ValueError(f"Unsupported paper persistence backend: {config.paper.persistence_backend}")


def _paper_deployment_registry(config: ExperimentConfig) -> PaperDeploymentRegistry:
    if config.paper.persistence_backend == "filesystem":
        return build_filesystem_paper_deployment_registry(config)
    if config.paper.persistence_backend == "sqlite":
        return build_sqlite_paper_deployment_registry(config)
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


def _ensure_hosted_phase(
    config: ExperimentConfig,
    hosted_context: PaperHostedExecutionContext | None,
    *,
    expected_phase: PaperHostedPhase,
) -> None:
    if hosted_context is None:
        return
    if hosted_context.phase != expected_phase:
        raise ValueError(
            "Hosted paper execution metadata phase mismatch: "
            f"expected {expected_phase}, got {hosted_context.phase}."
        )
    _paper_deployment_registry(config).record_phase_run(hosted_context)


def _paper_phase_context(
    execution_context: ExecutionContext | None,
    hosted_context: PaperHostedExecutionContext | None,
    *,
    phase: str,
    provider: str | None = None,
    details: dict[str, Any] | None = None,
) -> ExecutionContext | None:
    if hosted_context is None:
        return execution_context
    return hosted_root_execution_context(
        hosted_context,
        phase=phase,
        provider=provider,
        details=details,
    )


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
    execution_context: ExecutionContext | None = None,
    hosted_context: PaperHostedExecutionContext | None = None,
) -> dict[str, Any]:
    _ensure_hosted_phase(config, hosted_context, expected_phase="decision")
    context_details = hosted_execution_details(hosted_context, {"component": "paper_service"})
    decision_context = paper_execution_context(
        _paper_phase_context(
            execution_context,
            hosted_context,
            phase="paper-decision",
            provider=config.paper.data_provider,
            details=context_details,
        ),
        phase="paper-decision",
        deployment=hosted_context.deployment_id if hosted_context is not None else None,
        provider=config.paper.data_provider,
        refresh_execution_id=hosted_context is None,
        details=context_details,
    )
    emit_structured_log(
        LOGGER,
        logging.INFO,
        "Starting paper decision service.",
        event="paper.decision.start",
        execution_context=decision_context,
    )
    start_time = perf_counter()
    try:
        with bind_execution_context(decision_context):
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
            notification_path = sink.notify_decision(
                outcome=_decision_notification_outcome(result.status),
                status=result.status,
                proposal=result.proposal,
                now=now,
            )
    except Exception as exc:
        emit_structured_log(
            LOGGER,
            logging.ERROR,
            "Paper decision service failed.",
            event="paper.decision.error",
            execution_context=paper_execution_context(
                decision_context,
                phase="paper-decision",
                provider=config.paper.data_provider,
                outcome="error",
                duration_ms=duration_ms_since(start_time),
                details=context_details,
            ),
            exc_info=exc,
        )
        raise
    emit_structured_log(
        LOGGER,
        logging.INFO,
        "Finished paper decision service.",
        event="paper.decision.finish",
        execution_context=paper_execution_context(
            decision_context,
            phase="paper-decision",
            status=result.status,
            proposal=result.proposal,
            provider=config.paper.data_provider,
            outcome=_decision_notification_outcome(result.status),
            duration_ms=duration_ms_since(start_time),
            details=hosted_execution_details(
                hosted_context,
                {
                    "component": "paper_service",
                    "status_path": result.status_path,
                    "proposal_path": result.proposal_path,
                    "evidence_path": result.evidence_path,
                    "notification_path": str(notification_path),
                },
            ),
        ),
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
    execution_context: ExecutionContext | None = None,
    hosted_context: PaperHostedExecutionContext | None = None,
) -> dict[str, Any]:
    _ensure_hosted_phase(config, hosted_context, expected_phase="agent_approve")
    context_details = hosted_execution_details(
        hosted_context,
        {"component": "paper_service", "actor": actor, "decision": decision},
    )
    approval_context = paper_execution_context(
        _paper_phase_context(
            execution_context,
            hosted_context,
            phase="paper-approve",
            provider=provider,
            details=context_details,
        ),
        phase="paper-approve",
        deployment=hosted_context.deployment_id if hosted_context is not None else None,
        proposal={"proposal_id": proposal_id},
        provider=provider,
        refresh_execution_id=hosted_context is None,
        details=context_details,
    )
    emit_structured_log(
        LOGGER,
        logging.INFO,
        "Starting paper approval service.",
        event="paper.approval.start",
        execution_context=approval_context,
    )
    start_time = perf_counter()
    notification_path: str | None = None
    try:
        with bind_execution_context(approval_context):
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
                notification_path = str(
                    sink.notify_approval(
                        proposal=result.proposal,
                        approval_record=result.approval,
                        now=now,
                    )
                )
    except Exception as exc:
        emit_structured_log(
            LOGGER,
            logging.ERROR,
            "Paper approval service failed.",
            event="paper.approval.error",
            execution_context=paper_execution_context(
                approval_context,
                phase="paper-approve",
                proposal={"proposal_id": proposal_id},
                provider=provider,
                outcome="error",
                duration_ms=duration_ms_since(start_time),
                details=context_details,
            ),
            exc_info=exc,
        )
        raise
    emit_structured_log(
        LOGGER,
        logging.INFO,
        "Finished paper approval service.",
        event="paper.approval.finish",
        execution_context=paper_execution_context(
            approval_context,
            phase="paper-approve",
            status=result.status,
            proposal=result.proposal,
            approval=result.approval,
            provider=provider,
            duration_ms=duration_ms_since(start_time),
            details=hosted_execution_details(
                hosted_context,
                {
                    "component": "paper_service",
                    "status_path": result.status_path,
                    "proposal_path": result.proposal_path,
                    "approval_path": result.approval_path,
                    "notification_path": notification_path,
                },
            ),
        ),
    )
    return result.as_legacy_payload()


def reconcile_latest_submission_status(
    config: ExperimentConfig,
    *,
    now: datetime | None = None,
    broker: PaperBroker | None = None,
    execution_context: ExecutionContext | None = None,
    hosted_context: PaperHostedExecutionContext | None = None,
) -> dict[str, Any] | None:
    _ensure_hosted_phase(config, hosted_context, expected_phase="reconcile")
    context_details = hosted_execution_details(hosted_context, {"component": "paper_service"})
    reconciliation_context = paper_execution_context(
        _paper_phase_context(
            execution_context,
            hosted_context,
            phase="paper-submit-reconcile",
            provider=config.paper.broker,
            details=context_details,
        ),
        phase="paper-submit-reconcile",
        deployment=hosted_context.deployment_id if hosted_context is not None else None,
        provider=config.paper.broker,
        refresh_execution_id=hosted_context is None,
        details=context_details,
    )
    emit_structured_log(
        LOGGER,
        logging.INFO,
        "Starting paper submission reconciliation service.",
        event="paper.reconcile.start",
        execution_context=reconciliation_context,
    )
    start_time = perf_counter()
    try:
        with bind_execution_context(reconciliation_context):
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
    except Exception as exc:
        emit_structured_log(
            LOGGER,
            logging.ERROR,
            "Paper submission reconciliation service failed.",
            event="paper.reconcile.error",
            execution_context=paper_execution_context(
                reconciliation_context,
                phase="paper-submit-reconcile",
                provider=config.paper.broker,
                outcome="error",
                duration_ms=duration_ms_since(start_time),
                details=context_details,
            ),
            exc_info=exc,
        )
        raise
    if result is None:
        emit_structured_log(
            LOGGER,
            logging.INFO,
            "Finished paper submission reconciliation service without updates.",
            event="paper.reconcile.finish",
            execution_context=paper_execution_context(
                reconciliation_context,
                phase="paper-submit-reconcile",
                provider=config.paper.broker,
                outcome="no_reconciliation_update",
                duration_ms=duration_ms_since(start_time),
                details=context_details,
            ),
        )
        return None
    emit_structured_log(
        LOGGER,
        logging.INFO,
        "Finished paper submission reconciliation service.",
        event="paper.reconcile.finish",
        execution_context=paper_execution_context(
            reconciliation_context,
            phase="paper-submit-reconcile",
            submission=result.submission,
            provider=config.paper.broker,
            outcome=result.poll_status or result.order_status,
            duration_ms=duration_ms_since(start_time),
            details=hosted_execution_details(
                hosted_context,
                {
                    "component": "paper_service",
                    "submission_path": result.submission_path,
                    "order_status_path": result.order_status_path,
                },
            ),
        ),
    )
    return result.as_legacy_payload()


def run_paper_submit(
    config: ExperimentConfig,
    *,
    now: datetime | None = None,
    broker: PaperBroker | None = None,
    notification_sink: PaperNotificationSink | None = None,
    retry_failed_submission: bool = False,
    execution_context: ExecutionContext | None = None,
    hosted_context: PaperHostedExecutionContext | None = None,
) -> dict[str, Any]:
    _ensure_hosted_phase(config, hosted_context, expected_phase="submit")
    context_details = hosted_execution_details(
        hosted_context,
        {"component": "paper_service", "retry_failed_submission": retry_failed_submission},
    )
    submission_context = paper_execution_context(
        _paper_phase_context(
            execution_context,
            hosted_context,
            phase="paper-submit",
            provider=config.paper.broker,
            details=context_details,
        ),
        phase="paper-submit",
        deployment=hosted_context.deployment_id if hosted_context is not None else None,
        provider=config.paper.broker,
        refresh_execution_id=hosted_context is None,
        details=context_details,
    )
    emit_structured_log(
        LOGGER,
        logging.INFO,
        "Starting paper submission service.",
        event="paper.submit.start",
        execution_context=submission_context,
    )
    start_time = perf_counter()
    try:
        with bind_execution_context(submission_context):
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
            notification_path = sink.notify_submission(
                outcome=str(result.status.get("status", "")),
                status=result.status,
                proposal=result.proposal,
                submission=result.submission,
                now=now,
            )
    except Exception as exc:
        emit_structured_log(
            LOGGER,
            logging.ERROR,
            "Paper submission service failed.",
            event="paper.submit.error",
            execution_context=paper_execution_context(
                submission_context,
                phase="paper-submit",
                provider=config.paper.broker,
                outcome="error",
                duration_ms=duration_ms_since(start_time),
                details=context_details,
            ),
            exc_info=exc,
        )
        raise
    emit_structured_log(
        LOGGER,
        logging.INFO,
        "Finished paper submission service.",
        event="paper.submit.finish",
        execution_context=paper_execution_context(
            submission_context,
            phase="paper-submit",
            status=result.status,
            proposal=result.proposal,
            submission=result.submission,
            provider=config.paper.broker,
            duration_ms=duration_ms_since(start_time),
            details=hosted_execution_details(
                hosted_context,
                {
                    "component": "paper_service",
                    "status_path": result.status_path,
                    "submission_path": result.submission_path,
                    "notification_path": str(notification_path),
                },
            ),
        ),
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
