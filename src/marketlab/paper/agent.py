from __future__ import annotations

import json
import logging
import time
from datetime import datetime
from pathlib import Path
from time import perf_counter
from typing import Any

from marketlab.config import ExperimentConfig
from marketlab.log import (
    ExecutionContext,
    bind_execution_context,
    duration_ms_since,
    emit_structured_log,
)
from marketlab.paper.approval_clients import build_default_paper_approval_client
from marketlab.paper.contracts import (
    PaperApprovalClient,
    PaperApprovalEvaluationRequest,
    PaperApprovalResult,
    PaperBroker,
    PaperHostedExecutionContext,
    PaperNotificationSink,
)
from marketlab.paper.notifications import (
    PaperLoopStageError,
    build_error_fingerprint,
    build_telegram_paper_notification_sink,
)
from marketlab.paper.observability import (
    hosted_execution_details,
    hosted_root_execution_context,
    paper_execution_context,
)
from marketlab.paper.service import (
    APPROVAL_PENDING,
    _now_utc,
    _paper_broker_factory,
    decide_paper_proposal,
    read_paper_evidence,
    validate_paper_trading_config,
)
from marketlab.paper.state import PaperStateStore, _load_paper_state

LOGGER = logging.getLogger(__name__)


def _json_dump(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return path


def _worker_state_path(config: ExperimentConfig) -> Path:
    return config.paper_state_dir / "agent_worker.json"


def _load_worker_state(config: ExperimentConfig) -> dict[str, Any]:
    return _load_paper_state(_worker_state_path(config), LOGGER)


def _save_worker_state(config: ExperimentConfig, payload: dict[str, Any]) -> Path:
    return _json_dump(_worker_state_path(config), payload)


def _clear_worker_error_state(state: dict[str, Any]) -> None:
    for key in (
        "last_error_fingerprint",
        "last_error_stage",
        "last_error_type",
        "last_error_message",
        "last_error_proposal_id",
        "last_error_trade_date",
        "last_error_alert_at",
    ):
        state.pop(key, None)


def _paper_notification_sink(config: ExperimentConfig) -> PaperNotificationSink:
    return build_telegram_paper_notification_sink(config)


def _paper_approval_client(config: ExperimentConfig) -> PaperApprovalClient:
    return build_default_paper_approval_client(config)


def _derive_agent_approval_context(
    hosted_context: PaperHostedExecutionContext | None,
    *,
    proposal_id: str,
) -> PaperHostedExecutionContext | None:
    if hosted_context is None:
        return None
    return hosted_context.derive(
        phase="agent_approve",
        suffix=f"agent_approve:{proposal_id}",
    )


def _notify_worker_error(
    config: ExperimentConfig,
    *,
    state: dict[str, Any],
    exc: Exception,
    now: datetime | None = None,
    notification_sink: PaperNotificationSink | None = None,
) -> Path | None:
    if isinstance(exc, PaperLoopStageError):
        stage = exc.stage
        root_error = exc.cause
        proposal_id = exc.proposal_id
        trade_date = exc.trade_date
    else:
        stage = "paper-approve"
        root_error = exc
        proposal_id = ""
        trade_date = ""

    fingerprint = build_error_fingerprint(
        loop_name="agent",
        stage=stage,
        exc=root_error,
        proposal_id=proposal_id,
        trade_date=trade_date,
    )
    state["last_checked_at"] = _now_utc(now).isoformat()
    state["last_result"] = "error"
    if state.get("last_error_fingerprint") == fingerprint:
        return None

    state["last_error_fingerprint"] = fingerprint
    state["last_error_stage"] = stage
    state["last_error_type"] = type(root_error).__name__
    state["last_error_message"] = str(root_error)
    state["last_error_proposal_id"] = proposal_id
    state["last_error_trade_date"] = trade_date
    state["last_error_alert_at"] = _now_utc(now).isoformat()
    sink = notification_sink or _paper_notification_sink(config)
    return sink.notify_error(
        loop_name="agent",
        stage=stage,
        exc=root_error,
        proposal_id=proposal_id,
        trade_date=trade_date,
        now=now,
    )


def _current_account_context(
    config: ExperimentConfig,
    *,
    broker: PaperBroker | None = None,
) -> dict[str, Any]:
    symbol = str(config.data.symbols[0])
    client = broker if broker is not None else _paper_broker_factory(config)()
    account = client.get_account()
    position = client.get_position(symbol)
    return {
        "account": account,
        "position": position,
    }


def run_agent_approval_iteration(
    config: ExperimentConfig,
    *,
    now: datetime | None = None,
    broker: PaperBroker | None = None,
    notification_sink: PaperNotificationSink | None = None,
    approval_client: PaperApprovalClient | None = None,
    execution_context: ExecutionContext | None = None,
    hosted_context: PaperHostedExecutionContext | None = None,
) -> dict[str, Any]:
    iteration_details = hosted_execution_details(
        hosted_context,
        {"component": "agent_iteration"},
    )
    iteration_context = paper_execution_context(
        (
            hosted_root_execution_context(
                hosted_context,
                phase="paper-approve",
                details=iteration_details,
            )
            if hosted_context is not None
            else execution_context
        ),
        phase="paper-approve",
        deployment=hosted_context.deployment_id if hosted_context is not None else "paper_agent",
        refresh_execution_id=hosted_context is None,
        details=iteration_details,
    )
    emit_structured_log(
        LOGGER,
        logging.INFO,
        "Starting paper approval worker iteration.",
        event="paper.agent.iteration.start",
        execution_context=iteration_context,
    )
    start_time = perf_counter()
    validate_paper_trading_config(config)
    state = _load_worker_state(config)
    events: list[dict[str, Any]] = []
    summary: dict[str, Any]
    try:
        with bind_execution_context(
            paper_execution_context(
                iteration_context,
                phase="paper-approve",
                deployment=hosted_context.deployment_id if hosted_context is not None else "paper_agent",
                details=iteration_details,
            )
        ):
            if config.paper.execution_mode != "agent_approval":
                _clear_worker_error_state(state)
                state["last_checked_at"] = _now_utc(now).isoformat()
                state["last_result"] = "execution_mode_not_agent_approval"
                state_path = _save_worker_state(config, state)
                summary = {
                    "agent_state_path": str(state_path),
                    "events": [],
                    "processed_count": 0,
                }
            else:
                store = PaperStateStore(config)
                proposals = sorted(
                    [
                        proposal
                        for proposal in store.list_proposals()
                        if proposal.get("approval_status", APPROVAL_PENDING) == APPROVAL_PENDING
                        and not store.trade_submission_path(proposal["effective_date"]).exists()
                    ],
                    key=lambda proposal: (
                        proposal.get("effective_date", ""),
                        proposal.get("proposal_id", ""),
                    ),
                )
                current_status = store.read_status()
                account_context = _current_account_context(config, broker=broker) if proposals else {}
                client = approval_client or _paper_approval_client(config)

                for proposal in proposals:
                    try:
                        evidence = read_paper_evidence(config, proposal_id=proposal["proposal_id"])
                    except FileNotFoundError as exc:
                        result = decide_paper_proposal(
                            config,
                            proposal_id=proposal["proposal_id"],
                            decision="reject",
                            actor="agent",
                            rationale=(
                                "Rejected because the approval worker could not read the persisted "
                                f"proposal evidence: {exc}."
                            ),
                            fallback_reason=str(exc),
                            now=now,
                            notification_sink=notification_sink,
                            hosted_context=_derive_agent_approval_context(
                                hosted_context,
                                proposal_id=str(proposal["proposal_id"]),
                            ),
                            execution_context=paper_execution_context(
                                iteration_context,
                                phase="paper-approve",
                                deployment=(
                                    hosted_context.deployment_id
                                    if hosted_context is not None
                                    else "paper_agent"
                                ),
                                proposal=proposal,
                                refresh_execution_id=hosted_context is None,
                            ),
                        )
                        approval_result = PaperApprovalResult.from_legacy(result)
                        events.append(
                            {
                                "proposal_id": proposal["proposal_id"],
                                "decision": "reject",
                                "provider": "",
                                "model": "",
                                "fallback_used": False,
                                "fallback_reason": str(exc),
                                "approval_path": approval_result.approval_path,
                            }
                        )
                        continue
                    try:
                        decision = client.evaluate(
                            PaperApprovalEvaluationRequest(
                                proposal=proposal,
                                evidence=evidence,
                                status=current_status,
                                account_context=account_context,
                            )
                        )
                        result = decide_paper_proposal(
                            config,
                            proposal_id=proposal["proposal_id"],
                            decision=decision.decision,
                            actor="agent",
                            rationale=decision.rationale,
                            provider=decision.provider,
                            model=decision.model,
                            fallback_used=decision.fallback_used,
                            fallback_reason=decision.fallback_reason,
                            now=now,
                            notification_sink=notification_sink,
                            hosted_context=_derive_agent_approval_context(
                                hosted_context,
                                proposal_id=str(proposal["proposal_id"]),
                            ),
                            execution_context=paper_execution_context(
                                iteration_context,
                                phase="paper-approve",
                                deployment=(
                                    hosted_context.deployment_id
                                    if hosted_context is not None
                                    else "paper_agent"
                                ),
                                proposal=proposal,
                                provider=decision.provider,
                                refresh_execution_id=hosted_context is None,
                            ),
                        )
                        approval_result = PaperApprovalResult.from_legacy(result)
                    except Exception as exc:
                        raise PaperLoopStageError(
                            loop_name="agent",
                            stage="paper-approve",
                            cause=exc,
                            proposal_id=str(proposal["proposal_id"]),
                            trade_date=str(proposal["effective_date"]),
                        ) from exc
                    events.append(
                        {
                            "proposal_id": proposal["proposal_id"],
                            "decision": decision.decision,
                            "provider": decision.provider,
                            "model": decision.model,
                            "fallback_used": decision.fallback_used,
                            "fallback_reason": decision.fallback_reason,
                            "approval_path": approval_result.approval_path,
                        }
                    )

                _clear_worker_error_state(state)
                state["last_checked_at"] = _now_utc(now).isoformat()
                state["last_processed_count"] = len(events)
                state["last_result"] = "processed" if events else "no_pending_proposals"
                state_path = _save_worker_state(config, state)
                summary = {
                    "agent_state_path": str(state_path),
                    "events": events,
                    "processed_count": len(events),
                }
    except Exception as exc:
        emit_structured_log(
            LOGGER,
            logging.ERROR,
            "Paper approval worker iteration failed.",
            event="paper.agent.iteration.error",
            execution_context=paper_execution_context(
                iteration_context,
                phase="paper-approve",
                deployment=hosted_context.deployment_id if hosted_context is not None else "paper_agent",
                outcome="error",
                duration_ms=duration_ms_since(start_time),
                details=iteration_details,
            ),
            exc_info=exc,
        )
        raise
    emit_structured_log(
        LOGGER,
        logging.INFO,
        "Finished paper approval worker iteration.",
        event="paper.agent.iteration.finish",
        execution_context=paper_execution_context(
            iteration_context,
            phase="paper-approve",
            deployment=hosted_context.deployment_id if hosted_context is not None else "paper_agent",
            outcome="processed" if summary["processed_count"] > 0 else state.get("last_result", "idle"),
            duration_ms=duration_ms_since(start_time),
            details=hosted_execution_details(
                hosted_context,
                {
                    "component": "agent_iteration",
                    "agent_state_path": summary["agent_state_path"],
                    "processed_count": summary["processed_count"],
                },
            ),
        ),
    )
    return summary


def run_agent_approval_loop(
    config: ExperimentConfig,
    *,
    once: bool = False,
    notification_sink: PaperNotificationSink | None = None,
    approval_client: PaperApprovalClient | None = None,
    hosted_context: PaperHostedExecutionContext | None = None,
) -> None:
    while True:
        loop_details = hosted_execution_details(
            hosted_context,
            {"component": "agent_loop", "once": once},
        )
        loop_context = paper_execution_context(
            (
                hosted_root_execution_context(
                    hosted_context,
                    phase="paper-approve",
                    details=loop_details,
                )
                if hosted_context is not None
                else None
            ),
            phase="paper-approve",
            deployment=hosted_context.deployment_id if hosted_context is not None else "paper_agent",
            refresh_execution_id=hosted_context is None,
            details=loop_details,
        )
        emit_structured_log(
            LOGGER,
            logging.INFO,
            "Starting paper approval worker loop run.",
            event="paper.agent.loop.start",
            execution_context=loop_context,
        )
        start_time = perf_counter()
        loop_error: Exception | None = None
        try:
            with bind_execution_context(loop_context):
                summary = run_agent_approval_iteration(
                    config,
                    notification_sink=notification_sink,
                    approval_client=approval_client,
                    execution_context=loop_context,
                    hosted_context=hosted_context,
                )
        except Exception as exc:
            loop_error = exc
            state = _load_worker_state(config)
            notification_path = _notify_worker_error(
                config,
                state=state,
                exc=exc,
                notification_sink=notification_sink,
            )
            state_path = _save_worker_state(config, state)
            summary = {
                "agent_state_path": str(state_path),
                "events": [],
                "processed_count": 0,
                "error": {
                    "type": type(exc).__name__,
                    "message": str(exc),
                    "notification_path": str(notification_path) if notification_path else "",
                    "duplicate_suppressed": notification_path is None,
                },
            }
            emit_structured_log(
                LOGGER,
                logging.ERROR,
                "Paper approval worker loop run failed.",
                event="paper.agent.loop.error",
                execution_context=paper_execution_context(
                    loop_context,
                    phase="paper-approve",
                    deployment=hosted_context.deployment_id if hosted_context is not None else "paper_agent",
                    outcome="error",
                    duration_ms=duration_ms_since(start_time),
                    details=hosted_execution_details(
                        hosted_context,
                        {
                            "component": "agent_loop",
                            "agent_state_path": summary["agent_state_path"],
                            "duplicate_suppressed": notification_path is None,
                        },
                    ),
                ),
                exc_info=exc,
            )
        else:
            emit_structured_log(
                LOGGER,
                logging.INFO,
                "Finished paper approval worker loop run.",
                event="paper.agent.loop.finish",
                execution_context=paper_execution_context(
                    loop_context,
                    phase="paper-approve",
                    deployment=hosted_context.deployment_id if hosted_context is not None else "paper_agent",
                    outcome="processed" if summary["processed_count"] > 0 else "idle",
                    duration_ms=duration_ms_since(start_time),
                    details=hosted_execution_details(
                        hosted_context,
                        {
                            "component": "agent_loop",
                            "agent_state_path": summary["agent_state_path"],
                            "processed_count": summary["processed_count"],
                        },
                    ),
                ),
            )
        print(json.dumps(summary, indent=2, sort_keys=True))
        if once:
            if loop_error is not None:
                raise loop_error
            return
        time.sleep(config.paper.poll_interval_seconds)
