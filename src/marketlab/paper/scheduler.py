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
from marketlab.paper.contracts import (
    PaperDecisionResult,
    PaperHostedExecutionContext,
    PaperHostedPhase,
    PaperNotificationSink,
    PaperReconciliationResult,
    PaperSubmissionResult,
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
    PaperStateStore,
    _clock_value,
    _local_now,
    _now_utc,
    reconcile_latest_submission_status,
    run_paper_decision,
    run_paper_submit,
)
from marketlab.paper.state import _load_paper_state

LOGGER = logging.getLogger(__name__)


def _scheduler_state_path(config: ExperimentConfig) -> Path:
    return config.paper_state_dir / "scheduler.json"


def _load_scheduler_state(config: ExperimentConfig) -> dict[str, Any]:
    return _load_paper_state(_scheduler_state_path(config), LOGGER)


def _save_scheduler_state(config: ExperimentConfig, payload: dict[str, Any]) -> Path:
    path = _scheduler_state_path(config)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return path


def _clear_scheduler_error_state(state: dict[str, Any]) -> None:
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


def _derive_hosted_phase_context(
    hosted_context: PaperHostedExecutionContext | None,
    *,
    phase: PaperHostedPhase,
    suffix: str,
) -> PaperHostedExecutionContext | None:
    if hosted_context is None:
        return None
    return hosted_context.derive(phase=phase, suffix=f"{phase}:{suffix}")


def _notify_scheduler_error(
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
        stage = "paper-scheduler"
        root_error = exc
        proposal_id = ""
        trade_date = ""

    fingerprint = build_error_fingerprint(
        loop_name="scheduler",
        stage=stage,
        exc=root_error,
        proposal_id=proposal_id,
        trade_date=trade_date,
    )
    state["last_checked_at"] = _now_utc(now).isoformat()
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
        loop_name="scheduler",
        stage=stage,
        exc=root_error,
        proposal_id=proposal_id,
        trade_date=trade_date,
        now=now,
    )


def run_scheduler_iteration(
    config: ExperimentConfig,
    *,
    now: datetime | None = None,
    notification_sink: PaperNotificationSink | None = None,
    execution_context: ExecutionContext | None = None,
    hosted_context: PaperHostedExecutionContext | None = None,
) -> dict[str, Any]:
    iteration_details = hosted_execution_details(
        hosted_context,
        {"component": "scheduler_iteration"},
    )
    iteration_context = paper_execution_context(
        (
            hosted_root_execution_context(
                hosted_context,
                phase="paper-scheduler",
                details=iteration_details,
            )
            if hosted_context is not None
            else execution_context
        ),
        phase="paper-scheduler",
        deployment=hosted_context.deployment_id if hosted_context is not None else "paper_scheduler",
        refresh_execution_id=hosted_context is None,
        details=iteration_details,
    )
    emit_structured_log(
        LOGGER,
        logging.INFO,
        "Starting paper scheduler iteration.",
        event="paper.scheduler.iteration.start",
        execution_context=iteration_context,
    )
    start_time = perf_counter()
    local_now = _local_now(config, now)
    market_date = local_now.date().isoformat()
    decision_clock = _clock_value(config.paper.decision_time)
    submission_clock = _clock_value(config.paper.submission_time)
    state = _load_scheduler_state(config)
    events: list[dict[str, Any]] = []
    try:
        with bind_execution_context(
            paper_execution_context(
                iteration_context,
                phase="paper-scheduler",
                deployment=hosted_context.deployment_id if hosted_context is not None else "paper_scheduler",
                trade_date=market_date,
                details=iteration_details,
            )
        ):
            if local_now.time() >= decision_clock and state.get("last_decision_market_date") != market_date:
                try:
                    phase_hosted_context = _derive_hosted_phase_context(
                        hosted_context,
                        phase="decision",
                        suffix=market_date,
                    )
                    result = run_paper_decision(
                        config,
                        now=now,
                        notification_sink=notification_sink,
                        hosted_context=phase_hosted_context,
                        execution_context=paper_execution_context(
                            iteration_context,
                            phase="paper-decision",
                            deployment=(
                                hosted_context.deployment_id
                                if hosted_context is not None
                                else "paper_scheduler"
                            ),
                            trade_date=market_date,
                            provider=config.paper.data_provider,
                            refresh_execution_id=phase_hosted_context is None,
                        ),
                    )
                    decision_result = PaperDecisionResult.from_legacy(result)
                except Exception as exc:
                    raise PaperLoopStageError(
                        loop_name="scheduler",
                        stage="paper-decision",
                        cause=exc,
                    ) from exc
                events.append({"phase": "decision", **decision_result.as_legacy_payload()})
                state["last_decision_market_date"] = market_date
                state["last_decision_at"] = _now_utc(now).isoformat()

            if local_now.time() >= submission_clock and state.get("last_submission_market_date") != market_date:
                try:
                    phase_hosted_context = _derive_hosted_phase_context(
                        hosted_context,
                        phase="submit",
                        suffix=market_date,
                    )
                    result = run_paper_submit(
                        config,
                        now=now,
                        notification_sink=notification_sink,
                        hosted_context=phase_hosted_context,
                        execution_context=paper_execution_context(
                            iteration_context,
                            phase="paper-submit",
                            deployment=(
                                hosted_context.deployment_id
                                if hosted_context is not None
                                else "paper_scheduler"
                            ),
                            trade_date=market_date,
                            provider=config.paper.broker,
                            refresh_execution_id=phase_hosted_context is None,
                        ),
                    )
                    submission_result = PaperSubmissionResult.from_legacy(result)
                except Exception as exc:
                    proposal = PaperStateStore(config).latest_proposal()
                    raise PaperLoopStageError(
                        loop_name="scheduler",
                        stage="paper-submit",
                        cause=exc,
                        proposal_id=str((proposal or {}).get("proposal_id", "")),
                        trade_date=str((proposal or {}).get("effective_date", "")),
                    ) from exc
                events.append({"phase": "submission", **submission_result.as_legacy_payload()})
                state["last_submission_market_date"] = market_date
                state["last_submission_at"] = _now_utc(now).isoformat()

            try:
                phase_hosted_context = _derive_hosted_phase_context(
                    hosted_context,
                    phase="reconcile",
                    suffix=market_date,
                )
                reconciliation = reconcile_latest_submission_status(
                    config,
                    now=now,
                    hosted_context=phase_hosted_context,
                    execution_context=paper_execution_context(
                        iteration_context,
                        phase="paper-submit-reconcile",
                        deployment=(
                            hosted_context.deployment_id
                            if hosted_context is not None
                            else "paper_scheduler"
                        ),
                        trade_date=market_date,
                        provider=config.paper.broker,
                        refresh_execution_id=phase_hosted_context is None,
                    ),
                )
            except Exception as exc:
                proposal = PaperStateStore(config).latest_proposal()
                raise PaperLoopStageError(
                    loop_name="scheduler",
                    stage="paper-submit-reconcile",
                    cause=exc,
                    proposal_id=str((proposal or {}).get("proposal_id", "")),
                    trade_date=str((proposal or {}).get("effective_date", "")),
                ) from exc
            if reconciliation is not None:
                reconciliation_result = PaperReconciliationResult.from_legacy(reconciliation)
                events.append({"phase": "submission_reconcile", **reconciliation_result.as_legacy_payload()})

            _clear_scheduler_error_state(state)
            state["last_checked_at"] = _now_utc(now).isoformat()
            state["last_checked_market_date"] = market_date
            state_path = _save_scheduler_state(config, state)
            summary = {
                "scheduler_state_path": str(state_path),
                "market_date": market_date,
                "events": events,
            }
    except Exception as exc:
        emit_structured_log(
            LOGGER,
            logging.ERROR,
            "Paper scheduler iteration failed.",
            event="paper.scheduler.iteration.error",
            execution_context=paper_execution_context(
                iteration_context,
                phase="paper-scheduler",
                deployment=hosted_context.deployment_id if hosted_context is not None else "paper_scheduler",
                trade_date=market_date,
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
        "Finished paper scheduler iteration.",
        event="paper.scheduler.iteration.finish",
        execution_context=paper_execution_context(
            iteration_context,
            phase="paper-scheduler",
            deployment=hosted_context.deployment_id if hosted_context is not None else "paper_scheduler",
            trade_date=market_date,
            outcome="events_recorded" if events else "idle",
            duration_ms=duration_ms_since(start_time),
            details=hosted_execution_details(
                hosted_context,
                {
                    "component": "scheduler_iteration",
                    "event_count": len(events),
                    "scheduler_state_path": summary["scheduler_state_path"],
                },
            ),
        ),
    )
    return summary


def run_scheduler_loop(
    config: ExperimentConfig,
    *,
    once: bool = False,
    notification_sink: PaperNotificationSink | None = None,
    hosted_context: PaperHostedExecutionContext | None = None,
) -> None:
    while True:
        loop_details = hosted_execution_details(
            hosted_context,
            {"component": "scheduler_loop", "once": once},
        )
        loop_context = paper_execution_context(
            (
                hosted_root_execution_context(
                    hosted_context,
                    phase="paper-scheduler",
                    details=loop_details,
                )
                if hosted_context is not None
                else None
            ),
            phase="paper-scheduler",
            deployment=hosted_context.deployment_id if hosted_context is not None else "paper_scheduler",
            refresh_execution_id=hosted_context is None,
            details=loop_details,
        )
        emit_structured_log(
            LOGGER,
            logging.INFO,
            "Starting paper scheduler loop run.",
            event="paper.scheduler.loop.start",
            execution_context=loop_context,
        )
        start_time = perf_counter()
        loop_error: Exception | None = None
        try:
            with bind_execution_context(loop_context):
                summary = run_scheduler_iteration(
                    config,
                    notification_sink=notification_sink,
                    execution_context=loop_context,
                    hosted_context=hosted_context,
                )
        except Exception as exc:
            loop_error = exc
            state = _load_scheduler_state(config)
            notification_path = _notify_scheduler_error(
                config,
                state=state,
                exc=exc,
                notification_sink=notification_sink,
            )
            state_path = _save_scheduler_state(config, state)
            summary = {
                "scheduler_state_path": str(state_path),
                "events": [],
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
                "Paper scheduler loop run failed.",
                event="paper.scheduler.loop.error",
                execution_context=paper_execution_context(
                    loop_context,
                    phase="paper-scheduler",
                    deployment=hosted_context.deployment_id if hosted_context is not None else "paper_scheduler",
                    outcome="error",
                    duration_ms=duration_ms_since(start_time),
                    details=hosted_execution_details(
                        hosted_context,
                        {
                            "component": "scheduler_loop",
                            "scheduler_state_path": summary["scheduler_state_path"],
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
                "Finished paper scheduler loop run.",
                event="paper.scheduler.loop.finish",
                execution_context=paper_execution_context(
                    loop_context,
                    phase="paper-scheduler",
                    deployment=hosted_context.deployment_id if hosted_context is not None else "paper_scheduler",
                    outcome="events_recorded" if summary["events"] else "idle",
                    duration_ms=duration_ms_since(start_time),
                    details=hosted_execution_details(
                        hosted_context,
                        {
                            "component": "scheduler_loop",
                            "scheduler_state_path": summary["scheduler_state_path"],
                            "event_count": len(summary["events"]),
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
