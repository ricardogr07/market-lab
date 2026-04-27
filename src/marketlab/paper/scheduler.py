from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from marketlab.config import ExperimentConfig
from marketlab.paper.contracts import (
    PaperDecisionResult,
    PaperReconciliationResult,
    PaperSubmissionResult,
)
from marketlab.paper.notifications import (
    PaperLoopStageError,
    TelegramTransport,
    build_error_fingerprint,
    build_error_message,
)
from marketlab.paper.service import (
    PaperStateStore,
    _clock_value,
    _local_now,
    _now_utc,
    _write_notification_record,
    reconcile_latest_submission_status,
    run_paper_decision,
    run_paper_submit,
)


def _scheduler_state_path(config: ExperimentConfig) -> Path:
    return config.paper_state_dir / "scheduler.json"


def _load_scheduler_state(config: ExperimentConfig) -> dict[str, Any]:
    path = _scheduler_state_path(config)
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


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


def _notify_scheduler_error(
    config: ExperimentConfig,
    *,
    state: dict[str, Any],
    exc: Exception,
    now: datetime | None = None,
    transport: TelegramTransport | None = None,
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
    store = PaperStateStore(config)
    return _write_notification_record(
        config,
        store,
        stage="paper-error",
        outcome="error",
        message=build_error_message(
            config,
            loop_name="scheduler",
            stage=stage,
            exc=root_error,
            proposal_id=proposal_id,
            trade_date=trade_date,
        ),
        details={
            "experiment_name": config.experiment_name,
            "loop": "scheduler",
            "failed_stage": stage,
            "proposal_id": proposal_id,
            "trade_date": trade_date,
            "exception_type": type(root_error).__name__,
            "exception_message": str(root_error),
        },
        proposal_id=proposal_id,
        trade_date=trade_date,
        now=now,
        transport=transport,
    )


def run_scheduler_iteration(
    config: ExperimentConfig,
    *,
    now: datetime | None = None,
    notification_transport: TelegramTransport | None = None,
) -> dict[str, Any]:
    local_now = _local_now(config, now)
    market_date = local_now.date().isoformat()
    decision_clock = _clock_value(config.paper.decision_time)
    submission_clock = _clock_value(config.paper.submission_time)
    state = _load_scheduler_state(config)
    events: list[dict[str, Any]] = []

    if local_now.time() >= decision_clock and state.get("last_decision_market_date") != market_date:
        try:
            result = run_paper_decision(
                config,
                now=now,
                notification_transport=notification_transport,
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
            result = run_paper_submit(
                config,
                now=now,
                notification_transport=notification_transport,
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
        reconciliation = reconcile_latest_submission_status(config, now=now)
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
    return {
        "scheduler_state_path": str(state_path),
        "market_date": market_date,
        "events": events,
    }


def run_scheduler_loop(
    config: ExperimentConfig,
    *,
    once: bool = False,
    notification_transport: TelegramTransport | None = None,
) -> None:
    while True:
        loop_error: Exception | None = None
        try:
            summary = run_scheduler_iteration(
                config,
                notification_transport=notification_transport,
            )
        except Exception as exc:
            loop_error = exc
            state = _load_scheduler_state(config)
            notification_path = _notify_scheduler_error(
                config,
                state=state,
                exc=exc,
                transport=notification_transport,
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
        print(json.dumps(summary, indent=2, sort_keys=True))
        if once:
            if loop_error is not None:
                raise loop_error
            return
        time.sleep(config.paper.poll_interval_seconds)
