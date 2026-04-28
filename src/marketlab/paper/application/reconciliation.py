from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

from marketlab.config import ExperimentConfig
from marketlab.paper.alpaca import AlpacaPaperBrokerClient
from marketlab.paper.contracts import (
    PaperBroker,
    PaperReconciliationRequest,
    PaperReconciliationResult,
)
from marketlab.paper.core import (
    SUBMISSION_SUBMITTED,
    TERMINAL_ORDER_STATUSES,
    _now_utc,
    validate_paper_trading_config,
)
from marketlab.paper.state import PaperStateStore, _json_dump, _json_load


def _poll_order_status(
    *,
    broker_client: PaperBroker,
    order_id: str,
    fallback_status: str,
    client_order_id: str,
) -> tuple[dict[str, Any], str]:
    try:
        order_status = broker_client.get_order(order_id)
        poll_status = "observed"
    except RuntimeError as exc:
        order_status = {
            "id": order_id,
            "client_order_id": client_order_id,
            "status": fallback_status,
            "poll_error": str(exc),
        }
        poll_status = "timeout"
    return order_status, poll_status


def _latest_submitted_proposal_requiring_reconciliation(
    store: PaperStateStore,
) -> tuple[dict[str, Any], dict[str, Any], Path] | None:
    for proposal in store.list_proposals():
        trade_date = str(proposal.get("effective_date", ""))
        if trade_date == "":
            continue
        submission_path = store.trade_submission_path(trade_date)
        if not submission_path.exists():
            continue
        submission = _json_load(submission_path)
        if submission.get("status") != SUBMISSION_SUBMITTED:
            continue
        order_status = str(submission.get("order_status", "")).lower()
        if order_status in TERMINAL_ORDER_STATUSES:
            continue
        return proposal, submission, submission_path
    return None


def _refresh_submission_order_status(
    store: PaperStateStore,
    *,
    proposal: dict[str, Any],
    submission: dict[str, Any],
    broker_client: PaperBroker,
    now: datetime | None = None,
) -> dict[str, Any] | None:
    if submission.get("status") != SUBMISSION_SUBMITTED:
        return None

    order_id = str(submission.get("order_id", "")).strip()
    if order_id == "":
        return None

    current_order_status = str(submission.get("order_status", "")).lower()
    if current_order_status in TERMINAL_ORDER_STATUSES:
        return None

    order_status, poll_status = _poll_order_status(
        broker_client=broker_client,
        order_id=order_id,
        fallback_status=current_order_status or "unknown",
        client_order_id=str(submission.get("client_order_id", "")),
    )
    refreshed_order_status = str(order_status.get("status", current_order_status or "unknown")).lower()
    current_poll_status = str(submission.get("poll_status", "")).lower()
    trade_date = str(submission["trade_date"])
    order_status_path = store.trade_order_status_path(trade_date)
    if (
        refreshed_order_status == current_order_status
        and poll_status == current_poll_status
        and order_status_path.exists()
    ):
        return None

    _json_dump(order_status_path, order_status)
    refreshed_submission = dict(submission)
    refreshed_submission["order_status"] = refreshed_order_status
    refreshed_submission["poll_status"] = poll_status
    refreshed_submission["order_status_path"] = str(order_status_path)
    refreshed_submission["updated_at"] = _now_utc(now).isoformat()
    submission_path = store.trade_submission_path(trade_date)
    _json_dump(submission_path, refreshed_submission)
    status = {
        "event": "paper-submit",
        "status": refreshed_submission["status"],
        "proposal_id": refreshed_submission["proposal_id"],
        "submission_path": str(submission_path),
        "order_status": refreshed_order_status,
        "updated_at": _now_utc(now).isoformat(),
    }
    store.write_status(status)
    return refreshed_submission


class ReconciliationService:
    def __init__(self, config: ExperimentConfig) -> None:
        self._config = config

    def run(
        self,
        request: PaperReconciliationRequest,
    ) -> PaperReconciliationResult | None:
        validate_paper_trading_config(self._config)
        store = PaperStateStore(self._config)
        latest_submitted = _latest_submitted_proposal_requiring_reconciliation(store)
        if latest_submitted is None:
            return None

        proposal, submission, submission_path = latest_submitted
        trade_date = str(submission["trade_date"])
        broker_client = request.broker or AlpacaPaperBrokerClient()
        refreshed_submission = _refresh_submission_order_status(
            store,
            proposal=proposal,
            submission=submission,
            broker_client=broker_client,
            now=request.now,
        )
        if refreshed_submission is None:
            return None

        return PaperReconciliationResult(
            proposal_id=str(proposal["proposal_id"]),
            submission_path=str(submission_path),
            order_status_path=str(store.trade_order_status_path(trade_date)),
            order_status=str(refreshed_submission["order_status"]),
            poll_status=str(refreshed_submission.get("poll_status", "")),
            submission=refreshed_submission,
        )
