from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

from marketlab.config import ExperimentConfig
from marketlab.paper.contracts import (
    PaperBroker,
    PaperBrokerFactory,
    PaperReconciliationRequest,
    PaperReconciliationResult,
    PaperTradeRepository,
    PaperUnitOfWorkFactory,
)
from marketlab.paper.core import (
    SUBMISSION_SUBMITTED,
    TERMINAL_ORDER_STATUSES,
    _now_utc,
    validate_paper_trading_config,
)


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
    trades: PaperTradeRepository,
) -> tuple[dict[str, Any], dict[str, Any]] | None:
    for proposal in trades.list_proposals():
        trade_date = str(proposal.get("effective_date", ""))
        if trade_date == "":
            continue
        submission = trades.get_submission(trade_date)
        if submission is None:
            continue
        if submission.get("status") != SUBMISSION_SUBMITTED:
            continue
        order_status = str(submission.get("order_status", "")).lower()
        if order_status in TERMINAL_ORDER_STATUSES:
            continue
        return proposal, submission
    return None


def _refresh_submission_order_status(
    *,
    submission: dict[str, Any],
    order_status_path: Path,
    has_order_status_path: bool,
    broker_client: PaperBroker,
    now: datetime | None = None,
) -> tuple[dict[str, Any], dict[str, Any]] | None:
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
    if (
        refreshed_order_status == current_order_status
        and poll_status == current_poll_status
        and has_order_status_path
    ):
        return None

    refreshed_submission = dict(submission)
    refreshed_submission["order_status"] = refreshed_order_status
    refreshed_submission["poll_status"] = poll_status
    refreshed_submission["order_status_path"] = str(order_status_path)
    refreshed_submission["updated_at"] = _now_utc(now).isoformat()
    return refreshed_submission, order_status


class ReconciliationService:
    def __init__(
        self,
        config: ExperimentConfig,
        *,
        uow_factory: PaperUnitOfWorkFactory,
        broker_factory: PaperBrokerFactory | None = None,
    ) -> None:
        self._config = config
        self._uow_factory = uow_factory
        self._broker_factory = broker_factory

    def _broker(self, request: PaperReconciliationRequest) -> PaperBroker:
        if request.broker is not None:
            return request.broker
        if self._broker_factory is None:
            raise RuntimeError("ReconciliationService requires a broker or broker_factory.")
        return self._broker_factory()

    def run(
        self,
        request: PaperReconciliationRequest,
    ) -> PaperReconciliationResult | None:
        validate_paper_trading_config(self._config)
        with self._uow_factory() as uow:
            latest_submitted = _latest_submitted_proposal_requiring_reconciliation(uow.trades)
        if latest_submitted is None:
            return None

        proposal, submission = latest_submitted
        trade_date = str(submission["trade_date"])
        with self._uow_factory() as path_uow:
            order_status_path = path_uow.trades.trade_order_status_path(trade_date)
        broker_client = self._broker(request)
        refreshed = _refresh_submission_order_status(
            submission=submission,
            order_status_path=order_status_path,
            has_order_status_path=str(submission.get("order_status_path", "")).strip() != "",
            broker_client=broker_client,
            now=request.now,
        )
        if refreshed is None:
            return None
        refreshed_submission, order_status = refreshed
        with self._uow_factory() as uow:
            order_status_path = uow.trades.save_order_status(
                trade_date=trade_date,
                order_status=order_status,
            )
            submission_path = uow.trades.save_submission(
                trade_date=trade_date,
                submission=refreshed_submission,
            )
            status = {
                "event": "paper-submit",
                "status": refreshed_submission["status"],
                "proposal_id": refreshed_submission["proposal_id"],
                "submission_path": str(submission_path),
                "order_status": str(refreshed_submission["order_status"]),
                "updated_at": _now_utc(request.now).isoformat(),
            }
            uow.status.write_status(status)
            uow.commit()

        return PaperReconciliationResult(
            proposal_id=str(proposal["proposal_id"]),
            submission_path=str(submission_path),
            order_status_path=str(order_status_path),
            order_status=str(refreshed_submission["order_status"]),
            poll_status=str(refreshed_submission.get("poll_status", "")),
            submission=refreshed_submission,
        )
