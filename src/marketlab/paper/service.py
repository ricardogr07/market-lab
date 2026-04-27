from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

from marketlab.config import ExperimentConfig
from marketlab.paper.alpaca import AlpacaPaperBrokerClient
from marketlab.paper.application.approval import ApprovalService
from marketlab.paper.application.decision import DecisionService
from marketlab.paper.contracts import (
    PaperApprovalRequest,
    PaperBroker,
    PaperDecisionRequest,
    PaperHistoryProvider,
)
from marketlab.paper.core import (
    ALPACA_MIN_NOTIONAL_ORDER,
    APPROVAL_APPROVED,
    APPROVAL_PENDING,
    APPROVAL_REJECTED,
    FAILED_ORDER_STATUSES,
    SUBMISSION_NOOP,
    SUBMISSION_SKIPPED,
    SUBMISSION_SUBMITTED,
    TERMINAL_ORDER_STATUSES,
    _buy_order_notional,
    _client_order_id,
    _clock_value,
    _local_now,
    _now_utc,
    _paper_symbol,
    _position_market_value,
    _rounded_notional,
    _safe_float,
    validate_paper_trading_config,
)
from marketlab.paper.notifications import (
    TelegramTransport,
    notify_paper_submission,
    write_notification_record,
)
from marketlab.paper.state import PaperStateStore, _json_dump, _json_load

_notify_paper_submission = notify_paper_submission
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


def _submission_gate_status(
    config: ExperimentConfig,
    proposal: dict[str, Any],
) -> tuple[str, str]:
    if config.paper.execution_mode == "autonomous":
        return "ready", ""

    approval_status = proposal.get("approval_status", APPROVAL_PENDING)
    if approval_status == APPROVAL_REJECTED:
        return SUBMISSION_SKIPPED, "rejected"
    if approval_status != APPROVAL_APPROVED:
        return SUBMISSION_SKIPPED, "missing_approval"

    required_actor = "agent" if config.paper.execution_mode == "agent_approval" else "manual"
    if proposal.get("approval_actor") != required_actor:
        return SUBMISSION_SKIPPED, "wrong_actor"
    return "ready", ""


def _poll_order_status(
    *,
    broker_client: AlpacaPaperBrokerClient,
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
    config: ExperimentConfig,
    store: PaperStateStore,
    *,
    proposal: dict[str, Any],
    submission: dict[str, Any],
    broker_client: AlpacaPaperBrokerClient,
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
    if (
        refreshed_order_status == current_order_status
        and poll_status == current_poll_status
        and store.trade_order_status_path(str(proposal["effective_date"])).exists()
    ):
        return None

    trade_date = str(submission["trade_date"])
    _json_dump(store.trade_order_status_path(trade_date), order_status)
    refreshed_submission = dict(submission)
    refreshed_submission["order_status"] = refreshed_order_status
    refreshed_submission["poll_status"] = poll_status
    refreshed_submission["order_status_path"] = str(store.trade_order_status_path(trade_date))
    refreshed_submission["updated_at"] = _now_utc(now).isoformat()
    _json_dump(store.trade_submission_path(trade_date), refreshed_submission)
    status = {
        "event": "paper-submit",
        "status": refreshed_submission["status"],
        "proposal_id": refreshed_submission["proposal_id"],
        "submission_path": str(store.trade_submission_path(trade_date)),
        "order_status": refreshed_order_status,
        "updated_at": _now_utc(now).isoformat(),
    }
    store.write_status(status)
    return refreshed_submission


def reconcile_latest_submission_status(
    config: ExperimentConfig,
    *,
    now: datetime | None = None,
    broker: PaperBroker | None = None,
) -> dict[str, Any] | None:
    validate_paper_trading_config(config)
    store = PaperStateStore(config)
    latest_submitted = _latest_submitted_proposal_requiring_reconciliation(store)
    if latest_submitted is None:
        return None
    proposal, submission, submission_path = latest_submitted
    trade_date = str(submission["trade_date"])
    broker_client = broker or AlpacaPaperBrokerClient()
    refreshed_submission = _refresh_submission_order_status(
        config,
        store,
        proposal=proposal,
        submission=submission,
        broker_client=broker_client,
        now=now,
    )
    if refreshed_submission is None:
        return None

    return {
        "proposal_id": proposal["proposal_id"],
        "submission_path": str(submission_path),
        "order_status_path": str(store.trade_order_status_path(trade_date)),
        "order_status": refreshed_submission["order_status"],
        "poll_status": refreshed_submission.get("poll_status", ""),
    }


def _backup_submission_attempt_artifacts(
    store: PaperStateStore,
    *,
    trade_date: str,
    now: datetime | None = None,
) -> None:
    timestamp = _now_utc(now).strftime("%Y%m%dT%H%M%S%fZ")
    for path in (
        store.trade_submission_path(trade_date),
        store.trade_order_status_path(trade_date),
        store.trade_order_preview_path(trade_date),
        store.trade_account_snapshot_path(trade_date),
    ):
        if not path.exists():
            continue
        backup_path = path.with_name(f"{path.stem}.retry-backup.{timestamp}.bak")
        path.rename(backup_path)


def run_paper_submit(
    config: ExperimentConfig,
    *,
    now: datetime | None = None,
    broker: PaperBroker | None = None,
    notification_transport: TelegramTransport | None = None,
    retry_failed_submission: bool = False,
) -> dict[str, Any]:
    validate_paper_trading_config(config)
    paper_symbol = _paper_symbol(config)
    store = PaperStateStore(config)
    proposal = store.latest_proposal()
    if proposal is None:
        status = {
            "event": "paper-submit",
            "status": SUBMISSION_SKIPPED,
            "reason": "no_proposal",
            "updated_at": _now_utc(now).isoformat(),
        }
        status_path = store.write_status(status)
        _notify_paper_submission(
            config,
            store,
            outcome=SUBMISSION_SKIPPED,
            status=status,
            now=now,
            transport=notification_transport,
        )
        return {"status_path": str(status_path), "status": status}

    trade_date = proposal["effective_date"]
    submission_path = store.trade_submission_path(trade_date)
    if submission_path.exists():
        submission = _json_load(submission_path)
        broker_client = broker or AlpacaPaperBrokerClient()
        refreshed_submission = _refresh_submission_order_status(
            config,
            store,
            proposal=proposal,
            submission=submission,
            broker_client=broker_client,
            now=now,
        )
        if refreshed_submission is not None:
            submission = refreshed_submission
        order_status = str(submission.get("order_status", "")).lower()
        if not retry_failed_submission or order_status not in FAILED_ORDER_STATUSES:
            status = {
                "event": "paper-submit",
                "status": "existing_submission",
                "proposal_id": proposal["proposal_id"],
                "submission_path": str(submission_path),
                "order_status": submission.get("order_status", ""),
                "updated_at": _now_utc(now).isoformat(),
            }
            status_path = store.write_status(status)
            _notify_paper_submission(
                config,
                store,
                outcome="existing_submission",
                status=status,
                proposal=proposal,
                submission=submission,
                now=now,
                transport=notification_transport,
            )
            return {
                "submission_path": str(submission_path),
                "status_path": str(status_path),
                "status": status,
                "submission": submission,
            }

        _backup_submission_attempt_artifacts(store, trade_date=trade_date, now=now)
        retry_suffix = _now_utc(now).strftime("retry%H%M%S")
    else:
        local_now = _local_now(config, now)
        submission_clock = _clock_value(config.paper.submission_time)
        if local_now.time() < submission_clock:
            raise RuntimeError(
                "paper-submit is only allowed at or after "
                f"{config.paper.submission_time} {config.paper.schedule_timezone}."
            )
        broker_client = broker or AlpacaPaperBrokerClient()
        retry_suffix = ""

    gate_status, gate_reason = _submission_gate_status(config, proposal)
    if gate_status != "ready":
        submission = {
            "proposal_id": proposal["proposal_id"],
            "trade_date": trade_date,
            "status": gate_status,
            "reason": gate_reason,
            "updated_at": _now_utc(now).isoformat(),
        }
        _json_dump(submission_path, submission)
        status = {
            "event": "paper-submit",
            "status": gate_status,
            "reason": gate_reason,
            "proposal_id": proposal["proposal_id"],
            "submission_path": str(submission_path),
            "updated_at": _now_utc(now).isoformat(),
        }
        status_path = store.write_status(status)
        _notify_paper_submission(
            config,
            store,
            outcome=gate_status,
            status=status,
            proposal=proposal,
            submission=submission,
            now=now,
            transport=notification_transport,
        )
        return {
            "submission_path": str(submission_path),
            "status_path": str(status_path),
            "status": status,
            "submission": submission,
        }

    account = broker_client.get_account()
    _json_dump(store.trade_account_snapshot_path(trade_date), account)
    position = broker_client.get_position(paper_symbol)
    current_qty = _safe_float((position or {}).get("qty"))
    current_market_value = _position_market_value(position, reference_price=float(proposal["reference_price"]))
    equity = _safe_float(account.get("equity"))
    buying_power = _safe_float(account.get("buying_power"), default=_safe_float(account.get("cash"), default=equity))
    reference_price = float(proposal["reference_price"])
    target_weight = _safe_float(proposal.get("target_weight"))
    current_signed_market_value = current_qty * reference_price
    hold_existing_long = (
        target_weight > 0.0
        and current_qty > 0.0
        and current_market_value >= ALPACA_MIN_NOTIONAL_ORDER
    )
    desired_notional = 0.0
    order_notional = 0.0
    gap_notional = 0.0
    desired_qty = 0.0
    if target_weight > 0.0:
        if hold_existing_long:
            desired_notional = current_market_value
            desired_qty = current_qty
        else:
            desired_notional, order_notional = _buy_order_notional(
                equity=equity,
                buying_power=buying_power,
                current_market_value=current_signed_market_value,
                target_weight=target_weight,
            )
            gap_notional = max(desired_notional - current_signed_market_value, 0.0)
            order_notional = _rounded_notional(order_notional)
            desired_qty = desired_notional / reference_price if reference_price > 0.0 else 0.0
    delta_qty = round(desired_qty - current_qty, 6)
    if delta_qty > 1e-6:
        side = "buy"
    elif delta_qty < -1e-6:
        side = "sell"
    else:
        side = "none"
    order_preview = {
        "proposal_id": proposal["proposal_id"],
        "trade_date": trade_date,
        "symbol": paper_symbol,
        "equity": equity,
        "buying_power": buying_power,
        "reference_price": reference_price,
        "current_qty": current_qty,
        "current_market_value": current_market_value,
        "desired_notional": desired_notional,
        "order_notional": order_notional,
        "desired_qty": desired_qty,
        "delta_qty": delta_qty,
        "side": side,
        "updated_at": _now_utc(now).isoformat(),
    }
    _json_dump(store.trade_order_preview_path(trade_date), order_preview)

    if side == "none" or (side == "buy" and order_notional < ALPACA_MIN_NOTIONAL_ORDER):
        buy_reason = "already_at_target"
        if _rounded_notional(gap_notional) >= ALPACA_MIN_NOTIONAL_ORDER:
            buy_reason = "insufficient_buying_power"
        submission = {
            "proposal_id": proposal["proposal_id"],
            "trade_date": trade_date,
            "status": SUBMISSION_NOOP,
            "reason": "already_at_target" if side == "none" else buy_reason,
            "order_preview_path": str(store.trade_order_preview_path(trade_date)),
            "updated_at": _now_utc(now).isoformat(),
        }
        _json_dump(submission_path, submission)
        status = {
            "event": "paper-submit",
            "status": SUBMISSION_NOOP,
            "proposal_id": proposal["proposal_id"],
            "submission_path": str(submission_path),
            "updated_at": _now_utc(now).isoformat(),
        }
        status_path = store.write_status(status)
        _notify_paper_submission(
            config,
            store,
            outcome=SUBMISSION_NOOP,
            status=status,
            proposal=proposal,
            submission=submission,
            now=now,
            transport=notification_transport,
        )
        return {
            "submission_path": str(submission_path),
            "status_path": str(status_path),
            "status": status,
            "submission": submission,
        }

    client_order_id = _client_order_id(proposal["proposal_id"], retry_suffix=retry_suffix)
    if side == "buy":
        order = broker_client.submit_notional_day_market_order(
            symbol=paper_symbol,
            notional=order_notional,
            side=side,
            client_order_id=client_order_id,
        )
    else:
        order = broker_client.submit_fractional_day_market_order(
            symbol=paper_symbol,
            qty=abs(delta_qty),
            side=side,
            client_order_id=client_order_id,
        )
    order_status, poll_status = _poll_order_status(
        broker_client=broker_client,
        order_id=str(order["id"]),
        fallback_status=str(order.get("status", "unknown")),
        client_order_id=str(order.get("client_order_id", client_order_id)),
    )
    _json_dump(store.trade_order_status_path(trade_date), order_status)

    submission = {
        "proposal_id": proposal["proposal_id"],
        "trade_date": trade_date,
        "status": SUBMISSION_SUBMITTED,
        "side": side,
        "qty": abs(delta_qty) if side == "sell" else None,
        "notional": order_notional if side == "buy" else None,
        "order_id": order["id"],
        "client_order_id": order.get("client_order_id", client_order_id),
        "order_status": str(order_status.get("status", order.get("status", "unknown"))).lower(),
        "poll_status": poll_status,
        "order_preview_path": str(store.trade_order_preview_path(trade_date)),
        "account_snapshot_path": str(store.trade_account_snapshot_path(trade_date)),
        "order_status_path": str(store.trade_order_status_path(trade_date)),
        "updated_at": _now_utc(now).isoformat(),
    }
    _json_dump(submission_path, submission)
    status = {
        "event": "paper-submit",
        "status": SUBMISSION_SUBMITTED,
        "proposal_id": proposal["proposal_id"],
        "submission_path": str(submission_path),
        "order_status": submission["order_status"],
        "updated_at": _now_utc(now).isoformat(),
    }
    status_path = store.write_status(status)
    _notify_paper_submission(
        config,
        store,
        outcome=SUBMISSION_SUBMITTED,
        status=status,
        proposal=proposal,
        submission=submission,
        now=now,
        transport=notification_transport,
    )
    return {
        "submission_path": str(submission_path),
        "status_path": str(status_path),
        "status": status,
        "submission": submission,
    }


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
