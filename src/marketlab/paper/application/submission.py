from __future__ import annotations

from datetime import datetime

from marketlab.config import ExperimentConfig
from marketlab.paper.alpaca import AlpacaPaperBrokerClient
from marketlab.paper.contracts import (
    PaperSubmissionRequest,
    PaperSubmissionResult,
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
from marketlab.paper.notifications import notify_paper_submission
from marketlab.paper.state import PaperStateStore, _json_dump, _json_load

from .reconciliation import _poll_order_status, _refresh_submission_order_status


def _submission_gate_status(
    config: ExperimentConfig,
    proposal: dict[str, object],
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


class SubmissionService:
    def __init__(self, config: ExperimentConfig) -> None:
        self._config = config

    def run(self, request: PaperSubmissionRequest) -> PaperSubmissionResult:
        config = self._config
        validate_paper_trading_config(config)
        paper_symbol = _paper_symbol(config)
        store = PaperStateStore(config)
        proposal = store.latest_proposal()
        if proposal is None:
            status = {
                "event": "paper-submit",
                "status": SUBMISSION_SKIPPED,
                "reason": "no_proposal",
                "updated_at": _now_utc(request.now).isoformat(),
            }
            status_path = store.write_status(status)
            notify_paper_submission(
                config,
                store,
                outcome=SUBMISSION_SKIPPED,
                status=status,
                now=request.now,
                transport=request.notification_transport,
            )
            return PaperSubmissionResult(
                status_path=str(status_path),
                status=status,
            )

        trade_date = str(proposal["effective_date"])
        submission_path = store.trade_submission_path(trade_date)
        if submission_path.exists():
            submission = _json_load(submission_path)
            broker_client = request.broker or AlpacaPaperBrokerClient()
            refreshed_submission = _refresh_submission_order_status(
                store,
                proposal=proposal,
                submission=submission,
                broker_client=broker_client,
                now=request.now,
            )
            if refreshed_submission is not None:
                submission = refreshed_submission
            order_status = str(submission.get("order_status", "")).lower()
            if not request.retry_failed_submission or order_status not in FAILED_ORDER_STATUSES:
                status = {
                    "event": "paper-submit",
                    "status": "existing_submission",
                    "proposal_id": proposal["proposal_id"],
                    "submission_path": str(submission_path),
                    "order_status": submission.get("order_status", ""),
                    "updated_at": _now_utc(request.now).isoformat(),
                }
                status_path = store.write_status(status)
                notify_paper_submission(
                    config,
                    store,
                    outcome="existing_submission",
                    status=status,
                    proposal=proposal,
                    submission=submission,
                    now=request.now,
                    transport=request.notification_transport,
                )
                return PaperSubmissionResult(
                    proposal_id=str(proposal["proposal_id"]),
                    submission_path=str(submission_path),
                    status_path=str(status_path),
                    status=status,
                    submission=submission,
                )

            _backup_submission_attempt_artifacts(store, trade_date=trade_date, now=request.now)
            retry_suffix = _now_utc(request.now).strftime("retry%H%M%S")
        else:
            local_now = _local_now(config, request.now)
            submission_clock = _clock_value(config.paper.submission_time)
            if local_now.time() < submission_clock:
                raise RuntimeError(
                    "paper-submit is only allowed at or after "
                    f"{config.paper.submission_time} {config.paper.schedule_timezone}."
                )
            broker_client = request.broker or AlpacaPaperBrokerClient()
            retry_suffix = ""

        gate_status, gate_reason = _submission_gate_status(config, proposal)
        if gate_status != "ready":
            submission = {
                "proposal_id": proposal["proposal_id"],
                "trade_date": trade_date,
                "status": gate_status,
                "reason": gate_reason,
                "updated_at": _now_utc(request.now).isoformat(),
            }
            _json_dump(submission_path, submission)
            status = {
                "event": "paper-submit",
                "status": gate_status,
                "reason": gate_reason,
                "proposal_id": proposal["proposal_id"],
                "submission_path": str(submission_path),
                "updated_at": _now_utc(request.now).isoformat(),
            }
            status_path = store.write_status(status)
            notify_paper_submission(
                config,
                store,
                outcome=gate_status,
                status=status,
                proposal=proposal,
                submission=submission,
                now=request.now,
                transport=request.notification_transport,
            )
            return PaperSubmissionResult(
                proposal_id=str(proposal["proposal_id"]),
                submission_path=str(submission_path),
                status_path=str(status_path),
                status=status,
                submission=submission,
            )

        account = broker_client.get_account()
        _json_dump(store.trade_account_snapshot_path(trade_date), account)
        position = broker_client.get_position(paper_symbol)
        current_qty = _safe_float((position or {}).get("qty"))
        current_market_value = _position_market_value(
            position,
            reference_price=float(proposal["reference_price"]),
        )
        equity = _safe_float(account.get("equity"))
        buying_power = _safe_float(
            account.get("buying_power"),
            default=_safe_float(account.get("cash"), default=equity),
        )
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
            "updated_at": _now_utc(request.now).isoformat(),
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
                "updated_at": _now_utc(request.now).isoformat(),
            }
            _json_dump(submission_path, submission)
            status = {
                "event": "paper-submit",
                "status": SUBMISSION_NOOP,
                "proposal_id": proposal["proposal_id"],
                "submission_path": str(submission_path),
                "updated_at": _now_utc(request.now).isoformat(),
            }
            status_path = store.write_status(status)
            notify_paper_submission(
                config,
                store,
                outcome=SUBMISSION_NOOP,
                status=status,
                proposal=proposal,
                submission=submission,
                now=request.now,
                transport=request.notification_transport,
            )
            return PaperSubmissionResult(
                proposal_id=str(proposal["proposal_id"]),
                submission_path=str(submission_path),
                status_path=str(status_path),
                status=status,
                submission=submission,
            )

        client_order_id = _client_order_id(str(proposal["proposal_id"]), retry_suffix=retry_suffix)
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
            "updated_at": _now_utc(request.now).isoformat(),
        }
        _json_dump(submission_path, submission)
        status = {
            "event": "paper-submit",
            "status": SUBMISSION_SUBMITTED,
            "proposal_id": proposal["proposal_id"],
            "submission_path": str(submission_path),
            "order_status": submission["order_status"],
            "updated_at": _now_utc(request.now).isoformat(),
        }
        status_path = store.write_status(status)
        notify_paper_submission(
            config,
            store,
            outcome=SUBMISSION_SUBMITTED,
            status=status,
            proposal=proposal,
            submission=submission,
            now=request.now,
            transport=request.notification_transport,
        )
        return PaperSubmissionResult(
            proposal_id=str(proposal["proposal_id"]),
            submission_path=str(submission_path),
            status_path=str(status_path),
            status=status,
            submission=submission,
        )
