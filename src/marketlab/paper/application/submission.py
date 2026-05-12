from __future__ import annotations

from marketlab.config import ExperimentConfig
from marketlab.paper.contracts import (
    PaperArtifactStore,
    PaperBroker,
    PaperBrokerFactory,
    PaperSubmissionRequest,
    PaperSubmissionResult,
    PaperUnitOfWorkFactory,
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


def _order_time_in_force(config: ExperimentConfig) -> str:
    if config.paper.order_type == "crypto_market_gtc":
        return "gtc"
    if config.paper.order_type == "crypto_market_ioc":
        return "ioc"
    return "day"


class SubmissionService:
    def __init__(
        self,
        config: ExperimentConfig,
        *,
        uow_factory: PaperUnitOfWorkFactory,
        artifact_store: PaperArtifactStore,
        broker_factory: PaperBrokerFactory | None = None,
    ) -> None:
        self._config = config
        self._uow_factory = uow_factory
        self._artifact_store = artifact_store
        self._broker_factory = broker_factory

    def _broker(self, request: PaperSubmissionRequest) -> PaperBroker:
        if request.broker is not None:
            return request.broker
        if self._broker_factory is None:
            raise RuntimeError("SubmissionService requires a broker or broker_factory.")
        return self._broker_factory()

    def run(self, request: PaperSubmissionRequest) -> PaperSubmissionResult:
        config = self._config
        validate_paper_trading_config(config)
        paper_symbol = _paper_symbol(config)
        with self._uow_factory() as uow:
            proposal = uow.trades.get_latest_proposal()
            if proposal is None:
                status = {
                    "event": "paper-submit",
                    "status": SUBMISSION_SKIPPED,
                    "reason": "no_proposal",
                    "updated_at": _now_utc(request.now).isoformat(),
                }
                status_path = uow.status.write_status(status)
                uow.commit()
                return PaperSubmissionResult(
                    status_path=str(status_path),
                    status=status,
                )
            trade_date = str(proposal["effective_date"])
            submission_path = uow.trades.trade_submission_path(trade_date)
            order_status_path = uow.trades.trade_order_status_path(trade_date)
            has_order_status_path = uow.trades.order_status_path_exists(trade_date)
            submission = uow.trades.get_submission(trade_date)

        if submission is not None:
            broker_client = self._broker(request)
            refreshed = _refresh_submission_order_status(
                submission=submission,
                order_status_path=order_status_path,
                has_order_status_path=has_order_status_path,
                broker_client=broker_client,
                now=request.now,
            )
            if refreshed is not None:
                submission, refreshed_order_status = refreshed
                with self._uow_factory() as uow:
                    order_status_path = uow.trades.save_order_status(
                        trade_date=trade_date,
                        order_status=refreshed_order_status,
                    )
                    submission_path = uow.trades.save_submission(
                        trade_date=trade_date,
                        submission=submission,
                    )
                    uow.commit()

            existing_order_status = str(submission.get("order_status", "")).lower()
            if (
                not request.retry_failed_submission
                or existing_order_status not in FAILED_ORDER_STATUSES
            ):
                with self._uow_factory() as uow:
                    status = {
                        "event": "paper-submit",
                        "status": "existing_submission",
                        "proposal_id": proposal["proposal_id"],
                        "submission_path": str(submission_path),
                        "order_status": submission.get("order_status", ""),
                        "updated_at": _now_utc(request.now).isoformat(),
                    }
                    status_path = uow.status.write_status(status)
                    uow.commit()
                return PaperSubmissionResult(
                    proposal_id=str(proposal["proposal_id"]),
                    submission_path=str(submission_path),
                    status_path=str(status_path),
                    status=status,
                    submission=submission,
                    proposal=proposal,
                )

            with self._uow_factory() as uow:
                uow.trades.backup_submission_attempt_artifacts(
                    trade_date=trade_date,
                    now=request.now,
                )
            retry_suffix = _now_utc(request.now).strftime("retry%H%M%S")
        else:
            local_now = _local_now(config, request.now)
            submission_clock = _clock_value(config.paper.submission_time)
            if local_now.time() < submission_clock:
                raise RuntimeError(
                    "paper-submit is only allowed at or after "
                    f"{config.paper.submission_time} {config.paper.schedule_timezone}."
                )
            broker_client = self._broker(request)
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
            with self._uow_factory() as uow:
                submission_path = uow.trades.save_submission(
                    trade_date=trade_date,
                    submission=submission,
                )
                status = {
                    "event": "paper-submit",
                    "status": gate_status,
                    "reason": gate_reason,
                    "proposal_id": proposal["proposal_id"],
                    "submission_path": str(submission_path),
                    "updated_at": _now_utc(request.now).isoformat(),
                }
                status_path = uow.status.write_status(status)
                uow.commit()
            return PaperSubmissionResult(
                proposal_id=str(proposal["proposal_id"]),
                submission_path=str(submission_path),
                status_path=str(status_path),
                status=status,
                submission=submission,
                proposal=proposal,
            )

        account = broker_client.get_account()
        account_snapshot_path = self._artifact_store.write_trade_account_snapshot(
            trade_date=trade_date,
            payload=account,
        )
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
            config.paper.position_sizing == "full_equity_fractional"
            and target_weight > 0.0
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
                if config.paper.position_sizing == "target_weight_fractional":
                    gap_notional = desired_notional - current_signed_market_value
                    if gap_notional < 0.0:
                        order_notional = 0.0
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
        order_preview_path = self._artifact_store.write_trade_order_preview(
            trade_date=trade_date,
            payload=order_preview,
        )

        minimum_trade_notional = abs(delta_qty) * reference_price
        if (
            side == "none"
            or (side == "buy" and order_notional < ALPACA_MIN_NOTIONAL_ORDER)
            or (side == "sell" and minimum_trade_notional < ALPACA_MIN_NOTIONAL_ORDER)
        ):
            buy_reason = "already_at_target"
            if _rounded_notional(gap_notional) >= ALPACA_MIN_NOTIONAL_ORDER:
                buy_reason = "insufficient_buying_power"
            submission = {
                "proposal_id": proposal["proposal_id"],
                "trade_date": trade_date,
                "status": SUBMISSION_NOOP,
                "reason": "already_at_target" if side == "none" else buy_reason,
                "order_preview_path": str(order_preview_path),
                "updated_at": _now_utc(request.now).isoformat(),
            }
            with self._uow_factory() as uow:
                submission_path = uow.trades.save_submission(
                    trade_date=trade_date,
                    submission=submission,
                )
                status = {
                    "event": "paper-submit",
                    "status": SUBMISSION_NOOP,
                    "proposal_id": proposal["proposal_id"],
                    "submission_path": str(submission_path),
                    "updated_at": _now_utc(request.now).isoformat(),
                }
                status_path = uow.status.write_status(status)
                uow.commit()
            return PaperSubmissionResult(
                proposal_id=str(proposal["proposal_id"]),
                submission_path=str(submission_path),
                status_path=str(status_path),
                status=status,
                submission=submission,
                proposal=proposal,
            )

        client_order_id = _client_order_id(str(proposal["proposal_id"]), retry_suffix=retry_suffix)
        time_in_force = _order_time_in_force(config)
        if side == "buy":
            if time_in_force == "day":
                order = broker_client.submit_notional_day_market_order(
                    symbol=paper_symbol,
                    notional=order_notional,
                    side=side,
                    client_order_id=client_order_id,
                )
            else:
                order = broker_client.submit_notional_day_market_order(
                    symbol=paper_symbol,
                    notional=order_notional,
                    side=side,
                    client_order_id=client_order_id,
                    time_in_force=time_in_force,
                )
        else:
            if time_in_force == "day":
                order = broker_client.submit_fractional_day_market_order(
                    symbol=paper_symbol,
                    qty=abs(delta_qty),
                    side=side,
                    client_order_id=client_order_id,
                )
            else:
                order = broker_client.submit_fractional_day_market_order(
                    symbol=paper_symbol,
                    qty=abs(delta_qty),
                    side=side,
                    client_order_id=client_order_id,
                    time_in_force=time_in_force,
                )
        order_status, poll_status = _poll_order_status(
            broker_client=broker_client,
            order_id=str(order["id"]),
            fallback_status=str(order.get("status", "unknown")),
            client_order_id=str(order.get("client_order_id", client_order_id)),
        )

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
            "order_preview_path": str(order_preview_path),
            "account_snapshot_path": str(account_snapshot_path),
            "order_status_path": str(order_status_path),
            "updated_at": _now_utc(request.now).isoformat(),
        }
        with self._uow_factory() as uow:
            order_status_path = uow.trades.save_order_status(
                trade_date=trade_date,
                order_status=order_status,
            )
            submission["order_status_path"] = str(order_status_path)
            submission_path = uow.trades.save_submission(
                trade_date=trade_date,
                submission=submission,
            )
            status = {
                "event": "paper-submit",
                "status": SUBMISSION_SUBMITTED,
                "proposal_id": proposal["proposal_id"],
                "submission_path": str(submission_path),
                "order_status": submission["order_status"],
                "updated_at": _now_utc(request.now).isoformat(),
            }
            status_path = uow.status.write_status(status)
            uow.commit()
        return PaperSubmissionResult(
            proposal_id=str(proposal["proposal_id"]),
            submission_path=str(submission_path),
            status_path=str(status_path),
            status=status,
            submission=submission,
            proposal=proposal,
        )
