from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Callable, Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib import error, request

from marketlab.config import ExperimentConfig
from marketlab.env import load_env_file
from marketlab.paper.state import PaperStateStore

TELEGRAM_API_BASE_URL_ENV = "MARKETLAB_TELEGRAM_API_BASE_URL"
TELEGRAM_BOT_TOKEN_ENV = "TELEGRAM_BOT_TOKEN"
TELEGRAM_CHAT_ID_ENV = "TELEGRAM_CHAT_ID"
DEFAULT_TELEGRAM_API_BASE_URL = "https://api.telegram.org"
DEFAULT_TELEGRAM_TIMEOUT_SECONDS = 10
DELIVERY_DELIVERED = "delivered"
DELIVERY_SKIPPED_DISABLED = "skipped_disabled"
DELIVERY_SKIPPED_MISSING_CREDENTIALS = "skipped_missing_credentials"
DELIVERY_FAILED = "failed_delivery"

TelegramTransport = Callable[[str, dict[str, Any], int], tuple[int, str]]


class PaperLoopStageError(RuntimeError):
    def __init__(
        self,
        *,
        loop_name: str,
        stage: str,
        cause: Exception,
        proposal_id: str = "",
        trade_date: str = "",
    ) -> None:
        super().__init__(str(cause))
        self.loop_name = loop_name
        self.stage = stage
        self.cause = cause
        self.proposal_id = proposal_id
        self.trade_date = trade_date


def _now_utc(now: datetime | None = None) -> datetime:
    if now is None:
        return datetime.now(UTC)
    if now.tzinfo is None:
        return now.replace(tzinfo=UTC)
    return now.astimezone(UTC)


def _default_transport(url: str, payload: dict[str, Any], timeout_seconds: int) -> tuple[int, str]:
    encoded = json.dumps(payload).encode("utf-8")
    telegram_request = request.Request(
        url,
        data=encoded,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with request.urlopen(telegram_request, timeout=timeout_seconds) as response:
        return response.status, response.read().decode("utf-8")


def _notification_symbol(
    config: ExperimentConfig,
    proposal: Mapping[str, Any] | None = None,
) -> str:
    if proposal is not None and str(proposal.get("symbol", "")).strip() != "":
        return str(proposal["symbol"])
    symbols = [str(symbol).strip() for symbol in config.data.symbols if str(symbol).strip() != ""]
    if len(symbols) == 1:
        return symbols[0]
    return ""


def _format_number(value: Any) -> str:
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        text = f"{value:.6f}"
        return text.rstrip("0").rstrip(".") or "0"
    return str(value)


def _append_line(lines: list[str], label: str, value: Any) -> None:
    if value is None:
        return
    text = str(value).strip()
    if text == "":
        return
    lines.append(f"{label}: {text}")


def _message_lines(stage: str, *, config: ExperimentConfig, proposal: Mapping[str, Any] | None) -> list[str]:
    lines = [stage]
    _append_line(lines, "experiment", config.experiment_name)
    _append_line(lines, "symbol", _notification_symbol(config, proposal))
    return lines


def build_decision_message(
    config: ExperimentConfig,
    *,
    outcome: str,
    status: Mapping[str, Any],
    proposal: Mapping[str, Any] | None = None,
) -> str:
    lines = _message_lines("paper-decision", config=config, proposal=proposal)
    _append_line(lines, "outcome", outcome)
    _append_line(lines, "market_date", status.get("market_date"))
    _append_line(lines, "signal_date", (proposal or {}).get("signal_date"))
    _append_line(lines, "effective_date", (proposal or {}).get("effective_date"))
    _append_line(lines, "latest_signal_date", status.get("latest_signal_date"))
    _append_line(lines, "proposal_id", (proposal or {}).get("proposal_id"))
    _append_line(lines, "decision", (proposal or {}).get("decision"))
    target_weight = (proposal or {}).get("target_weight")
    if target_weight is not None:
        _append_line(lines, "target_weight", _format_number(target_weight))
    long_vote_count = (proposal or {}).get("long_vote_count")
    cash_vote_count = (proposal or {}).get("cash_vote_count")
    threshold = config.paper.consensus_min_long_votes
    if long_vote_count is not None or cash_vote_count is not None:
        votes = (
            f"long={_format_number(long_vote_count or 0)} "
            f"cash={_format_number(cash_vote_count or 0)} "
            f"threshold={threshold}"
        )
        _append_line(lines, "votes", votes)
    reference_price = (proposal or {}).get("reference_price")
    if reference_price is not None:
        _append_line(lines, "reference_price", _format_number(reference_price))
    _append_line(lines, "reason", status.get("reason"))
    return "\n".join(lines)


def build_approval_message(
    config: ExperimentConfig,
    *,
    proposal: Mapping[str, Any],
    approval: Mapping[str, Any],
) -> str:
    lines = _message_lines("paper-approve", config=config, proposal=proposal)
    _append_line(lines, "outcome", approval.get("approval_status"))
    _append_line(lines, "signal_date", proposal.get("signal_date"))
    _append_line(lines, "effective_date", proposal.get("effective_date"))
    _append_line(lines, "proposal_id", proposal.get("proposal_id"))
    _append_line(lines, "actor", approval.get("actor"))
    _append_line(lines, "provider", approval.get("provider"))
    _append_line(lines, "model", approval.get("model"))
    _append_line(lines, "fallback_used", _format_number(approval.get("fallback_used", False)))
    _append_line(lines, "fallback_reason", approval.get("fallback_reason"))
    _append_line(lines, "rationale", approval.get("rationale"))
    return "\n".join(lines)


def build_submission_message(
    config: ExperimentConfig,
    *,
    outcome: str,
    status: Mapping[str, Any],
    proposal: Mapping[str, Any] | None = None,
    submission: Mapping[str, Any] | None = None,
) -> str:
    lines = _message_lines("paper-submit", config=config, proposal=proposal)
    _append_line(lines, "outcome", outcome)
    _append_line(lines, "signal_date", (proposal or {}).get("signal_date"))
    _append_line(lines, "effective_date", (proposal or {}).get("effective_date"))
    _append_line(lines, "trade_date", (submission or {}).get("trade_date"))
    _append_line(lines, "proposal_id", (proposal or {}).get("proposal_id") or (submission or {}).get("proposal_id"))
    _append_line(lines, "reason", (submission or {}).get("reason") or status.get("reason"))
    _append_line(lines, "side", (submission or {}).get("side"))
    qty = (submission or {}).get("qty")
    if qty is not None:
        _append_line(lines, "qty", _format_number(qty))
    notional = (submission or {}).get("notional")
    if notional is not None:
        _append_line(lines, "notional", _format_number(notional))
    _append_line(lines, "order_id", (submission or {}).get("order_id"))
    _append_line(lines, "order_status", (submission or {}).get("order_status"))
    return "\n".join(lines)


def build_error_message(
    config: ExperimentConfig,
    *,
    loop_name: str,
    stage: str,
    exc: Exception,
    proposal_id: str = "",
    trade_date: str = "",
) -> str:
    lines = _message_lines("paper-error", config=config, proposal=None)
    _append_line(lines, "loop", loop_name)
    _append_line(lines, "failed_stage", stage)
    _append_line(lines, "proposal_id", proposal_id)
    _append_line(lines, "trade_date", trade_date)
    _append_line(lines, "exception_type", type(exc).__name__)
    _append_line(lines, "exception_message", str(exc))
    return "\n".join(lines)


def build_error_fingerprint(
    *,
    loop_name: str,
    stage: str,
    exc: Exception,
    proposal_id: str = "",
    trade_date: str = "",
) -> str:
    raw = "|".join(
        [
            loop_name,
            stage,
            type(exc).__name__,
            str(exc),
            proposal_id,
            trade_date,
        ]
    )
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _telegram_api_base_url() -> str:
    configured = os.environ.get(TELEGRAM_API_BASE_URL_ENV, "").strip()
    if configured != "":
        return configured.rstrip("/")
    return DEFAULT_TELEGRAM_API_BASE_URL


def _safe_json(text: str) -> Any:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return text


def deliver_telegram_notification(
    config: ExperimentConfig,
    *,
    stage: str,
    outcome: str,
    message: str,
    details: Mapping[str, Any] | None = None,
    proposal_id: str = "",
    trade_date: str = "",
    now: datetime | None = None,
    transport: TelegramTransport | None = None,
) -> dict[str, Any]:
    timestamp = _now_utc(now).isoformat()
    record: dict[str, Any] = {
        "channel": "telegram",
        "stage": stage,
        "outcome": outcome,
        "proposal_id": proposal_id,
        "trade_date": trade_date,
        "message": message,
        "details": dict(details or {}),
        "requested_at": timestamp,
    }
    if not config.paper.notifications.telegram.enabled:
        record["delivery_status"] = DELIVERY_SKIPPED_DISABLED
        return record

    load_env_file()
    bot_token = os.environ.get(TELEGRAM_BOT_TOKEN_ENV, "").strip()
    chat_id = os.environ.get(TELEGRAM_CHAT_ID_ENV, "").strip()
    api_base_url = _telegram_api_base_url()
    record["api_base_url"] = api_base_url
    if bot_token == "" or chat_id == "":
        record["delivery_status"] = DELIVERY_SKIPPED_MISSING_CREDENTIALS
        record["missing_credentials"] = {
            TELEGRAM_BOT_TOKEN_ENV: bot_token == "",
            TELEGRAM_CHAT_ID_ENV: chat_id == "",
        }
        return record

    payload = {
        "chat_id": chat_id,
        "text": message,
        "disable_web_page_preview": True,
    }
    record["request"] = payload
    sender = transport or _default_transport
    url = f"{api_base_url}/bot{bot_token}/sendMessage"

    try:
        response_status, response_body = sender(url, payload, DEFAULT_TELEGRAM_TIMEOUT_SECONDS)
        response_payload = _safe_json(response_body)
        record["response_status"] = response_status
        record["response_body"] = response_payload
        if not 200 <= response_status < 300:
            raise RuntimeError(f"Telegram API responded with HTTP {response_status}.")
        if isinstance(response_payload, dict) and response_payload.get("ok") is False:
            raise RuntimeError(
                f"Telegram API rejected the message: {response_payload.get('description', 'unknown error')}"
            )
        record["delivery_status"] = DELIVERY_DELIVERED
    except (RuntimeError, error.URLError, error.HTTPError, OSError, ValueError) as exc:
        record["delivery_status"] = DELIVERY_FAILED
        record["error"] = f"{type(exc).__name__}: {exc}"

    return record


def write_notification_record(
    config: ExperimentConfig,
    store: PaperStateStore,
    *,
    stage: str,
    outcome: str,
    message: str,
    details: Mapping[str, Any],
    proposal_id: str = "",
    trade_date: str = "",
    now: datetime | None = None,
    transport: TelegramTransport | None = None,
) -> Path:
    record = deliver_telegram_notification(
        config,
        stage=stage,
        outcome=outcome,
        message=message,
        details=details,
        proposal_id=proposal_id,
        trade_date=trade_date,
        now=now,
        transport=transport,
    )
    return store.write_notification_record(
        stage=stage,
        outcome=outcome,
        payload=record,
        now=now,
    )


def notify_paper_decision(
    config: ExperimentConfig,
    store: PaperStateStore,
    *,
    outcome: str,
    status: Mapping[str, Any],
    proposal: Mapping[str, Any] | None = None,
    now: datetime | None = None,
    transport: TelegramTransport | None = None,
) -> Path:
    details: dict[str, Any] = {
        "experiment_name": config.experiment_name,
        "market_date": status.get("market_date"),
        "latest_signal_date": status.get("latest_signal_date"),
        "reason": status.get("reason"),
    }
    if proposal is not None:
        details.update(
            {
                "symbol": proposal.get("symbol"),
                "signal_date": proposal.get("signal_date"),
                "effective_date": proposal.get("effective_date"),
                "decision": proposal.get("decision"),
                "target_weight": proposal.get("target_weight"),
                "long_vote_count": proposal.get("long_vote_count"),
                "cash_vote_count": proposal.get("cash_vote_count"),
                "threshold": config.paper.consensus_min_long_votes,
                "reference_price": proposal.get("reference_price"),
            }
        )
    return write_notification_record(
        config,
        store,
        stage="paper-decision",
        outcome=outcome,
        message=build_decision_message(
            config,
            outcome=outcome,
            status=status,
            proposal=proposal,
        ),
        details=details,
        proposal_id=str((proposal or {}).get("proposal_id", "")),
        trade_date=str((proposal or {}).get("effective_date", "")),
        now=now,
        transport=transport,
    )


def notify_paper_approval(
    config: ExperimentConfig,
    store: PaperStateStore,
    *,
    proposal: Mapping[str, Any],
    approval_record: Mapping[str, Any],
    now: datetime | None = None,
    transport: TelegramTransport | None = None,
) -> Path:
    outcome = str(approval_record["approval_status"])
    details = {
        "experiment_name": config.experiment_name,
        "symbol": proposal.get("symbol"),
        "signal_date": proposal.get("signal_date"),
        "effective_date": proposal.get("effective_date"),
        "actor": approval_record.get("actor"),
        "provider": approval_record.get("provider"),
        "model": approval_record.get("model"),
        "fallback_used": approval_record.get("fallback_used", False),
        "fallback_reason": approval_record.get("fallback_reason"),
        "rationale": approval_record.get("rationale"),
    }
    return write_notification_record(
        config,
        store,
        stage="paper-approve",
        outcome=outcome,
        message=build_approval_message(
            config,
            proposal=proposal,
            approval=approval_record,
        ),
        details=details,
        proposal_id=str(proposal["proposal_id"]),
        trade_date=str(proposal["effective_date"]),
        now=now,
        transport=transport,
    )


def notify_paper_submission(
    config: ExperimentConfig,
    store: PaperStateStore,
    *,
    outcome: str,
    status: Mapping[str, Any],
    proposal: Mapping[str, Any] | None = None,
    submission: Mapping[str, Any] | None = None,
    now: datetime | None = None,
    transport: TelegramTransport | None = None,
) -> Path:
    details = {
        "experiment_name": config.experiment_name,
        "symbol": (proposal or {}).get("symbol"),
        "signal_date": (proposal or {}).get("signal_date"),
        "effective_date": (proposal or {}).get("effective_date"),
        "reason": (submission or {}).get("reason") or status.get("reason"),
        "side": (submission or {}).get("side"),
        "qty": (submission or {}).get("qty"),
        "notional": (submission or {}).get("notional"),
        "order_id": (submission or {}).get("order_id"),
        "order_status": (submission or {}).get("order_status"),
    }
    return write_notification_record(
        config,
        store,
        stage="paper-submit",
        outcome=outcome,
        message=build_submission_message(
            config,
            outcome=outcome,
            status=status,
            proposal=proposal,
            submission=submission,
        ),
        details=details,
        proposal_id=str((proposal or {}).get("proposal_id") or (submission or {}).get("proposal_id") or ""),
        trade_date=str((submission or {}).get("trade_date") or (proposal or {}).get("effective_date") or ""),
        now=now,
        transport=transport,
    )
