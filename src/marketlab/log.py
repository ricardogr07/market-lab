from __future__ import annotations

import json
import logging
import sys
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from typing import Any
from uuid import uuid4

STRUCTURED_LOG_KEYS = (
    "timestamp",
    "level",
    "logger",
    "message",
    "event",
    "execution_id",
    "correlation_id",
    "phase",
    "deployment",
    "trade_date",
    "proposal_id",
    "order_id",
    "provider",
    "outcome",
    "duration_ms",
    "details",
)


@dataclass(frozen=True, slots=True)
class ExecutionContext:
    execution_id: str
    correlation_id: str
    phase: str | None = None
    deployment: str | None = None
    trade_date: str | None = None
    proposal_id: str | None = None
    order_id: str | None = None
    provider: str | None = None
    outcome: str | None = None
    duration_ms: float | None = None
    details: dict[str, Any] | None = None


_CURRENT_EXECUTION_CONTEXT: ContextVar[ExecutionContext | None] = ContextVar(
    "marketlab_execution_context",
    default=None,
)


class StructuredLogFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "timestamp": datetime.fromtimestamp(record.created, UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "event": getattr(record, "event", None),
            "execution_id": getattr(record, "execution_id", None),
            "correlation_id": getattr(record, "correlation_id", None),
            "phase": getattr(record, "phase", None),
            "deployment": getattr(record, "deployment", None),
            "trade_date": getattr(record, "trade_date", None),
            "proposal_id": getattr(record, "proposal_id", None),
            "order_id": getattr(record, "order_id", None),
            "provider": getattr(record, "provider", None),
            "outcome": getattr(record, "outcome", None),
            "duration_ms": getattr(record, "duration_ms", None),
            "details": self._details(record),
        }
        return json.dumps(payload, default=_json_default)

    def _details(self, record: logging.LogRecord) -> dict[str, Any] | None:
        details = _json_mapping(getattr(record, "details", None))
        if (
            record.exc_info
            and record.exc_info[0] is not None
            and record.exc_info[1] is not None
        ):
            if details is None:
                details = {}
            details["exception"] = self.formatException(record.exc_info)
        if record.stack_info:
            if details is None:
                details = {}
            details["stack"] = self.formatStack(record.stack_info)
        return details


class DynamicStderrStreamHandler(logging.Handler):
    terminator = "\n"

    def emit(self, record: logging.LogRecord) -> None:
        try:
            message = self.format(record)
            stream = sys.stderr
            stream.write(message + self.terminator)
            self.flush()
        except Exception:
            self.handleError(record)

    def flush(self) -> None:
        stream = sys.stderr
        if hasattr(stream, "flush"):
            stream.flush()


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, set):
        return sorted(value)
    return str(value)


def _json_mapping(details: Any) -> dict[str, Any] | None:
    if details is None:
        return None
    if isinstance(details, Mapping):
        return {str(key): value for key, value in details.items()}
    return {"value": details}


def _log_level(level: int | str) -> int:
    if isinstance(level, int):
        return level
    resolved = logging.getLevelNamesMapping().get(str(level).upper())
    if resolved is None:
        raise ValueError(f"Unsupported log level: {level}")
    return int(resolved)


def configure_logging(level: int | str = logging.INFO) -> None:
    handler = DynamicStderrStreamHandler()
    handler.setFormatter(StructuredLogFormatter())
    logging.basicConfig(
        level=_log_level(level),
        handlers=[handler],
        force=True,
    )


def create_execution_context(
    *,
    phase: str | None = None,
    deployment: str | None = None,
    trade_date: str | None = None,
    proposal_id: str | None = None,
    order_id: str | None = None,
    provider: str | None = None,
    outcome: str | None = None,
    duration_ms: float | None = None,
    details: Mapping[str, Any] | None = None,
    correlation_id: str | None = None,
    execution_id: str | None = None,
) -> ExecutionContext:
    resolved_execution_id = execution_id or uuid4().hex
    resolved_correlation_id = correlation_id or resolved_execution_id
    return ExecutionContext(
        execution_id=resolved_execution_id,
        correlation_id=resolved_correlation_id,
        phase=phase,
        deployment=deployment,
        trade_date=trade_date,
        proposal_id=proposal_id,
        order_id=order_id,
        provider=provider,
        outcome=outcome,
        duration_ms=duration_ms,
        details=_json_mapping(details),
    )


def child_execution_context(
    context: ExecutionContext | None = None,
    *,
    phase: str | None = None,
    deployment: str | None = None,
    trade_date: str | None = None,
    proposal_id: str | None = None,
    order_id: str | None = None,
    provider: str | None = None,
    outcome: str | None = None,
    duration_ms: float | None = None,
    details: Mapping[str, Any] | None = None,
    refresh_execution_id: bool = False,
) -> ExecutionContext:
    base_context = context or current_execution_context() or create_execution_context()
    return ExecutionContext(
        execution_id=uuid4().hex if refresh_execution_id else base_context.execution_id,
        correlation_id=base_context.correlation_id,
        phase=phase if phase is not None else base_context.phase,
        deployment=deployment if deployment is not None else base_context.deployment,
        trade_date=trade_date if trade_date is not None else base_context.trade_date,
        proposal_id=proposal_id if proposal_id is not None else base_context.proposal_id,
        order_id=order_id if order_id is not None else base_context.order_id,
        provider=provider if provider is not None else base_context.provider,
        outcome=outcome if outcome is not None else base_context.outcome,
        duration_ms=duration_ms if duration_ms is not None else base_context.duration_ms,
        details=_json_mapping(details) if details is not None else base_context.details,
    )


def current_execution_context() -> ExecutionContext | None:
    return _CURRENT_EXECUTION_CONTEXT.get()


@contextmanager
def bind_execution_context(context: ExecutionContext | None) -> Iterator[ExecutionContext | None]:
    token = _CURRENT_EXECUTION_CONTEXT.set(context)
    try:
        yield context
    finally:
        _CURRENT_EXECUTION_CONTEXT.reset(token)


def duration_ms_since(start_time: float) -> float:
    return round((perf_counter() - start_time) * 1000.0, 3)


def emit_structured_log(
    logger: logging.Logger,
    level: int,
    message: str,
    *,
    event: str,
    execution_context: ExecutionContext | None = None,
    exc_info: BaseException | tuple[type[BaseException], BaseException, Any] | bool | None = None,
) -> None:
    context = execution_context or current_execution_context()
    logger.log(
        level,
        message,
        extra={
            "event": event,
            "execution_id": context.execution_id if context is not None else None,
            "correlation_id": context.correlation_id if context is not None else None,
            "phase": context.phase if context is not None else None,
            "deployment": context.deployment if context is not None else None,
            "trade_date": context.trade_date if context is not None else None,
            "proposal_id": context.proposal_id if context is not None else None,
            "order_id": context.order_id if context is not None else None,
            "provider": context.provider if context is not None else None,
            "outcome": context.outcome if context is not None else None,
            "duration_ms": context.duration_ms if context is not None else None,
            "details": context.details if context is not None else None,
        },
        exc_info=exc_info,
    )
