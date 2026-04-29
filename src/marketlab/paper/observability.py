from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from marketlab.log import (
    ExecutionContext,
    child_execution_context,
    create_execution_context,
    current_execution_context,
)


def _first_non_empty(*values: object) -> str | None:
    for value in values:
        text = str(value or "").strip()
        if text != "":
            return text
    return None


def root_execution_context(
    *,
    deployment: str,
    phase: str,
    proposal_id: str | None = None,
    trade_date: str | None = None,
    provider: str | None = None,
    details: Mapping[str, Any] | None = None,
) -> ExecutionContext:
    return create_execution_context(
        deployment=deployment,
        phase=phase,
        proposal_id=proposal_id,
        trade_date=trade_date,
        provider=provider,
        details=details,
    )


def paper_execution_context(
    execution_context: ExecutionContext | None = None,
    *,
    phase: str,
    deployment: str | None = None,
    trade_date: str | None = None,
    status: Mapping[str, Any] | None = None,
    proposal: Mapping[str, Any] | None = None,
    approval: Mapping[str, Any] | None = None,
    submission: Mapping[str, Any] | None = None,
    provider: str | None = None,
    outcome: str | None = None,
    duration_ms: float | None = None,
    details: Mapping[str, Any] | None = None,
    refresh_execution_id: bool = False,
) -> ExecutionContext:
    base_context = execution_context or current_execution_context()
    return child_execution_context(
        base_context,
        phase=phase,
        deployment=deployment,
        trade_date=_first_non_empty(
            trade_date,
            (submission or {}).get("trade_date"),
            (approval or {}).get("trade_date"),
            (proposal or {}).get("effective_date"),
            (status or {}).get("trade_date"),
            (status or {}).get("market_date"),
        ),
        proposal_id=_first_non_empty(
            (proposal or {}).get("proposal_id"),
            (approval or {}).get("proposal_id"),
            (submission or {}).get("proposal_id"),
            (status or {}).get("proposal_id"),
        ),
        order_id=_first_non_empty((submission or {}).get("order_id")),
        provider=_first_non_empty(
            provider,
            (approval or {}).get("provider"),
            (proposal or {}).get("broker"),
            (proposal or {}).get("data_provider"),
        ),
        outcome=_first_non_empty(
            outcome,
            (approval or {}).get("approval_status"),
            (submission or {}).get("status"),
            (status or {}).get("status"),
        ),
        duration_ms=duration_ms,
        details=details,
        refresh_execution_id=refresh_execution_id,
    )
