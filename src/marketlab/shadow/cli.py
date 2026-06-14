from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from marketlab.data.panel import load_panel_csv
from marketlab.shadow import (
    ShadowDecisionEvaluation,
    ShadowDecisionRequest,
    run_shadow_decision,
    run_shadow_scheduler,
    shadow_bars_from_panel,
    verify_shadow_contract,
)
from marketlab.shadow.decision import ShadowDecisionStatus


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="phase9-shadow-decision")
    parser.add_argument("--config", required=True)
    parser.add_argument("--evaluation", required=True)
    parser.add_argument("--panel")
    parser.add_argument("--as-of")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    contract = verify_shadow_contract(args.config)
    evaluation = _load_evaluation(Path(args.evaluation))
    panel_path = Path(args.panel).resolve() if args.panel else contract.config.prepared_panel_path
    panel = load_panel_csv(panel_path)
    as_of = _parse_as_of(args.as_of)
    result = run_shadow_decision(
        ShadowDecisionRequest(
            contract=contract,
            as_of=as_of,
            bars=shadow_bars_from_panel(panel),
        ),
        evaluator=lambda context: evaluation,
    )
    print(result.path)
    return 0


def scheduler_main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="phase9-shadow-scheduler")
    parser.add_argument("--config", required=True)
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--as-of")
    args = parser.parse_args(argv)
    if not args.once:
        parser.error("--once is required; the scheduler does not run a resident loop.")
    result = run_shadow_scheduler(
        args.config,
        as_of=_parse_as_of(args.as_of),
    )
    payload = {
        "attempts": [str(write.path) for write in result.attempts],
        "decision": str(result.decision.path) if result.decision is not None else None,
        "decision_evidence": (
            str(result.decision_evidence.path)
            if result.decision_evidence is not None
            else None
        ),
        "label_evidence": [str(write.path) for write in result.label_evidence],
    }
    print(json.dumps(payload, sort_keys=True))
    return 0


def _load_evaluation(path: Path) -> ShadowDecisionEvaluation:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Unable to read shadow evaluation: {path}") from exc
    if not isinstance(payload, dict):
        raise RuntimeError("Shadow evaluation must contain a JSON object.")
    input_payload = payload.get("input_payload", {})
    if not isinstance(input_payload, dict):
        raise RuntimeError("Shadow evaluation input_payload must contain a JSON object.")
    return ShadowDecisionEvaluation(
        status=_status_value(payload),
        selection_source=_string_value(payload, "selection_source"),
        fallback_mode=_string_value(payload, "fallback_mode"),
        target_allocation=_optional_float(payload.get("target_allocation")),
        reason=_optional_string(payload.get("reason")),
        input_payload=input_payload,
    )


def _parse_as_of(value: str | None) -> datetime:
    if value is None:
        return datetime.now(UTC)
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise RuntimeError("--as-of must be an ISO-8601 datetime.") from exc
    if parsed.tzinfo is None:
        raise RuntimeError("--as-of must include an explicit timezone.")
    return parsed.astimezone(UTC)


def _string_value(payload: dict[str, Any], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or value.strip() == "":
        raise RuntimeError(f"Shadow evaluation {key} must be a non-empty string.")
    return value


def _status_value(payload: dict[str, Any]) -> ShadowDecisionStatus:
    value = _string_value(payload, "status")
    if value == "success":
        return "success"
    if value == "skipped":
        return "skipped"
    if value == "failed":
        return "failed"
    raise RuntimeError(f"Unsupported shadow evaluation status: {value!r}.")


def _optional_string(value: object) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise RuntimeError("Shadow evaluation reason must be a string or null.")
    return value


def _optional_float(value: object) -> float | None:
    if value is None:
        return None
    try:
        return float(str(value))
    except (TypeError, ValueError) as exc:
        raise RuntimeError(
            "Shadow evaluation target_allocation must be numeric or null."
        ) from exc
