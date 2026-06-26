from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

from marketlab.config import ExperimentConfig
from marketlab.paper.core import _paper_symbol, validate_paper_trading_config

_REQUIRED_TRADE_ARTIFACTS = {
    "decision_proposal": "proposal.json",
    "decision_evidence": "evidence.json",
    "approval": "approval.json",
    "submission": "submission.json",
    "reconciliation": "order_status.json",
}
_ISSUE_FILES = {
    "alerts": "alerts.json",
    "dead_letters": "dead_letters.json",
    "failed_jobs": "failed_jobs.json",
    "non_terminal_orders": "non_terminal_orders.json",
}
_RESOLVED_STATUSES = frozenset({"accepted", "expected", "resolved"})


class PaperCloseoutReportError(RuntimeError):
    """Raised when closeout inputs cannot produce a safe report."""


@dataclass(frozen=True, slots=True)
class _TradeEvidence:
    trade_date: date
    surface: str
    path: Path
    payload: Any


def build_paper_closeout_report(
    config: ExperimentConfig,
    *,
    paper_prod_state_dir: str | Path,
    paper_prod_artifact_dir: str | Path,
    start_date: str,
    end_date: str,
    min_trading_days: int = 10,
    rollback_evidence_path: str | Path | None = None,
    report_path: str | Path | None = None,
    markdown_path: str | Path | None = None,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Build a deterministic QQQ paper-prod post-cutover closeout report."""

    validate_paper_trading_config(config)
    symbol = _paper_symbol(config)
    if symbol != "QQQ":
        raise ValueError("paper-closeout-report is bounded to the QQQ paper config.")
    if min_trading_days < 1:
        raise PaperCloseoutReportError("min_trading_days must be at least 1.")
    start = _parse_date(start_date, "start_date")
    end = _parse_date(end_date, "end_date")
    if start > end:
        raise PaperCloseoutReportError("start_date must be on or before end_date.")

    state_root = Path(paper_prod_state_dir)
    artifact_root = Path(paper_prod_artifact_dir)
    _require_dir(state_root, "paper_prod_state_dir")
    _require_dir(artifact_root, "paper_prod_artifact_dir")

    trade_dates = _trade_dates(state_root, start=start, end=end)
    trade_evidence, missing_evidence = _collect_trade_evidence(
        state_root=state_root,
        trade_dates=trade_dates,
    )
    notification_inventory = _inventory(artifact_root / "notifications")
    report_inventory = _inventory(artifact_root / "reports")
    observed_trade_dates = _observed_trade_dates(
        trade_evidence,
        notification_inventory=notification_inventory,
        report_inventory=report_inventory,
    )
    consecutive_days = _max_consecutive_weekdays(observed_trade_dates)
    evidence_window_passed = consecutive_days >= min_trading_days

    duplicate_submissions = _duplicate_submission_identifiers(trade_evidence)
    unresolved_operational_items = _unresolved_operational_items(artifact_root)
    rollback_evidence = _load_rollback_evidence(rollback_evidence_path)
    rollback_accepted = bool(rollback_evidence["accepted"])
    manifest = _manifest_entries(
        trade_evidence,
        notification_inventory=notification_inventory,
        report_inventory=report_inventory,
        rollback_evidence=rollback_evidence,
        unresolved_operational_items=unresolved_operational_items,
        duplicate_submissions=duplicate_submissions,
        missing_evidence=missing_evidence,
    )
    report = {
        "command": "paper-closeout-report",
        "experiment_name": config.experiment_name,
        "symbol": symbol,
        "paper_prod_state_dir": _path_text(state_root),
        "paper_prod_artifact_dir": _path_text(artifact_root),
        "window": {
            "start_date": start.isoformat(),
            "end_date": end.isoformat(),
            "min_trading_days": min_trading_days,
            "observed_trade_dates": [value.isoformat() for value in observed_trade_dates],
            "max_consecutive_weekdays": consecutive_days,
            "evidence_window_passed": evidence_window_passed,
        },
        "counts": {
            "trade_dates": len(trade_dates),
            "observed_trade_dates": len(observed_trade_dates),
            "missing_evidence": len(missing_evidence),
            "duplicate_broker_submissions": len(duplicate_submissions),
            "unresolved_operational_items": len(unresolved_operational_items),
        },
        "missing_evidence": missing_evidence,
        "duplicate_broker_submissions": duplicate_submissions,
        "unresolved_operational_items": unresolved_operational_items,
        "rollback_evidence": rollback_evidence,
        "accepted": (
            evidence_window_passed
            and not missing_evidence
            and not duplicate_submissions
            and not unresolved_operational_items
            and rollback_accepted
        ),
        "generated_at": _timestamp(now),
        "aggregate_checksum": _aggregate_checksum(manifest),
        "manifest": manifest,
    }
    if report_path is not None:
        report["report_path"] = _path_text(Path(report_path))
    if markdown_path is not None:
        report["markdown_path"] = _path_text(Path(markdown_path))
    if report_path is not None:
        _write_json_report(Path(report_path), report)
    if markdown_path is not None:
        _write_markdown_report(Path(markdown_path), report)
    return report


def _collect_trade_evidence(
    *,
    state_root: Path,
    trade_dates: list[date],
) -> tuple[list[_TradeEvidence], list[dict[str, str]]]:
    evidence: list[_TradeEvidence] = []
    missing: list[dict[str, str]] = []
    for trade_date in trade_dates:
        trade_key = trade_date.isoformat()
        trade_root = state_root / "trades" / trade_key
        for surface, filename in _REQUIRED_TRADE_ARTIFACTS.items():
            path = trade_root / filename
            if not path.exists():
                missing.append(
                    {
                        "trade_date": trade_key,
                        "surface": surface,
                        "path": _path_text(path),
                    }
                )
                continue
            evidence.append(
                _TradeEvidence(
                    trade_date=trade_date,
                    surface=surface,
                    path=path,
                    payload=_load_json_object(path),
                )
            )
    return evidence, missing


def _observed_trade_dates(
    evidence: list[_TradeEvidence],
    *,
    notification_inventory: list[dict[str, str]],
    report_inventory: list[dict[str, str]],
) -> list[date]:
    surfaces_by_date: dict[date, set[str]] = {}
    for item in evidence:
        surfaces_by_date.setdefault(item.trade_date, set()).add(item.surface)
    notification_dates = _inventory_dates(notification_inventory)
    report_dates = _inventory_dates(report_inventory)
    required_surfaces = set(_REQUIRED_TRADE_ARTIFACTS)
    return sorted(
        trade_date
        for trade_date, surfaces in surfaces_by_date.items()
        if surfaces == required_surfaces
        and trade_date in notification_dates
        and trade_date in report_dates
    )


def _inventory_dates(entries: list[dict[str, str]]) -> set[date]:
    values: set[date] = set()
    for entry in entries:
        path_text = entry["path"]
        for part in path_text.replace("\\", "/").split("/"):
            try:
                values.add(date.fromisoformat(part.removesuffix(".json").removesuffix(".md")))
            except ValueError:
                continue
    return values


def _duplicate_submission_identifiers(evidence: list[_TradeEvidence]) -> list[dict[str, Any]]:
    values: dict[str, list[dict[str, str]]] = {}
    for item in evidence:
        if item.surface != "submission" or not isinstance(item.payload, dict):
            continue
        for field_name in ("broker_order_id", "order_id", "client_order_id"):
            identifier = str(item.payload.get(field_name, "")).strip()
            if identifier == "":
                continue
            key = f"{field_name}:{identifier}"
            values.setdefault(key, []).append(
                {
                    "trade_date": item.trade_date.isoformat(),
                    "path": _path_text(item.path),
                }
            )
    return [
        {
            "identifier": key.split(":", maxsplit=1)[1],
            "field": key.split(":", maxsplit=1)[0],
            "occurrences": occurrences,
        }
        for key, occurrences in sorted(values.items())
        if len(occurrences) > 1
    ]


def _unresolved_operational_items(artifact_root: Path) -> list[dict[str, Any]]:
    unresolved: list[dict[str, Any]] = []
    for issue_type, filename in _ISSUE_FILES.items():
        path = artifact_root / filename
        if not path.exists():
            continue
        for index, item in enumerate(_load_issue_items(path)):
            status = str(item.get("status", "")).strip()
            if status not in _RESOLVED_STATUSES:
                unresolved.append(
                    {
                        "type": issue_type,
                        "index": index,
                        "status": status or "unresolved",
                        "path": _path_text(path),
                        "item": item,
                    }
                )
    return unresolved


def _load_issue_items(path: Path) -> list[dict[str, Any]]:
    payload = _load_json(path)
    if isinstance(payload, dict):
        items = payload.get("items", [])
    else:
        items = payload
    if not isinstance(items, list):
        raise PaperCloseoutReportError(f"Expected {path} to contain a JSON list or items list.")
    entries: list[dict[str, Any]] = []
    for index, item in enumerate(items):
        if not isinstance(item, dict):
            raise PaperCloseoutReportError(f"Expected item {index} in {path} to be a JSON object.")
        entries.append(item)
    return entries


def _load_rollback_evidence(path: str | Path | None) -> dict[str, Any]:
    if path is None:
        return {
            "path": None,
            "accepted": False,
            "status": "missing",
        }
    rollback_path = Path(path)
    if not rollback_path.exists():
        raise PaperCloseoutReportError(f"rollback_evidence file does not exist: {path}")
    payload = _load_json_object(rollback_path)
    status = str(payload.get("status", "")).strip()
    accepted = bool(payload.get("accepted") is True or status in _RESOLVED_STATUSES)
    return {
        "path": _path_text(rollback_path),
        "accepted": accepted,
        "status": status or ("accepted" if accepted else "unresolved"),
        "payload_checksum": _payload_checksum(payload),
    }


def _trade_dates(root: Path, *, start: date, end: date) -> list[date]:
    trades_root = root / "trades"
    if not trades_root.exists():
        return []
    values: list[date] = []
    for item in trades_root.iterdir():
        if not item.is_dir():
            continue
        try:
            trade_date = date.fromisoformat(item.name)
        except ValueError:
            raise PaperCloseoutReportError(f"Invalid trade-date directory: {item}") from None
        if start <= trade_date <= end:
            values.append(trade_date)
    return sorted(values)


def _inventory(root: Path) -> list[dict[str, str]]:
    if not root.exists():
        return []
    return [
        {
            "path": path.relative_to(root).as_posix(),
            "sha256": _file_checksum(path),
        }
        for path in sorted(item for item in root.rglob("*") if item.is_file())
    ]


def _manifest_entries(
    evidence: list[_TradeEvidence],
    *,
    notification_inventory: list[dict[str, str]],
    report_inventory: list[dict[str, str]],
    rollback_evidence: dict[str, Any],
    unresolved_operational_items: list[dict[str, Any]],
    duplicate_submissions: list[dict[str, Any]],
    missing_evidence: list[dict[str, str]],
) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = [
        {
            "surface": item.surface,
            "key": item.trade_date.isoformat(),
            "path": _path_text(item.path),
            "checksum": _payload_checksum(item.payload),
        }
        for item in evidence
    ]
    entries.extend(
        {
            "surface": "notifications",
            "key": entry["path"],
            "path": entry["path"],
            "checksum": entry["sha256"],
        }
        for entry in notification_inventory
    )
    entries.extend(
        {
            "surface": "reports",
            "key": entry["path"],
            "path": entry["path"],
            "checksum": entry["sha256"],
        }
        for entry in report_inventory
    )
    entries.append(
        {
            "surface": "rollback",
            "key": str(rollback_evidence["status"]),
            "path": rollback_evidence["path"],
            "checksum": rollback_evidence.get("payload_checksum"),
        }
    )
    entries.append(
        {
            "surface": "closeout-summary",
            "key": "gates",
            "checksum": _payload_checksum(
                {
                    "unresolved_operational_items": unresolved_operational_items,
                    "duplicate_submissions": duplicate_submissions,
                    "missing_evidence": missing_evidence,
                }
            ),
            "path": None,
        }
    )
    return sorted(entries, key=lambda entry: (str(entry["surface"]), str(entry["key"])))


def _max_consecutive_weekdays(values: list[date]) -> int:
    if not values:
        return 0
    observed = set(values)
    best = 0
    for start in values:
        if _previous_weekday(start) in observed:
            continue
        count = 0
        current = start
        while current in observed:
            count += 1
            current = _next_weekday(current)
        best = max(best, count)
    return best


def _previous_weekday(value: date) -> date:
    current = value - timedelta(days=1)
    while current.weekday() >= 5:
        current -= timedelta(days=1)
    return current


def _next_weekday(value: date) -> date:
    current = value + timedelta(days=1)
    while current.weekday() >= 5:
        current += timedelta(days=1)
    return current


def _load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise PaperCloseoutReportError(f"Malformed JSON in {path}: {exc.msg}") from exc


def _load_json_object(path: Path) -> dict[str, Any]:
    payload = _load_json(path)
    if not isinstance(payload, dict):
        raise PaperCloseoutReportError(f"Expected a JSON object in {path}.")
    return payload


def _write_json_report(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_markdown_report(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# QQQ Post-Cutover Closeout Report",
        "",
        f"- Experiment: {report['experiment_name']}",
        f"- Symbol: {report['symbol']}",
        f"- Window: {report['window']['start_date']} to {report['window']['end_date']}",
        f"- Evidence window passed: {report['window']['evidence_window_passed']}",
        f"- Rollback evidence accepted: {report['rollback_evidence']['accepted']}",
        f"- Accepted: {report['accepted']}",
        f"- Missing evidence: {report['counts']['missing_evidence']}",
        f"- Duplicate broker submissions: {report['counts']['duplicate_broker_submissions']}",
        f"- Unresolved operational items: {report['counts']['unresolved_operational_items']}",
        "",
        "## Observed Trade Dates",
        "",
    ]
    observed_dates = report["window"]["observed_trade_dates"]
    lines.extend(observed_dates if observed_dates else ["No complete trade dates observed."])
    lines.extend(["", "## Blocking Items", ""])
    if report["accepted"]:
        lines.append("No blocking closeout items detected.")
    else:
        for item in report["missing_evidence"]:
            lines.append(f"- missing {item['surface']} for {item['trade_date']}")
        for item in report["duplicate_broker_submissions"]:
            lines.append(f"- duplicate {item['field']} {item['identifier']}")
        for item in report["unresolved_operational_items"]:
            lines.append(f"- unresolved {item['type']} item at {item['path']}")
        if not report["rollback_evidence"]["accepted"]:
            lines.append("- rollback rehearsal evidence is missing or not accepted")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _parse_date(value: str, field_name: str) -> date:
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise PaperCloseoutReportError(f"{field_name} must be an ISO-8601 date.") from exc


def _require_dir(path: Path, field_name: str) -> None:
    if not path.exists() or not path.is_dir():
        raise PaperCloseoutReportError(f"{field_name} must be an existing directory: {path}")


def _payload_checksum(payload: Any) -> str:
    data = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def _aggregate_checksum(entries: list[dict[str, Any]]) -> str:
    data = json.dumps(entries, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def _file_checksum(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _timestamp(now: datetime | None) -> str:
    value = datetime.now(UTC) if now is None else now
    if value.tzinfo is None:
        value = value.replace(tzinfo=UTC)
    return value.astimezone(UTC).isoformat()


def _path_text(path: Path) -> str:
    return path.as_posix()
