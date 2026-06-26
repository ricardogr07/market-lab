from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

from marketlab.config import ExperimentConfig
from marketlab.paper.core import _paper_symbol, validate_paper_trading_config

_TRADE_ARTIFACTS = {
    "proposal": "proposal.json",
    "evidence": "evidence.json",
    "approval": "approval.json",
    "submission": "submission.json",
    "order_preview": "order_preview.json",
    "account_snapshot": "account_snapshot.json",
    "order_status": "order_status.json",
}
_EXPLANATION_STATUSES = frozenset({"accepted", "expected", "blocking"})


class PaperParityReportError(RuntimeError):
    """Raised when parity inputs cannot produce a safe comparison report."""


@dataclass(frozen=True, slots=True)
class _ArtifactComparison:
    surface: str
    key: str
    local_path: Path | None
    shadow_path: Path | None
    local_payload: Any
    shadow_payload: Any


def build_paper_parity_report(
    config: ExperimentConfig,
    *,
    local_state_dir: str | Path,
    shadow_state_dir: str | Path,
    start_date: str,
    end_date: str,
    min_trading_days: int = 10,
    explanations_path: str | Path | None = None,
    report_path: str | Path | None = None,
    markdown_path: str | Path | None = None,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Compare authoritative local QQQ artifacts with shadow/UAT artifacts."""

    validate_paper_trading_config(config)
    symbol = _paper_symbol(config)
    if symbol != "QQQ":
        raise ValueError("paper-parity-report is bounded to the QQQ paper config.")
    if min_trading_days < 1:
        raise PaperParityReportError("min_trading_days must be at least 1.")
    start = _parse_date(start_date, "start_date")
    end = _parse_date(end_date, "end_date")
    if start > end:
        raise PaperParityReportError("start_date must be on or before end_date.")

    local_root = Path(local_state_dir)
    shadow_root = Path(shadow_state_dir)
    _require_dir(local_root, "local_state_dir")
    _require_dir(shadow_root, "shadow_state_dir")
    explanations = _load_explanations(explanations_path)
    comparisons = _collect_comparisons(
        local_root=local_root,
        shadow_root=shadow_root,
        start=start,
        end=end,
    )
    differences = [
        _difference_entry(comparison, explanations)
        for comparison in comparisons
        if comparison.local_payload != comparison.shadow_payload
    ]
    observed_trade_dates = _observed_trade_dates(
        comparisons,
        start=start,
        end=end,
    )
    consecutive_days = _max_consecutive_weekdays(observed_trade_dates)
    evidence_window_passed = consecutive_days >= min_trading_days
    unresolved = [
        difference
        for difference in differences
        if difference["explanation_status"] not in {"accepted", "expected"}
    ]
    manifest = _manifest_entries(comparisons)
    report = {
        "command": "paper-parity-report",
        "experiment_name": config.experiment_name,
        "symbol": symbol,
        "local_state_dir": _path_text(local_root),
        "shadow_state_dir": _path_text(shadow_root),
        "window": {
            "start_date": start.isoformat(),
            "end_date": end.isoformat(),
            "min_trading_days": min_trading_days,
            "observed_trade_dates": [value.isoformat() for value in observed_trade_dates],
            "max_consecutive_weekdays": consecutive_days,
            "evidence_window_passed": evidence_window_passed,
        },
        "counts": _counts(comparisons, differences),
        "differences": differences,
        "unresolved_difference_count": len(unresolved),
        "accepted": evidence_window_passed and not unresolved,
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


def _collect_comparisons(
    *,
    local_root: Path,
    shadow_root: Path,
    start: date,
    end: date,
) -> list[_ArtifactComparison]:
    trade_dates = sorted(
        date_value
        for date_value in (_trade_dates(local_root) | _trade_dates(shadow_root))
        if start <= date_value <= end
    )
    comparisons: list[_ArtifactComparison] = []
    for trade_date in trade_dates:
        trade_key = trade_date.isoformat()
        for surface, filename in _TRADE_ARTIFACTS.items():
            comparisons.append(
                _comparison(
                    surface=surface,
                    key=trade_key,
                    local_path=local_root / "trades" / trade_key / filename,
                    shadow_path=shadow_root / "trades" / trade_key / filename,
                )
            )
    comparisons.append(
        _comparison(
            surface="status",
            key="latest",
            local_path=local_root / "status.json",
            shadow_path=shadow_root / "status.json",
        )
    )
    comparisons.append(
        _inventory_comparison(
            surface="notifications",
            local_root=local_root / "notifications",
            shadow_root=shadow_root / "notifications",
        )
    )
    comparisons.append(
        _inventory_comparison(
            surface="reports",
            local_root=local_root.parent / "reports",
            shadow_root=shadow_root.parent / "reports",
        )
    )
    return comparisons


def _comparison(
    *,
    surface: str,
    key: str,
    local_path: Path,
    shadow_path: Path,
) -> _ArtifactComparison:
    return _ArtifactComparison(
        surface=surface,
        key=key,
        local_path=local_path if local_path.exists() else None,
        shadow_path=shadow_path if shadow_path.exists() else None,
        local_payload=_load_json_if_exists(local_path),
        shadow_payload=_load_json_if_exists(shadow_path),
    )


def _inventory_comparison(
    *,
    surface: str,
    local_root: Path,
    shadow_root: Path,
) -> _ArtifactComparison:
    return _ArtifactComparison(
        surface=surface,
        key="inventory",
        local_path=local_root if local_root.exists() else None,
        shadow_path=shadow_root if shadow_root.exists() else None,
        local_payload=_inventory(local_root),
        shadow_payload=_inventory(shadow_root),
    )


def _load_json_if_exists(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise PaperParityReportError(f"Malformed JSON in {path}: {exc.msg}") from exc
    if not isinstance(payload, dict):
        raise PaperParityReportError(f"Expected a JSON object in {path}.")
    return payload


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


def _difference_entry(
    comparison: _ArtifactComparison,
    explanations: dict[str, dict[str, str]],
) -> dict[str, Any]:
    status = _difference_status(comparison)
    diff_id = _difference_id(comparison)
    explanation = explanations.get(diff_id, {})
    explanation_status = str(explanation.get("status", "unexplained")).strip()
    if explanation_status not in _EXPLANATION_STATUSES:
        explanation_status = "unexplained"
    return {
        "id": diff_id,
        "surface": comparison.surface,
        "key": comparison.key,
        "status": status,
        "local_path": None if comparison.local_path is None else _path_text(comparison.local_path),
        "shadow_path": None if comparison.shadow_path is None else _path_text(comparison.shadow_path),
        "explanation_status": explanation_status,
        "explanation": str(explanation.get("explanation", "")).strip(),
    }


def _difference_status(comparison: _ArtifactComparison) -> str:
    if comparison.local_payload is None:
        return "missing_local"
    if comparison.shadow_payload is None:
        return "missing_shadow"
    if comparison.surface in {"notifications", "reports"}:
        return "inventory_mismatch"
    return "payload_mismatch"


def _difference_id(comparison: _ArtifactComparison) -> str:
    payload = "|".join((comparison.surface, comparison.key, _difference_status(comparison)))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _manifest_entries(comparisons: list[_ArtifactComparison]) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for comparison in comparisons:
        entries.append(
            {
                "surface": comparison.surface,
                "key": comparison.key,
                "local_path": None
                if comparison.local_path is None
                else _path_text(comparison.local_path),
                "shadow_path": None
                if comparison.shadow_path is None
                else _path_text(comparison.shadow_path),
                "local_checksum": _payload_checksum(comparison.local_payload),
                "shadow_checksum": _payload_checksum(comparison.shadow_payload),
                "matched": comparison.local_payload == comparison.shadow_payload,
            }
        )
    return sorted(entries, key=lambda entry: (entry["surface"], entry["key"]))


def _observed_trade_dates(
    comparisons: list[_ArtifactComparison],
    *,
    start: date,
    end: date,
) -> list[date]:
    observed_surfaces: dict[date, set[str]] = {}
    for comparison in comparisons:
        if comparison.surface not in {"proposal", "evidence"}:
            continue
        try:
            trade_date = date.fromisoformat(comparison.key)
        except ValueError:
            continue
        if not (start <= trade_date <= end):
            continue
        if comparison.local_payload is not None and comparison.shadow_payload is not None:
            observed_surfaces.setdefault(trade_date, set()).add(comparison.surface)
    return sorted(
        trade_date
        for trade_date, surfaces in observed_surfaces.items()
        if surfaces == {"proposal", "evidence"}
    )


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


def _trade_dates(root: Path) -> set[date]:
    trades_root = root / "trades"
    if not trades_root.exists():
        return set()
    values: set[date] = set()
    for item in trades_root.iterdir():
        if not item.is_dir():
            continue
        try:
            values.add(date.fromisoformat(item.name))
        except ValueError:
            raise PaperParityReportError(f"Invalid trade-date directory: {item}") from None
    return values


def _load_explanations(path: str | Path | None) -> dict[str, dict[str, str]]:
    if path is None:
        return {}
    payload = _load_json_if_exists(Path(path))
    if payload is None:
        raise PaperParityReportError(f"Explanations file does not exist: {path}")
    differences = payload.get("differences", payload)
    if not isinstance(differences, dict):
        raise PaperParityReportError("Parity explanations must be a JSON object.")
    explanations: dict[str, dict[str, str]] = {}
    for key, value in differences.items():
        if not isinstance(value, dict):
            raise PaperParityReportError(f"Explanation for {key!r} must be an object.")
        explanations[str(key)] = {
            "status": str(value.get("status", "")),
            "explanation": str(value.get("explanation", "")),
        }
    return explanations


def _counts(
    comparisons: list[_ArtifactComparison],
    differences: list[dict[str, Any]],
) -> dict[str, Any]:
    by_surface: dict[str, dict[str, int]] = {}
    for comparison in comparisons:
        status = "matched" if comparison.local_payload == comparison.shadow_payload else "different"
        by_surface.setdefault(comparison.surface, {"matched": 0, "different": 0})[status] += 1
    return {
        "compared": len(comparisons),
        "matched": len(comparisons) - len(differences),
        "different": len(differences),
        "by_surface": by_surface,
    }


def _write_json_report(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_markdown_report(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# QQQ UAT Parity Report",
        "",
        f"- Experiment: {report['experiment_name']}",
        f"- Symbol: {report['symbol']}",
        f"- Window: {report['window']['start_date']} to {report['window']['end_date']}",
        f"- Evidence window passed: {report['window']['evidence_window_passed']}",
        f"- Accepted: {report['accepted']}",
        f"- Unresolved differences: {report['unresolved_difference_count']}",
        "",
        "## Differences",
        "",
    ]
    differences = report["differences"]
    if not differences:
        lines.append("No differences detected.")
    else:
        lines.extend(
            [
                "| id | surface | key | status | explanation_status |",
                "| --- | --- | --- | --- | --- |",
            ]
        )
        lines.extend(
            "| {id} | {surface} | {key} | {status} | {explanation_status} |".format(
                **difference
            )
            for difference in differences
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _parse_date(value: str, field_name: str) -> date:
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise PaperParityReportError(f"{field_name} must be an ISO-8601 date.") from exc


def _require_dir(path: Path, field_name: str) -> None:
    if not path.exists() or not path.is_dir():
        raise PaperParityReportError(f"{field_name} must be an existing directory: {path}")


def _payload_checksum(payload: Any) -> str | None:
    if payload is None:
        return None
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
