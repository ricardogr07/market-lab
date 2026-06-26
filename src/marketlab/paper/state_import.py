from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from marketlab.config import ExperimentConfig

ImportAction = Literal["imported", "skipped", "artifact_only", "conflicting"]


class PaperStateImportError(RuntimeError):
    """Raised when filesystem paper state cannot be imported safely."""


def postgres_dsn_from_environment() -> str:
    from marketlab.paper.persistence.postgres import (
        postgres_dsn_from_environment as _dsn,
    )

    return _dsn()


def _connect(dsn: str, *, autocommit: bool = False) -> Any:
    from marketlab.paper.persistence.postgres import _connect as _postgres_connect

    return _postgres_connect(dsn, autocommit=autocommit)


def _jsonb(payload: dict[str, Any]) -> Any:
    from psycopg.types.json import Jsonb

    return Jsonb(payload)


@dataclass(frozen=True, slots=True)
class _SourceRecord:
    surface: str
    key: str
    target: str
    payload: dict[str, Any]
    paths: tuple[Path, ...]


@dataclass(frozen=True, slots=True)
class _ArtifactRecord:
    surface: str
    key: str
    target: str
    path: Path


def import_paper_state(
    config: ExperimentConfig,
    *,
    source_state_dir: str | Path,
    source_inbox_dir: str | Path,
    apply: bool = False,
    report_path: str | Path | None = None,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Import QQQ filesystem paper state into the configured PostgreSQL backend.

    The command is intentionally narrow: it reads local JSON review artifacts,
    preflights the target database for idempotent conflicts, and writes only
    canonical PostgreSQL control state when explicitly asked to apply.
    """

    if config.paper.persistence_backend != "postgres":
        raise ValueError("paper-state-import requires paper.persistence_backend='postgres'.")

    started_at = _timestamp(now)
    source_state_root = Path(source_state_dir)
    source_inbox_root = Path(source_inbox_dir)
    source_reports_root = source_state_root.parent / "reports"
    if not source_state_root.exists():
        raise PaperStateImportError(f"Source state directory does not exist: {source_state_root}")
    if not source_inbox_root.exists():
        raise PaperStateImportError(f"Source inbox directory does not exist: {source_inbox_root}")

    source_records = _collect_source_records(source_state_root, source_inbox_root)
    artifact_records = _collect_artifact_records(source_state_root, source_reports_root)
    dsn = postgres_dsn_from_environment()
    actions = _preflight_records(dsn, source_records)
    conflicts = [entry for entry in actions if entry["action"] == "conflicting"]
    if conflicts:
        raise PaperStateImportError(
            "Conflicting paper state already exists in PostgreSQL: "
            + ", ".join(f"{entry['surface']}:{entry['key']}" for entry in conflicts)
        )

    if apply:
        _apply_records(dsn, source_records, actions)

    manifest_entries = [
        *_record_manifest_entries(source_records, actions),
        *_artifact_manifest_entries(artifact_records),
    ]
    completed_at = _timestamp(now)
    aggregate_checksum = _aggregate_checksum(manifest_entries)
    report = {
        "command": "paper-state-import",
        "mode": "apply" if apply else "dry-run",
        "experiment_name": config.experiment_name,
        "source_state_dir": _path_text(source_state_root),
        "source_inbox_dir": _path_text(source_inbox_root),
        "source_reports_dir": _path_text(source_reports_root),
        "started_at": started_at,
        "completed_at": completed_at,
        "aggregate_checksum": aggregate_checksum,
        "counts": _counts(manifest_entries),
        "manifest": manifest_entries,
    }
    if report_path is not None:
        output_path = Path(report_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        report["report_path"] = _path_text(output_path)
    return report


def _timestamp(now: datetime | None) -> str:
    value = datetime.now(UTC) if now is None else now
    if value.tzinfo is None:
        value = value.replace(tzinfo=UTC)
    return value.astimezone(UTC).isoformat()


def _collect_source_records(
    source_state_root: Path,
    source_inbox_root: Path,
) -> list[_SourceRecord]:
    proposals = _proposal_records(source_state_root, source_inbox_root)
    records = [*proposals]
    records.extend(
        _trade_records(
            source_state_root,
            surface="evidence",
            filename="evidence.json",
            table="paper_evidence",
            identity_field="effective_date",
        )
    )
    records.extend(
        _trade_records(
            source_state_root,
            surface="approvals",
            filename="approval.json",
            table="paper_approvals",
            identity_field="trade_date",
        )
    )
    records.extend(
        _trade_records(
            source_state_root,
            surface="submissions",
            filename="submission.json",
            table="paper_submissions",
            identity_field="trade_date",
        )
    )
    records.extend(_order_status_records(source_state_root))
    status_path = source_state_root / "status.json"
    if status_path.exists():
        records.append(
            _SourceRecord(
                surface="status",
                key="status",
                target="paper_status",
                payload=_load_json_object(status_path),
                paths=(status_path,),
            )
        )
    return records


def _proposal_records(source_state_root: Path, source_inbox_root: Path) -> list[_SourceRecord]:
    by_id: dict[str, tuple[dict[str, Any], list[Path]]] = {}
    proposal_paths = [
        *sorted(source_inbox_root.glob("*.json")),
        *sorted((source_state_root / "trades").glob("*/proposal.json")),
    ]
    for path in proposal_paths:
        payload = _load_json_object(path)
        proposal_id = _required_text(payload, "proposal_id", path)
        _required_text(payload, "effective_date", path)
        _required_text(payload, "created_at", path)
        existing = by_id.get(proposal_id)
        if existing is None:
            by_id[proposal_id] = (payload, [path])
            continue
        existing_payload, paths = existing
        if existing_payload != payload:
            raise PaperStateImportError(
                f"Duplicate proposal_id {proposal_id!r} has conflicting payloads."
            )
        paths.append(path)
    return [
        _SourceRecord(
            surface="proposals",
            key=proposal_id,
            target="paper_proposals",
            payload=payload,
            paths=tuple(paths),
        )
        for proposal_id, (payload, paths) in sorted(by_id.items())
    ]


def _trade_records(
    source_state_root: Path,
    *,
    surface: str,
    filename: str,
    table: str,
    identity_field: str,
) -> list[_SourceRecord]:
    records: dict[str, _SourceRecord] = {}
    for path in sorted((source_state_root / "trades").glob(f"*/{filename}")):
        payload = _load_json_object(path)
        trade_date = path.parent.name
        identity = _required_text(payload, identity_field, path)
        if identity != trade_date:
            raise PaperStateImportError(
                f"{path} has {identity_field}={identity!r}, expected {trade_date!r}."
            )
        _required_text(payload, "proposal_id", path)
        if trade_date in records:
            raise PaperStateImportError(f"Duplicate {surface} record for {trade_date}.")
        records[trade_date] = _SourceRecord(
            surface=surface,
            key=trade_date,
            target=table,
            payload=payload,
            paths=(path,),
        )
    return [records[key] for key in sorted(records)]


def _order_status_records(source_state_root: Path) -> list[_SourceRecord]:
    records: dict[str, _SourceRecord] = {}
    for path in sorted((source_state_root / "trades").glob("*/order_status.json")):
        payload = _load_json_object(path)
        trade_date = path.parent.name
        _required_text(payload, "status", path)
        if trade_date in records:
            raise PaperStateImportError(f"Duplicate order_status record for {trade_date}.")
        records[trade_date] = _SourceRecord(
            surface="order_statuses",
            key=trade_date,
            target="paper_order_statuses",
            payload=payload,
            paths=(path,),
        )
    return [records[key] for key in sorted(records)]


def _collect_artifact_records(
    source_state_root: Path,
    source_reports_root: Path,
) -> list[_ArtifactRecord]:
    records: list[_ArtifactRecord] = []
    for path in sorted((source_state_root / "notifications").glob("*.json")):
        records.append(
            _ArtifactRecord(
                surface="notifications",
                key=path.stem,
                target="blob_review_artifact",
                path=path,
            )
        )
    if source_reports_root.exists():
        for path in sorted(item for item in source_reports_root.rglob("*") if item.is_file()):
            records.append(
                _ArtifactRecord(
                    surface="reports",
                    key=path.relative_to(source_reports_root).as_posix(),
                    target="blob_review_artifact",
                    path=path,
                )
            )
    return records


def _load_json_object(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise PaperStateImportError(f"Malformed JSON in {path}: {exc.msg}") from exc
    if not isinstance(payload, dict):
        raise PaperStateImportError(f"Expected a JSON object in {path}.")
    return payload


def _required_text(payload: dict[str, Any], field: str, path: Path) -> str:
    value = str(payload.get(field, "")).strip()
    if value == "":
        raise PaperStateImportError(f"{path} is missing required field {field!r}.")
    return value


def _preflight_records(dsn: str, records: list[_SourceRecord]) -> list[dict[str, str]]:
    connection = _connect(dsn, autocommit=True)
    try:
        return [_preflight_record(connection, record) for record in records]
    finally:
        connection.close()


def _preflight_record(connection: Any, record: _SourceRecord) -> dict[str, str]:
    existing = _load_existing_payload(connection, record)
    action: ImportAction
    if existing is None:
        action = "imported"
    elif existing == record.payload:
        action = "skipped"
    else:
        action = "conflicting"
    return {
        "surface": record.surface,
        "key": record.key,
        "target": record.target,
        "action": action,
    }


def _load_existing_payload(connection: Any, record: _SourceRecord) -> dict[str, Any] | None:
    if record.surface == "proposals":
        row = connection.execute(
            "SELECT payload_json FROM paper_proposals WHERE proposal_id = %s",
            (record.key,),
        ).fetchone()
    elif record.surface == "evidence":
        row = connection.execute(
            "SELECT payload_json FROM paper_evidence WHERE trade_date = %s",
            (record.key,),
        ).fetchone()
    elif record.surface == "approvals":
        row = connection.execute(
            "SELECT payload_json FROM paper_approvals WHERE trade_date = %s",
            (record.key,),
        ).fetchone()
    elif record.surface == "submissions":
        row = connection.execute(
            "SELECT payload_json FROM paper_submissions WHERE trade_date = %s",
            (record.key,),
        ).fetchone()
    elif record.surface == "order_statuses":
        row = connection.execute(
            "SELECT payload_json FROM paper_order_statuses WHERE trade_date = %s",
            (record.key,),
        ).fetchone()
    elif record.surface == "status":
        row = connection.execute(
            "SELECT payload_json FROM paper_status WHERE singleton_key = 1"
        ).fetchone()
    else:  # pragma: no cover - source records are built from known surfaces.
        raise AssertionError(f"Unsupported import surface: {record.surface}")
    if row is None:
        return None
    payload = row["payload_json"]
    if not isinstance(payload, dict):
        raise PaperStateImportError(
            f"Existing PostgreSQL payload for {record.surface}:{record.key} is not an object."
        )
    return dict(payload)


def _apply_records(
    dsn: str,
    records: list[_SourceRecord],
    actions: list[dict[str, str]],
) -> None:
    action_by_record = {
        (action["surface"], action["key"]): action["action"] for action in actions
    }
    connection = _connect(dsn, autocommit=False)
    try:
        with connection.transaction():
            for record in records:
                if action_by_record[(record.surface, record.key)] != "imported":
                    continue
                _insert_record(connection, record)
    finally:
        connection.close()


def _insert_record(connection: Any, record: _SourceRecord) -> None:
    payload = _jsonb(record.payload)
    if record.surface == "proposals":
        connection.execute(
            """
            INSERT INTO paper_proposals (
                proposal_id, effective_date, created_at, payload_json
            )
            VALUES (%s, %s, %s, %s)
            """,
            (
                record.key,
                str(record.payload["effective_date"]),
                str(record.payload["created_at"]),
                payload,
            ),
        )
    elif record.surface == "evidence":
        connection.execute(
            """
            INSERT INTO paper_evidence (trade_date, proposal_id, payload_json)
            VALUES (%s, %s, %s)
            """,
            (record.key, str(record.payload["proposal_id"]), payload),
        )
    elif record.surface == "approvals":
        connection.execute(
            """
            INSERT INTO paper_approvals (trade_date, proposal_id, payload_json)
            VALUES (%s, %s, %s)
            """,
            (record.key, str(record.payload["proposal_id"]), payload),
        )
    elif record.surface == "submissions":
        connection.execute(
            """
            INSERT INTO paper_submissions (trade_date, proposal_id, payload_json)
            VALUES (%s, %s, %s)
            """,
            (record.key, str(record.payload["proposal_id"]), payload),
        )
    elif record.surface == "order_statuses":
        connection.execute(
            """
            INSERT INTO paper_order_statuses (trade_date, payload_json)
            VALUES (%s, %s)
            """,
            (record.key, payload),
        )
    elif record.surface == "status":
        connection.execute(
            """
            INSERT INTO paper_status (singleton_key, payload_json)
            VALUES (1, %s)
            """,
            (payload,),
        )
    else:  # pragma: no cover - source records are built from known surfaces.
        raise AssertionError(f"Unsupported import surface: {record.surface}")


def _record_manifest_entries(
    records: list[_SourceRecord],
    actions: list[dict[str, str]],
) -> list[dict[str, Any]]:
    action_by_record = {
        (action["surface"], action["key"]): action["action"] for action in actions
    }
    entries: list[dict[str, Any]] = []
    for record in records:
        for path in record.paths:
            entries.append(
                {
                    "surface": record.surface,
                    "key": record.key,
                    "target": record.target,
                    "source_path": _path_text(path),
                    "sha256": _file_checksum(path),
                    "action": action_by_record[(record.surface, record.key)],
                }
            )
    return sorted(entries, key=lambda entry: (entry["surface"], entry["key"], entry["source_path"]))


def _artifact_manifest_entries(records: list[_ArtifactRecord]) -> list[dict[str, Any]]:
    entries = [
        {
            "surface": record.surface,
            "key": record.key,
            "target": record.target,
            "source_path": _path_text(record.path),
            "sha256": _file_checksum(record.path),
            "action": "artifact_only",
        }
        for record in records
    ]
    return sorted(entries, key=lambda entry: (entry["surface"], entry["key"], entry["source_path"]))


def _file_checksum(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _aggregate_checksum(entries: list[dict[str, Any]]) -> str:
    payload = json.dumps(entries, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _counts(entries: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    counts: dict[str, dict[str, int]] = {}
    seen: set[tuple[str, str, str]] = set()
    for entry in entries:
        surface = str(entry["surface"])
        action = str(entry["action"])
        key = str(entry["key"])
        count_key = (surface, action, key)
        if count_key in seen:
            continue
        seen.add(count_key)
        counts.setdefault(
            surface,
            {
                "imported": 0,
                "skipped": 0,
                "artifact_only": 0,
                "conflicting": 0,
            },
        )
        counts[surface][action] += 1
    return counts


def _path_text(path: Path) -> str:
    return path.as_posix()
