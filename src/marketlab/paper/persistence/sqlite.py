from __future__ import annotations

import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any, cast

from marketlab.config import ExperimentConfig
from marketlab.paper.contracts import (
    PaperDeploymentRecord,
    PaperDeploymentRegistry,
    PaperDeploymentRegistryConflictError,
    PaperHostedExecutionContext,
    PaperOutboxConflictError,
    PaperOutboxDeliveryStatus,
    PaperOutboxRecord,
    PaperOutboxRepository,
    PaperPhaseRunRecord,
    PaperStatusRepository,
    PaperTradeRepository,
    PaperUnitOfWork,
    PaperUnitOfWorkFactory,
)
from marketlab.paper.core import _now_utc
from marketlab.paper.state import PaperStateStore, _json_dump

_SCHEMA_STATEMENTS = (
    """
    CREATE TABLE IF NOT EXISTS paper_proposals (
        proposal_id TEXT PRIMARY KEY,
        effective_date TEXT NOT NULL,
        created_at TEXT NOT NULL,
        payload_json TEXT NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS paper_evidence (
        trade_date TEXT PRIMARY KEY,
        proposal_id TEXT NOT NULL,
        payload_json TEXT NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS paper_approvals (
        trade_date TEXT PRIMARY KEY,
        proposal_id TEXT NOT NULL,
        payload_json TEXT NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS paper_submissions (
        trade_date TEXT PRIMARY KEY,
        proposal_id TEXT NOT NULL,
        payload_json TEXT NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS paper_order_statuses (
        trade_date TEXT PRIMARY KEY,
        payload_json TEXT NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS paper_status (
        singleton_key INTEGER PRIMARY KEY CHECK (singleton_key = 1),
        payload_json TEXT NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS paper_deployment_records (
        environment TEXT NOT NULL,
        deployment_id TEXT NOT NULL,
        phase TEXT NOT NULL,
        execution_id TEXT NOT NULL,
        idempotency_key TEXT NOT NULL,
        payload_json TEXT NOT NULL,
        PRIMARY KEY (environment, deployment_id, phase, execution_id)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS paper_phase_run_records (
        idempotency_key TEXT PRIMARY KEY,
        phase TEXT NOT NULL,
        payload_json TEXT NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS paper_outbox (
        message_id TEXT PRIMARY KEY,
        event_type TEXT NOT NULL,
        payload_json TEXT NOT NULL,
        created_at TEXT NOT NULL,
        delivery_status TEXT NOT NULL,
        delivery_attempts INTEGER NOT NULL,
        delivered_at TEXT,
        last_error TEXT
    )
    """,
    """
    CREATE INDEX IF NOT EXISTS paper_outbox_pending_idx
        ON paper_outbox (delivery_status, created_at, message_id)
    """,
    """
    CREATE INDEX IF NOT EXISTS paper_outbox_pending_event_idx
        ON paper_outbox (delivery_status, event_type, created_at, message_id)
    """,
)


def _payload_json(payload: dict[str, Any]) -> str:
    return json.dumps(payload, sort_keys=True)


def _row_payload(row: sqlite3.Row | None) -> dict[str, Any] | None:
    if row is None:
        return None
    return json.loads(str(row["payload_json"]))


def _ensure_schema(connection: sqlite3.Connection) -> None:
    for statement in _SCHEMA_STATEMENTS:
        connection.execute(statement)


def _metadata_json(payload: dict[str, str]) -> str:
    return json.dumps(payload, sort_keys=True)


def _ensure_identical_payload(
    *,
    row: sqlite3.Row | None,
    payload: dict[str, str],
    conflict_target: str,
) -> bool:
    if row is None:
        return False
    existing = json.loads(str(row["payload_json"]))
    if existing != payload:
        raise PaperDeploymentRegistryConflictError(
            f"Hosted execution idempotency conflict for {conflict_target}."
        )
    return True


def _snapshot_artifacts(paths: list[Path]) -> dict[Path, bytes | None]:
    snapshots: dict[Path, bytes | None] = {}
    for path in paths:
        snapshots[path] = path.read_bytes() if path.exists() else None
    return snapshots


def _restore_artifacts(snapshots: dict[Path, bytes | None]) -> None:
    for path, payload in snapshots.items():
        if payload is None:
            if path.exists():
                path.unlink()
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)


class SQLitePaperTradeRepository(PaperTradeRepository):
    def __init__(
        self,
        store: PaperStateStore,
        connection: sqlite3.Connection,
        artifact_writes: dict[Path, dict[str, Any]],
    ) -> None:
        self._store = store
        self._connection = connection
        self._artifact_writes = artifact_writes

    def list_proposals(self) -> list[dict[str, Any]]:
        rows = self._connection.execute(
            """
            SELECT payload_json
            FROM paper_proposals
            ORDER BY effective_date DESC, created_at DESC, proposal_id DESC
            """
        ).fetchall()
        return [json.loads(str(row["payload_json"])) for row in rows]

    def get_latest_proposal(self) -> dict[str, Any] | None:
        row = self._connection.execute(
            """
            SELECT payload_json
            FROM paper_proposals
            ORDER BY effective_date DESC, created_at DESC, proposal_id DESC
            LIMIT 1
            """
        ).fetchone()
        return _row_payload(row)

    def get_proposal(self, proposal_id: str) -> dict[str, Any] | None:
        row = self._connection.execute(
            "SELECT payload_json FROM paper_proposals WHERE proposal_id = ?",
            (proposal_id,),
        ).fetchone()
        return _row_payload(row)

    def get_evidence(self, trade_date: str) -> dict[str, Any] | None:
        row = self._connection.execute(
            "SELECT payload_json FROM paper_evidence WHERE trade_date = ?",
            (trade_date,),
        ).fetchone()
        return _row_payload(row)

    def get_submission(self, trade_date: str) -> dict[str, Any] | None:
        row = self._connection.execute(
            "SELECT payload_json FROM paper_submissions WHERE trade_date = ?",
            (trade_date,),
        ).fetchone()
        return _row_payload(row)

    def save_evidence(self, evidence: dict[str, Any]) -> Path:
        trade_date = str(evidence["effective_date"])
        proposal_id = str(evidence["proposal_id"])
        self._connection.execute(
            """
            INSERT INTO paper_evidence (trade_date, proposal_id, payload_json)
            VALUES (?, ?, ?)
            ON CONFLICT(trade_date) DO UPDATE SET
                proposal_id = excluded.proposal_id,
                payload_json = excluded.payload_json
            """,
            (trade_date, proposal_id, _payload_json(evidence)),
        )
        path = self._store.trade_evidence_path(trade_date)
        self._artifact_writes[path] = dict(evidence)
        return path

    def save_proposal(self, proposal: dict[str, Any]) -> Path:
        trade_date = str(proposal["effective_date"])
        proposal_id = str(proposal["proposal_id"])
        created_at = str(proposal.get("created_at", ""))
        self._connection.execute(
            """
            INSERT INTO paper_proposals (proposal_id, effective_date, created_at, payload_json)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(proposal_id) DO UPDATE SET
                effective_date = excluded.effective_date,
                created_at = excluded.created_at,
                payload_json = excluded.payload_json
            """,
            (proposal_id, trade_date, created_at, _payload_json(proposal)),
        )
        proposal_path = self._store.trade_proposal_path(trade_date)
        inbox_path = self._store.inbox_proposal_path(proposal_id)
        self._artifact_writes[proposal_path] = dict(proposal)
        self._artifact_writes[inbox_path] = dict(proposal)
        return proposal_path

    def save_approval(self, *, trade_date: str, approval: dict[str, Any]) -> Path:
        proposal_id = str(approval["proposal_id"])
        self._connection.execute(
            """
            INSERT INTO paper_approvals (trade_date, proposal_id, payload_json)
            VALUES (?, ?, ?)
            ON CONFLICT(trade_date) DO UPDATE SET
                proposal_id = excluded.proposal_id,
                payload_json = excluded.payload_json
            """,
            (trade_date, proposal_id, _payload_json(approval)),
        )
        path = self._store.trade_approval_path(trade_date)
        self._artifact_writes[path] = dict(approval)
        return path

    def save_submission(self, *, trade_date: str, submission: dict[str, Any]) -> Path:
        proposal_id = str(submission["proposal_id"])
        self._connection.execute(
            """
            INSERT INTO paper_submissions (trade_date, proposal_id, payload_json)
            VALUES (?, ?, ?)
            ON CONFLICT(trade_date) DO UPDATE SET
                proposal_id = excluded.proposal_id,
                payload_json = excluded.payload_json
            """,
            (trade_date, proposal_id, _payload_json(submission)),
        )
        path = self._store.trade_submission_path(trade_date)
        self._artifact_writes[path] = dict(submission)
        return path

    def save_order_status(self, *, trade_date: str, order_status: dict[str, Any]) -> Path:
        self._connection.execute(
            """
            INSERT INTO paper_order_statuses (trade_date, payload_json)
            VALUES (?, ?)
            ON CONFLICT(trade_date) DO UPDATE SET
                payload_json = excluded.payload_json
            """,
            (trade_date, _payload_json(order_status)),
        )
        path = self._store.trade_order_status_path(trade_date)
        self._artifact_writes[path] = dict(order_status)
        return path

    def proposal_path(self, proposal_id: str) -> Path:
        return self._store.inbox_proposal_path(proposal_id)

    def trade_evidence_path(self, trade_date: str) -> Path:
        return self._store.trade_evidence_path(trade_date)

    def trade_submission_path(self, trade_date: str) -> Path:
        return self._store.trade_submission_path(trade_date)

    def trade_order_status_path(self, trade_date: str) -> Path:
        return self._store.trade_order_status_path(trade_date)

    def order_status_path_exists(self, trade_date: str) -> bool:
        path = self._store.trade_order_status_path(trade_date)
        return path in self._artifact_writes or path.exists()

    def backup_submission_attempt_artifacts(
        self,
        *,
        trade_date: str,
        now: datetime | None = None,
    ) -> None:
        timestamp = _now_utc(now).strftime("%Y%m%dT%H%M%S%fZ")
        renamed_paths: list[tuple[Path, Path]] = []
        for path in (
            self._store.trade_submission_path(trade_date),
            self._store.trade_order_status_path(trade_date),
            self._store.trade_order_preview_path(trade_date),
            self._store.trade_account_snapshot_path(trade_date),
        ):
            if not path.exists():
                continue
            backup_path = path.with_name(f"{path.stem}.retry-backup.{timestamp}.bak")
            path.rename(backup_path)
            renamed_paths.append((path, backup_path))
        try:
            self._connection.execute(
                "DELETE FROM paper_submissions WHERE trade_date = ?",
                (trade_date,),
            )
            self._connection.execute(
                "DELETE FROM paper_order_statuses WHERE trade_date = ?",
                (trade_date,),
            )
            if self._connection.in_transaction:
                self._connection.commit()
        except Exception:
            if self._connection.in_transaction:
                self._connection.rollback()
            for path, backup_path in reversed(renamed_paths):
                if backup_path.exists():
                    backup_path.rename(path)
            raise


class SQLitePaperStatusRepository(PaperStatusRepository):
    def __init__(
        self,
        store: PaperStateStore,
        connection: sqlite3.Connection,
        artifact_writes: dict[Path, dict[str, Any]],
    ) -> None:
        self._store = store
        self._connection = connection
        self._artifact_writes = artifact_writes

    @property
    def status_path(self) -> Path:
        return self._store.status_path

    def read_status(self) -> dict[str, Any] | None:
        row = self._connection.execute(
            "SELECT payload_json FROM paper_status WHERE singleton_key = 1"
        ).fetchone()
        return _row_payload(row)

    def write_status(self, payload: dict[str, Any]) -> Path:
        self._connection.execute(
            """
            INSERT INTO paper_status (singleton_key, payload_json)
            VALUES (1, ?)
            ON CONFLICT(singleton_key) DO UPDATE SET
                payload_json = excluded.payload_json
            """,
            (_payload_json(payload),),
        )
        self._artifact_writes[self.status_path] = dict(payload)
        return self.status_path


def _outbox_record(row: sqlite3.Row | None) -> PaperOutboxRecord | None:
    if row is None:
        return None
    payload = json.loads(str(row["payload_json"]))
    if not isinstance(payload, dict):
        raise TypeError("SQLite paper outbox payload_json must decode to an object.")
    return PaperOutboxRecord(
        message_id=str(row["message_id"]),
        event_type=str(row["event_type"]),
        payload=payload,
        created_at=str(row["created_at"]),
        delivery_status=cast(PaperOutboxDeliveryStatus, str(row["delivery_status"])),
        delivery_attempts=int(row["delivery_attempts"]),
        delivered_at=None if row["delivered_at"] is None else str(row["delivered_at"]),
        last_error=None if row["last_error"] is None else str(row["last_error"]),
    )


class SQLitePaperOutboxRepository(PaperOutboxRepository):
    def __init__(self, connection: sqlite3.Connection) -> None:
        self._connection = connection

    def enqueue(
        self,
        *,
        message_id: str,
        event_type: str,
        payload: dict[str, Any],
        created_at: str,
    ) -> PaperOutboxRecord:
        record = PaperOutboxRecord(
            message_id=message_id,
            event_type=event_type,
            payload=dict(payload),
            created_at=created_at,
        )
        inserted = self._connection.execute(
            """
            INSERT OR IGNORE INTO paper_outbox (
                message_id, event_type, payload_json, created_at,
                delivery_status, delivery_attempts, delivered_at, last_error
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                record.message_id,
                record.event_type,
                _payload_json(record.payload),
                record.created_at,
                record.delivery_status,
                record.delivery_attempts,
                record.delivered_at,
                record.last_error,
            ),
        )
        if inserted.rowcount == 1:
            return record
        existing = self.get(message_id)
        if existing is None:  # pragma: no cover - the primary key conflict should be visible immediately.
            raise RuntimeError("Paper outbox enqueue did not persist or return an existing record.")
        if existing.event_type != event_type or existing.payload != payload:
            raise PaperOutboxConflictError(
                f"Paper outbox message ID {message_id!r} was reused with different event data."
            )
        return existing

    def get(self, message_id: str) -> PaperOutboxRecord | None:
        row = self._connection.execute(
            """
            SELECT message_id, event_type, payload_json, created_at,
                   delivery_status, delivery_attempts, delivered_at, last_error
            FROM paper_outbox
            WHERE message_id = ?
            """,
            (message_id,),
        ).fetchone()
        return _outbox_record(row)

    def list_pending(
        self,
        *,
        limit: int = 100,
        event_types: frozenset[str] | None = None,
    ) -> list[PaperOutboxRecord]:
        if limit < 1:
            raise ValueError("Paper outbox limit must be at least 1.")
        if event_types == frozenset():
            return []
        event_type_clause = ""
        parameters: list[str | int] = []
        if event_types is not None:
            placeholders = ", ".join("?" for _ in event_types)
            event_type_clause = f" AND event_type IN ({placeholders})"
            parameters.extend(sorted(event_types))
        parameters.append(limit)
        rows = self._connection.execute(
            f"""
            SELECT message_id, event_type, payload_json, created_at,
                   delivery_status, delivery_attempts, delivered_at, last_error
            FROM paper_outbox
            WHERE delivery_status IN ('pending', 'failed')
            {event_type_clause}
            ORDER BY created_at, message_id
            LIMIT ?
            """,
            parameters,
        ).fetchall()
        return [record for row in rows if (record := _outbox_record(row)) is not None]

    def mark_delivered(
        self,
        *,
        message_id: str,
        delivered_at: str,
    ) -> PaperOutboxRecord:
        record = self._required_record(message_id)
        if record.delivery_status == "delivered":
            return record
        updated = PaperOutboxRecord(
            message_id=record.message_id,
            event_type=record.event_type,
            payload=record.payload,
            created_at=record.created_at,
            delivery_status="delivered",
            delivery_attempts=record.delivery_attempts + 1,
            delivered_at=delivered_at,
        )
        self._save_delivery(updated)
        return updated

    def mark_failed(
        self,
        *,
        message_id: str,
        error: str,
    ) -> PaperOutboxRecord:
        record = self._required_record(message_id)
        if record.delivery_status == "delivered":
            return record
        normalized_error = error.strip()
        if normalized_error == "":
            raise ValueError("Paper outbox error must not be empty.")
        updated = PaperOutboxRecord(
            message_id=record.message_id,
            event_type=record.event_type,
            payload=record.payload,
            created_at=record.created_at,
            delivery_status="failed",
            delivery_attempts=record.delivery_attempts + 1,
            last_error=normalized_error,
        )
        self._save_delivery(updated)
        return updated

    def _required_record(self, message_id: str) -> PaperOutboxRecord:
        record = self.get(message_id)
        if record is None:
            raise KeyError(f"Unknown paper outbox message_id: {message_id}")
        return record

    def _save_delivery(self, record: PaperOutboxRecord) -> None:
        self._connection.execute(
            """
            UPDATE paper_outbox
            SET delivery_status = ?, delivery_attempts = ?, delivered_at = ?, last_error = ?
            WHERE message_id = ?
            """,
            (
                record.delivery_status,
                record.delivery_attempts,
                record.delivered_at,
                record.last_error,
                record.message_id,
            ),
        )


class SQLitePaperDeploymentRegistry(PaperDeploymentRegistry):
    def __init__(self, config: ExperimentConfig) -> None:
        self._db_path = config.paper_sqlite_db_path
        self._db_path.parent.mkdir(parents=True, exist_ok=True)

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self._db_path)
        connection.row_factory = sqlite3.Row
        _ensure_schema(connection)
        return connection

    def record_deployment(
        self,
        context: PaperHostedExecutionContext,
    ) -> PaperDeploymentRecord:
        payload = context.as_metadata()
        payload_json = _metadata_json(payload)
        connection = self._connect()
        try:
            with connection:
                row = connection.execute(
                    """
                    SELECT payload_json
                    FROM paper_deployment_records
                    WHERE environment = ?
                      AND deployment_id = ?
                      AND phase = ?
                      AND execution_id = ?
                    """,
                    (
                        context.environment,
                        context.deployment_id,
                        context.phase,
                        context.execution_id,
                    ),
                ).fetchone()
                if not _ensure_identical_payload(
                    row=row,
                    payload=payload,
                    conflict_target=(
                        f"deployment {context.environment}/{context.deployment_id}/"
                        f"{context.phase}/{context.execution_id}"
                    ),
                ):
                    connection.execute(
                        """
                        INSERT INTO paper_deployment_records (
                            environment,
                            deployment_id,
                            phase,
                            execution_id,
                            idempotency_key,
                            payload_json
                        )
                        VALUES (?, ?, ?, ?, ?, ?)
                        """,
                        (
                            context.environment,
                            context.deployment_id,
                            context.phase,
                            context.execution_id,
                            context.idempotency_key,
                            payload_json,
                        ),
                    )
        finally:
            connection.close()
        return PaperDeploymentRecord.from_context(context)

    def record_phase_run(
        self,
        context: PaperHostedExecutionContext,
    ) -> PaperPhaseRunRecord:
        payload = context.as_metadata()
        payload_json = _metadata_json(payload)
        connection = self._connect()
        try:
            with connection:
                row = connection.execute(
                    """
                    SELECT payload_json
                    FROM paper_phase_run_records
                    WHERE idempotency_key = ?
                    """,
                    (context.idempotency_key,),
                ).fetchone()
                if not _ensure_identical_payload(
                    row=row,
                    payload=payload,
                    conflict_target=f"idempotency key {context.idempotency_key}",
                ):
                    connection.execute(
                        """
                        INSERT INTO paper_phase_run_records (
                            idempotency_key,
                            phase,
                            payload_json
                        )
                        VALUES (?, ?, ?)
                        """,
                        (
                            context.idempotency_key,
                            context.phase,
                            payload_json,
                        ),
                    )
                row = connection.execute(
                    """
                    SELECT payload_json
                    FROM paper_deployment_records
                    WHERE environment = ?
                      AND deployment_id = ?
                      AND phase = ?
                      AND execution_id = ?
                    """,
                    (
                        context.environment,
                        context.deployment_id,
                        context.phase,
                        context.execution_id,
                    ),
                ).fetchone()
                if not _ensure_identical_payload(
                    row=row,
                    payload=payload,
                    conflict_target=(
                        f"deployment {context.environment}/{context.deployment_id}/"
                        f"{context.phase}/{context.execution_id}"
                    ),
                ):
                    connection.execute(
                        """
                        INSERT INTO paper_deployment_records (
                            environment,
                            deployment_id,
                            phase,
                            execution_id,
                            idempotency_key,
                            payload_json
                        )
                        VALUES (?, ?, ?, ?, ?, ?)
                        """,
                        (
                            context.environment,
                            context.deployment_id,
                            context.phase,
                            context.execution_id,
                            context.idempotency_key,
                            payload_json,
                        ),
                    )
        finally:
            connection.close()
        return PaperPhaseRunRecord.from_context(context)


class SQLitePaperUnitOfWork(PaperUnitOfWork):
    def __init__(self, config: ExperimentConfig) -> None:
        self._store = PaperStateStore(config)
        self._db_path = config.paper_sqlite_db_path
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._connection = sqlite3.connect(self._db_path, isolation_level=None)
        self._connection.row_factory = sqlite3.Row
        _ensure_schema(self._connection)
        self._artifact_writes: dict[Path, dict[str, Any]] = {}
        self._trades = SQLitePaperTradeRepository(
            self._store,
            self._connection,
            self._artifact_writes,
        )
        self._status = SQLitePaperStatusRepository(
            self._store,
            self._connection,
            self._artifact_writes,
        )
        self._outbox = SQLitePaperOutboxRepository(self._connection)
        self._committed = False

    @property
    def trades(self) -> PaperTradeRepository:
        return self._trades

    @property
    def status(self) -> PaperStatusRepository:
        return self._status

    @property
    def outbox(self) -> PaperOutboxRepository:
        return self._outbox

    def commit(self) -> None:
        snapshots = _snapshot_artifacts(list(self._artifact_writes))
        try:
            for path, payload in self._artifact_writes.items():
                _json_dump(path, payload)
            if self._connection.in_transaction:
                self._connection.commit()
            self._committed = True
        except Exception:
            if self._connection.in_transaction:
                self._connection.rollback()
            _restore_artifacts(snapshots)
            raise
        finally:
            self._artifact_writes.clear()

    def rollback(self) -> None:
        if self._connection.in_transaction:
            self._connection.rollback()
        self._artifact_writes.clear()

    def __enter__(self) -> SQLitePaperUnitOfWork:
        self._connection.execute("BEGIN")
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        try:
            if exc_type is not None or not self._committed:
                self.rollback()
        finally:
            self._connection.close()


class SQLitePaperUnitOfWorkFactory(PaperUnitOfWorkFactory):
    def __init__(self, config: ExperimentConfig) -> None:
        self._config = config

    def __call__(self) -> PaperUnitOfWork:
        return SQLitePaperUnitOfWork(self._config)


def build_sqlite_paper_uow_factory(config: ExperimentConfig) -> PaperUnitOfWorkFactory:
    return SQLitePaperUnitOfWorkFactory(config)


def build_sqlite_paper_deployment_registry(config: ExperimentConfig) -> PaperDeploymentRegistry:
    return SQLitePaperDeploymentRegistry(config)
