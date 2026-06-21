from __future__ import annotations

import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any

from marketlab.config import ExperimentConfig
from marketlab.paper.contracts import (
    PaperDeploymentRecord,
    PaperDeploymentRegistry,
    PaperDeploymentRegistryConflictError,
    PaperHostedExecutionContext,
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
        self._committed = False

    @property
    def trades(self) -> PaperTradeRepository:
        return self._trades

    @property
    def status(self) -> PaperStatusRepository:
        return self._status

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
