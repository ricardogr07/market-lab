from __future__ import annotations

import hashlib
import os
import re
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime
from importlib import resources
from pathlib import Path
from typing import Any

import psycopg
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb

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

POSTGRES_DSN_ENV = "MARKETLAB_PAPER_POSTGRES_DSN"
_MIGRATIONS_PACKAGE = "marketlab.paper.persistence.migrations"
_MIGRATION_PATTERN = re.compile(r"^(?P<version>[1-9][0-9]*)_(?P<name>[a-z0-9_]+)\\.sql$")
_MIGRATION_LOCK_KEY = int.from_bytes(
    hashlib.sha256(b"marketlab.paper.postgres.migrations").digest()[:8],
    byteorder="big",
    signed=True,
)


class PostgreSQLMigrationError(RuntimeError):
    """Raised when the paper PostgreSQL schema cannot be upgraded safely."""


@dataclass(frozen=True, slots=True)
class PostgreSQLMigration:
    version: int
    name: str
    sql: str
    checksum: str


def postgres_dsn_from_environment() -> str:
    """Return the one supported PostgreSQL configuration surface without logging it."""

    dsn = os.environ.get(POSTGRES_DSN_ENV, "").strip()
    if dsn == "":
        raise ValueError(f"{POSTGRES_DSN_ENV} must be set for paper.persistence_backend='postgres'.")
    return dsn


def load_postgres_migrations() -> tuple[PostgreSQLMigration, ...]:
    """Load immutable, numbered SQL migrations packaged with MarketLab."""

    migrations: list[PostgreSQLMigration] = []
    for resource in resources.files(_MIGRATIONS_PACKAGE).iterdir():
        match = _MIGRATION_PATTERN.match(resource.name)
        if match is None:
            continue
        sql = resource.read_text(encoding="utf-8")
        migrations.append(
            PostgreSQLMigration(
                version=int(match.group("version")),
                name=resource.name,
                sql=sql,
                checksum=hashlib.sha256(sql.encode("utf-8")).hexdigest(),
            )
        )
    migrations.sort(key=lambda migration: migration.version)
    versions = [migration.version for migration in migrations]
    if not migrations:
        raise PostgreSQLMigrationError("No packaged PostgreSQL paper migrations were found.")
    if len(set(versions)) != len(versions):
        raise PostgreSQLMigrationError("PostgreSQL paper migration versions must be unique.")
    return tuple(migrations)


def _connect(dsn: str, *, autocommit: bool = False) -> Any:
    return psycopg.connect(dsn, autocommit=autocommit, row_factory=dict_row)


def _create_migration_ledger(connection: Any) -> None:
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS marketlab_paper_schema_migrations (
            version INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            checksum TEXT NOT NULL,
            applied_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
        )
        """
    )


def apply_postgres_migrations(
    *,
    dsn: str | None = None,
    migrations: Iterable[PostgreSQLMigration] | None = None,
) -> int:
    """Apply new paper schema migrations under a PostgreSQL advisory lock.

    Historical migration checksums are verified on every invocation. The runner
    only moves forward; restoring a database and adding a corrective migration
    are the supported rollback path.
    """

    resolved_dsn = dsn if dsn is not None else postgres_dsn_from_environment()
    ordered_migrations = tuple(migrations if migrations is not None else load_postgres_migrations())
    versions = [migration.version for migration in ordered_migrations]
    if not ordered_migrations or versions != sorted(versions) or len(set(versions)) != len(versions):
        raise PostgreSQLMigrationError("PostgreSQL paper migrations must be unique and ordered.")

    connection = _connect(resolved_dsn, autocommit=True)
    locked = False
    try:
        connection.execute("SELECT pg_advisory_lock(%s)", (_MIGRATION_LOCK_KEY,))
        locked = True
        _create_migration_ledger(connection)
        rows = connection.execute(
            "SELECT version, name, checksum FROM marketlab_paper_schema_migrations ORDER BY version"
        ).fetchall()
        applied = {int(row["version"]): row for row in rows}
        known_versions = set(versions)
        unknown_versions = sorted(set(applied) - known_versions)
        if unknown_versions:
            raise PostgreSQLMigrationError(
                "Database contains PostgreSQL paper migrations unavailable to this build: "
                + ", ".join(str(version) for version in unknown_versions)
            )

        for migration in ordered_migrations:
            recorded = applied.get(migration.version)
            if recorded is not None:
                if (
                    str(recorded["name"]) != migration.name
                    or str(recorded["checksum"]) != migration.checksum
                ):
                    raise PostgreSQLMigrationError(
                        "PostgreSQL paper migration checksum mismatch for "
                        f"version {migration.version}; historical migrations are append-only."
                    )
                continue
            with connection.transaction():
                connection.execute(migration.sql)
                connection.execute(
                    """
                    INSERT INTO marketlab_paper_schema_migrations (version, name, checksum)
                    VALUES (%s, %s, %s)
                    """,
                    (migration.version, migration.name, migration.checksum),
                )
        return ordered_migrations[-1].version
    finally:
        if locked:
            connection.execute("SELECT pg_advisory_unlock(%s)", (_MIGRATION_LOCK_KEY,))
        connection.close()


def migrate_paper_postgres_database(config: ExperimentConfig) -> int:
    """Run the explicit paper database migration command for a PostgreSQL config."""

    if config.paper.persistence_backend != "postgres":
        raise ValueError("paper-db-migrate requires paper.persistence_backend='postgres'.")
    return apply_postgres_migrations()


def _row_payload(row: dict[str, Any] | None) -> dict[str, Any] | None:
    if row is None:
        return None
    payload = row["payload_json"]
    if not isinstance(payload, dict):
        raise TypeError("PostgreSQL paper payload_json must decode to an object.")
    return dict(payload)


def _snapshot_artifacts(paths: list[Path]) -> dict[Path, bytes | None]:
    return {path: path.read_bytes() if path.exists() else None for path in paths}


def _restore_artifacts(snapshots: dict[Path, bytes | None]) -> None:
    for path, payload in snapshots.items():
        if payload is None:
            if path.exists():
                path.unlink()
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)


def _ensure_identical_payload(
    *,
    row: dict[str, Any] | None,
    payload: dict[str, str],
    conflict_target: str,
) -> bool:
    if row is None:
        return False
    existing = _row_payload(row)
    if existing != payload:
        raise PaperDeploymentRegistryConflictError(
            f"Hosted execution idempotency conflict for {conflict_target}."
        )
    return True


class PostgreSQLPaperTradeRepository(PaperTradeRepository):
    def __init__(
        self,
        store: PaperStateStore,
        connection: Any,
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
        payloads: list[dict[str, Any]] = []
        for row in rows:
            payload = _row_payload(row)
            if payload is not None:
                payloads.append(payload)
        return payloads

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
            "SELECT payload_json FROM paper_proposals WHERE proposal_id = %s",
            (proposal_id,),
        ).fetchone()
        return _row_payload(row)

    def get_evidence(self, trade_date: str) -> dict[str, Any] | None:
        row = self._connection.execute(
            "SELECT payload_json FROM paper_evidence WHERE trade_date = %s",
            (trade_date,),
        ).fetchone()
        return _row_payload(row)

    def get_submission(self, trade_date: str) -> dict[str, Any] | None:
        row = self._connection.execute(
            "SELECT payload_json FROM paper_submissions WHERE trade_date = %s",
            (trade_date,),
        ).fetchone()
        return _row_payload(row)

    def save_evidence(self, evidence: dict[str, Any]) -> Path:
        trade_date = str(evidence["effective_date"])
        proposal_id = str(evidence["proposal_id"])
        self._connection.execute(
            """
            INSERT INTO paper_evidence (trade_date, proposal_id, payload_json)
            VALUES (%s, %s, %s)
            ON CONFLICT(trade_date) DO UPDATE SET
                proposal_id = EXCLUDED.proposal_id,
                payload_json = EXCLUDED.payload_json
            """,
            (trade_date, proposal_id, Jsonb(evidence)),
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
            VALUES (%s, %s, %s, %s)
            ON CONFLICT(proposal_id) DO UPDATE SET
                effective_date = EXCLUDED.effective_date,
                created_at = EXCLUDED.created_at,
                payload_json = EXCLUDED.payload_json
            """,
            (proposal_id, trade_date, created_at, Jsonb(proposal)),
        )
        proposal_path = self._store.trade_proposal_path(trade_date)
        inbox_path = self._store.inbox_proposal_path(proposal_id)
        self._artifact_writes[proposal_path] = dict(proposal)
        self._artifact_writes[inbox_path] = dict(proposal)
        return proposal_path

    def save_approval(self, *, trade_date: str, approval: dict[str, Any]) -> Path:
        self._save_trade_payload("paper_approvals", trade_date, approval)
        path = self._store.trade_approval_path(trade_date)
        self._artifact_writes[path] = dict(approval)
        return path

    def save_submission(self, *, trade_date: str, submission: dict[str, Any]) -> Path:
        self._save_trade_payload("paper_submissions", trade_date, submission)
        path = self._store.trade_submission_path(trade_date)
        self._artifact_writes[path] = dict(submission)
        return path

    def _save_trade_payload(
        self,
        table_name: str,
        trade_date: str,
        payload: dict[str, Any],
    ) -> None:
        proposal_id = str(payload["proposal_id"])
        self._connection.execute(
            f"""
            INSERT INTO {table_name} (trade_date, proposal_id, payload_json)
            VALUES (%s, %s, %s)
            ON CONFLICT(trade_date) DO UPDATE SET
                proposal_id = EXCLUDED.proposal_id,
                payload_json = EXCLUDED.payload_json
            """,
            (trade_date, proposal_id, Jsonb(payload)),
        )

    def save_order_status(self, *, trade_date: str, order_status: dict[str, Any]) -> Path:
        self._connection.execute(
            """
            INSERT INTO paper_order_statuses (trade_date, payload_json)
            VALUES (%s, %s)
            ON CONFLICT(trade_date) DO UPDATE SET
                payload_json = EXCLUDED.payload_json
            """,
            (trade_date, Jsonb(order_status)),
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
            self._connection.execute("DELETE FROM paper_submissions WHERE trade_date = %s", (trade_date,))
            self._connection.execute("DELETE FROM paper_order_statuses WHERE trade_date = %s", (trade_date,))
            self._connection.commit()
        except Exception:
            self._connection.rollback()
            for path, backup_path in reversed(renamed_paths):
                if backup_path.exists():
                    backup_path.rename(path)
            raise


class PostgreSQLPaperStatusRepository(PaperStatusRepository):
    def __init__(
        self,
        store: PaperStateStore,
        connection: Any,
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
            VALUES (1, %s)
            ON CONFLICT(singleton_key) DO UPDATE SET
                payload_json = EXCLUDED.payload_json
            """,
            (Jsonb(payload),),
        )
        self._artifact_writes[self.status_path] = dict(payload)
        return self.status_path


class PostgreSQLPaperDeploymentRegistry(PaperDeploymentRegistry):
    def __init__(self, config: ExperimentConfig) -> None:
        self._dsn = postgres_dsn_from_environment()

    def _record_deployment(self, connection: Any, context: PaperHostedExecutionContext) -> None:
        payload = context.as_metadata()
        row = connection.execute(
            """
            INSERT INTO paper_deployment_records (
                environment, deployment_id, phase, execution_id, idempotency_key, payload_json
            )
            VALUES (%s, %s, %s, %s, %s, %s)
            ON CONFLICT (environment, deployment_id, phase, execution_id) DO NOTHING
            RETURNING payload_json
            """,
            (
                context.environment,
                context.deployment_id,
                context.phase,
                context.execution_id,
                context.idempotency_key,
                Jsonb(payload),
            ),
        ).fetchone()
        if row is None:
            row = connection.execute(
                """
                SELECT payload_json
                FROM paper_deployment_records
                WHERE environment = %s
                  AND deployment_id = %s
                  AND phase = %s
                  AND execution_id = %s
                """,
                (
                    context.environment,
                    context.deployment_id,
                    context.phase,
                    context.execution_id,
                ),
            ).fetchone()
        _ensure_identical_payload(
            row=row,
            payload=payload,
            conflict_target=(
                f"deployment {context.environment}/{context.deployment_id}/"
                f"{context.phase}/{context.execution_id}"
            ),
        )

    def record_deployment(
        self,
        context: PaperHostedExecutionContext,
    ) -> PaperDeploymentRecord:
        connection = _connect(self._dsn, autocommit=True)
        try:
            with connection.transaction():
                self._record_deployment(connection, context)
        finally:
            connection.close()
        return PaperDeploymentRecord.from_context(context)

    def record_phase_run(
        self,
        context: PaperHostedExecutionContext,
    ) -> PaperPhaseRunRecord:
        payload = context.as_metadata()
        connection = _connect(self._dsn, autocommit=True)
        try:
            with connection.transaction():
                row = connection.execute(
                    """
                    INSERT INTO paper_phase_run_records (idempotency_key, phase, payload_json)
                    VALUES (%s, %s, %s)
                    ON CONFLICT (idempotency_key) DO NOTHING
                    RETURNING payload_json
                    """,
                    (context.idempotency_key, context.phase, Jsonb(payload)),
                ).fetchone()
                if row is None:
                    row = connection.execute(
                        """
                        SELECT payload_json
                        FROM paper_phase_run_records
                        WHERE idempotency_key = %s
                        """,
                        (context.idempotency_key,),
                    ).fetchone()
                _ensure_identical_payload(
                    row=row,
                    payload=payload,
                    conflict_target=f"idempotency key {context.idempotency_key}",
                )
                self._record_deployment(connection, context)
        finally:
            connection.close()
        return PaperPhaseRunRecord.from_context(context)


class PostgreSQLPaperUnitOfWork(PaperUnitOfWork):
    def __init__(self, config: ExperimentConfig) -> None:
        self._store = PaperStateStore(config)
        self._connection = _connect(postgres_dsn_from_environment())
        self._artifact_writes: dict[Path, dict[str, Any]] = {}
        self._trades = PostgreSQLPaperTradeRepository(
            self._store,
            self._connection,
            self._artifact_writes,
        )
        self._status = PostgreSQLPaperStatusRepository(
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
            self._connection.commit()
            self._committed = True
        except Exception:
            self._connection.rollback()
            _restore_artifacts(snapshots)
            raise
        finally:
            self._artifact_writes.clear()

    def rollback(self) -> None:
        self._connection.rollback()
        self._artifact_writes.clear()

    def __enter__(self) -> PostgreSQLPaperUnitOfWork:
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        try:
            if exc_type is not None or not self._committed:
                self.rollback()
        finally:
            self._connection.close()


class PostgreSQLPaperUnitOfWorkFactory(PaperUnitOfWorkFactory):
    def __init__(self, config: ExperimentConfig) -> None:
        self._config = config

    def __call__(self) -> PaperUnitOfWork:
        return PostgreSQLPaperUnitOfWork(self._config)


def build_postgres_paper_uow_factory(config: ExperimentConfig) -> PaperUnitOfWorkFactory:
    return PostgreSQLPaperUnitOfWorkFactory(config)


def build_postgres_paper_deployment_registry(config: ExperimentConfig) -> PaperDeploymentRegistry:
    return PostgreSQLPaperDeploymentRegistry(config)
