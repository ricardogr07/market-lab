from __future__ import annotations

import hashlib
import threading
import time
from dataclasses import replace
from pathlib import Path

import pytest
from tests._postgres import postgres_dsn_from_environment, reset_postgres_database

psycopg = pytest.importorskip("psycopg")

from marketlab.paper.persistence.postgres import (  # noqa: E402
    POSTGRES_DSN_ENV,
    PostgreSQLMigration,
    PostgreSQLMigrationError,
    apply_postgres_migrations,
    load_postgres_migrations,
)

ROOT = Path(__file__).resolve().parents[2]
FIXTURE_SQL = ROOT / "tests" / "fixtures" / "postgres" / "restored_paper_fixture.sql"


def _dsn() -> str:
    dsn = postgres_dsn_from_environment()
    if dsn is None:
        pytest.skip(f"{POSTGRES_DSN_ENV} is required for PostgreSQL integration tests.")
    return dsn


@pytest.fixture(autouse=True)
def _clean_postgres_database() -> None:
    dsn = _dsn()
    reset_postgres_database(dsn)
    try:
        yield
    finally:
        reset_postgres_database(dsn)


def _ledger_rows(dsn: str) -> list[tuple[int, str, str]]:
    with psycopg.connect(dsn) as connection:
        rows = connection.execute(
            """
            SELECT version, name, checksum
            FROM marketlab_paper_schema_migrations
            ORDER BY version
            """
        ).fetchall()
    return [(int(row[0]), str(row[1]), str(row[2])) for row in rows]


def test_postgres_migrations_install_clean_schema_and_rerun_as_a_noop() -> None:
    dsn = _dsn()
    migrations = load_postgres_migrations()

    assert apply_postgres_migrations(dsn=dsn) == migrations[-1].version
    first_rows = _ledger_rows(dsn)
    assert [row[0] for row in first_rows] == [migration.version for migration in migrations]

    assert apply_postgres_migrations(dsn=dsn) == migrations[-1].version
    assert _ledger_rows(dsn) == first_rows


def test_postgres_migrations_upgrade_a_restored_qqq_fixture_in_order() -> None:
    dsn = _dsn()
    migrations = load_postgres_migrations()
    assert len(migrations) >= 2

    assert apply_postgres_migrations(dsn=dsn, migrations=(migrations[0],)) == 1
    with psycopg.connect(dsn) as connection:
        connection.execute(FIXTURE_SQL.read_text(encoding="utf-8"))
        connection.commit()

    assert apply_postgres_migrations(dsn=dsn, migrations=migrations) == migrations[-1].version
    with psycopg.connect(dsn) as connection:
        row = connection.execute(
            "SELECT payload_json->>'symbol' FROM paper_proposals WHERE proposal_id = %s",
            ("restored-qqq-proposal",),
        ).fetchone()
        index_row = connection.execute(
            "SELECT 1 FROM pg_indexes WHERE indexname = 'paper_proposals_order_idx'"
        ).fetchone()
    assert row == ("QQQ",)
    assert index_row == (1,)


def test_postgres_migrations_reject_tampered_historical_checksum() -> None:
    dsn = _dsn()
    migrations = load_postgres_migrations()
    apply_postgres_migrations(dsn=dsn)
    tampered_first = replace(migrations[0], sql=migrations[0].sql + "\n-- edited\n", checksum="0" * 64)

    with pytest.raises(PostgreSQLMigrationError, match="checksum mismatch"):
        apply_postgres_migrations(dsn=dsn, migrations=(tampered_first, *migrations[1:]))


def test_postgres_migrations_serialize_concurrent_runners_with_an_advisory_lock() -> None:
    dsn = _dsn()
    first = load_postgres_migrations()[0]
    slow_second_sql = "SELECT pg_sleep(0.25);"
    slow_second = PostgreSQLMigration(
        version=2,
        name="002_lock_serialization.sql",
        sql=slow_second_sql,
        checksum=hashlib.sha256(slow_second_sql.encode("utf-8")).hexdigest(),
    )
    barrier = threading.Barrier(2)
    failures: list[Exception] = []
    durations: list[float] = []

    def _apply() -> None:
        try:
            barrier.wait()
            started = time.monotonic()
            apply_postgres_migrations(dsn=dsn, migrations=(first, slow_second))
            durations.append(time.monotonic() - started)
        except Exception as exc:  # pragma: no cover - assertion below reports it.
            failures.append(exc)

    threads = [threading.Thread(target=_apply), threading.Thread(target=_apply)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert failures == []
    assert len(durations) == 2
    assert max(durations) >= 0.25
    assert [row[0] for row in _ledger_rows(dsn)] == [1, 2]
