from __future__ import annotations

import os

POSTGRES_DSN_ENV = "MARKETLAB_PAPER_POSTGRES_DSN"


def postgres_dsn_from_environment() -> str | None:
    dsn = os.environ.get(POSTGRES_DSN_ENV, "").strip()
    return dsn or None


def reset_postgres_database(dsn: str) -> None:
    import psycopg

    with psycopg.connect(dsn, autocommit=True) as connection:
        connection.execute("DROP SCHEMA public CASCADE")
        connection.execute("CREATE SCHEMA public")
