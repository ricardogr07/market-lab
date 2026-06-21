from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
COMPOSE_FILE = ROOT / "docker" / "compose.postgres.yml"
_TEST_DB_SCHEME = "postgresql"
_TEST_DB_USER = "postgres"
_TEST_DB_HOST = "127.0.0.1"
_TEST_DB_PORT = 55432
_TEST_DB_NAME = "marketlab_test"


def _compose(*args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["docker", "compose", "--ansi", "never", "-f", str(COMPOSE_FILE), *args],
        cwd=ROOT,
        check=check,
        text=True,
    )


def main() -> int:
    environment = os.environ.copy()
    environment["MARKETLAB_PAPER_POSTGRES_DSN"] = (
        f"{_TEST_DB_SCHEME}://{_TEST_DB_USER}@{_TEST_DB_HOST}:"
        f"{_TEST_DB_PORT}/{_TEST_DB_NAME}"
    )
    try:
        _compose("up", "--detach", "--wait")
        subprocess.run(
            [
                sys.executable,
                "-m",
                "pytest",
                "-q",
                "tests/unit/test_paper_persistence.py",
                "tests/integration/test_postgres_persistence.py",
            ],
            cwd=ROOT,
            check=True,
            env=environment,
        )
    finally:
        _compose("down", "--volumes", "--remove-orphans", check=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
