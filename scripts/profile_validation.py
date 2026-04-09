from __future__ import annotations

import argparse
import subprocess
import sys
import time
from dataclasses import dataclass
from typing import Callable

VALID_ENVS = ("lint", "docs", "py312", "package", "integration")
DEFAULT_ENVS = VALID_ENVS
CI_PYTHON = (3, 12)


@dataclass(slots=True)
class LaneResult:
    env: str
    exit_code: int
    elapsed_seconds: float


def resolve_envs(envs: list[str] | None) -> list[str]:
    if not envs:
        return list(DEFAULT_ENVS)
    return list(envs)


def python_warning(version_info: tuple[int, int] | None = None) -> str | None:
    major, minor = version_info or sys.version_info[:2]
    if (major, minor) == CI_PYTHON:
        return None
    return (
        f"Warning: current interpreter is Python {major}.{minor}. "
        "Use `py -3.12 ...` to match the local CI defaults and GitHub Actions."
    )


def build_tox_command(env: str) -> list[str]:
    return [sys.executable, "-m", "tox", "-e", env]


def default_runner(command: list[str]) -> int:
    completed = subprocess.run(command, check=False)
    return completed.returncode


def build_summary_table(results: list[LaneResult]) -> str:
    lines = [
        "Validation Summary",
        f"{'env':<16}{'exit':>8}{'seconds':>12}",
        f"{'-' * 16}{'-' * 8}{'-' * 12}",
    ]
    for result in results:
        lines.append(
            f"{result.env:<16}{result.exit_code:>8}{result.elapsed_seconds:>12.2f}"
        )
    if results:
        total_seconds = sum(result.elapsed_seconds for result in results)
        lines.append(f"{'total':<16}{'':>8}{total_seconds:>12.2f}")
    return "\n".join(lines)


def run_validation(
    envs: list[str],
    *,
    runner: Callable[[list[str]], int] = default_runner,
    out: Callable[[str], None] = print,
    version_info: tuple[int, int] | None = None,
) -> int:
    warning = python_warning(version_info)
    if warning is not None:
        out(warning)

    results: list[LaneResult] = []
    first_failure = 0

    for env in envs:
        command = build_tox_command(env)
        out(f"[start] {env}: {' '.join(command)}")
        started_at = time.perf_counter()
        exit_code = runner(command)
        elapsed_seconds = time.perf_counter() - started_at
        out(f"[finish] {env}: exit={exit_code} elapsed={elapsed_seconds:.2f}s")
        results.append(
            LaneResult(
                env=env,
                exit_code=exit_code,
                elapsed_seconds=elapsed_seconds,
            )
        )
        if exit_code != 0:
            first_failure = exit_code
            break

    out(build_summary_table(results))
    return first_failure


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Profile local validation lanes one tox environment at a time."
    )
    parser.add_argument(
        "--env",
        action="append",
        choices=VALID_ENVS,
        dest="envs",
        help="Run only the selected tox env. Repeat to run multiple envs in order.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    envs = resolve_envs(args.envs)
    return run_validation(envs)


if __name__ == "__main__":
    raise SystemExit(main())
