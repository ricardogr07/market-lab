from __future__ import annotations

import os
import subprocess
import sys
import threading
from collections import deque
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable
from uuid import uuid4

from marketlab.config import ExperimentConfig, load_config
from marketlab.mcp.workspace import WorkspaceSandbox

ALLOWED_COMMANDS = ("prepare-data", "backtest", "train-models", "run-experiment")
FINAL_JOB_STATES = {"succeeded", "failed", "cancelled"}


def _utc_now() -> str:
    return datetime.now(tz=UTC).isoformat()


def _last_non_empty_line(path: Path) -> str | None:
    if not path.exists():
        return None
    lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return lines[-1] if lines else None


def _network_required(config: ExperimentConfig) -> bool:
    if config.prepared_panel_path.exists():
        return False
    raw_symbol_paths = [config.cache_dir / f"{symbol}.csv" for symbol in config.data.symbols]
    return not all(path.exists() for path in raw_symbol_paths)


def _command_builder(command: str, config_path: Path) -> list[str]:
    return [sys.executable, "-m", "marketlab.cli", command, "--config", str(config_path)]


@dataclass(slots=True)
class RunPlan:
    id: str
    command: str
    config_path: str
    created_at: str
    network_required: bool
    summary: dict[str, Any]
    started: bool = False


@dataclass(slots=True)
class JobRecord:
    id: str
    plan_id: str
    command: str
    config_path: str
    status: str
    created_at: str
    log_path: str
    network_required: bool
    started_at: str | None = None
    finished_at: str | None = None
    return_code: int | None = None
    result_path: str | None = None
    result_kind: str | None = None
    error_message: str | None = None
    cancel_requested: bool = False


class MarketLabJobManager:
    def __init__(
        self,
        *,
        sandbox: WorkspaceSandbox,
        allow_network: bool,
        process_factory: Callable[..., subprocess.Popen[str]] | None = None,
        command_builder: Callable[[str, Path], list[str]] | None = None,
    ) -> None:
        self._sandbox = sandbox
        self._allow_network = allow_network
        self._process_factory = process_factory or subprocess.Popen
        self._command_builder = command_builder or _command_builder
        self._plans: dict[str, RunPlan] = {}
        self._jobs: dict[str, JobRecord] = {}
        self._queue: deque[str] = deque()
        self._lock = threading.RLock()
        self._wake_event = threading.Event()
        self._shutdown_event = threading.Event()
        self._active_job_id: str | None = None
        self._active_process: subprocess.Popen[str] | None = None
        self._worker = threading.Thread(
            target=self._run_worker,
            name="marketlab-mcp-job-worker",
            daemon=True,
        )
        self._worker.start()

    def create_plan(self, *, command: str, config_path: str | Path) -> dict[str, Any]:
        if command not in ALLOWED_COMMANDS:
            allowed = ", ".join(ALLOWED_COMMANDS)
            raise ValueError(f"Unsupported command {command!r}. Expected one of: {allowed}")

        resolved_config = self._sandbox.resolve_workspace_path(config_path)
        config = load_config(resolved_config)
        self._sandbox.validate_execution_paths(config)
        network_required = _network_required(config)
        if network_required and not self._allow_network:
            raise ValueError(
                f"{command} requires network access because the prepared panel or raw symbol cache is not available."
            )

        plan = RunPlan(
            id=uuid4().hex,
            command=command,
            config_path=str(resolved_config),
            created_at=_utc_now(),
            network_required=network_required,
            summary={
                "experiment_name": config.experiment_name,
                "output_dir": str(config.output_dir),
                "prepared_panel_path": str(config.prepared_panel_path),
                "symbol_count": len(config.data.symbols),
            },
        )
        with self._lock:
            self._plans[plan.id] = plan
        return asdict(plan)

    def start_job(self, plan_id: str) -> dict[str, Any]:
        with self._lock:
            plan = self._plans.get(plan_id)
            if plan is None:
                raise KeyError(f"Unknown plan_id: {plan_id}")
            if plan.started:
                raise ValueError(f"Plan {plan_id} has already been started.")

            job = JobRecord(
                id=uuid4().hex,
                plan_id=plan.id,
                command=plan.command,
                config_path=plan.config_path,
                status="queued",
                created_at=_utc_now(),
                log_path=str(self._sandbox.logs_root / f"{plan.id}.log"),
                network_required=plan.network_required,
            )
            plan.started = True
            self._jobs[job.id] = job
            self._queue.append(job.id)
            self._wake_event.set()
            return self._serialize_job(job)

    def list_jobs(self) -> dict[str, Any]:
        with self._lock:
            return {
                "active_job_id": self._active_job_id,
                "queued_job_ids": list(self._queue),
                "jobs": [self._serialize_job(job) for job in self._jobs.values()],
            }

    def get_job(self, job_id: str) -> dict[str, Any]:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                raise KeyError(f"Unknown job_id: {job_id}")
            return self._serialize_job(job)

    def cancel_job(self, job_id: str) -> dict[str, Any]:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                raise KeyError(f"Unknown job_id: {job_id}")
            if job.status in FINAL_JOB_STATES:
                return self._serialize_job(job)
            if job.status == "queued":
                job.status = "cancelled"
                job.finished_at = _utc_now()
                job.error_message = "Cancelled before execution."
                return self._serialize_job(job)
            if job.status == "running" and self._active_process is not None:
                job.cancel_requested = True
                job.status = "cancelling"
                self._active_process.terminate()
                return self._serialize_job(job)
            return self._serialize_job(job)

    def tail_logs(self, job_id: str, lines: int = 40) -> dict[str, Any]:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                raise KeyError(f"Unknown job_id: {job_id}")
            log_path = Path(job.log_path)
            content = ""
            if log_path.exists():
                log_lines = log_path.read_text(encoding="utf-8").splitlines()
                content = "\n".join(log_lines[-max(lines, 1) :])
            return {
                "job_id": job_id,
                "status": job.status,
                "log_path": job.log_path,
                "log_tail": content,
            }

    def close(self) -> None:
        with self._lock:
            active_process = self._active_process
        if active_process is not None and active_process.poll() is None:
            active_process.terminate()
        self._shutdown_event.set()
        self._wake_event.set()
        self._worker.join(timeout=5.0)

    def _run_worker(self) -> None:
        while not self._shutdown_event.is_set():
            self._wake_event.wait(timeout=0.1)
            job_id = self._next_job_id()
            if job_id is None:
                self._wake_event.clear()
                continue
            self._execute_job(job_id)

    def _next_job_id(self) -> str | None:
        with self._lock:
            while self._queue:
                job_id = self._queue.popleft()
                job = self._jobs[job_id]
                if job.status == "cancelled":
                    continue
                self._active_job_id = job_id
                return job_id
        return None

    def _execute_job(self, job_id: str) -> None:
        with self._lock:
            job = self._jobs[job_id]
            job.status = "running"
            job.started_at = _utc_now()
            log_path = Path(job.log_path)
            log_path.parent.mkdir(parents=True, exist_ok=True)

        command = self._command_builder(job.command, Path(job.config_path))
        env = os.environ.copy()
        try:
            with Path(job.log_path).open("w", encoding="utf-8") as log_handle:
                process = self._process_factory(
                    command,
                    cwd=self._sandbox.workspace_root,
                    stdout=log_handle,
                    stderr=subprocess.STDOUT,
                    text=True,
                    env=env,
                )
                with self._lock:
                    self._active_process = process
                return_code = process.wait()
        except Exception as exc:
            with self._lock:
                job.return_code = -1
                job.finished_at = _utc_now()
                job.status = "failed"
                job.error_message = str(exc)
                self._active_process = None
                self._active_job_id = None
                if self._queue:
                    self._wake_event.set()
            return

        last_line = _last_non_empty_line(Path(job.log_path))
        with self._lock:
            job.return_code = return_code
            job.finished_at = _utc_now()
            if job.cancel_requested:
                job.status = "cancelled"
                job.error_message = "Cancelled while running."
            elif return_code == 0:
                job.status = "succeeded"
                if last_line:
                    job.result_path = last_line
                    job.result_kind = "panel_path" if job.command == "prepare-data" else "run_dir"
            else:
                job.status = "failed"
                job.error_message = last_line or f"Command exited with status {return_code}."
            self._active_process = None
            self._active_job_id = None
            if self._queue:
                self._wake_event.set()

    def _serialize_job(self, job: JobRecord) -> dict[str, Any]:
        queued_position = None
        if job.status == "queued":
            with self._lock:
                try:
                    queued_position = list(self._queue).index(job.id) + 1
                except ValueError:
                    queued_position = None
        payload = asdict(job)
        payload["queued_position"] = queued_position
        return payload
