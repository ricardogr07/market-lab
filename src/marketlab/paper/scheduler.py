from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from marketlab.config import ExperimentConfig
from marketlab.paper.service import (
    _clock_value,
    _local_now,
    _now_utc,
    run_paper_decision,
    run_paper_submit,
)


def _scheduler_state_path(config: ExperimentConfig) -> Path:
    return config.paper_state_dir / "scheduler.json"


def _load_scheduler_state(config: ExperimentConfig) -> dict[str, Any]:
    path = _scheduler_state_path(config)
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _save_scheduler_state(config: ExperimentConfig, payload: dict[str, Any]) -> Path:
    path = _scheduler_state_path(config)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return path


def run_scheduler_iteration(
    config: ExperimentConfig,
    *,
    now: datetime | None = None,
) -> dict[str, Any]:
    local_now = _local_now(config, now)
    market_date = local_now.date().isoformat()
    decision_clock = _clock_value(config.paper.decision_time)
    submission_clock = _clock_value(config.paper.submission_time)
    state = _load_scheduler_state(config)
    events: list[dict[str, Any]] = []

    if local_now.time() >= decision_clock and state.get("last_decision_market_date") != market_date:
        result = run_paper_decision(config, now=now)
        events.append({"phase": "decision", **result})
        state["last_decision_market_date"] = market_date
        state["last_decision_at"] = _now_utc(now).isoformat()

    if local_now.time() >= submission_clock and state.get("last_submission_market_date") != market_date:
        result = run_paper_submit(config, now=now)
        events.append({"phase": "submission", **result})
        state["last_submission_market_date"] = market_date
        state["last_submission_at"] = _now_utc(now).isoformat()

    state["last_checked_at"] = _now_utc(now).isoformat()
    state["last_checked_market_date"] = market_date
    state_path = _save_scheduler_state(config, state)
    return {
        "scheduler_state_path": str(state_path),
        "market_date": market_date,
        "events": events,
    }


def run_scheduler_loop(config: ExperimentConfig, *, once: bool = False) -> None:
    while True:
        summary = run_scheduler_iteration(config)
        print(json.dumps(summary, indent=2, sort_keys=True))
        if once:
            return
        time.sleep(config.paper.poll_interval_seconds)
