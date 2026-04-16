from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest
from tests._paper_fakes import build_phase7_paper_config

from marketlab.paper import scheduler


def _capture_transport(calls: list[dict[str, object]]):
    def _transport(url: str, payload: dict[str, object], timeout_seconds: int) -> tuple[int, str]:
        calls.append(
            {
                "url": url,
                "payload": payload,
                "timeout_seconds": timeout_seconds,
            }
        )
        return 200, '{"ok": true, "result": {"message_id": 1}}'

    return _transport


def _configure_notification_env(monkeypatch) -> None:
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "bot-token")
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "chat-id")


def test_scheduler_iteration_runs_each_phase_once_per_market_date(
    monkeypatch,
    tmp_path: Path,
) -> None:
    config = build_phase7_paper_config(tmp_path)
    events: list[str] = []

    def _fake_decision(*args, **kwargs):
        events.append("decision")
        return {"proposal_path": "proposal.json", "status_path": "status.json", "status": {}}

    def _fake_submit(*args, **kwargs):
        events.append("submission")
        return {"submission_path": "submission.json", "status_path": "status.json", "status": {}}

    monkeypatch.setattr(scheduler, "run_paper_decision", _fake_decision)
    monkeypatch.setattr(scheduler, "run_paper_submit", _fake_submit)

    first = scheduler.run_scheduler_iteration(
        config,
        now=datetime(2026, 4, 10, 23, 10, tzinfo=UTC),
    )
    second = scheduler.run_scheduler_iteration(
        config,
        now=datetime(2026, 4, 10, 23, 20, tzinfo=UTC),
    )

    assert [event["phase"] for event in first["events"]] == ["decision", "submission"]
    assert second["events"] == []
    assert events == ["decision", "submission"]


def test_scheduler_loop_deduplicates_repeated_error_alerts_until_recovery(
    monkeypatch,
    tmp_path: Path,
) -> None:
    config = build_phase7_paper_config(tmp_path, telegram_enabled=True)
    calls: list[dict[str, object]] = []
    _configure_notification_env(monkeypatch)

    def _failing_iteration(*args, **kwargs):
        raise RuntimeError("decision phase failed")

    def _successful_iteration(*args, **kwargs):
        state = scheduler._load_scheduler_state(config)
        scheduler._clear_scheduler_error_state(state)
        state["last_checked_at"] = datetime(2026, 4, 10, 23, 20, tzinfo=UTC).isoformat()
        state_path = scheduler._save_scheduler_state(config, state)
        return {
            "scheduler_state_path": str(state_path),
            "events": [],
            "market_date": "2026-04-10",
        }

    monkeypatch.setattr(scheduler, "run_scheduler_iteration", _failing_iteration)
    with pytest.raises(RuntimeError, match="decision phase failed"):
        scheduler.run_scheduler_loop(
            config,
            once=True,
            notification_transport=_capture_transport(calls),
        )
    with pytest.raises(RuntimeError, match="decision phase failed"):
        scheduler.run_scheduler_loop(
            config,
            once=True,
            notification_transport=_capture_transport(calls),
        )

    monkeypatch.setattr(scheduler, "run_scheduler_iteration", _successful_iteration)
    scheduler.run_scheduler_loop(
        config,
        once=True,
        notification_transport=_capture_transport(calls),
    )

    monkeypatch.setattr(scheduler, "run_scheduler_iteration", _failing_iteration)
    with pytest.raises(RuntimeError, match="decision phase failed"):
        scheduler.run_scheduler_loop(
            config,
            once=True,
            notification_transport=_capture_transport(calls),
        )

    records = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted((config.paper_state_dir / "notifications").glob("*.json"))
    ]
    assert len(calls) == 2
    assert len(records) == 2
    assert all(record["stage"] == "paper-error" for record in records)


def test_scheduler_loop_once_propagates_iteration_failures_after_notifying(
    monkeypatch,
    tmp_path: Path,
) -> None:
    config = build_phase7_paper_config(tmp_path, telegram_enabled=True)
    calls: list[dict[str, object]] = []
    _configure_notification_env(monkeypatch)

    def _failing_iteration(*args, **kwargs):
        raise RuntimeError("submit phase failed")

    monkeypatch.setattr(scheduler, "run_scheduler_iteration", _failing_iteration)

    with pytest.raises(RuntimeError, match="submit phase failed"):
        scheduler.run_scheduler_loop(
            config,
            once=True,
            notification_transport=_capture_transport(calls),
        )

    records = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted((config.paper_state_dir / "notifications").glob("*.json"))
    ]
    assert len(calls) == 1
    assert len(records) == 1
    assert records[0]["stage"] == "paper-error"
