from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest
from tests._paper_fakes import FakePaperNotificationSink, build_phase7_paper_config

from marketlab.log import configure_logging
from marketlab.paper import scheduler
from marketlab.paper.contracts import PaperHostedExecutionContext
from marketlab.paper.notifications import build_telegram_paper_notification_sink


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


def _stderr_records(stderr: str) -> list[dict[str, object]]:
    return [
        json.loads(line)
        for line in stderr.splitlines()
        if line.strip() != ""
    ]


def _hosted_context(**overrides: str) -> PaperHostedExecutionContext:
    payload = {
        "deployment_id": "qqq-paper-dev",
        "environment": "dev",
        "phase": "decision",
        "execution_id": "scheduler-exec",
        "correlation_id": "scheduler-corr",
        "idempotency_key": "scheduler-idem",
        "trigger_source": "container-app-job",
        "requested_at": "2026-06-19T12:00:00+00:00",
        "config_version": "config-v1",
        "image_digest": "sha256:abc123",
    }
    payload.update(overrides)
    return PaperHostedExecutionContext.from_metadata(payload)


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

    def _fake_reconcile(*args, **kwargs):
        return None

    monkeypatch.setattr(scheduler, "run_paper_decision", _fake_decision)
    monkeypatch.setattr(scheduler, "run_paper_submit", _fake_submit)
    monkeypatch.setattr(scheduler, "reconcile_latest_submission_status", _fake_reconcile)

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


def test_scheduler_iteration_forwards_injected_notification_sink(
    monkeypatch,
    tmp_path: Path,
) -> None:
    config = build_phase7_paper_config(tmp_path)
    sink = FakePaperNotificationSink()
    forwarded_sinks: list[object] = []

    def _fake_decision(*args, **kwargs):
        forwarded_sinks.append(kwargs["notification_sink"])
        return {"proposal_path": "proposal.json", "status_path": "status.json", "status": {}}

    def _fake_submit(*args, **kwargs):
        forwarded_sinks.append(kwargs["notification_sink"])
        return {"submission_path": "submission.json", "status_path": "status.json", "status": {}}

    monkeypatch.setattr(scheduler, "run_paper_decision", _fake_decision)
    monkeypatch.setattr(scheduler, "run_paper_submit", _fake_submit)
    monkeypatch.setattr(scheduler, "reconcile_latest_submission_status", lambda *args, **kwargs: None)

    scheduler.run_scheduler_iteration(
        config,
        now=datetime(2026, 4, 10, 23, 10, tzinfo=UTC),
        notification_sink=sink,
    )

    assert forwarded_sinks == [sink, sink]


def test_scheduler_iteration_derives_hosted_contexts_for_due_phases(
    monkeypatch,
    tmp_path: Path,
) -> None:
    config = build_phase7_paper_config(tmp_path)
    captured: list[PaperHostedExecutionContext | None] = []

    def _fake_decision(*args, **kwargs):
        captured.append(kwargs["hosted_context"])
        return {"proposal_path": "proposal.json", "status_path": "status.json", "status": {}}

    def _fake_submit(*args, **kwargs):
        captured.append(kwargs["hosted_context"])
        return {"submission_path": "submission.json", "status_path": "status.json", "status": {}}

    def _fake_reconcile(*args, **kwargs):
        captured.append(kwargs["hosted_context"])
        return None

    monkeypatch.setattr(scheduler, "run_paper_decision", _fake_decision)
    monkeypatch.setattr(scheduler, "run_paper_submit", _fake_submit)
    monkeypatch.setattr(scheduler, "reconcile_latest_submission_status", _fake_reconcile)

    scheduler.run_scheduler_iteration(
        config,
        now=datetime(2026, 4, 10, 23, 10, tzinfo=UTC),
        hosted_context=_hosted_context(),
    )

    assert [context.phase if context is not None else "" for context in captured] == [
        "decision",
        "submit",
        "reconcile",
    ]
    assert [context.idempotency_key if context is not None else "" for context in captured] == [
        "scheduler-idem:decision:2026-04-10",
        "scheduler-idem:submit:2026-04-10",
        "scheduler-idem:reconcile:2026-04-10",
    ]
    assert all(context.correlation_id == "scheduler-corr" for context in captured if context is not None)


def test_scheduler_iteration_appends_submission_reconciliation_events(
    monkeypatch,
    tmp_path: Path,
) -> None:
    config = build_phase7_paper_config(tmp_path)

    monkeypatch.setattr(
        scheduler,
        "reconcile_latest_submission_status",
        lambda *args, **kwargs: {
            "proposal_id": "proposal-1",
            "order_status": "rejected",
            "submission_path": "submission.json",
            "order_status_path": "order_status.json",
            "poll_status": "observed",
        },
    )

    result = scheduler.run_scheduler_iteration(
        config,
        now=datetime(2026, 4, 10, 18, 0, tzinfo=UTC),
    )

    assert [event["phase"] for event in result["events"]] == ["submission_reconcile"]


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
            notification_sink=build_telegram_paper_notification_sink(
                config,
                transport=_capture_transport(calls),
            ),
        )
    with pytest.raises(RuntimeError, match="decision phase failed"):
        scheduler.run_scheduler_loop(
            config,
            once=True,
            notification_sink=build_telegram_paper_notification_sink(
                config,
                transport=_capture_transport(calls),
            ),
        )

    monkeypatch.setattr(scheduler, "run_scheduler_iteration", _successful_iteration)
    scheduler.run_scheduler_loop(
        config,
        once=True,
        notification_sink=build_telegram_paper_notification_sink(
            config,
            transport=_capture_transport(calls),
        ),
    )

    monkeypatch.setattr(scheduler, "run_scheduler_iteration", _failing_iteration)
    with pytest.raises(RuntimeError, match="decision phase failed"):
        scheduler.run_scheduler_loop(
            config,
            once=True,
            notification_sink=build_telegram_paper_notification_sink(
                config,
                transport=_capture_transport(calls),
            ),
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
            notification_sink=build_telegram_paper_notification_sink(
                config,
                transport=_capture_transport(calls),
            ),
        )

    records = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted((config.paper_state_dir / "notifications").glob("*.json"))
    ]
    assert len(calls) == 1
    assert len(records) == 1
    assert records[0]["stage"] == "paper-error"


def test_scheduler_loop_logs_start_and_error_with_shared_correlation_id(
    monkeypatch,
    capsys,
    tmp_path: Path,
) -> None:
    configure_logging()
    config = build_phase7_paper_config(tmp_path)

    monkeypatch.setattr(
        scheduler,
        "run_scheduler_iteration",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("scheduler boom")),
    )

    with pytest.raises(RuntimeError, match="scheduler boom"):
        scheduler.run_scheduler_loop(
            config,
            once=True,
            notification_sink=FakePaperNotificationSink(),
        )

    records = _stderr_records(capsys.readouterr().err)
    start_record = next(record for record in records if record["event"] == "paper.scheduler.loop.start")
    error_record = next(record for record in records if record["event"] == "paper.scheduler.loop.error")

    assert start_record["correlation_id"] == error_record["correlation_id"]
    assert start_record["deployment"] == "paper_scheduler"
    assert error_record["phase"] == "paper-scheduler"
    assert error_record["outcome"] == "error"
