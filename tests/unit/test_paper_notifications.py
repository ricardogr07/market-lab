from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from tests._paper_fakes import build_phase7_paper_config

from marketlab.paper.notifications import (
    DELIVERY_DELIVERED,
    DELIVERY_FAILED,
    DELIVERY_SKIPPED_DISABLED,
    DELIVERY_SKIPPED_MISSING_CREDENTIALS,
    deliver_telegram_notification,
)
from marketlab.paper.service import PaperStateStore


def test_disabled_telegram_notification_records_skipped_disabled(
    monkeypatch,
    tmp_path: Path,
) -> None:
    config = build_phase7_paper_config(tmp_path, telegram_enabled=False)
    store = PaperStateStore(config)
    monkeypatch.delenv("TELEGRAM_BOT_TOKEN", raising=False)
    monkeypatch.delenv("TELEGRAM_CHAT_ID", raising=False)

    record = deliver_telegram_notification(
        config,
        stage="paper-decision",
        outcome="proposal_created",
        message="paper-decision\noutcome: proposal_created",
        now=datetime(2026, 4, 10, 20, 10, tzinfo=UTC),
    )
    record_path = store.write_notification_record(
        stage="paper-decision",
        outcome="proposal_created",
        payload=record,
        now=datetime(2026, 4, 10, 20, 10, tzinfo=UTC),
    )

    persisted = json.loads(record_path.read_text(encoding="utf-8"))
    assert record["delivery_status"] == DELIVERY_SKIPPED_DISABLED
    assert persisted["delivery_status"] == DELIVERY_SKIPPED_DISABLED


def test_enabled_telegram_notification_without_credentials_records_skip(
    monkeypatch,
    tmp_path: Path,
) -> None:
    config = build_phase7_paper_config(tmp_path, telegram_enabled=True)
    store = PaperStateStore(config)
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("MARKETLAB_ENV_FILE", raising=False)
    monkeypatch.delenv("TELEGRAM_BOT_TOKEN", raising=False)
    monkeypatch.delenv("TELEGRAM_CHAT_ID", raising=False)

    record = deliver_telegram_notification(
        config,
        stage="paper-approve",
        outcome="approved",
        message="paper-approve\noutcome: approved",
        now=datetime(2026, 4, 10, 20, 20, tzinfo=UTC),
    )
    record_path = store.write_notification_record(
        stage="paper-approve",
        outcome="approved",
        payload=record,
        now=datetime(2026, 4, 10, 20, 20, tzinfo=UTC),
    )

    persisted = json.loads(record_path.read_text(encoding="utf-8"))
    assert record["delivery_status"] == DELIVERY_SKIPPED_MISSING_CREDENTIALS
    assert persisted["missing_credentials"]["TELEGRAM_BOT_TOKEN"] is True
    assert persisted["missing_credentials"]["TELEGRAM_CHAT_ID"] is True


def test_successful_telegram_delivery_records_request_and_audit(
    monkeypatch,
    tmp_path: Path,
) -> None:
    config = build_phase7_paper_config(tmp_path, telegram_enabled=True)
    store = PaperStateStore(config)
    calls: list[dict[str, object]] = []

    def _transport(url: str, payload: dict[str, object], timeout_seconds: int) -> tuple[int, str]:
        calls.append(
            {
                "url": url,
                "payload": payload,
                "timeout_seconds": timeout_seconds,
            }
        )
        return 200, '{"ok": true, "result": {"message_id": 42}}'

    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "bot-token")
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "chat-id")
    monkeypatch.setenv("MARKETLAB_TELEGRAM_API_BASE_URL", "http://127.0.0.1:9999")

    record = deliver_telegram_notification(
        config,
        stage="paper-submit",
        outcome="submitted",
        message="paper-submit\noutcome: submitted",
        details={"order_id": "order-1"},
        proposal_id="proposal-1",
        trade_date="2026-04-13",
        now=datetime(2026, 4, 10, 23, 5, tzinfo=UTC),
        transport=_transport,
    )
    record_path = store.write_notification_record(
        stage="paper-submit",
        outcome="submitted",
        payload=record,
        now=datetime(2026, 4, 10, 23, 5, tzinfo=UTC),
    )

    persisted = json.loads(record_path.read_text(encoding="utf-8"))
    assert record["delivery_status"] == DELIVERY_DELIVERED
    assert len(calls) == 1
    assert calls[0]["url"] == "http://127.0.0.1:9999/botbot-token/sendMessage"
    assert calls[0]["payload"] == {
        "chat_id": "chat-id",
        "text": "paper-submit\noutcome: submitted",
        "disable_web_page_preview": True,
    }
    assert persisted["request"]["chat_id"] == "chat-id"
    assert persisted["response_body"]["result"]["message_id"] == 42


def test_failed_telegram_delivery_records_failure_without_raising(
    monkeypatch,
    tmp_path: Path,
) -> None:
    config = build_phase7_paper_config(tmp_path, telegram_enabled=True)
    store = PaperStateStore(config)

    def _transport(url: str, payload: dict[str, object], timeout_seconds: int) -> tuple[int, str]:
        raise RuntimeError("connection refused")

    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "bot-token")
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "chat-id")

    record = deliver_telegram_notification(
        config,
        stage="paper-error",
        outcome="error",
        message="paper-error\nloop: scheduler",
        now=datetime(2026, 4, 10, 23, 6, tzinfo=UTC),
        transport=_transport,
    )
    record_path = store.write_notification_record(
        stage="paper-error",
        outcome="error",
        payload=record,
        now=datetime(2026, 4, 10, 23, 6, tzinfo=UTC),
    )

    persisted = json.loads(record_path.read_text(encoding="utf-8"))
    assert record["delivery_status"] == DELIVERY_FAILED
    assert "connection refused" in record["error"]
    assert persisted["delivery_status"] == DELIVERY_FAILED
