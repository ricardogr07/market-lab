from __future__ import annotations

import json
import os
from datetime import UTC, datetime
from pathlib import Path

import pytest
from tests._paper_fakes import build_phase7_paper_config

from marketlab.env import load_env_file
from marketlab.paper.notifications import (
    DELIVERY_DELIVERED,
    deliver_telegram_notification,
)
from marketlab.paper.service import PaperStateStore

pytestmark = [pytest.mark.network]

REPO_ROOT = Path(__file__).resolve().parents[2]


def _require_real_telegram_env() -> None:
    if os.getenv("MARKETLAB_RUN_REAL_TELEGRAM") != "1":
        pytest.skip("Set MARKETLAB_RUN_REAL_TELEGRAM=1 to run the real Telegram smoke test.")

    load_env_file()
    missing = [
        name
        for name in ("TELEGRAM_BOT_TOKEN", "TELEGRAM_CHAT_ID")
        if os.getenv(name, "").strip() == ""
    ]
    if missing:
        joined = ", ".join(missing)
        pytest.skip(f"Real Telegram smoke test requires configured env vars: {joined}.")


def test_real_telegram_smoke_sends_one_message_and_persists_audit_record(
    monkeypatch,
    tmp_path: Path,
) -> None:
    monkeypatch.chdir(REPO_ROOT)
    monkeypatch.delenv("MARKETLAB_TELEGRAM_API_BASE_URL", raising=False)
    _require_real_telegram_env()

    config = build_phase7_paper_config(
        tmp_path,
        execution_mode="agent_approval",
        telegram_enabled=True,
    )
    store = PaperStateStore(config)
    now = datetime.now(UTC)
    timestamp = now.strftime("%Y-%m-%d %H:%M:%S UTC")
    message = "\n".join(
        [
            "paper-submit",
            "experiment: telegram_smoke_test",
            "symbol: TEST",
            "outcome: smoke_test",
            f"note: MarketLab real Telegram smoke test at {timestamp}",
        ]
    )

    record = deliver_telegram_notification(
        config,
        stage="paper-submit",
        outcome="smoke_test",
        message=message,
        details={
            "experiment_name": "telegram_smoke_test",
            "symbol": "TEST",
            "note": f"MarketLab real Telegram smoke test at {timestamp}",
        },
        proposal_id="telegram-smoke-test",
        trade_date=now.date().isoformat(),
        now=now,
    )
    record_path = store.write_notification_record(
        stage="paper-submit",
        outcome="smoke_test",
        payload=record,
        now=now,
    )

    persisted = json.loads(record_path.read_text(encoding="utf-8"))
    assert record["delivery_status"] == DELIVERY_DELIVERED, record
    assert persisted["delivery_status"] == DELIVERY_DELIVERED
    assert persisted["response_body"]["ok"] is True
    assert persisted["request"]["text"] == message
