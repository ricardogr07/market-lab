from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from tests._paper_fakes import (
    FakeAlpacaBroker,
    FakeAlpacaProvider,
    build_phase7_paper_config,
)
from tests._telegram_fakes import FakeTelegramServer

from marketlab.paper.agent import run_agent_approval_iteration
from marketlab.paper.service import (
    PaperStateStore,
    run_paper_decision,
    run_paper_submit,
)


def test_local_paper_flow_sends_decision_approval_and_submit_notifications(
    monkeypatch,
    tmp_path: Path,
) -> None:
    config = build_phase7_paper_config(
        tmp_path,
        execution_mode="agent_approval",
        telegram_enabled=True,
    )
    provider = FakeAlpacaProvider(symbol="VOO")
    broker = FakeAlpacaBroker(symbol="VOO")

    with FakeTelegramServer() as telegram_server:
        monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "bot-token")
        monkeypatch.setenv("TELEGRAM_CHAT_ID", "chat-id")
        monkeypatch.setenv("MARKETLAB_TELEGRAM_API_BASE_URL", telegram_server.base_url)

        decision = run_paper_decision(
            config,
            now=datetime(2026, 4, 10, 20, 10, tzinfo=UTC),
            provider=provider,
            broker=broker,
        )
        approval = run_agent_approval_iteration(
            config,
            now=datetime(2026, 4, 10, 20, 30, tzinfo=UTC),
            broker=broker,
        )
        submission = run_paper_submit(
            config,
            now=datetime(2026, 4, 10, 23, 5, tzinfo=UTC),
            broker=broker,
        )

    records = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted(PaperStateStore(config).notifications_root.glob("*.json"))
    ]
    messages = [request.payload["text"] for request in telegram_server.requests]

    assert decision["status"]["status"] == "proposal_created"
    assert approval["processed_count"] == 1
    assert submission["submission"]["status"] in {"submitted", "no_trade_required"}
    assert len(telegram_server.requests) == 3
    assert messages[0].startswith("paper-decision")
    assert messages[1].startswith("paper-approve")
    assert messages[2].startswith("paper-submit")
    assert [record["stage"] for record in records] == [
        "paper-decision",
        "paper-approve",
        "paper-submit",
    ]
