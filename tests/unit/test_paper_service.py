from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest
from tests._paper_fakes import (
    FakeAlpacaBroker,
    FakeAlpacaProvider,
    build_phase7_paper_config,
)

from marketlab.paper.service import (
    PaperStateStore,
    decide_paper_proposal,
    get_paper_status,
    read_paper_evidence,
    reconcile_latest_submission_status,
    run_paper_decision,
    run_paper_submit,
)


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


def _notification_records(config) -> list[dict[str, object]]:
    store = PaperStateStore(config)
    return [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted(store.notifications_root.glob("*.json"))
    ]


def _configure_notification_env(monkeypatch) -> None:
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "bot-token")
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "chat-id")


def test_run_paper_decision_writes_latest_daily_proposal(monkeypatch, tmp_path: Path) -> None:
    config = build_phase7_paper_config(
        tmp_path,
        execution_mode="agent_approval",
        telegram_enabled=True,
    )
    provider = FakeAlpacaProvider(symbol="VOO")
    broker = FakeAlpacaBroker(symbol="VOO")
    calls: list[dict[str, object]] = []
    _configure_notification_env(monkeypatch)

    result = run_paper_decision(
        config,
        now=datetime(2026, 4, 10, 20, 10, tzinfo=UTC),
        provider=provider,
        broker=broker,
        notification_transport=_capture_transport(calls),
    )

    proposal_path = Path(result["proposal_path"])
    proposal = json.loads(proposal_path.read_text(encoding="utf-8"))
    evidence = read_paper_evidence(config, proposal_id=result["proposal_id"])

    assert proposal["proposal_id"].startswith("2026-04-13-VOO-2026-04-10")
    assert proposal["signal_date"] == "2026-04-10"
    assert proposal["effective_date"] == "2026-04-13"
    assert proposal["approval_status"] == "pending"
    assert proposal["decision_policy"] == "consensus_vote"
    assert proposal["decision"] in {"long", "cash"}
    assert proposal["target_weight"] in {0.0, 1.0}
    assert proposal["long_vote_count"] + proposal["cash_vote_count"] == 6
    assert proposal["train_rows"] >= 200
    assert len(evidence["models"]) == 6
    assert evidence["decision"] == proposal["decision"]
    assert len(calls) == 1
    assert calls[0]["payload"]["text"].startswith("paper-decision")
    records = _notification_records(config)
    assert records[-1]["stage"] == "paper-decision"
    assert records[-1]["outcome"] == "proposal_created"


def test_run_paper_decision_supports_configured_single_symbol(tmp_path: Path) -> None:
    config = build_phase7_paper_config(tmp_path, execution_mode="agent_approval", symbol="QQQ")
    provider = FakeAlpacaProvider(symbol="QQQ")
    broker = FakeAlpacaBroker(symbol="QQQ")

    result = run_paper_decision(
        config,
        now=datetime(2026, 4, 10, 20, 10, tzinfo=UTC),
        provider=provider,
        broker=broker,
    )

    proposal_path = Path(result["proposal_path"])
    proposal = json.loads(proposal_path.read_text(encoding="utf-8"))

    assert proposal["proposal_id"].startswith("2026-04-13-QQQ-2026-04-10")
    assert proposal["symbol"] == "QQQ"
    assert proposal["evidence_path"].endswith("evidence.json")


def test_run_paper_decision_rejects_non_single_symbol_config(tmp_path: Path) -> None:
    config = build_phase7_paper_config(tmp_path, execution_mode="agent_approval")
    config.data.symbols = ["QQQ", "VOO"]

    with pytest.raises(RuntimeError, match="exactly one configured symbol"):
        run_paper_decision(
            config,
            now=datetime(2026, 4, 10, 20, 10, tzinfo=UTC),
            provider=FakeAlpacaProvider(symbol="QQQ"),
            broker=FakeAlpacaBroker(symbol="QQQ"),
        )


def test_run_paper_decision_rejects_empty_symbol_config(tmp_path: Path) -> None:
    config = build_phase7_paper_config(tmp_path, execution_mode="agent_approval")
    config.data.symbols = []

    with pytest.raises(RuntimeError, match="exactly one configured symbol"):
        run_paper_decision(
            config,
            now=datetime(2026, 4, 10, 20, 10, tzinfo=UTC),
            provider=FakeAlpacaProvider(symbol="QQQ"),
            broker=FakeAlpacaBroker(symbol="QQQ"),
        )


def test_decide_paper_proposal_records_agent_approval(monkeypatch, tmp_path: Path) -> None:
    config = build_phase7_paper_config(
        tmp_path,
        execution_mode="agent_approval",
        telegram_enabled=True,
    )
    proposal_result = run_paper_decision(
        config,
        now=datetime(2026, 4, 10, 20, 10, tzinfo=UTC),
        provider=FakeAlpacaProvider(symbol="VOO"),
        broker=FakeAlpacaBroker(symbol="VOO"),
    )
    calls: list[dict[str, object]] = []
    _configure_notification_env(monkeypatch)

    decision_result = decide_paper_proposal(
        config,
        proposal_id=proposal_result["proposal_id"],
        decision="approve",
        actor="agent",
        now=datetime(2026, 4, 10, 20, 20, tzinfo=UTC),
        notification_transport=_capture_transport(calls),
    )

    approval_path = Path(decision_result["approval_path"])
    approval = json.loads(approval_path.read_text(encoding="utf-8"))
    proposal = PaperStateStore(config).load_proposal(proposal_result["proposal_id"])

    assert approval["decision"] == "approve"
    assert approval["actor"] == "agent"
    assert proposal["approval_status"] == "approved"
    assert proposal["approval_actor"] == "agent"
    assert len(calls) == 1
    assert "outcome: approved" in calls[0]["payload"]["text"]
    records = _notification_records(config)
    assert records[-1]["stage"] == "paper-approve"
    assert records[-1]["outcome"] == "approved"


def test_run_paper_submit_skips_missing_agent_approval(monkeypatch, tmp_path: Path) -> None:
    config = build_phase7_paper_config(
        tmp_path,
        execution_mode="agent_approval",
        telegram_enabled=True,
    )
    proposal_result = run_paper_decision(
        config,
        now=datetime(2026, 4, 10, 20, 10, tzinfo=UTC),
        provider=FakeAlpacaProvider(symbol="VOO"),
        broker=FakeAlpacaBroker(symbol="VOO"),
    )
    calls: list[dict[str, object]] = []
    _configure_notification_env(monkeypatch)

    submission_result = run_paper_submit(
        config,
        now=datetime(2026, 4, 10, 23, 5, tzinfo=UTC),
        broker=FakeAlpacaBroker(symbol="VOO"),
        notification_transport=_capture_transport(calls),
    )

    submission = submission_result["submission"]

    assert submission["proposal_id"] == proposal_result["proposal_id"]
    assert submission["status"] == "skipped"
    assert submission["reason"] == "missing_approval"
    assert len(calls) == 1
    assert "reason: missing_approval" in calls[0]["payload"]["text"]
    records = _notification_records(config)
    assert records[-1]["stage"] == "paper-submit"
    assert records[-1]["outcome"] == "skipped"


def test_run_paper_submit_places_fractional_order_after_agent_approval(monkeypatch, tmp_path: Path) -> None:
    config = build_phase7_paper_config(
        tmp_path,
        execution_mode="agent_approval",
        telegram_enabled=True,
    )
    provider = FakeAlpacaProvider(symbol="VOO")
    broker = FakeAlpacaBroker(symbol="VOO")
    proposal_result = run_paper_decision(
        config,
        now=datetime(2026, 4, 10, 20, 10, tzinfo=UTC),
        provider=provider,
        broker=broker,
    )
    decide_paper_proposal(
        config,
        proposal_id=proposal_result["proposal_id"],
        decision="approve",
        actor="agent",
        now=datetime(2026, 4, 10, 20, 20, tzinfo=UTC),
    )
    calls: list[dict[str, object]] = []
    _configure_notification_env(monkeypatch)

    submission_result = run_paper_submit(
        config,
        now=datetime(2026, 4, 10, 23, 5, tzinfo=UTC),
        broker=broker,
        notification_transport=_capture_transport(calls),
    )

    submission = submission_result["submission"]

    assert submission["status"] in {"submitted", "no_trade_required"}
    assert len(calls) == 1
    if submission["status"] == "submitted":
        assert broker.submitted_orders
        order = broker.submitted_orders[-1]
        assert submission["order_status"] == "accepted"
        if submission["side"] == "buy":
            assert "notional" in order
            assert submission["notional"] is not None
            assert submission["qty"] is None
        else:
            assert "qty" in order
        assert "outcome: submitted" in calls[0]["payload"]["text"]
    else:
        assert "outcome: no_trade_required" in calls[0]["payload"]["text"]


def test_run_paper_submit_uses_notional_buy_buffer_for_long_entries(tmp_path: Path) -> None:
    config = build_phase7_paper_config(tmp_path, execution_mode="agent_approval", symbol="QQQ")
    broker = FakeAlpacaBroker(symbol="QQQ", equity=1000.0, buying_power=1000.0, cash=1000.0)
    proposal_result = run_paper_decision(
        config,
        now=datetime(2026, 4, 10, 20, 10, tzinfo=UTC),
        provider=FakeAlpacaProvider(symbol="QQQ"),
        broker=broker,
    )
    store = PaperStateStore(config)
    proposal = store.load_proposal(proposal_result["proposal_id"])
    proposal["decision"] = "long"
    proposal["target_weight"] = 1.0
    proposal["reference_price"] = 640.41
    store.update_proposal(proposal)
    decide_paper_proposal(
        config,
        proposal_id=proposal_result["proposal_id"],
        decision="approve",
        actor="agent",
        now=datetime(2026, 4, 10, 20, 20, tzinfo=UTC),
    )

    submission_result = run_paper_submit(
        config,
        now=datetime(2026, 4, 10, 23, 5, tzinfo=UTC),
        broker=broker,
    )

    submission = submission_result["submission"]
    assert submission["status"] == "submitted"
    assert submission["side"] == "buy"
    assert submission["notional"] == pytest.approx(990.0)
    assert submission["qty"] is None
    assert broker.submitted_orders[-1]["notional"] == "990.00"


def test_reconcile_latest_submission_status_refreshes_broker_rejection(tmp_path: Path) -> None:
    config = build_phase7_paper_config(tmp_path, execution_mode="agent_approval", symbol="QQQ")
    broker = FakeAlpacaBroker(symbol="QQQ", order_status="accepted")
    proposal_result = run_paper_decision(
        config,
        now=datetime(2026, 4, 10, 20, 10, tzinfo=UTC),
        provider=FakeAlpacaProvider(symbol="QQQ"),
        broker=broker,
    )
    store = PaperStateStore(config)
    proposal = store.load_proposal(proposal_result["proposal_id"])
    proposal["decision"] = "long"
    proposal["target_weight"] = 1.0
    proposal["reference_price"] = 640.41
    store.update_proposal(proposal)
    decide_paper_proposal(
        config,
        proposal_id=proposal_result["proposal_id"],
        decision="approve",
        actor="agent",
        now=datetime(2026, 4, 10, 20, 20, tzinfo=UTC),
    )
    run_paper_submit(
        config,
        now=datetime(2026, 4, 10, 23, 5, tzinfo=UTC),
        broker=broker,
    )

    broker.order_status = "rejected"
    reconciliation = reconcile_latest_submission_status(
        config,
        now=datetime(2026, 4, 11, 14, 0, tzinfo=UTC),
        broker=broker,
    )

    submission = json.loads(
        (PaperStateStore(config).trade_submission_path("2026-04-13")).read_text(encoding="utf-8")
    )
    assert reconciliation is not None
    assert reconciliation["order_status"] == "rejected"
    assert submission["order_status"] == "rejected"


def test_run_paper_submit_can_retry_failed_submission(tmp_path: Path) -> None:
    config = build_phase7_paper_config(tmp_path, execution_mode="agent_approval", symbol="QQQ")
    broker = FakeAlpacaBroker(symbol="QQQ", order_status="rejected")
    proposal_result = run_paper_decision(
        config,
        now=datetime(2026, 4, 10, 20, 10, tzinfo=UTC),
        provider=FakeAlpacaProvider(symbol="QQQ"),
        broker=broker,
    )
    store = PaperStateStore(config)
    proposal = store.load_proposal(proposal_result["proposal_id"])
    proposal["decision"] = "long"
    proposal["target_weight"] = 1.0
    proposal["reference_price"] = 640.41
    store.update_proposal(proposal)
    decide_paper_proposal(
        config,
        proposal_id=proposal_result["proposal_id"],
        decision="approve",
        actor="agent",
        now=datetime(2026, 4, 10, 20, 20, tzinfo=UTC),
    )
    first_submission = run_paper_submit(
        config,
        now=datetime(2026, 4, 10, 23, 5, tzinfo=UTC),
        broker=broker,
    )["submission"]

    broker.order_status = "accepted"
    second_submission = run_paper_submit(
        config,
        now=datetime(2026, 4, 11, 14, 0, tzinfo=UTC),
        broker=broker,
        retry_failed_submission=True,
    )["submission"]

    trade_dir = PaperStateStore(config).trade_dir("2026-04-13")
    assert first_submission["order_status"] == "rejected"
    assert second_submission["order_status"] == "accepted"
    assert second_submission["client_order_id"] != first_submission["client_order_id"]
    assert list(trade_dir.glob("submission.retry-backup.*.bak"))


def test_run_paper_decision_notifies_existing_proposal(monkeypatch, tmp_path: Path) -> None:
    config = build_phase7_paper_config(
        tmp_path,
        execution_mode="agent_approval",
        telegram_enabled=True,
    )
    provider = FakeAlpacaProvider(symbol="VOO")
    broker = FakeAlpacaBroker(symbol="VOO")
    run_paper_decision(
        config,
        now=datetime(2026, 4, 10, 20, 10, tzinfo=UTC),
        provider=provider,
        broker=broker,
    )
    calls: list[dict[str, object]] = []
    _configure_notification_env(monkeypatch)

    result = run_paper_decision(
        config,
        now=datetime(2026, 4, 10, 20, 11, tzinfo=UTC),
        provider=provider,
        broker=broker,
        notification_transport=_capture_transport(calls),
    )

    assert result["status"]["status"] == "existing_proposal"
    assert len(calls) == 1
    assert "outcome: existing_proposal" in calls[0]["payload"]["text"]


def test_decide_paper_proposal_notifies_manual_rejection(monkeypatch, tmp_path: Path) -> None:
    config = build_phase7_paper_config(
        tmp_path,
        execution_mode="manual_approval",
        telegram_enabled=True,
    )
    proposal_result = run_paper_decision(
        config,
        now=datetime(2026, 4, 10, 20, 10, tzinfo=UTC),
        provider=FakeAlpacaProvider(symbol="VOO"),
        broker=FakeAlpacaBroker(symbol="VOO"),
    )
    calls: list[dict[str, object]] = []
    _configure_notification_env(monkeypatch)

    decision_result = decide_paper_proposal(
        config,
        proposal_id=proposal_result["proposal_id"],
        decision="reject",
        actor="manual",
        rationale="Manual hold for review.",
        now=datetime(2026, 4, 10, 20, 20, tzinfo=UTC),
        notification_transport=_capture_transport(calls),
    )

    approval = json.loads(Path(decision_result["approval_path"]).read_text(encoding="utf-8"))
    assert approval["approval_status"] == "rejected"
    assert len(calls) == 1
    assert "actor: manual" in calls[0]["payload"]["text"]
    assert "outcome: rejected" in calls[0]["payload"]["text"]


def test_get_paper_status_returns_latest_proposal_summary(tmp_path: Path) -> None:
    config = build_phase7_paper_config(tmp_path, execution_mode="autonomous", symbol="QQQ")
    run_paper_decision(
        config,
        now=datetime(2026, 4, 10, 20, 10, tzinfo=UTC),
        provider=FakeAlpacaProvider(symbol="QQQ"),
        broker=FakeAlpacaBroker(symbol="QQQ"),
    )

    status = get_paper_status(config)

    assert status["latest_proposal"] is not None
    assert status["latest_proposal"]["symbol"] == "QQQ"
