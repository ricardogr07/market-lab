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
    run_paper_decision,
    run_paper_submit,
)


def test_run_paper_decision_writes_latest_daily_proposal(tmp_path: Path) -> None:
    config = build_phase7_paper_config(tmp_path, execution_mode="agent_approval")
    provider = FakeAlpacaProvider(symbol="VOO")
    broker = FakeAlpacaBroker(symbol="VOO")

    result = run_paper_decision(
        config,
        now=datetime(2026, 4, 10, 20, 10, tzinfo=UTC),
        provider=provider,
        broker=broker,
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


def test_decide_paper_proposal_records_agent_approval(tmp_path: Path) -> None:
    config = build_phase7_paper_config(tmp_path, execution_mode="agent_approval")
    proposal_result = run_paper_decision(
        config,
        now=datetime(2026, 4, 10, 20, 10, tzinfo=UTC),
        provider=FakeAlpacaProvider(symbol="VOO"),
        broker=FakeAlpacaBroker(symbol="VOO"),
    )

    decision_result = decide_paper_proposal(
        config,
        proposal_id=proposal_result["proposal_id"],
        decision="approve",
        actor="agent",
        now=datetime(2026, 4, 10, 20, 20, tzinfo=UTC),
    )

    approval_path = Path(decision_result["approval_path"])
    approval = json.loads(approval_path.read_text(encoding="utf-8"))
    proposal = PaperStateStore(config).load_proposal(proposal_result["proposal_id"])

    assert approval["decision"] == "approve"
    assert approval["actor"] == "agent"
    assert proposal["approval_status"] == "approved"
    assert proposal["approval_actor"] == "agent"


def test_run_paper_submit_skips_missing_agent_approval(tmp_path: Path) -> None:
    config = build_phase7_paper_config(tmp_path, execution_mode="agent_approval")
    proposal_result = run_paper_decision(
        config,
        now=datetime(2026, 4, 10, 20, 10, tzinfo=UTC),
        provider=FakeAlpacaProvider(symbol="VOO"),
        broker=FakeAlpacaBroker(symbol="VOO"),
    )

    submission_result = run_paper_submit(
        config,
        now=datetime(2026, 4, 10, 23, 5, tzinfo=UTC),
        broker=FakeAlpacaBroker(symbol="VOO"),
    )

    submission = submission_result["submission"]

    assert submission["proposal_id"] == proposal_result["proposal_id"]
    assert submission["status"] == "skipped"
    assert submission["reason"] == "missing_approval"


def test_run_paper_submit_places_fractional_order_after_agent_approval(tmp_path: Path) -> None:
    config = build_phase7_paper_config(tmp_path, execution_mode="agent_approval")
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

    submission_result = run_paper_submit(
        config,
        now=datetime(2026, 4, 10, 23, 5, tzinfo=UTC),
        broker=broker,
    )

    submission = submission_result["submission"]

    assert submission["status"] in {"submitted", "no_trade_required"}
    if submission["status"] == "submitted":
        assert broker.submitted_orders
        assert submission["order_status"] == "accepted"


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
