from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from tests._paper_fakes import (
    FakeAlpacaBroker,
    FakeAlpacaProvider,
    build_paper_history_frame,
    build_phase7_paper_config,
)

from marketlab.paper.report import run_paper_report
from marketlab.paper.service import (
    decide_paper_proposal,
    run_paper_decision,
    run_paper_submit,
)


def test_run_paper_report_writes_summary_and_decision_journal(tmp_path: Path) -> None:
    config = build_phase7_paper_config(tmp_path, execution_mode="agent_approval", symbol="QQQ")
    broker = FakeAlpacaBroker(symbol="QQQ")
    provider = FakeAlpacaProvider(symbol="QQQ")
    report_provider = FakeAlpacaProvider(
        frame=build_paper_history_frame(end_date="2026-04-13"),
        symbol="QQQ",
    )
    decision = run_paper_decision(
        config,
        now=datetime(2026, 4, 10, 20, 10, tzinfo=UTC),
        provider=provider,
        broker=broker,
    )
    decide_paper_proposal(
        config,
        proposal_id=decision["proposal_id"],
        decision="approve",
        actor="agent",
        provider="deterministic_consensus",
        model="deterministic_consensus",
        rationale="Approve the consensus proposal.",
        now=datetime(2026, 4, 10, 20, 20, tzinfo=UTC),
    )
    run_paper_submit(
        config,
        now=datetime(2026, 4, 10, 23, 5, tzinfo=UTC),
        broker=broker,
    )

    report = run_paper_report(
        config,
        start_date="2026-04-13",
        end_date="2026-04-13",
        provider=report_provider,
    )

    summary_path = Path(report["summary_path"])
    journal_path = Path(report["decision_journal_path"])
    report_path = Path(report["report_path"])

    summary_text = summary_path.read_text(encoding="utf-8")
    journal_text = journal_path.read_text(encoding="utf-8")
    markdown_text = report_path.read_text(encoding="utf-8")

    assert "paper_realized" in summary_text
    assert "consensus" in summary_text
    assert "buy_hold" in summary_text
    assert "sma" in summary_text
    assert "model_logistic_regression" in summary_text
    assert decision["proposal_id"] in journal_text
    assert "Decision Journal" in markdown_text
