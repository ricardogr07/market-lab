from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from tests._paper_fakes import build_phase7_paper_config

from marketlab.paper import scheduler


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
