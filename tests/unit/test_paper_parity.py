from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest
from tests._paper_fakes import build_phase7_paper_config

from marketlab.paper.parity import PaperParityReportError, build_paper_parity_report


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")


def _seed_trade_day(root: Path, trade_date: str, *, target_weight: float = 1.0) -> None:
    base = root / "trades" / trade_date
    _write_json(
        base / "proposal.json",
        {
            "proposal_id": f"qqq-{trade_date}",
            "trade_date": trade_date,
            "symbol": "QQQ",
            "target_weight": target_weight,
        },
    )
    _write_json(
        base / "evidence.json",
        {
            "proposal_id": f"qqq-{trade_date}",
            "trade_date": trade_date,
            "votes": {"long": 6, "cash": 0},
        },
    )
    _write_json(
        base / "approval.json",
        {
            "proposal_id": f"qqq-{trade_date}",
            "decision": "approve",
            "actor": "agent",
        },
    )


def _seed_matching_window(local_root: Path, shadow_root: Path) -> list[str]:
    trade_dates = [
        "2026-06-01",
        "2026-06-02",
        "2026-06-03",
        "2026-06-04",
        "2026-06-05",
        "2026-06-08",
        "2026-06-09",
        "2026-06-10",
        "2026-06-11",
        "2026-06-12",
    ]
    for trade_date in trade_dates:
        _seed_trade_day(local_root, trade_date)
        _seed_trade_day(shadow_root, trade_date)
    _write_json(local_root / "status.json", {"status": "proposal_created"})
    _write_json(shadow_root / "status.json", {"status": "proposal_created"})
    return trade_dates


def test_paper_parity_report_accepts_ten_day_matching_window(tmp_path: Path) -> None:
    config = build_phase7_paper_config(tmp_path, symbol="QQQ")
    local_root = tmp_path / "local" / "state"
    shadow_root = tmp_path / "shadow" / "state"
    _seed_matching_window(local_root, shadow_root)
    report_path = tmp_path / "reports" / "parity.json"
    markdown_path = tmp_path / "reports" / "parity.md"

    report = build_paper_parity_report(
        config,
        local_state_dir=local_root,
        shadow_state_dir=shadow_root,
        start_date="2026-06-01",
        end_date="2026-06-12",
        report_path=report_path,
        markdown_path=markdown_path,
        now=datetime(2026, 6, 15, 12, 0, tzinfo=UTC),
    )

    assert report["accepted"] is True
    assert report["window"]["evidence_window_passed"] is True
    assert report["window"]["max_consecutive_weekdays"] == 10
    assert report["unresolved_difference_count"] == 0
    assert report["differences"] == []
    assert len(report["aggregate_checksum"]) == 64
    assert json.loads(report_path.read_text(encoding="utf-8"))["accepted"] is True
    assert "# QQQ UAT Parity Report" in markdown_path.read_text(encoding="utf-8")


def test_paper_parity_report_requires_explanations_for_differences(
    tmp_path: Path,
) -> None:
    config = build_phase7_paper_config(tmp_path, symbol="QQQ")
    local_root = tmp_path / "local" / "state"
    shadow_root = tmp_path / "shadow" / "state"
    _seed_matching_window(local_root, shadow_root)
    _seed_trade_day(shadow_root, "2026-06-03", target_weight=0.0)

    report = build_paper_parity_report(
        config,
        local_state_dir=local_root,
        shadow_state_dir=shadow_root,
        start_date="2026-06-01",
        end_date="2026-06-12",
    )

    assert report["accepted"] is False
    assert report["unresolved_difference_count"] == 1
    difference = report["differences"][0]
    assert difference["surface"] == "proposal"
    assert difference["status"] == "payload_mismatch"
    assert difference["explanation_status"] == "unexplained"

    explanations_path = tmp_path / "explanations.json"
    _write_json(
        explanations_path,
        {
            difference["id"]: {
                "status": "accepted",
                "explanation": "Expected UAT no-submit preview drift.",
            }
        },
    )
    explained = build_paper_parity_report(
        config,
        local_state_dir=local_root,
        shadow_state_dir=shadow_root,
        start_date="2026-06-01",
        end_date="2026-06-12",
        explanations_path=explanations_path,
    )

    assert explained["accepted"] is True
    assert explained["unresolved_difference_count"] == 0
    assert explained["differences"][0]["explanation_status"] == "accepted"


def test_paper_parity_report_rejects_short_evidence_window(tmp_path: Path) -> None:
    config = build_phase7_paper_config(tmp_path, symbol="QQQ")
    local_root = tmp_path / "local" / "state"
    shadow_root = tmp_path / "shadow" / "state"
    for root in (local_root, shadow_root):
        _seed_trade_day(root, "2026-06-01")
        _seed_trade_day(root, "2026-06-02")

    report = build_paper_parity_report(
        config,
        local_state_dir=local_root,
        shadow_state_dir=shadow_root,
        start_date="2026-06-01",
        end_date="2026-06-12",
    )

    assert report["accepted"] is False
    assert report["window"]["evidence_window_passed"] is False
    assert report["window"]["max_consecutive_weekdays"] == 2


def test_paper_parity_report_rejects_malformed_json(tmp_path: Path) -> None:
    config = build_phase7_paper_config(tmp_path, symbol="QQQ")
    local_root = tmp_path / "local" / "state"
    shadow_root = tmp_path / "shadow" / "state"
    _seed_trade_day(local_root, "2026-06-01")
    _seed_trade_day(shadow_root, "2026-06-01")
    (shadow_root / "trades" / "2026-06-01" / "proposal.json").write_text(
        "{bad",
        encoding="utf-8",
    )

    with pytest.raises(PaperParityReportError, match="Malformed JSON"):
        build_paper_parity_report(
            config,
            local_state_dir=local_root,
            shadow_state_dir=shadow_root,
            start_date="2026-06-01",
            end_date="2026-06-12",
        )


def test_paper_parity_report_requires_qqq_paper_config(tmp_path: Path) -> None:
    config = build_phase7_paper_config(tmp_path, symbol="VOO")
    local_root = tmp_path / "local" / "state"
    shadow_root = tmp_path / "shadow" / "state"
    local_root.mkdir(parents=True)
    shadow_root.mkdir(parents=True)

    with pytest.raises(ValueError, match="QQQ"):
        build_paper_parity_report(
            config,
            local_state_dir=local_root,
            shadow_state_dir=shadow_root,
            start_date="2026-06-01",
            end_date="2026-06-12",
        )
