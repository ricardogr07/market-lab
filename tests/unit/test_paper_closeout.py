from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest
from tests._paper_fakes import build_phase7_paper_config

from marketlab.paper.closeout import (
    PaperCloseoutReportError,
    build_paper_closeout_report,
)

TRADE_DATES = [
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


def _write_json(path: Path, payload: dict[str, object] | list[object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")


def _seed_trade_day(
    state_root: Path,
    artifact_root: Path,
    trade_date: str,
    *,
    order_id: str | None = None,
) -> None:
    base = state_root / "trades" / trade_date
    _write_json(
        base / "proposal.json",
        {
            "proposal_id": f"qqq-{trade_date}",
            "effective_date": trade_date,
            "symbol": "QQQ",
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
            "approval_status": "approved",
        },
    )
    _write_json(
        base / "submission.json",
        {
            "proposal_id": f"qqq-{trade_date}",
            "trade_date": trade_date,
            "status": "submitted",
            "order_id": order_id or f"order-{trade_date}",
            "client_order_id": f"client-{trade_date}",
        },
    )
    _write_json(
        base / "order_status.json",
        {
            "id": order_id or f"order-{trade_date}",
            "status": "filled",
        },
    )
    _write_json(
        artifact_root / "notifications" / trade_date / "submission.json",
        {"trade_date": trade_date, "delivery_status": "delivered"},
    )
    (artifact_root / "reports" / trade_date / "paper_report.md").parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    (artifact_root / "reports" / trade_date / "paper_report.md").write_text(
        f"# QQQ {trade_date}\n",
        encoding="utf-8",
    )


def _seed_matching_closeout_window(
    state_root: Path,
    artifact_root: Path,
) -> Path:
    for trade_date in TRADE_DATES:
        _seed_trade_day(state_root, artifact_root, trade_date)
    for filename in (
        "alerts.json",
        "dead_letters.json",
        "failed_jobs.json",
        "non_terminal_orders.json",
    ):
        _write_json(artifact_root / filename, {"items": []})
    rollback_path = artifact_root / "rollback-evidence.json"
    _write_json(
        rollback_path,
        {
            "status": "accepted",
            "evidence_uri": "https://example.invalid/rollback/qqq",
        },
    )
    return rollback_path


def test_paper_closeout_report_accepts_ten_day_observation_window(
    tmp_path: Path,
) -> None:
    config = build_phase7_paper_config(tmp_path, symbol="QQQ")
    state_root = tmp_path / "export" / "state"
    artifact_root = tmp_path / "export" / "artifacts"
    rollback_path = _seed_matching_closeout_window(state_root, artifact_root)
    report_path = tmp_path / "closeout.json"
    markdown_path = tmp_path / "closeout.md"

    report = build_paper_closeout_report(
        config,
        paper_prod_state_dir=state_root,
        paper_prod_artifact_dir=artifact_root,
        start_date="2026-06-01",
        end_date="2026-06-12",
        rollback_evidence_path=rollback_path,
        report_path=report_path,
        markdown_path=markdown_path,
        now=datetime(2026, 6, 15, 12, 0, tzinfo=UTC),
    )

    assert report["accepted"] is True
    assert report["window"]["evidence_window_passed"] is True
    assert report["window"]["max_consecutive_weekdays"] == 10
    assert report["counts"]["duplicate_broker_submissions"] == 0
    assert report["counts"]["unresolved_operational_items"] == 0
    assert report["counts"]["missing_evidence"] == 0
    assert len(report["aggregate_checksum"]) == 64
    assert json.loads(report_path.read_text(encoding="utf-8"))["accepted"] is True
    assert "# QQQ Post-Cutover Closeout Report" in markdown_path.read_text(
        encoding="utf-8"
    )


def test_paper_closeout_report_rejects_short_observation_window(tmp_path: Path) -> None:
    config = build_phase7_paper_config(tmp_path, symbol="QQQ")
    state_root = tmp_path / "export" / "state"
    artifact_root = tmp_path / "export" / "artifacts"
    rollback_path = _seed_matching_closeout_window(state_root, artifact_root)

    report = build_paper_closeout_report(
        config,
        paper_prod_state_dir=state_root,
        paper_prod_artifact_dir=artifact_root,
        start_date="2026-06-01",
        end_date="2026-06-12",
        min_trading_days=11,
        rollback_evidence_path=rollback_path,
    )

    assert report["accepted"] is False
    assert report["window"]["evidence_window_passed"] is False
    assert report["window"]["max_consecutive_weekdays"] == 10


def test_paper_closeout_report_blocks_missing_expected_evidence(
    tmp_path: Path,
) -> None:
    config = build_phase7_paper_config(tmp_path, symbol="QQQ")
    state_root = tmp_path / "export" / "state"
    artifact_root = tmp_path / "export" / "artifacts"
    rollback_path = _seed_matching_closeout_window(state_root, artifact_root)
    (state_root / "trades" / "2026-06-03" / "approval.json").unlink()

    report = build_paper_closeout_report(
        config,
        paper_prod_state_dir=state_root,
        paper_prod_artifact_dir=artifact_root,
        start_date="2026-06-01",
        end_date="2026-06-12",
        rollback_evidence_path=rollback_path,
    )

    assert report["accepted"] is False
    assert report["counts"]["missing_evidence"] == 1
    assert report["missing_evidence"][0]["surface"] == "approval"


def test_paper_closeout_report_detects_duplicate_broker_submissions(
    tmp_path: Path,
) -> None:
    config = build_phase7_paper_config(tmp_path, symbol="QQQ")
    state_root = tmp_path / "export" / "state"
    artifact_root = tmp_path / "export" / "artifacts"
    rollback_path = _seed_matching_closeout_window(state_root, artifact_root)
    _seed_trade_day(state_root, artifact_root, "2026-06-04", order_id="duplicate-order")
    _seed_trade_day(state_root, artifact_root, "2026-06-05", order_id="duplicate-order")

    report = build_paper_closeout_report(
        config,
        paper_prod_state_dir=state_root,
        paper_prod_artifact_dir=artifact_root,
        start_date="2026-06-01",
        end_date="2026-06-12",
        rollback_evidence_path=rollback_path,
    )

    assert report["accepted"] is False
    assert report["duplicate_broker_submissions"][0]["identifier"] == "duplicate-order"


@pytest.mark.parametrize(
    ("filename", "issue_type"),
    [
        ("alerts.json", "alerts"),
        ("dead_letters.json", "dead_letters"),
        ("failed_jobs.json", "failed_jobs"),
        ("non_terminal_orders.json", "non_terminal_orders"),
    ],
)
def test_paper_closeout_report_blocks_unresolved_operational_items(
    tmp_path: Path,
    filename: str,
    issue_type: str,
) -> None:
    config = build_phase7_paper_config(tmp_path, symbol="QQQ")
    state_root = tmp_path / "export" / "state"
    artifact_root = tmp_path / "export" / "artifacts"
    rollback_path = _seed_matching_closeout_window(state_root, artifact_root)
    _write_json(
        artifact_root / filename,
        {"items": [{"id": "item-1", "status": "open"}]},
    )

    report = build_paper_closeout_report(
        config,
        paper_prod_state_dir=state_root,
        paper_prod_artifact_dir=artifact_root,
        start_date="2026-06-01",
        end_date="2026-06-12",
        rollback_evidence_path=rollback_path,
    )

    assert report["accepted"] is False
    assert report["unresolved_operational_items"][0]["type"] == issue_type


def test_paper_closeout_report_requires_accepted_rollback_evidence(
    tmp_path: Path,
) -> None:
    config = build_phase7_paper_config(tmp_path, symbol="QQQ")
    state_root = tmp_path / "export" / "state"
    artifact_root = tmp_path / "export" / "artifacts"
    _seed_matching_closeout_window(state_root, artifact_root)

    missing_rollback = build_paper_closeout_report(
        config,
        paper_prod_state_dir=state_root,
        paper_prod_artifact_dir=artifact_root,
        start_date="2026-06-01",
        end_date="2026-06-12",
    )

    assert missing_rollback["accepted"] is False
    assert missing_rollback["rollback_evidence"]["status"] == "missing"

    rejected_path = artifact_root / "rollback-rejected.json"
    _write_json(rejected_path, {"status": "blocking"})
    rejected_rollback = build_paper_closeout_report(
        config,
        paper_prod_state_dir=state_root,
        paper_prod_artifact_dir=artifact_root,
        start_date="2026-06-01",
        end_date="2026-06-12",
        rollback_evidence_path=rejected_path,
    )

    assert rejected_rollback["accepted"] is False
    assert rejected_rollback["rollback_evidence"]["accepted"] is False


def test_paper_closeout_report_rejects_malformed_json(tmp_path: Path) -> None:
    config = build_phase7_paper_config(tmp_path, symbol="QQQ")
    state_root = tmp_path / "export" / "state"
    artifact_root = tmp_path / "export" / "artifacts"
    rollback_path = _seed_matching_closeout_window(state_root, artifact_root)
    (state_root / "trades" / "2026-06-01" / "proposal.json").write_text(
        "{bad",
        encoding="utf-8",
    )

    with pytest.raises(PaperCloseoutReportError, match="Malformed JSON"):
        build_paper_closeout_report(
            config,
            paper_prod_state_dir=state_root,
            paper_prod_artifact_dir=artifact_root,
            start_date="2026-06-01",
            end_date="2026-06-12",
            rollback_evidence_path=rollback_path,
        )


def test_paper_closeout_report_requires_qqq_paper_config(tmp_path: Path) -> None:
    config = build_phase7_paper_config(tmp_path, symbol="VOO")
    state_root = tmp_path / "export" / "state"
    artifact_root = tmp_path / "export" / "artifacts"
    state_root.mkdir(parents=True)
    artifact_root.mkdir(parents=True)

    with pytest.raises(ValueError, match="QQQ"):
        build_paper_closeout_report(
            config,
            paper_prod_state_dir=state_root,
            paper_prod_artifact_dir=artifact_root,
            start_date="2026-06-01",
            end_date="2026-06-12",
        )
