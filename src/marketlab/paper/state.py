from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from marketlab.config import ExperimentConfig
from marketlab.paper.core import _now_utc


def _json_dump(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return path


def _json_load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


class PaperStateStore:
    def __init__(self, config: ExperimentConfig) -> None:
        self._config = config
        self.inbox_root = config.paper_approval_inbox_dir
        self.state_root = config.paper_state_dir
        self.notifications_root = self.state_root / "notifications"
        self.trades_root = self.state_root / "trades"
        self.reports_root = self.state_root.parent / "reports"
        self.status_path = self.state_root / "status.json"
        for root in (
            self.inbox_root,
            self.state_root,
            self.notifications_root,
            self.trades_root,
            self.reports_root,
        ):
            root.mkdir(parents=True, exist_ok=True)

    def trade_dir(self, trade_date: str) -> Path:
        path = self.trades_root / trade_date
        path.mkdir(parents=True, exist_ok=True)
        return path

    def trade_proposal_path(self, trade_date: str) -> Path:
        return self.trade_dir(trade_date) / "proposal.json"

    def trade_evidence_path(self, trade_date: str) -> Path:
        return self.trade_dir(trade_date) / "evidence.json"

    def trade_approval_path(self, trade_date: str) -> Path:
        return self.trade_dir(trade_date) / "approval.json"

    def trade_submission_path(self, trade_date: str) -> Path:
        return self.trade_dir(trade_date) / "submission.json"

    def trade_account_snapshot_path(self, trade_date: str) -> Path:
        return self.trade_dir(trade_date) / "account_snapshot.json"

    def trade_order_preview_path(self, trade_date: str) -> Path:
        return self.trade_dir(trade_date) / "order_preview.json"

    def trade_order_status_path(self, trade_date: str) -> Path:
        return self.trade_dir(trade_date) / "order_status.json"

    def inbox_proposal_path(self, proposal_id: str) -> Path:
        return self.inbox_root / f"{proposal_id}.json"

    def report_dir(self, start_date: str, end_date: str) -> Path:
        path = self.reports_root / f"{start_date}_{end_date}"
        path.mkdir(parents=True, exist_ok=True)
        return path

    def notification_record_path(
        self,
        *,
        stage: str,
        outcome: str,
        now: datetime | None = None,
    ) -> Path:
        timestamp = _now_utc(now).strftime("%Y%m%dT%H%M%S%fZ")
        stem = f"{timestamp}_{stage}_{outcome}".replace(" ", "_")
        path = self.notifications_root / f"{stem}.json"
        suffix = 1
        while path.exists():
            path = self.notifications_root / f"{stem}_{suffix}.json"
            suffix += 1
        return path

    def write_notification_record(
        self,
        *,
        stage: str,
        outcome: str,
        payload: dict[str, Any],
        now: datetime | None = None,
    ) -> Path:
        return _json_dump(
            self.notification_record_path(stage=stage, outcome=outcome, now=now),
            payload,
        )

    def write_status(self, payload: dict[str, Any]) -> Path:
        return _json_dump(self.status_path, payload)

    def read_status(self) -> dict[str, Any] | None:
        if not self.status_path.exists():
            return None
        return _json_load(self.status_path)

    def save_evidence(self, evidence: dict[str, Any]) -> Path:
        return _json_dump(self.trade_evidence_path(evidence["effective_date"]), evidence)

    def load_evidence(self, trade_date: str) -> dict[str, Any]:
        return _json_load(self.trade_evidence_path(trade_date))

    def save_proposal(self, proposal: dict[str, Any]) -> Path:
        trade_date = proposal["effective_date"]
        proposal_path = self.trade_proposal_path(trade_date)
        inbox_path = self.inbox_proposal_path(proposal["proposal_id"])
        _json_dump(proposal_path, proposal)
        _json_dump(inbox_path, proposal)
        return proposal_path

    def update_proposal(self, proposal: dict[str, Any]) -> Path:
        return self.save_proposal(proposal)

    def load_proposal(self, proposal_id: str) -> dict[str, Any]:
        path = self.inbox_proposal_path(proposal_id)
        if not path.exists():
            raise FileNotFoundError(f"Unknown proposal_id: {proposal_id}")
        return _json_load(path)

    def list_proposals(self) -> list[dict[str, Any]]:
        proposals = [_json_load(path) for path in sorted(self.inbox_root.glob("*.json"))]
        return sorted(
            proposals,
            key=lambda proposal: (
                proposal.get("effective_date", ""),
                proposal.get("created_at", ""),
                proposal.get("proposal_id", ""),
            ),
            reverse=True,
        )

    def latest_proposal(self) -> dict[str, Any] | None:
        proposals = self.list_proposals()
        if not proposals:
            return None
        return proposals[0]
