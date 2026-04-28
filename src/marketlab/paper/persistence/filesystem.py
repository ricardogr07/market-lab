from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

from marketlab.config import ExperimentConfig
from marketlab.paper.contracts import (
    PaperArtifactStore,
    PaperStatusRepository,
    PaperTradeRepository,
    PaperUnitOfWork,
    PaperUnitOfWorkFactory,
)
from marketlab.paper.core import _now_utc
from marketlab.paper.state import PaperStateStore, _json_dump, _json_load


class FilesystemPaperTradeRepository(PaperTradeRepository):
    def __init__(
        self,
        store: PaperStateStore,
        pending_writes: dict[Path, dict[str, Any]],
    ) -> None:
        self._store = store
        self._pending_writes = pending_writes

    def _load_path(self, path: Path) -> dict[str, Any] | None:
        payload = self._pending_writes.get(path)
        if payload is not None:
            return dict(payload)
        if not path.exists():
            return None
        return _json_load(path)

    def _stage(self, path: Path, payload: dict[str, Any]) -> Path:
        self._pending_writes[path] = dict(payload)
        return path

    def list_proposals(self) -> list[dict[str, Any]]:
        proposal_paths = {
            *sorted(self._store.inbox_root.glob("*.json")),
            *(
                path
                for path in self._pending_writes
                if path.parent == self._store.inbox_root and path.suffix == ".json"
            ),
        }
        proposals = [
            proposal
            for path in sorted(proposal_paths)
            if (proposal := self._load_path(path)) is not None
        ]
        return sorted(
            proposals,
            key=lambda proposal: (
                proposal.get("effective_date", ""),
                proposal.get("created_at", ""),
                proposal.get("proposal_id", ""),
            ),
            reverse=True,
        )

    def get_latest_proposal(self) -> dict[str, Any] | None:
        proposals = self.list_proposals()
        if not proposals:
            return None
        return proposals[0]

    def get_proposal(self, proposal_id: str) -> dict[str, Any] | None:
        return self._load_path(self._store.inbox_proposal_path(proposal_id))

    def get_evidence(self, trade_date: str) -> dict[str, Any] | None:
        return self._load_path(self._store.trade_evidence_path(trade_date))

    def get_submission(self, trade_date: str) -> dict[str, Any] | None:
        return self._load_path(self._store.trade_submission_path(trade_date))

    def save_evidence(self, evidence: dict[str, Any]) -> Path:
        return self._stage(self._store.trade_evidence_path(str(evidence["effective_date"])), evidence)

    def save_proposal(self, proposal: dict[str, Any]) -> Path:
        trade_date = str(proposal["effective_date"])
        proposal_id = str(proposal["proposal_id"])
        proposal_path = self._store.trade_proposal_path(trade_date)
        inbox_path = self._store.inbox_proposal_path(proposal_id)
        self._stage(proposal_path, proposal)
        self._stage(inbox_path, proposal)
        return proposal_path

    def save_approval(self, *, trade_date: str, approval: dict[str, Any]) -> Path:
        return self._stage(self._store.trade_approval_path(trade_date), approval)

    def save_submission(self, *, trade_date: str, submission: dict[str, Any]) -> Path:
        return self._stage(self._store.trade_submission_path(trade_date), submission)

    def save_order_status(self, *, trade_date: str, order_status: dict[str, Any]) -> Path:
        return self._stage(self._store.trade_order_status_path(trade_date), order_status)

    def proposal_path(self, proposal_id: str) -> Path:
        return self._store.inbox_proposal_path(proposal_id)

    def trade_evidence_path(self, trade_date: str) -> Path:
        return self._store.trade_evidence_path(trade_date)

    def trade_submission_path(self, trade_date: str) -> Path:
        return self._store.trade_submission_path(trade_date)

    def trade_order_status_path(self, trade_date: str) -> Path:
        return self._store.trade_order_status_path(trade_date)

    def backup_submission_attempt_artifacts(
        self,
        *,
        trade_date: str,
        now: datetime | None = None,
    ) -> None:
        timestamp = _now_utc(now).strftime("%Y%m%dT%H%M%S%fZ")
        for path in (
            self._store.trade_submission_path(trade_date),
            self._store.trade_order_status_path(trade_date),
            self._store.trade_order_preview_path(trade_date),
            self._store.trade_account_snapshot_path(trade_date),
        ):
            if not path.exists():
                continue
            backup_path = path.with_name(f"{path.stem}.retry-backup.{timestamp}.bak")
            path.rename(backup_path)


class FilesystemPaperStatusRepository(PaperStatusRepository):
    def __init__(
        self,
        store: PaperStateStore,
        pending_writes: dict[Path, dict[str, Any]],
    ) -> None:
        self._store = store
        self._pending_writes = pending_writes

    @property
    def status_path(self) -> Path:
        return self._store.status_path

    def read_status(self) -> dict[str, Any] | None:
        payload = self._pending_writes.get(self.status_path)
        if payload is not None:
            return dict(payload)
        if not self.status_path.exists():
            return None
        return _json_load(self.status_path)

    def write_status(self, payload: dict[str, Any]) -> Path:
        self._pending_writes[self.status_path] = dict(payload)
        return self.status_path


class FilesystemPaperUnitOfWork(PaperUnitOfWork):
    def __init__(self, config: ExperimentConfig) -> None:
        self._store = PaperStateStore(config)
        self._pending_writes: dict[Path, dict[str, Any]] = {}
        self._trades = FilesystemPaperTradeRepository(self._store, self._pending_writes)
        self._status = FilesystemPaperStatusRepository(self._store, self._pending_writes)
        self._committed = False

    @property
    def trades(self) -> PaperTradeRepository:
        return self._trades

    @property
    def status(self) -> PaperStatusRepository:
        return self._status

    def commit(self) -> None:
        for path, payload in self._pending_writes.items():
            _json_dump(path, payload)
        self._pending_writes.clear()
        self._committed = True

    def rollback(self) -> None:
        self._pending_writes.clear()

    def __enter__(self) -> FilesystemPaperUnitOfWork:
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        if exc_type is not None or not self._committed:
            self.rollback()


class FilesystemPaperUnitOfWorkFactory(PaperUnitOfWorkFactory):
    def __init__(self, config: ExperimentConfig) -> None:
        self._config = config

    def __call__(self) -> PaperUnitOfWork:
        return FilesystemPaperUnitOfWork(self._config)


def build_filesystem_paper_uow_factory(config: ExperimentConfig) -> PaperUnitOfWorkFactory:
    return FilesystemPaperUnitOfWorkFactory(config)


class FilesystemPaperArtifactStore(PaperArtifactStore):
    def __init__(self, config: ExperimentConfig) -> None:
        self._store = PaperStateStore(config)

    def write_trade_account_snapshot(
        self,
        *,
        trade_date: str,
        payload: dict[str, Any],
    ) -> Path:
        return _json_dump(self._store.trade_account_snapshot_path(trade_date), payload)

    def write_trade_order_preview(
        self,
        *,
        trade_date: str,
        payload: dict[str, Any],
    ) -> Path:
        return _json_dump(self._store.trade_order_preview_path(trade_date), payload)


def build_filesystem_paper_artifact_store(config: ExperimentConfig) -> PaperArtifactStore:
    return FilesystemPaperArtifactStore(config)
