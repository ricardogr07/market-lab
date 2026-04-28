from __future__ import annotations

import copy
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
from tests._paper_fakes import FakeAlpacaBroker, build_phase7_paper_config

import marketlab.paper.service as service_module
from marketlab.paper.application import ReconciliationService
from marketlab.paper.contracts import (
    PaperReconciliationRequest,
    PaperStatusRepository,
    PaperTradeRepository,
    PaperUnitOfWork,
    PaperUnitOfWorkFactory,
)
from marketlab.paper.persistence import (
    build_filesystem_paper_uow_factory,
    write_trade_account_snapshot,
    write_trade_order_preview,
)
from marketlab.paper.state import PaperStateStore


@dataclass
class _InMemoryStoreState:
    status: dict[str, Any] | None = None
    proposals_by_id: dict[str, dict[str, Any]] = field(default_factory=dict)
    evidence_by_trade_date: dict[str, dict[str, Any]] = field(default_factory=dict)
    approvals_by_trade_date: dict[str, dict[str, Any]] = field(default_factory=dict)
    submissions_by_trade_date: dict[str, dict[str, Any]] = field(default_factory=dict)
    order_status_by_trade_date: dict[str, dict[str, Any]] = field(default_factory=dict)
    backup_calls: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class _PendingState:
    status: dict[str, Any] | None = None
    proposals_by_id: dict[str, dict[str, Any]] = field(default_factory=dict)
    evidence_by_trade_date: dict[str, dict[str, Any]] = field(default_factory=dict)
    approvals_by_trade_date: dict[str, dict[str, Any]] = field(default_factory=dict)
    submissions_by_trade_date: dict[str, dict[str, Any]] = field(default_factory=dict)
    order_status_by_trade_date: dict[str, dict[str, Any]] = field(default_factory=dict)


class InMemoryPaperTradeRepository(PaperTradeRepository):
    def __init__(self, root: Path, state: _InMemoryStoreState, pending: _PendingState) -> None:
        self._root = root
        self._state = state
        self._pending = pending

    def _proposal_payload(self, proposal_id: str) -> dict[str, Any] | None:
        if proposal_id in self._pending.proposals_by_id:
            return copy.deepcopy(self._pending.proposals_by_id[proposal_id])
        payload = self._state.proposals_by_id.get(proposal_id)
        if payload is None:
            return None
        return copy.deepcopy(payload)

    def _trade_payload(
        self,
        *,
        trade_date: str,
        pending_map: dict[str, dict[str, Any]],
        state_map: dict[str, dict[str, Any]],
    ) -> dict[str, Any] | None:
        if trade_date in pending_map:
            return copy.deepcopy(pending_map[trade_date])
        payload = state_map.get(trade_date)
        if payload is None:
            return None
        return copy.deepcopy(payload)

    def list_proposals(self) -> list[dict[str, Any]]:
        proposal_ids = set(self._state.proposals_by_id) | set(self._pending.proposals_by_id)
        proposals = [
            proposal
            for proposal_id in proposal_ids
            if (proposal := self._proposal_payload(proposal_id)) is not None
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
        return self._proposal_payload(proposal_id)

    def get_evidence(self, trade_date: str) -> dict[str, Any] | None:
        return self._trade_payload(
            trade_date=trade_date,
            pending_map=self._pending.evidence_by_trade_date,
            state_map=self._state.evidence_by_trade_date,
        )

    def get_submission(self, trade_date: str) -> dict[str, Any] | None:
        return self._trade_payload(
            trade_date=trade_date,
            pending_map=self._pending.submissions_by_trade_date,
            state_map=self._state.submissions_by_trade_date,
        )

    def save_evidence(self, evidence: dict[str, Any]) -> Path:
        trade_date = str(evidence["effective_date"])
        self._pending.evidence_by_trade_date[trade_date] = copy.deepcopy(evidence)
        return self.trade_evidence_path(trade_date)

    def save_proposal(self, proposal: dict[str, Any]) -> Path:
        proposal_id = str(proposal["proposal_id"])
        self._pending.proposals_by_id[proposal_id] = copy.deepcopy(proposal)
        return self._trade_path(str(proposal["effective_date"])) / "proposal.json"

    def save_approval(self, *, trade_date: str, approval: dict[str, Any]) -> Path:
        self._pending.approvals_by_trade_date[trade_date] = copy.deepcopy(approval)
        return self._trade_path(trade_date) / "approval.json"

    def save_submission(self, *, trade_date: str, submission: dict[str, Any]) -> Path:
        self._pending.submissions_by_trade_date[trade_date] = copy.deepcopy(submission)
        return self.trade_submission_path(trade_date)

    def save_order_status(self, *, trade_date: str, order_status: dict[str, Any]) -> Path:
        self._pending.order_status_by_trade_date[trade_date] = copy.deepcopy(order_status)
        return self.trade_order_status_path(trade_date)

    def proposal_path(self, proposal_id: str) -> Path:
        return self._root / "artifacts" / "paper" / "inbox" / f"{proposal_id}.json"

    def trade_evidence_path(self, trade_date: str) -> Path:
        return self._trade_path(trade_date) / "evidence.json"

    def trade_submission_path(self, trade_date: str) -> Path:
        return self._trade_path(trade_date) / "submission.json"

    def trade_order_status_path(self, trade_date: str) -> Path:
        return self._trade_path(trade_date) / "order_status.json"

    def backup_submission_attempt_artifacts(
        self,
        *,
        trade_date: str,
        now: datetime | None = None,
    ) -> None:
        self._state.backup_calls.append(
            {
                "trade_date": trade_date,
                "timestamp": datetime.now(UTC).isoformat() if now is None else now.isoformat(),
            }
        )

    def _trade_path(self, trade_date: str) -> Path:
        return self._root / "artifacts" / "paper" / "state" / "trades" / trade_date


class InMemoryPaperStatusRepository(PaperStatusRepository):
    def __init__(self, root: Path, state: _InMemoryStoreState, pending: _PendingState) -> None:
        self._root = root
        self._state = state
        self._pending = pending

    @property
    def status_path(self) -> Path:
        return self._root / "artifacts" / "paper" / "state" / "status.json"

    def read_status(self) -> dict[str, Any] | None:
        if self._pending.status is not None:
            return copy.deepcopy(self._pending.status)
        if self._state.status is None:
            return None
        return copy.deepcopy(self._state.status)

    def write_status(self, payload: dict[str, Any]) -> Path:
        self._pending.status = copy.deepcopy(payload)
        return self.status_path


class InMemoryPaperUnitOfWork(PaperUnitOfWork):
    def __init__(self, factory: InMemoryPaperUnitOfWorkFactory) -> None:
        self._factory = factory
        self._pending = _PendingState()
        self._trades = InMemoryPaperTradeRepository(factory.root, factory.state, self._pending)
        self._status = InMemoryPaperStatusRepository(factory.root, factory.state, self._pending)
        self._committed = False

    @property
    def trades(self) -> PaperTradeRepository:
        return self._trades

    @property
    def status(self) -> PaperStatusRepository:
        return self._status

    def commit(self) -> None:
        self._factory.state.proposals_by_id.update(copy.deepcopy(self._pending.proposals_by_id))
        self._factory.state.evidence_by_trade_date.update(copy.deepcopy(self._pending.evidence_by_trade_date))
        self._factory.state.approvals_by_trade_date.update(copy.deepcopy(self._pending.approvals_by_trade_date))
        self._factory.state.submissions_by_trade_date.update(copy.deepcopy(self._pending.submissions_by_trade_date))
        self._factory.state.order_status_by_trade_date.update(copy.deepcopy(self._pending.order_status_by_trade_date))
        if self._pending.status is not None:
            self._factory.state.status = copy.deepcopy(self._pending.status)
        self._pending = _PendingState()
        self._factory.commit_count += 1
        self._committed = True

    def rollback(self) -> None:
        self._pending = _PendingState()

    def __enter__(self) -> InMemoryPaperUnitOfWork:
        self._factory.active_count += 1
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        if exc_type is not None or not self._committed:
            self.rollback()
        self._factory.active_count -= 1


class InMemoryPaperUnitOfWorkFactory(PaperUnitOfWorkFactory):
    def __init__(self, root: Path) -> None:
        self.root = root
        self.state = _InMemoryStoreState()
        self.active_count = 0
        self.commit_count = 0

    def __call__(self) -> PaperUnitOfWork:
        return InMemoryPaperUnitOfWork(self)


def _proposal_payload(
    *,
    proposal_id: str,
    trade_date: str,
    created_at: str,
    approval_status: str = "approved",
    approval_actor: str = "agent",
) -> dict[str, Any]:
    return {
        "proposal_id": proposal_id,
        "experiment_name": "phase7_paper_fixture",
        "symbol": "QQQ",
        "signal_date": "2026-04-10",
        "effective_date": trade_date,
        "reference_price": 640.41,
        "execution_mode": "agent_approval",
        "approval_status": approval_status,
        "approval_actor": approval_actor,
        "submission_status": "pending",
        "min_score_threshold": 0.55,
        "train_rows": 250,
        "train_start": "2023-04-10",
        "train_end": "2026-04-09",
        "train_positive_rate": 0.55,
        "created_at": created_at,
        "data_provider": "alpaca",
        "broker": "alpaca",
        "evidence_path": f"/memory/{trade_date}/evidence.json",
        "decision_policy": "consensus_vote",
        "consensus_rule": {
            "type": "consensus_vote",
            "min_long_votes": 4,
            "model_count": 6,
        },
        "long_vote_count": 4,
        "cash_vote_count": 2,
        "decision": "long",
        "target_weight": 1.0,
    }


def _evidence_payload(*, proposal_id: str, trade_date: str) -> dict[str, Any]:
    return {
        "proposal_id": proposal_id,
        "symbol": "QQQ",
        "signal_date": "2026-04-10",
        "effective_date": trade_date,
        "feature_columns": ["feature_1"],
        "train_rows": 250,
        "train_start": "2023-04-10",
        "train_end": "2026-04-09",
        "train_positive_rate": 0.55,
        "min_score_threshold": 0.55,
        "reference_price": 640.41,
        "models": [
            {
                "model_name": "logistic_regression",
                "estimator_label": "logreg",
                "score": 0.61,
                "vote": "long",
                "target_weight": 1.0,
            }
        ],
        "decision_policy": "consensus_vote",
        "consensus_rule": {
            "type": "consensus_vote",
            "min_long_votes": 4,
            "model_count": 6,
        },
        "long_vote_count": 4,
        "cash_vote_count": 2,
        "decision": "long",
        "target_weight": 1.0,
        "created_at": "2026-04-10T20:10:00+00:00",
    }


def _build_factory(
    *,
    adapter_kind: str,
    tmp_path: Path,
    config,
) -> PaperUnitOfWorkFactory:
    if adapter_kind == "filesystem":
        return build_filesystem_paper_uow_factory(config)
    if adapter_kind == "memory":
        return InMemoryPaperUnitOfWorkFactory(tmp_path / "memory-root")
    raise ValueError(f"Unknown adapter kind: {adapter_kind}")


@pytest.mark.parametrize("adapter_kind", ["filesystem", "memory"])
def test_paper_repository_contract_stages_until_commit(adapter_kind: str, tmp_path: Path) -> None:
    config = build_phase7_paper_config(tmp_path / adapter_kind, symbol="QQQ")
    factory = _build_factory(adapter_kind=adapter_kind, tmp_path=tmp_path, config=config)
    proposal = _proposal_payload(
        proposal_id="proposal-1",
        trade_date="2026-04-13",
        created_at="2026-04-10T20:10:00+00:00",
    )
    evidence = _evidence_payload(
        proposal_id=proposal["proposal_id"],
        trade_date="2026-04-13",
    )
    status = {
        "event": "paper-decision",
        "status": "proposal_created",
        "proposal_id": proposal["proposal_id"],
    }

    with factory() as uow:
        uow.trades.save_evidence(evidence)
        uow.trades.save_proposal(proposal)
        uow.status.write_status(status)
        assert uow.trades.get_proposal(proposal["proposal_id"]) == proposal
        assert uow.trades.get_evidence("2026-04-13") == evidence
        assert uow.status.read_status() == status

    with factory() as uow:
        assert uow.trades.get_proposal(proposal["proposal_id"]) is None
        assert uow.trades.get_evidence("2026-04-13") is None
        assert uow.status.read_status() is None


@pytest.mark.parametrize("adapter_kind", ["filesystem", "memory"])
def test_paper_repository_contract_persists_and_orders_records(adapter_kind: str, tmp_path: Path) -> None:
    config = build_phase7_paper_config(tmp_path / f"{adapter_kind}-persist", symbol="QQQ")
    factory = _build_factory(adapter_kind=adapter_kind, tmp_path=tmp_path, config=config)
    older = _proposal_payload(
        proposal_id="proposal-older",
        trade_date="2026-04-13",
        created_at="2026-04-10T20:10:00+00:00",
    )
    newer = _proposal_payload(
        proposal_id="proposal-newer",
        trade_date="2026-04-14",
        created_at="2026-04-11T20:10:00+00:00",
    )
    submission = {
        "proposal_id": newer["proposal_id"],
        "trade_date": "2026-04-14",
        "status": "submitted",
        "order_status": "accepted",
    }
    order_status = {
        "id": "order-1",
        "status": "accepted",
    }
    status = {
        "event": "paper-submit",
        "status": "submitted",
        "proposal_id": newer["proposal_id"],
    }

    with factory() as uow:
        uow.trades.save_proposal(older)
        uow.trades.save_evidence(_evidence_payload(proposal_id=older["proposal_id"], trade_date="2026-04-13"))
        uow.commit()

    with factory() as uow:
        uow.trades.save_proposal(newer)
        uow.trades.save_evidence(_evidence_payload(proposal_id=newer["proposal_id"], trade_date="2026-04-14"))
        uow.trades.save_submission(trade_date="2026-04-14", submission=submission)
        uow.trades.save_order_status(trade_date="2026-04-14", order_status=order_status)
        uow.status.write_status(status)
        uow.commit()

    with factory() as uow:
        proposals = uow.trades.list_proposals()
        assert [proposal["proposal_id"] for proposal in proposals] == [
            newer["proposal_id"],
            older["proposal_id"],
        ]
        assert uow.trades.get_latest_proposal()["proposal_id"] == newer["proposal_id"]
        assert uow.trades.get_submission("2026-04-14") == submission
        assert uow.status.read_status() == status


def test_filesystem_trade_repository_retry_backup_preserves_attempt_artifacts(tmp_path: Path) -> None:
    config = build_phase7_paper_config(tmp_path, symbol="QQQ")
    factory = build_filesystem_paper_uow_factory(config)
    trade_date = "2026-04-13"
    store = PaperStateStore(config)

    with factory() as uow:
        uow.trades.save_submission(
            trade_date=trade_date,
            submission={"proposal_id": "proposal-1", "trade_date": trade_date, "status": "submitted"},
        )
        uow.trades.save_order_status(
            trade_date=trade_date,
            order_status={"id": "order-1", "status": "accepted"},
        )
        uow.commit()
    write_trade_order_preview(
        config,
        trade_date=trade_date,
        payload={"proposal_id": "proposal-1", "trade_date": trade_date, "side": "buy"},
    )
    write_trade_account_snapshot(
        config,
        trade_date=trade_date,
        payload={"equity": "1000.00"},
    )

    with factory() as uow:
        uow.trades.backup_submission_attempt_artifacts(
            trade_date=trade_date,
            now=datetime(2026, 4, 10, 23, 10, tzinfo=UTC),
        )

    trade_dir = store.trade_dir(trade_date)
    backup_files = sorted(path.name for path in trade_dir.glob("*.retry-backup.*.bak"))
    assert len(backup_files) == 4
    assert not store.trade_submission_path(trade_date).exists()
    assert not store.trade_order_status_path(trade_date).exists()
    assert not store.trade_order_preview_path(trade_date).exists()
    assert not store.trade_account_snapshot_path(trade_date).exists()


def test_read_helpers_use_repository_boundary(monkeypatch, tmp_path: Path) -> None:
    config = build_phase7_paper_config(tmp_path, symbol="QQQ")
    factory = InMemoryPaperUnitOfWorkFactory(tmp_path / "memory-read")
    proposal = _proposal_payload(
        proposal_id="proposal-read",
        trade_date="2026-04-13",
        created_at="2026-04-10T20:10:00+00:00",
        approval_status="pending",
        approval_actor="",
    )
    evidence = _evidence_payload(proposal_id=proposal["proposal_id"], trade_date="2026-04-13")
    status = {"event": "paper-decision", "status": "proposal_created", "proposal_id": proposal["proposal_id"]}

    with factory() as uow:
        uow.trades.save_proposal(proposal)
        uow.trades.save_evidence(evidence)
        uow.status.write_status(status)
        uow.commit()

    monkeypatch.setattr(service_module, "_paper_uow_factory", lambda _config: factory)

    proposals = service_module.list_paper_proposals(config)
    loaded_proposal = service_module.read_paper_proposal(config, proposal_id=proposal["proposal_id"])
    loaded_evidence = service_module.read_paper_evidence(config, proposal_id=proposal["proposal_id"])
    summary = service_module.get_paper_status(config)

    assert [item["proposal_id"] for item in proposals] == [proposal["proposal_id"]]
    assert loaded_proposal == proposal
    assert loaded_evidence == evidence
    assert summary["latest_proposal"]["proposal_id"] == proposal["proposal_id"]
    assert summary["pending_proposal_count"] == 1
    assert summary["status"] == status


def _seed_submission_ready_proposal(factory: InMemoryPaperUnitOfWorkFactory) -> dict[str, Any]:
    proposal = _proposal_payload(
        proposal_id="proposal-submit",
        trade_date="2026-04-13",
        created_at="2026-04-10T20:10:00+00:00",
        approval_status="approved",
        approval_actor="agent",
    )
    with factory() as uow:
        uow.trades.save_proposal(proposal)
        uow.commit()
    return proposal


class _BoundaryBroker(FakeAlpacaBroker):
    def __init__(self, *, factory: InMemoryPaperUnitOfWorkFactory) -> None:
        super().__init__(
            symbol="QQQ",
            equity=1000.0,
            buying_power=1000.0,
            cash=1000.0,
        )
        self._factory = factory
        self.calls: list[str] = []

    def get_account(self) -> dict[str, object]:
        assert self._factory.active_count == 0
        self.calls.append("get_account")
        return super().get_account()

    def get_position(self, symbol: str) -> dict[str, object] | None:
        assert self._factory.active_count == 0
        self.calls.append("get_position")
        return super().get_position(symbol)

    def submit_notional_day_market_order(
        self,
        *,
        symbol: str,
        notional: float,
        side: str,
        client_order_id: str,
    ) -> dict[str, object]:
        assert self._factory.active_count == 0
        self.calls.append("submit_notional_day_market_order")
        return super().submit_notional_day_market_order(
            symbol=symbol,
            notional=notional,
            side=side,
            client_order_id=client_order_id,
        )

    def get_order(self, order_id: str) -> dict[str, object]:
        assert self._factory.active_count == 0
        self.calls.append("get_order")
        return super().get_order(order_id)


def test_submission_service_and_wrapper_keep_broker_and_notifications_outside_uow(
    monkeypatch,
    tmp_path: Path,
) -> None:
    config = build_phase7_paper_config(tmp_path, symbol="QQQ")
    factory = InMemoryPaperUnitOfWorkFactory(tmp_path / "memory-submit")
    _seed_submission_ready_proposal(factory)
    broker = _BoundaryBroker(factory=factory)
    notification_events: list[str] = []

    monkeypatch.setattr(service_module, "_paper_uow_factory", lambda _config: factory)

    def _notify_submission(*args, **kwargs) -> None:
        assert factory.active_count == 0
        assert factory.commit_count >= 1
        notification_events.append("notified")

    monkeypatch.setattr(service_module, "notify_paper_submission", _notify_submission)

    result = service_module.run_paper_submit(
        config,
        now=datetime(2026, 4, 10, 23, 5, tzinfo=UTC),
        broker=broker,
    )

    assert result["status"]["status"] == "submitted"
    assert broker.calls == [
        "get_account",
        "get_position",
        "submit_notional_day_market_order",
        "get_order",
    ]
    assert notification_events == ["notified"]


class _ReconciliationBroker(FakeAlpacaBroker):
    def __init__(self, *, factory: InMemoryPaperUnitOfWorkFactory) -> None:
        super().__init__(
            symbol="QQQ",
            equity=1000.0,
            buying_power=1000.0,
            cash=1000.0,
            order_status="rejected",
        )
        self._factory = factory
        self.calls: list[str] = []

    def get_order(self, order_id: str) -> dict[str, object]:
        assert self._factory.active_count == 0
        self.calls.append("get_order")
        return {
            "id": order_id,
            "status": self.order_status,
            "client_order_id": "client-order-1",
        }


def test_reconciliation_service_polls_broker_outside_uow(tmp_path: Path) -> None:
    config = build_phase7_paper_config(tmp_path, symbol="QQQ")
    factory = InMemoryPaperUnitOfWorkFactory(tmp_path / "memory-reconcile")
    proposal = _seed_submission_ready_proposal(factory)
    with factory() as uow:
        uow.trades.save_submission(
            trade_date=str(proposal["effective_date"]),
            submission={
                "proposal_id": proposal["proposal_id"],
                "trade_date": str(proposal["effective_date"]),
                "status": "submitted",
                "order_id": "order-1",
                "client_order_id": "client-order-1",
                "order_status": "accepted",
                "poll_status": "observed",
            },
        )
        uow.commit()

    broker = _ReconciliationBroker(factory=factory)
    result = ReconciliationService(config, uow_factory=factory).run(
        PaperReconciliationRequest(
            now=datetime(2026, 4, 11, 14, 0, tzinfo=UTC),
            broker=broker,
        )
    )

    assert result is not None
    assert result.order_status == "rejected"
    assert broker.calls == ["get_order"]
