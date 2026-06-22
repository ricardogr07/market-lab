from __future__ import annotations

import copy
import json
import os
import sqlite3
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
from tests._paper_fakes import (
    FakeAlpacaBroker,
    FakePaperNotificationSink,
    build_phase7_paper_config,
)
from tests._postgres import postgres_dsn_from_environment, reset_postgres_database

import marketlab.paper.service as service_module
from marketlab.paper.application import ReconciliationService
from marketlab.paper.contracts import (
    PaperDeploymentRegistry,
    PaperDeploymentRegistryConflictError,
    PaperHostedExecutionContext,
    PaperOutboxConflictError,
    PaperOutboxRecord,
    PaperOutboxRepository,
    PaperReconciliationRequest,
    PaperStatusRepository,
    PaperTradeRepository,
    PaperUnitOfWork,
    PaperUnitOfWorkFactory,
)
from marketlab.paper.persistence import (
    apply_postgres_migrations,
    build_filesystem_paper_artifact_store,
    build_filesystem_paper_deployment_registry,
    build_filesystem_paper_uow_factory,
    build_postgres_paper_deployment_registry,
    build_postgres_paper_uow_factory,
    build_sqlite_paper_deployment_registry,
    build_sqlite_paper_uow_factory,
)
from marketlab.paper.persistence import (
    sqlite as sqlite_module,
)
from marketlab.paper.state import PaperStateStore, _json_load

PERSISTENCE_ADAPTER_KINDS = (
    ["filesystem", "sqlite", "postgres"]
    if os.environ.get("MARKETLAB_PAPER_POSTGRES_DSN", "").strip() != ""
    else ["filesystem", "sqlite"]
)


@pytest.fixture(autouse=True)
def _reset_postgres_persistence_database(request: pytest.FixtureRequest) -> None:
    callspec = getattr(request.node, "callspec", None)
    adapter_kind = None if callspec is None else callspec.params.get("adapter_kind")
    if adapter_kind != "postgres":
        yield
        return
    dsn = postgres_dsn_from_environment()
    assert dsn is not None
    reset_postgres_database(dsn)
    apply_postgres_migrations(dsn=dsn)
    try:
        yield
    finally:
        reset_postgres_database(dsn)


@pytest.fixture
def postgres_persistence_database() -> str:
    dsn = postgres_dsn_from_environment()
    if dsn is None:
        pytest.skip("MARKETLAB_PAPER_POSTGRES_DSN is required for PostgreSQL persistence tests.")
    reset_postgres_database(dsn)
    apply_postgres_migrations(dsn=dsn)
    try:
        yield dsn
    finally:
        reset_postgres_database(dsn)


@dataclass
class _InMemoryStoreState:
    status: dict[str, Any] | None = None
    proposals_by_id: dict[str, dict[str, Any]] = field(default_factory=dict)
    evidence_by_trade_date: dict[str, dict[str, Any]] = field(default_factory=dict)
    approvals_by_trade_date: dict[str, dict[str, Any]] = field(default_factory=dict)
    submissions_by_trade_date: dict[str, dict[str, Any]] = field(default_factory=dict)
    order_status_by_trade_date: dict[str, dict[str, Any]] = field(default_factory=dict)
    outbox_by_id: dict[str, PaperOutboxRecord] = field(default_factory=dict)
    backup_calls: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class _PendingState:
    status: dict[str, Any] | None = None
    proposals_by_id: dict[str, dict[str, Any]] = field(default_factory=dict)
    evidence_by_trade_date: dict[str, dict[str, Any]] = field(default_factory=dict)
    approvals_by_trade_date: dict[str, dict[str, Any]] = field(default_factory=dict)
    submissions_by_trade_date: dict[str, dict[str, Any]] = field(default_factory=dict)
    order_status_by_trade_date: dict[str, dict[str, Any]] = field(default_factory=dict)
    outbox_by_id: dict[str, PaperOutboxRecord] = field(default_factory=dict)


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

    def order_status_path_exists(self, trade_date: str) -> bool:
        return (
            trade_date in self._pending.order_status_by_trade_date
            or trade_date in self._state.order_status_by_trade_date
        )

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


class InMemoryPaperOutboxRepository(PaperOutboxRepository):
    def __init__(self, state: _InMemoryStoreState, pending: _PendingState) -> None:
        self._state = state
        self._pending = pending

    def enqueue(
        self,
        *,
        message_id: str,
        event_type: str,
        payload: dict[str, Any],
        created_at: str,
    ) -> PaperOutboxRecord:
        existing = self.get(message_id)
        if existing is not None:
            if existing.event_type != event_type or existing.payload != payload:
                raise PaperOutboxConflictError(
                    f"Paper outbox message ID {message_id!r} was reused with different event data."
                )
            return existing
        record = PaperOutboxRecord(
            message_id=message_id,
            event_type=event_type,
            payload=copy.deepcopy(payload),
            created_at=created_at,
        )
        self._pending.outbox_by_id[message_id] = record
        return record

    def get(self, message_id: str) -> PaperOutboxRecord | None:
        record = self._pending.outbox_by_id.get(message_id) or self._state.outbox_by_id.get(message_id)
        return copy.deepcopy(record) if record is not None else None

    def list_pending(
        self,
        *,
        limit: int = 100,
        event_types: frozenset[str] | None = None,
    ) -> list[PaperOutboxRecord]:
        if limit < 1:
            raise ValueError("Paper outbox limit must be at least 1.")
        records = {
            **self._state.outbox_by_id,
            **self._pending.outbox_by_id,
        }
        pending = [
            copy.deepcopy(record)
            for record in records.values()
            if record.delivery_status in {"pending", "failed"}
            and (event_types is None or record.event_type in event_types)
        ]
        return sorted(pending, key=lambda record: (record.created_at, record.message_id))[:limit]

    def mark_delivered(
        self,
        *,
        message_id: str,
        delivered_at: str,
    ) -> PaperOutboxRecord:
        record = self._required_record(message_id)
        if record.delivery_status == "delivered":
            return record
        updated = PaperOutboxRecord(
            message_id=record.message_id,
            event_type=record.event_type,
            payload=record.payload,
            created_at=record.created_at,
            delivery_status="delivered",
            delivery_attempts=record.delivery_attempts + 1,
            delivered_at=delivered_at,
        )
        self._pending.outbox_by_id[message_id] = updated
        return updated

    def mark_failed(
        self,
        *,
        message_id: str,
        error: str,
    ) -> PaperOutboxRecord:
        record = self._required_record(message_id)
        if record.delivery_status == "delivered":
            return record
        normalized_error = error.strip()
        if normalized_error == "":
            raise ValueError("Paper outbox error must not be empty.")
        updated = PaperOutboxRecord(
            message_id=record.message_id,
            event_type=record.event_type,
            payload=record.payload,
            created_at=record.created_at,
            delivery_status="failed",
            delivery_attempts=record.delivery_attempts + 1,
            last_error=normalized_error,
        )
        self._pending.outbox_by_id[message_id] = updated
        return updated

    def _required_record(self, message_id: str) -> PaperOutboxRecord:
        record = self.get(message_id)
        if record is None:
            raise KeyError(f"Unknown paper outbox message_id: {message_id}")
        return record


class InMemoryPaperUnitOfWork(PaperUnitOfWork):
    def __init__(self, factory: InMemoryPaperUnitOfWorkFactory) -> None:
        self._factory = factory
        self._pending = _PendingState()
        self._trades = InMemoryPaperTradeRepository(factory.root, factory.state, self._pending)
        self._status = InMemoryPaperStatusRepository(factory.root, factory.state, self._pending)
        self._outbox = InMemoryPaperOutboxRepository(factory.state, self._pending)
        self._committed = False

    @property
    def trades(self) -> PaperTradeRepository:
        return self._trades

    @property
    def status(self) -> PaperStatusRepository:
        return self._status

    @property
    def outbox(self) -> PaperOutboxRepository:
        return self._outbox

    def commit(self) -> None:
        self._factory.state.proposals_by_id.update(copy.deepcopy(self._pending.proposals_by_id))
        self._factory.state.evidence_by_trade_date.update(copy.deepcopy(self._pending.evidence_by_trade_date))
        self._factory.state.approvals_by_trade_date.update(copy.deepcopy(self._pending.approvals_by_trade_date))
        self._factory.state.submissions_by_trade_date.update(copy.deepcopy(self._pending.submissions_by_trade_date))
        self._factory.state.order_status_by_trade_date.update(copy.deepcopy(self._pending.order_status_by_trade_date))
        self._factory.state.outbox_by_id.update(copy.deepcopy(self._pending.outbox_by_id))
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
    if adapter_kind == "sqlite":
        return build_sqlite_paper_uow_factory(config)
    if adapter_kind == "postgres":
        return build_postgres_paper_uow_factory(config)
    if adapter_kind == "memory":
        return InMemoryPaperUnitOfWorkFactory(tmp_path / "memory-root")
    raise ValueError(f"Unknown adapter kind: {adapter_kind}")


def _build_registry(*, adapter_kind: str, config) -> PaperDeploymentRegistry:
    if adapter_kind == "filesystem":
        return build_filesystem_paper_deployment_registry(config)
    if adapter_kind == "sqlite":
        return build_sqlite_paper_deployment_registry(config)
    if adapter_kind == "postgres":
        return build_postgres_paper_deployment_registry(config)
    raise ValueError(f"Unknown adapter kind: {adapter_kind}")


def _hosted_context(**overrides: str) -> PaperHostedExecutionContext:
    payload = {
        "deployment_id": "qqq-paper-dev",
        "environment": "dev",
        "phase": "decision",
        "execution_id": "exec-1",
        "correlation_id": "corr-1",
        "idempotency_key": "idem-1",
        "trigger_source": "scheduler",
        "requested_at": "2026-06-19T12:00:00+00:00",
        "config_version": "config-v1",
        "image_digest": "sha256:abc123",
    }
    payload.update(overrides)
    return PaperHostedExecutionContext.from_metadata(payload)


@pytest.mark.parametrize("adapter_kind", PERSISTENCE_ADAPTER_KINDS)
def test_deployment_registry_accepts_identical_idempotency_replays(
    adapter_kind: str,
    tmp_path: Path,
) -> None:
    config = build_phase7_paper_config(
        tmp_path / f"{adapter_kind}-registry",
        symbol="QQQ",
        persistence_backend=adapter_kind,
    )
    registry = _build_registry(adapter_kind=adapter_kind, config=config)
    context = _hosted_context()

    first = registry.record_phase_run(context)
    second = registry.record_phase_run(context)

    assert first.as_metadata() == context.as_metadata()
    assert second.as_metadata() == context.as_metadata()


@pytest.mark.parametrize("adapter_kind", PERSISTENCE_ADAPTER_KINDS)
def test_deployment_registry_rejects_conflicting_idempotency_replays(
    adapter_kind: str,
    tmp_path: Path,
) -> None:
    config = build_phase7_paper_config(
        tmp_path / f"{adapter_kind}-registry-conflict",
        symbol="QQQ",
        persistence_backend=adapter_kind,
    )
    registry = _build_registry(adapter_kind=adapter_kind, config=config)
    registry.record_phase_run(_hosted_context())

    with pytest.raises(PaperDeploymentRegistryConflictError):
        registry.record_phase_run(_hosted_context(image_digest="sha256:changed"))


@pytest.mark.parametrize("adapter_kind", PERSISTENCE_ADAPTER_KINDS)
def test_deployment_registry_rejects_idempotency_replays_in_a_different_phase(
    adapter_kind: str,
    tmp_path: Path,
) -> None:
    config = build_phase7_paper_config(
        tmp_path / f"{adapter_kind}-registry-cross-phase-conflict",
        symbol="QQQ",
        persistence_backend=adapter_kind,
    )
    registry = _build_registry(adapter_kind=adapter_kind, config=config)
    registry.record_phase_run(_hosted_context())

    with pytest.raises(PaperDeploymentRegistryConflictError):
        registry.record_phase_run(
            _hosted_context(
                phase="submit",
                execution_id="submit-exec-1",
            )
        )


def test_filesystem_deployment_registry_rolls_back_phase_run_on_deployment_conflict(
    tmp_path: Path,
) -> None:
    config = build_phase7_paper_config(
        tmp_path / "filesystem-registry-deployment-conflict",
        symbol="QQQ",
    )
    registry = build_filesystem_paper_deployment_registry(config)
    registry.record_deployment(_hosted_context(idempotency_key="existing-deployment-key"))
    conflicting_context = _hosted_context(idempotency_key="rolled-back-phase-run-key")

    with pytest.raises(PaperDeploymentRegistryConflictError):
        registry.record_phase_run(conflicting_context)

    assert not (
        config.paper_state_dir
        / "phase-runs"
        / "rolled-back-phase-run-key.json"
    ).exists()


def test_filesystem_deployment_registry_writes_stable_layout(tmp_path: Path) -> None:
    config = build_phase7_paper_config(tmp_path / "filesystem-registry-layout", symbol="QQQ")
    registry = build_filesystem_paper_deployment_registry(config)
    context = _hosted_context(
        phase="submit",
        execution_id="exec/submit",
        idempotency_key="idem/submit",
    )

    registry.record_phase_run(context)

    phase_run_path = (
        config.paper_state_dir
        / "phase-runs"
        / "idem%2Fsubmit.json"
    )
    deployment_path = (
        config.paper_state_dir
        / "deployments"
        / "dev"
        / "qqq-paper-dev"
        / "submit"
        / "exec%2Fsubmit.json"
    )
    assert json.loads(phase_run_path.read_text(encoding="utf-8")) == context.as_metadata()
    assert json.loads(deployment_path.read_text(encoding="utf-8")) == context.as_metadata()


def test_sqlite_deployment_registry_uses_local_tables_only(tmp_path: Path) -> None:
    config = build_phase7_paper_config(
        tmp_path / "sqlite-registry-layout",
        symbol="QQQ",
        persistence_backend="sqlite",
    )
    registry = build_sqlite_paper_deployment_registry(config)
    context = _hosted_context(phase="reconcile", idempotency_key="idem-reconcile")

    registry.record_phase_run(context)

    with sqlite3.connect(config.paper_sqlite_db_path) as connection:
        phase_rows = connection.execute(
            "SELECT payload_json FROM paper_phase_run_records"
        ).fetchall()
        deployment_rows = connection.execute(
            "SELECT payload_json FROM paper_deployment_records"
        ).fetchall()
    assert [json.loads(row[0]) for row in phase_rows] == [context.as_metadata()]
    assert [json.loads(row[0]) for row in deployment_rows] == [context.as_metadata()]
    assert not (config.paper_state_dir / "phase-runs").exists()
    assert not (config.paper_state_dir / "deployments").exists()


@pytest.mark.parametrize("adapter_kind", PERSISTENCE_ADAPTER_KINDS)
def test_paper_repository_contract_stages_until_commit(adapter_kind: str, tmp_path: Path) -> None:
    config = build_phase7_paper_config(
        tmp_path / adapter_kind,
        symbol="QQQ",
        persistence_backend=adapter_kind,
    )
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


@pytest.mark.parametrize("adapter_kind", PERSISTENCE_ADAPTER_KINDS)
def test_paper_repository_contract_persists_and_orders_records(adapter_kind: str, tmp_path: Path) -> None:
    config = build_phase7_paper_config(
        tmp_path / f"{adapter_kind}-persist",
        symbol="QQQ",
        persistence_backend=adapter_kind,
    )
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
    approval = {
        "proposal_id": newer["proposal_id"],
        "trade_date": "2026-04-14",
        "decision": "approve",
        "approval_status": "approved",
        "actor": "agent",
        "timestamp": "2026-04-11T20:15:00+00:00",
        "provider": None,
        "model": None,
        "fallback_used": False,
        "fallback_reason": None,
        "rationale": None,
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
        uow.trades.save_approval(trade_date="2026-04-14", approval=approval)
        uow.trades.save_submission(trade_date="2026-04-14", submission=submission)
        uow.trades.save_order_status(trade_date="2026-04-14", order_status=order_status)
        uow.status.write_status(status)
        uow.commit()

    store = PaperStateStore(config)
    with factory() as uow:
        proposals = uow.trades.list_proposals()
        assert [proposal["proposal_id"] for proposal in proposals] == [
            newer["proposal_id"],
            older["proposal_id"],
        ]
        assert uow.trades.get_latest_proposal()["proposal_id"] == newer["proposal_id"]
        assert uow.trades.get_submission("2026-04-14") == submission
        assert uow.status.read_status() == status
    assert _json_load(store.trade_proposal_path("2026-04-14")) == newer
    assert _json_load(store.inbox_proposal_path(newer["proposal_id"])) == newer
    assert _json_load(store.trade_evidence_path("2026-04-14")) == _evidence_payload(
        proposal_id=newer["proposal_id"],
        trade_date="2026-04-14",
    )
    assert _json_load(store.trade_approval_path("2026-04-14")) == approval
    assert _json_load(store.trade_submission_path("2026-04-14")) == submission
    assert _json_load(store.trade_order_status_path("2026-04-14")) == order_status
    assert _json_load(store.status_path) == status


@pytest.mark.parametrize("adapter_kind", PERSISTENCE_ADAPTER_KINDS)
def test_trade_repository_retry_backup_preserves_attempt_artifacts(
    adapter_kind: str,
    tmp_path: Path,
) -> None:
    config = build_phase7_paper_config(
        tmp_path / f"{adapter_kind}-retry",
        symbol="QQQ",
        persistence_backend=adapter_kind,
    )
    factory = _build_factory(adapter_kind=adapter_kind, tmp_path=tmp_path, config=config)
    artifact_store = build_filesystem_paper_artifact_store(config)
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
    artifact_store.write_trade_order_preview(
        trade_date=trade_date,
        payload={"proposal_id": "proposal-1", "trade_date": trade_date, "side": "buy"},
    )
    artifact_store.write_trade_account_snapshot(
        trade_date=trade_date,
        payload={"equity": "1000.00"},
    )

    with factory() as uow:
        uow.trades.backup_submission_attempt_artifacts(
            trade_date=trade_date,
            now=datetime(2026, 4, 10, 23, 10, tzinfo=UTC),
        )
        assert uow.trades.get_submission(trade_date) is None

    trade_dir = store.trade_dir(trade_date)
    backup_files = sorted(path.name for path in trade_dir.glob("*.retry-backup.*.bak"))
    assert len(backup_files) == 4
    assert not store.trade_submission_path(trade_date).exists()
    assert not store.trade_order_status_path(trade_date).exists()
    assert not store.trade_order_preview_path(trade_date).exists()
    assert not store.trade_account_snapshot_path(trade_date).exists()


def test_sqlite_commit_rolls_back_when_artifact_mirror_write_fails(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    config = build_phase7_paper_config(
        tmp_path / "sqlite-failure",
        symbol="QQQ",
        persistence_backend="sqlite",
    )
    factory = build_sqlite_paper_uow_factory(config)
    proposal = _proposal_payload(
        proposal_id="proposal-rollback",
        trade_date="2026-04-13",
        created_at="2026-04-10T20:10:00+00:00",
    )

    original_json_dump = sqlite_module._json_dump

    def _failing_json_dump(path: Path, payload: dict[str, Any]) -> Path:
        if path.name == "proposal.json":
            raise PermissionError("simulated artifact write failure")
        return original_json_dump(path, payload)

    monkeypatch.setattr(sqlite_module, "_json_dump", _failing_json_dump)

    with pytest.raises(PermissionError, match="simulated artifact write failure"):
        with factory() as uow:
            uow.trades.save_proposal(proposal)
            uow.commit()

    with factory() as uow:
        assert uow.trades.get_proposal(proposal["proposal_id"]) is None

    store = PaperStateStore(config)
    assert not store.trade_proposal_path("2026-04-13").exists()
    assert not store.inbox_proposal_path(proposal["proposal_id"]).exists()


def test_postgres_commit_rolls_back_when_artifact_mirror_write_fails(
    monkeypatch: pytest.MonkeyPatch,
    postgres_persistence_database: str,
    tmp_path: Path,
) -> None:
    del postgres_persistence_database
    from marketlab.paper.persistence import postgres as postgres_module

    config = build_phase7_paper_config(
        tmp_path / "postgres-failure",
        symbol="QQQ",
        persistence_backend="postgres",
    )
    factory = build_postgres_paper_uow_factory(config)
    proposal = _proposal_payload(
        proposal_id="proposal-rollback",
        trade_date="2026-04-13",
        created_at="2026-04-10T20:10:00+00:00",
    )
    original_json_dump = postgres_module._json_dump

    def _failing_json_dump(path: Path, payload: dict[str, Any]) -> Path:
        if path.name == "proposal.json":
            raise PermissionError("simulated artifact write failure")
        return original_json_dump(path, payload)

    monkeypatch.setattr(postgres_module, "_json_dump", _failing_json_dump)

    with pytest.raises(PermissionError, match="simulated artifact write failure"):
        with factory() as uow:
            uow.trades.save_proposal(proposal)
            uow.commit()

    with factory() as uow:
        assert uow.trades.get_proposal(proposal["proposal_id"]) is None

    store = PaperStateStore(config)
    assert not store.trade_proposal_path("2026-04-13").exists()
    assert not store.inbox_proposal_path(proposal["proposal_id"]).exists()


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
    notification_sink = FakePaperNotificationSink()

    monkeypatch.setattr(service_module, "_paper_uow_factory", lambda _config: factory)
    monkeypatch.setattr(service_module, "_paper_notification_sink", lambda _config: notification_sink)

    result = service_module.run_paper_submit(
        config,
        now=datetime(2026, 4, 10, 23, 5, tzinfo=UTC),
        broker=broker,
    )

    assert len(notification_sink.submission_calls) == 1
    assert factory.active_count == 0
    assert factory.commit_count >= 1
    notification_events.append("notified")

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
