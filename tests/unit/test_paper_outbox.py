from __future__ import annotations

import os
from pathlib import Path

import pytest
from tests._paper_fakes import build_phase7_paper_config
from tests._postgres import postgres_dsn_from_environment, reset_postgres_database

from marketlab.paper.contracts import PaperOutboxConflictError
from marketlab.paper.persistence import (
    apply_postgres_migrations,
    build_filesystem_paper_uow_factory,
    build_postgres_paper_uow_factory,
    build_sqlite_paper_uow_factory,
)
from marketlab.paper.state import PaperStateStore, _json_load

PERSISTENCE_ADAPTER_KINDS = (
    ["filesystem", "sqlite", "postgres"]
    if os.environ.get("MARKETLAB_PAPER_POSTGRES_DSN", "").strip() != ""
    else ["filesystem", "sqlite"]
)


@pytest.fixture(autouse=True)
def _reset_postgres_outbox_database(request: pytest.FixtureRequest) -> None:
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


def _factory(*, adapter_kind: str, config):
    if adapter_kind == "filesystem":
        return build_filesystem_paper_uow_factory(config)
    if adapter_kind == "sqlite":
        return build_sqlite_paper_uow_factory(config)
    if adapter_kind == "postgres":
        return build_postgres_paper_uow_factory(config)
    raise ValueError(f"Unknown paper persistence adapter: {adapter_kind}")


@pytest.mark.parametrize("adapter_kind", PERSISTENCE_ADAPTER_KINDS)
def test_outbox_persists_idempotent_message_and_delivery_transition(
    adapter_kind: str,
    tmp_path: Path,
) -> None:
    config = build_phase7_paper_config(
        tmp_path / adapter_kind,
        symbol="QQQ",
        persistence_backend=adapter_kind,
    )
    factory = _factory(adapter_kind=adapter_kind, config=config)
    message_id = "approval-request:2026-04-13-qqq-2026-04-10"
    payload = {
        "proposal_id": "2026-04-13-qqq-2026-04-10",
        "correlation_id": "run-1",
    }

    with factory() as uow:
        first = uow.outbox.enqueue(
            message_id=message_id,
            event_type="paper.approval.requested",
            payload=payload,
            created_at="2026-04-10T20:20:00+00:00",
        )
        duplicate = uow.outbox.enqueue(
            message_id=message_id,
            event_type="paper.approval.requested",
            payload=payload,
            created_at="2026-04-10T21:20:00+00:00",
        )
        assert duplicate == first
        assert uow.outbox.list_pending() == [first]
        uow.commit()

    with factory() as uow:
        pending = uow.outbox.list_pending()
        assert pending == [first]
        delivered = uow.outbox.mark_delivered(
            message_id=message_id,
            delivered_at="2026-04-10T20:21:00+00:00",
        )
        assert delivered.delivery_status == "delivered"
        assert delivered.delivery_attempts == 1
        assert delivered.delivered_at == "2026-04-10T20:21:00+00:00"
        assert uow.outbox.list_pending() == []
        uow.commit()

    with factory() as uow:
        persisted = uow.outbox.get(message_id)
        assert persisted is not None
        assert persisted.delivery_status == "delivered"
        assert persisted.delivery_attempts == 1
        assert uow.outbox.list_pending() == []

    if adapter_kind == "filesystem":
        path = PaperStateStore(config).outbox_record_path(message_id)
        assert _json_load(path) == persisted.as_payload()


@pytest.mark.parametrize("adapter_kind", PERSISTENCE_ADAPTER_KINDS)
def test_outbox_filters_before_applying_the_delivery_limit(
    adapter_kind: str,
    tmp_path: Path,
) -> None:
    config = build_phase7_paper_config(
        tmp_path / adapter_kind,
        symbol="QQQ",
        persistence_backend=adapter_kind,
    )
    factory = _factory(adapter_kind=adapter_kind, config=config)
    with factory() as uow:
        uow.outbox.enqueue(
            message_id="notification-1",
            event_type="paper.notification.requested",
            payload={"stage": "decision"},
            created_at="2026-04-10T20:20:00+00:00",
        )
        approval = uow.outbox.enqueue(
            message_id="approval-1",
            event_type="paper.approval.requested",
            payload={"proposal_id": "proposal-1"},
            created_at="2026-04-10T20:21:00+00:00",
        )
        assert uow.outbox.list_pending(
            limit=1,
            event_types=frozenset(("paper.approval.requested",)),
        ) == [approval]
        uow.commit()


@pytest.mark.parametrize("adapter_kind", PERSISTENCE_ADAPTER_KINDS)
def test_outbox_rejects_conflicting_reuse_of_message_id(
    adapter_kind: str,
    tmp_path: Path,
) -> None:
    config = build_phase7_paper_config(
        tmp_path / adapter_kind,
        symbol="QQQ",
        persistence_backend=adapter_kind,
    )
    factory = _factory(adapter_kind=adapter_kind, config=config)

    with factory() as uow:
        uow.outbox.enqueue(
            message_id="notification:proposal-1",
            event_type="paper.notification.requested",
            payload={"proposal_id": "proposal-1"},
            created_at="2026-04-10T20:20:00+00:00",
        )
        uow.commit()

    with factory() as uow:
        with pytest.raises(PaperOutboxConflictError, match="reused with different event data"):
            uow.outbox.enqueue(
                message_id="notification:proposal-1",
                event_type="paper.notification.requested",
                payload={"proposal_id": "proposal-2"},
                created_at="2026-04-10T20:21:00+00:00",
            )


@pytest.mark.parametrize("adapter_kind", PERSISTENCE_ADAPTER_KINDS)
def test_outbox_rolls_back_uncommitted_enqueue(
    adapter_kind: str,
    tmp_path: Path,
) -> None:
    config = build_phase7_paper_config(
        tmp_path / adapter_kind,
        symbol="QQQ",
        persistence_backend=adapter_kind,
    )
    factory = _factory(adapter_kind=adapter_kind, config=config)

    with factory() as uow:
        uow.outbox.enqueue(
            message_id="uncommitted:proposal-1",
            event_type="paper.notification.requested",
            payload={"proposal_id": "proposal-1"},
            created_at="2026-04-10T20:20:00+00:00",
        )

    with factory() as uow:
        assert uow.outbox.get("uncommitted:proposal-1") is None
