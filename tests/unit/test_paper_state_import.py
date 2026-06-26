from __future__ import annotations

import json
from contextlib import nullcontext
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
from tests._paper_fakes import build_phase7_paper_config

import marketlab.paper.state_import as state_import
from marketlab.paper.state_import import PaperStateImportError, import_paper_state


class _FetchOneResult:
    def __init__(self, row: dict[str, Any] | None) -> None:
        self._row = row

    def fetchone(self) -> dict[str, Any] | None:
        return self._row


class _FakePostgres:
    def __init__(self) -> None:
        self.tables: dict[str, dict[str, dict[str, Any]]] = {
            "paper_proposals": {},
            "paper_evidence": {},
            "paper_approvals": {},
            "paper_submissions": {},
            "paper_order_statuses": {},
            "paper_status": {},
        }
        self.connect_calls: list[bool] = []

    def connect(self, dsn: str, *, autocommit: bool = False):
        assert dsn == "postgresql://fake"
        self.connect_calls.append(autocommit)
        return _FakeConnection(self)


class _FakeJsonb:
    def __init__(self, payload: dict[str, Any]) -> None:
        self.obj = payload


class _FakeConnection:
    def __init__(self, database: _FakePostgres) -> None:
        self._database = database

    def execute(self, sql: str, params: tuple[Any, ...] = ()):
        normalized = " ".join(sql.split())
        if normalized.startswith("SELECT payload_json FROM"):
            table = normalized.split(" FROM ")[1].split(" WHERE ")[0]
            key = "status" if table == "paper_status" else str(params[0])
            payload = self._database.tables[table].get(key)
            return _FetchOneResult(None if payload is None else {"payload_json": payload})
        if normalized.startswith("INSERT INTO"):
            table = normalized.split("INSERT INTO ")[1].split(" ")[0]
            self._insert(table, params)
            return _FetchOneResult(None)
        raise AssertionError(f"Unexpected SQL: {normalized}")

    def _insert(self, table: str, params: tuple[Any, ...]) -> None:
        if table == "paper_proposals":
            key = str(params[0])
            payload = params[3]
        elif table in {"paper_evidence", "paper_approvals", "paper_submissions"}:
            key = str(params[0])
            payload = params[2]
        elif table == "paper_order_statuses":
            key = str(params[0])
            payload = params[1]
        elif table == "paper_status":
            key = "status"
            payload = params[0]
        else:
            raise AssertionError(f"Unexpected insert table: {table}")
        self._database.tables[table][key] = dict(payload.obj)

    def transaction(self):
        return nullcontext()

    def close(self) -> None:
        return None


@pytest.fixture
def fake_postgres(monkeypatch: pytest.MonkeyPatch) -> _FakePostgres:
    database = _FakePostgres()
    monkeypatch.setattr(
        state_import,
        "postgres_dsn_from_environment",
        lambda: "postgresql://fake",
    )
    monkeypatch.setattr(state_import, "_connect", database.connect)
    monkeypatch.setattr(state_import, "_jsonb", lambda payload: _FakeJsonb(payload))
    return database


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _seed_source(root: Path) -> tuple[Path, Path, dict[str, Any]]:
    state_root = root / "artifacts" / "paper" / "state"
    inbox_root = root / "artifacts" / "paper" / "inbox"
    trade_date = "2026-04-13"
    proposal = {
        "proposal_id": "proposal-1",
        "effective_date": trade_date,
        "created_at": "2026-04-10T20:10:00+00:00",
        "symbol": "QQQ",
        "decision": "long",
    }
    evidence = {
        "proposal_id": "proposal-1",
        "effective_date": trade_date,
        "models": [],
    }
    approval = {
        "proposal_id": "proposal-1",
        "trade_date": trade_date,
        "approval_status": "approved",
    }
    submission = {
        "proposal_id": "proposal-1",
        "trade_date": trade_date,
        "status": "submitted",
    }
    order_status = {"id": "order-1", "status": "filled"}
    status = {"event": "paper-submit", "status": "submitted"}
    trade_root = state_root / "trades" / trade_date
    _write_json(inbox_root / "proposal-1.json", proposal)
    _write_json(trade_root / "proposal.json", proposal)
    _write_json(trade_root / "evidence.json", evidence)
    _write_json(trade_root / "approval.json", approval)
    _write_json(trade_root / "submission.json", submission)
    _write_json(trade_root / "order_status.json", order_status)
    _write_json(state_root / "status.json", status)
    _write_json(
        state_root / "notifications" / "notification-1.json",
        {"stage": "paper-submit", "outcome": "submitted"},
    )
    report_path = root / "artifacts" / "paper" / "reports" / "2026-04-01_2026-04-30"
    report_path.mkdir(parents=True)
    (report_path / "paper_report.md").write_text("# Report\n", encoding="utf-8")
    return state_root, inbox_root, proposal


def test_paper_state_import_dry_run_reports_checksums_without_mutating(
    fake_postgres: _FakePostgres,
    tmp_path: Path,
) -> None:
    state_root, inbox_root, _proposal = _seed_source(tmp_path)
    config = build_phase7_paper_config(
        tmp_path,
        symbol="QQQ",
        persistence_backend="postgres",
    )

    report = import_paper_state(
        config,
        source_state_dir=state_root,
        source_inbox_dir=inbox_root,
        now=datetime(2026, 6, 25, 12, 0, tzinfo=UTC),
    )

    assert report["mode"] == "dry-run"
    assert report["counts"]["proposals"]["imported"] == 1
    assert report["counts"]["notifications"]["artifact_only"] == 1
    assert report["counts"]["reports"]["artifact_only"] == 1
    assert len(str(report["aggregate_checksum"])) == 64
    assert {entry["action"] for entry in report["manifest"]} == {
        "imported",
        "artifact_only",
    }
    assert fake_postgres.tables["paper_proposals"] == {}


def test_paper_state_import_apply_is_idempotent_and_writes_report(
    fake_postgres: _FakePostgres,
    tmp_path: Path,
) -> None:
    state_root, inbox_root, proposal = _seed_source(tmp_path)
    config = build_phase7_paper_config(
        tmp_path,
        symbol="QQQ",
        persistence_backend="postgres",
    )
    report_path = tmp_path / "artifacts" / "paper" / "state" / "imports" / "import.json"

    first = import_paper_state(
        config,
        source_state_dir=state_root,
        source_inbox_dir=inbox_root,
        apply=True,
        report_path=report_path,
        now=datetime(2026, 6, 25, 12, 0, tzinfo=UTC),
    )
    second = import_paper_state(
        config,
        source_state_dir=state_root,
        source_inbox_dir=inbox_root,
        apply=True,
        now=datetime(2026, 6, 25, 12, 0, tzinfo=UTC),
    )

    assert first["mode"] == "apply"
    assert first["counts"]["proposals"]["imported"] == 1
    assert second["counts"]["proposals"]["skipped"] == 1
    assert fake_postgres.tables["paper_proposals"]["proposal-1"] == proposal
    assert fake_postgres.tables["paper_status"]["status"]["status"] == "submitted"
    assert json.loads(report_path.read_text(encoding="utf-8"))["mode"] == "apply"


def test_paper_state_import_rejects_existing_conflicting_payload(
    fake_postgres: _FakePostgres,
    tmp_path: Path,
) -> None:
    state_root, inbox_root, _proposal = _seed_source(tmp_path)
    fake_postgres.tables["paper_proposals"]["proposal-1"] = {
        "proposal_id": "proposal-1",
        "effective_date": "2026-04-13",
        "created_at": "2026-04-10T20:10:00+00:00",
        "symbol": "QQQ",
        "decision": "cash",
    }
    config = build_phase7_paper_config(
        tmp_path,
        symbol="QQQ",
        persistence_backend="postgres",
    )

    with pytest.raises(PaperStateImportError, match="proposals:proposal-1"):
        import_paper_state(
            config,
            source_state_dir=state_root,
            source_inbox_dir=inbox_root,
            apply=True,
        )

    assert fake_postgres.tables["paper_evidence"] == {}


def test_paper_state_import_rejects_malformed_json_and_missing_identity(
    fake_postgres: _FakePostgres,
    tmp_path: Path,
) -> None:
    del fake_postgres
    state_root, inbox_root, _proposal = _seed_source(tmp_path)
    bad_path = state_root / "trades" / "2026-04-14" / "proposal.json"
    bad_path.parent.mkdir(parents=True)
    bad_path.write_text("{not-json", encoding="utf-8")
    config = build_phase7_paper_config(
        tmp_path,
        symbol="QQQ",
        persistence_backend="postgres",
    )

    with pytest.raises(PaperStateImportError, match="Malformed JSON"):
        import_paper_state(config, source_state_dir=state_root, source_inbox_dir=inbox_root)

    bad_path.write_text(json.dumps({"proposal_id": "proposal-2"}), encoding="utf-8")
    with pytest.raises(PaperStateImportError, match="effective_date"):
        import_paper_state(config, source_state_dir=state_root, source_inbox_dir=inbox_root)


def test_paper_state_import_rejects_non_postgres_config(
    fake_postgres: _FakePostgres,
    tmp_path: Path,
) -> None:
    del fake_postgres
    state_root, inbox_root, _proposal = _seed_source(tmp_path)
    config = build_phase7_paper_config(tmp_path, symbol="QQQ")

    with pytest.raises(ValueError, match="paper-state-import requires"):
        import_paper_state(config, source_state_dir=state_root, source_inbox_dir=inbox_root)
