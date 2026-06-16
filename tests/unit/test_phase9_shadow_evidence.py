from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pytest

from marketlab.shadow import (
    ShadowAttemptStore,
    ShadowDecisionEvidenceStore,
    ShadowEvidenceConflictError,
    ShadowEvidenceError,
)


def test_attempt_store_is_append_only_and_idempotent(tmp_path: Path) -> None:
    store = ShadowAttemptStore(tmp_path / "attempts")
    record = {
        "attempt_id": "execution-1",
        "effective_date": "2026-06-11",
        "outcome": "success",
    }

    first = store.write(record)
    second = store.write(record)

    assert first.created is True
    assert second.created is False
    assert second.record == first.record
    assert store.list_for(date(2026, 6, 11)) == [first.record]


def test_attempt_store_preserves_conflicting_record(tmp_path: Path) -> None:
    store = ShadowAttemptStore(tmp_path / "attempts")
    first = store.write(
        {
            "attempt_id": "execution-1",
            "effective_date": "2026-06-11",
            "outcome": "success",
        }
    )
    original = first.path.read_bytes()

    with pytest.raises(ShadowEvidenceConflictError, match="original record"):
        store.write(
            {
                "attempt_id": "execution-1",
                "effective_date": "2026-06-11",
                "outcome": "failed",
            }
        )

    assert first.path.read_bytes() == original


def test_dated_evidence_detects_tampering(tmp_path: Path) -> None:
    store = ShadowDecisionEvidenceStore(tmp_path / "decisions")
    write = store.write(
        {
            "effective_date": "2026-06-11",
            "decision_fingerprint": "a" * 64,
        }
    )
    payload = json.loads(write.path.read_text(encoding="utf-8"))
    payload["decision_fingerprint"] = "b" * 64
    write.path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ShadowEvidenceError, match="fingerprint"):
        store.read(date(2026, 6, 11))
