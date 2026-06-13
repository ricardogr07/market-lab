from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any


class ShadowJournalError(RuntimeError):
    """Raised when the shadow decision journal cannot preserve its contract."""


class ShadowJournalConflictError(ShadowJournalError):
    """Raised when an effective date already contains a different decision."""


def canonical_fingerprint(payload: object) -> str:
    try:
        encoded = json.dumps(
            payload,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ShadowJournalError(
            "Shadow decision values must support deterministic JSON hashing."
        ) from exc
    return hashlib.sha256(encoded).hexdigest()


def normalize_record_fingerprint(payload: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(payload)
    normalized.pop("output_fingerprint", None)
    normalized["output_fingerprint"] = canonical_fingerprint(normalized)
    return normalized


@dataclass(frozen=True, slots=True)
class ShadowJournalWrite:
    path: Path
    record: dict[str, Any]
    created: bool


class ShadowDecisionJournal:
    def __init__(self, decisions_root: str | Path) -> None:
        self._decisions_root = Path(decisions_root)

    @property
    def decisions_root(self) -> Path:
        return self._decisions_root

    def path_for(self, effective_date: date) -> Path:
        return self._decisions_root / f"{effective_date.isoformat()}.json"

    def read(self, effective_date: date) -> dict[str, Any] | None:
        path = self.path_for(effective_date)
        if not path.exists():
            return None
        return self._load(path)

    def list(self) -> list[dict[str, Any]]:
        if not self._decisions_root.exists():
            return []
        return [self._load(path) for path in sorted(self._decisions_root.glob("*.json"))]

    def write(self, record: dict[str, Any]) -> ShadowJournalWrite:
        normalized = normalize_record_fingerprint(record)
        try:
            effective_date = date.fromisoformat(str(normalized["effective_date"]))
        except (KeyError, TypeError, ValueError) as exc:
            raise ShadowJournalError(
                "Shadow decision records require an ISO effective_date."
            ) from exc

        path = self.path_for(effective_date)
        path.parent.mkdir(parents=True, exist_ok=True)
        serialized = json.dumps(
            normalized,
            allow_nan=False,
            ensure_ascii=True,
            indent=2,
            sort_keys=True,
        ) + "\n"
        try:
            with path.open("x", encoding="utf-8", newline="\n") as handle:
                handle.write(serialized)
        except FileExistsError:
            existing = self._load(path)
            if existing == normalized:
                return ShadowJournalWrite(path=path, record=existing, created=False)
            raise ShadowJournalConflictError(
                f"Shadow decision conflict for effective date {effective_date.isoformat()}; "
                "the original journal record was preserved."
            ) from None
        except OSError as exc:
            raise ShadowJournalError(f"Unable to write shadow decision: {path}") from exc
        return ShadowJournalWrite(path=path, record=normalized, created=True)

    @staticmethod
    def _load(path: Path) -> dict[str, Any]:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ShadowJournalError(f"Unable to read shadow decision: {path}") from exc
        if not isinstance(payload, dict):
            raise ShadowJournalError(f"Shadow decision must contain a JSON object: {path}")
        try:
            filename_date = date.fromisoformat(path.stem)
            payload_date = date.fromisoformat(str(payload["effective_date"]))
        except (KeyError, TypeError, ValueError) as exc:
            raise ShadowJournalError(
                f"Shadow decision path and effective_date must use ISO dates: {path}"
            ) from exc
        if payload_date != filename_date:
            raise ShadowJournalError(
                f"Shadow decision effective_date does not match its path: {path}"
            )
        normalized = normalize_record_fingerprint(payload)
        if payload != normalized:
            raise ShadowJournalError(
                f"Shadow decision output fingerprint does not match its payload: {path}"
            )
        return payload
