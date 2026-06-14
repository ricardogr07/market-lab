from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

from marketlab.shadow.journal import ShadowJournalError, normalize_record_fingerprint

_IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}")


class ShadowEvidenceError(RuntimeError):
    """Raised when operational shadow evidence is malformed or unreadable."""


class ShadowEvidenceConflictError(ShadowEvidenceError):
    """Raised when an append-only evidence path already contains other bytes."""


@dataclass(frozen=True, slots=True)
class ShadowEvidenceWrite:
    path: Path
    record: dict[str, Any]
    created: bool


class _AppendOnlyStore:
    def _write(self, path: Path, record: dict[str, Any]) -> ShadowEvidenceWrite:
        try:
            normalized = normalize_record_fingerprint(record)
        except (TypeError, ValueError, ShadowJournalError) as exc:
            raise ShadowEvidenceError(
                f"Shadow evidence contains non-deterministic values: {path}"
            ) from exc
        serialized = json.dumps(
            normalized,
            allow_nan=False,
            ensure_ascii=True,
            indent=2,
            sort_keys=True,
        ) + "\n"
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with path.open("x", encoding="utf-8", newline="\n") as handle:
                handle.write(serialized)
        except FileExistsError:
            existing = self._load(path)
            if existing == normalized:
                return ShadowEvidenceWrite(path=path, record=existing, created=False)
            raise ShadowEvidenceConflictError(
                f"Shadow evidence conflict at {path}; the original record was preserved."
            ) from None
        except OSError as exc:
            raise ShadowEvidenceError(f"Unable to write shadow evidence: {path}") from exc
        return ShadowEvidenceWrite(path=path, record=normalized, created=True)

    @staticmethod
    def _load(path: Path) -> dict[str, Any]:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ShadowEvidenceError(f"Unable to read shadow evidence: {path}") from exc
        if not isinstance(payload, dict):
            raise ShadowEvidenceError(
                f"Shadow evidence must contain a JSON object: {path}"
            )
        try:
            normalized = normalize_record_fingerprint(payload)
        except (TypeError, ValueError, ShadowJournalError) as exc:
            raise ShadowEvidenceError(
                f"Shadow evidence contains non-deterministic values: {path}"
            ) from exc
        if payload != normalized:
            raise ShadowEvidenceError(
                f"Shadow evidence output fingerprint does not match its payload: {path}"
            )
        return payload


class ShadowAttemptStore(_AppendOnlyStore):
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)

    def path_for(self, effective_date: date, attempt_id: str) -> Path:
        if _IDENTIFIER.fullmatch(attempt_id) is None:
            raise ShadowEvidenceError("Shadow attempt_id contains unsupported characters.")
        return self.root / effective_date.isoformat() / f"{attempt_id}.json"

    def write(self, record: dict[str, Any]) -> ShadowEvidenceWrite:
        try:
            effective_date = date.fromisoformat(str(record["effective_date"]))
            attempt_id = str(record["attempt_id"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ShadowEvidenceError(
                "Shadow attempts require ISO effective_date and attempt_id."
            ) from exc
        return self._write(self.path_for(effective_date, attempt_id), record)

    def read(self, effective_date: date, attempt_id: str) -> dict[str, Any] | None:
        path = self.path_for(effective_date, attempt_id)
        if not path.exists():
            return None
        payload = self._load(path)
        if payload.get("effective_date") != effective_date.isoformat():
            raise ShadowEvidenceError(
                f"Shadow attempt effective_date does not match its path: {path}"
            )
        if payload.get("attempt_id") != attempt_id:
            raise ShadowEvidenceError(
                f"Shadow attempt attempt_id does not match its path: {path}"
            )
        return payload

    def list_for(self, effective_date: date) -> list[dict[str, Any]]:
        directory = self.root / effective_date.isoformat()
        if not directory.exists():
            return []
        records: list[dict[str, Any]] = []
        for path in sorted(directory.glob("*.json")):
            record = self.read(effective_date, path.stem)
            if record is not None:
                records.append(record)
        return records

    def list(self) -> list[dict[str, Any]]:
        if not self.root.exists():
            return []
        records: list[dict[str, Any]] = []
        for path in sorted(self.root.glob("*/*.json")):
            try:
                effective_date = date.fromisoformat(path.parent.name)
            except ValueError as exc:
                raise ShadowEvidenceError(
                    f"Shadow attempt directory must use an ISO date: {path.parent}"
                ) from exc
            record = self.read(effective_date, path.stem)
            if record is not None:
                records.append(record)
        return records


class _DatedEvidenceStore(_AppendOnlyStore):
    def __init__(self, root: str | Path, *, evidence_type: str) -> None:
        self.root = Path(root)
        self.evidence_type = evidence_type

    def path_for(self, effective_date: date) -> Path:
        return self.root / f"{effective_date.isoformat()}.json"

    def write(self, record: dict[str, Any]) -> ShadowEvidenceWrite:
        try:
            effective_date = date.fromisoformat(str(record["effective_date"]))
        except (KeyError, TypeError, ValueError) as exc:
            raise ShadowEvidenceError(
                f"Shadow {self.evidence_type} evidence requires an ISO effective_date."
            ) from exc
        return self._write(self.path_for(effective_date), record)

    def read(self, effective_date: date) -> dict[str, Any] | None:
        path = self.path_for(effective_date)
        if not path.exists():
            return None
        payload = self._load(path)
        if payload.get("effective_date") != effective_date.isoformat():
            raise ShadowEvidenceError(
                f"Shadow {self.evidence_type} effective_date does not match its path: {path}"
            )
        return payload

    def list(self) -> list[dict[str, Any]]:
        if not self.root.exists():
            return []
        records: list[dict[str, Any]] = []
        for path in sorted(self.root.glob("*.json")):
            try:
                effective_date = date.fromisoformat(path.stem)
            except ValueError as exc:
                raise ShadowEvidenceError(
                    f"Shadow {self.evidence_type} path must use an ISO date: {path}"
                ) from exc
            record = self.read(effective_date)
            if record is not None:
                records.append(record)
        return records


class ShadowDecisionEvidenceStore(_DatedEvidenceStore):
    def __init__(self, root: str | Path) -> None:
        super().__init__(root, evidence_type="decision")


class ShadowLabelEvidenceStore(_DatedEvidenceStore):
    def __init__(self, root: str | Path) -> None:
        super().__init__(root, evidence_type="label")
