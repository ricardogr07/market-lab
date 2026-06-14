from __future__ import annotations

import json
from collections import Counter
from datetime import date, timedelta
from pathlib import Path
from typing import Any

from marketlab.shadow.contract import verify_shadow_contract
from marketlab.shadow.evidence import (
    ShadowAttemptStore,
    ShadowDecisionEvidenceStore,
    ShadowLabelEvidenceStore,
)
from marketlab.shadow.journal import ShadowDecisionJournal


def build_shadow_status(
    config_path: str | Path,
    *,
    as_of: date,
    journal: ShadowDecisionJournal | None = None,
    attempt_store: ShadowAttemptStore | None = None,
    decision_evidence_store: ShadowDecisionEvidenceStore | None = None,
    label_evidence_store: ShadowLabelEvidenceStore | None = None,
) -> dict[str, Any]:
    contract = verify_shadow_contract(config_path)
    selected_journal = journal or ShadowDecisionJournal(
        contract.artifact_root / "decisions"
    )
    selected_attempts = attempt_store or ShadowAttemptStore(
        contract.artifact_root / "attempts"
    )
    selected_decision_evidence = (
        decision_evidence_store
        or ShadowDecisionEvidenceStore(contract.artifact_root / "evidence" / "decisions")
    )
    selected_labels = label_evidence_store or ShadowLabelEvidenceStore(
        contract.artifact_root / "evidence" / "labels"
    )

    dates: list[dict[str, object]] = []
    integrity_errors: list[dict[str, str]] = []
    for effective_date in _expected_dates(contract.protocol_start, contract.protocol_end, as_of):
        decision = _read(
            integrity_errors,
            effective_date,
            "decision",
            lambda effective_date=effective_date: selected_journal.read(effective_date),
        )
        attempts = _read(
            integrity_errors,
            effective_date,
            "attempt",
            lambda effective_date=effective_date: selected_attempts.list_for(effective_date),
        )
        decision_evidence = _read(
            integrity_errors,
            effective_date,
            "decision_evidence",
            lambda effective_date=effective_date: selected_decision_evidence.read(
                effective_date
            ),
        )
        label = _read(
            integrity_errors,
            effective_date,
            "label_evidence",
            lambda effective_date=effective_date: selected_labels.read(effective_date),
        )
        classification = _classification(
            effective_date=effective_date,
            as_of=as_of,
            decision=decision if isinstance(decision, dict) else None,
            attempts=attempts if isinstance(attempts, list) else [],
            decision_evidence=(
                decision_evidence if isinstance(decision_evidence, dict) else None
            ),
            label=label if isinstance(label, dict) else None,
        )
        dates.append(
            {
                "effective_date": effective_date.isoformat(),
                "classification": classification,
                "attempt_count": len(attempts) if isinstance(attempts, list) else 0,
                "decision_fingerprint": (
                    decision.get("output_fingerprint")
                    if isinstance(decision, dict)
                    else None
                ),
                "decision_evidence_fingerprint": (
                    decision_evidence.get("output_fingerprint")
                    if isinstance(decision_evidence, dict)
                    else None
                ),
                "label_evidence_fingerprint": (
                    label.get("output_fingerprint")
                    if isinstance(label, dict)
                    else None
                ),
            }
        )

    counts = Counter(str(item["classification"]) for item in dates)
    return {
        "schema_version": 1,
        "candidate_id": contract.candidate_id,
        "behavior_version": contract.behavior_version,
        "config_hash": contract.config_hash,
        "behavior_hash": contract.behavior_hash,
        "code_lock": contract.code_lock,
        "as_of": as_of.isoformat(),
        "expected_dates": len(dates),
        "counts": dict(sorted(counts.items())),
        "dates": dates,
        "integrity_errors": integrity_errors,
    }


def write_shadow_status(
    config_path: str | Path,
    *,
    as_of: date,
    output_path: str | Path | None = None,
    journal: ShadowDecisionJournal | None = None,
    attempt_store: ShadowAttemptStore | None = None,
    decision_evidence_store: ShadowDecisionEvidenceStore | None = None,
    label_evidence_store: ShadowLabelEvidenceStore | None = None,
) -> tuple[Path, dict[str, Any]]:
    contract = verify_shadow_contract(config_path)
    status = build_shadow_status(
        config_path,
        as_of=as_of,
        journal=journal,
        attempt_store=attempt_store,
        decision_evidence_store=decision_evidence_store,
        label_evidence_store=label_evidence_store,
    )
    path = (
        Path(output_path)
        if output_path is not None
        else contract.artifact_root / "state" / "status.json"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(status, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    temporary.replace(path)
    return path, status


def _expected_dates(start: date, end: date, as_of: date) -> list[date]:
    last_date = min(end, as_of)
    if last_date < start:
        return []
    return [
        start + timedelta(days=offset)
        for offset in range((last_date - start).days + 1)
    ]


def _read(
    errors: list[dict[str, str]],
    effective_date: date,
    record_type: str,
    reader,
) -> object:
    try:
        return reader()
    except Exception as exc:
        errors.append(
            {
                "effective_date": effective_date.isoformat(),
                "record_type": record_type,
                "error_type": type(exc).__name__,
                "reason": " ".join(str(exc).split())[:500],
            }
        )
        return None


def _classification(
    *,
    effective_date: date,
    as_of: date,
    decision: dict[str, Any] | None,
    attempts: list[dict[str, Any]],
    decision_evidence: dict[str, Any] | None,
    label: dict[str, Any] | None,
) -> str:
    if decision is not None:
        status = str(decision.get("status"))
        if status == "success":
            if decision_evidence is None or label is None:
                return "label-pending"
            return "successful"
        if status in {"skipped", "failed"}:
            return status
    outcomes = [str(attempt.get("outcome")) for attempt in attempts]
    for outcome in ("failed", "missed", "skipped"):
        if outcome in outcomes:
            return outcome
    return "pending" if effective_date == as_of else "missed"
