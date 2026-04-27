from __future__ import annotations

from tests._paper_fakes import FakeAlpacaBroker, FakeAlpacaProvider

from marketlab.paper.contracts import (
    PaperApprovalResult,
    PaperBroker,
    PaperDecisionResult,
    PaperHistoryProvider,
    PaperReconciliationResult,
    PaperSubmissionResult,
)


def test_paper_protocols_match_existing_fake_adapters() -> None:
    assert isinstance(FakeAlpacaProvider(), PaperHistoryProvider)
    assert isinstance(FakeAlpacaBroker(), PaperBroker)


def test_paper_decision_result_round_trips_legacy_payload() -> None:
    payload = {
        "proposal_id": "proposal-1",
        "proposal_path": "proposal.json",
        "evidence_path": "evidence.json",
        "status_path": "status.json",
        "status": {"event": "paper-decision", "status": "proposal_created"},
    }

    result = PaperDecisionResult.from_legacy(payload)

    assert result.as_legacy_payload() == payload


def test_paper_approval_result_round_trips_legacy_payload() -> None:
    payload = {
        "proposal_id": "proposal-1",
        "proposal_path": "proposal.json",
        "approval_path": "approval.json",
        "status_path": "status.json",
        "status": {"event": "paper-approve", "status": "approved"},
    }

    result = PaperApprovalResult.from_legacy(payload)

    assert result.as_legacy_payload() == payload


def test_paper_submission_result_round_trips_legacy_payload() -> None:
    payload = {
        "proposal_id": "proposal-1",
        "submission_path": "submission.json",
        "status_path": "status.json",
        "status": {"event": "paper-submit", "status": "submitted"},
    }

    result = PaperSubmissionResult.from_legacy(payload)

    assert result.as_legacy_payload() == payload


def test_paper_reconciliation_result_round_trips_legacy_payload() -> None:
    payload = {
        "proposal_id": "proposal-1",
        "submission_path": "submission.json",
        "order_status_path": "order_status.json",
        "order_status": "rejected",
        "poll_status": "observed",
    }

    result = PaperReconciliationResult.from_legacy(payload)

    assert result.as_legacy_payload() == payload
