from __future__ import annotations

from dataclasses import fields

import pytest
from tests._paper_fakes import (
    FakeAlpacaBroker,
    FakeAlpacaProvider,
    FakePaperApprovalClient,
    FakePaperNotificationSink,
)

from marketlab.paper.contracts import (
    PAPER_HOSTED_METADATA_FIELDS,
    PaperApprovalClient,
    PaperApprovalClientDecision,
    PaperApprovalRequest,
    PaperApprovalResult,
    PaperBroker,
    PaperDecisionRequest,
    PaperDecisionResult,
    PaperDeploymentRecord,
    PaperDeploymentRegistry,
    PaperHistoryProvider,
    PaperHostedExecutionContext,
    PaperNotificationSink,
    PaperPhaseRunRecord,
    PaperReconciliationRequest,
    PaperReconciliationResult,
    PaperSubmissionRequest,
    PaperSubmissionResult,
)


def test_paper_protocols_match_existing_fake_adapters() -> None:
    assert isinstance(FakeAlpacaProvider(), PaperHistoryProvider)
    assert isinstance(FakeAlpacaBroker(), PaperBroker)
    assert isinstance(FakePaperNotificationSink(), PaperNotificationSink)
    assert isinstance(FakePaperApprovalClient(), PaperApprovalClient)


def _hosted_metadata(**overrides: str) -> dict[str, str]:
    payload = {
        "deployment_id": "qqq-paper-dev",
        "environment": "dev",
        "phase": "decision",
        "execution_id": "exec-1",
        "correlation_id": "corr-1",
        "idempotency_key": "idem-1",
        "trigger_source": "cli",
        "requested_at": "2026-06-19T12:00:00+00:00",
        "config_version": "config-v1",
        "image_digest": "sha256:abc123",
    }
    payload.update(overrides)
    return payload


def test_hosted_execution_contracts_use_exact_metadata_fields() -> None:
    expected = list(PAPER_HOSTED_METADATA_FIELDS)

    assert [field.name for field in fields(PaperHostedExecutionContext)] == expected
    assert [field.name for field in fields(PaperDeploymentRecord)] == expected
    assert [field.name for field in fields(PaperPhaseRunRecord)] == expected


def test_hosted_execution_context_round_trips_metadata() -> None:
    metadata = _hosted_metadata(phase="submit", idempotency_key="submit-1")

    context = PaperHostedExecutionContext.from_metadata(metadata)
    deployment = PaperDeploymentRecord.from_context(context)
    phase_run = PaperPhaseRunRecord.from_metadata(metadata)

    assert context.as_metadata() == metadata
    assert deployment.as_metadata() == metadata
    assert phase_run.as_metadata() == metadata


@pytest.mark.parametrize(
    ("field_name", "value", "match"),
    [
        ("environment", "prod", "Unsupported paper hosted environment"),
        ("phase", "scheduler", "Unsupported paper hosted phase"),
        ("requested_at", "not-a-date", "requested_at must be an ISO-8601"),
    ],
)
def test_hosted_execution_context_rejects_invalid_metadata(
    field_name: str,
    value: str,
    match: str,
) -> None:
    with pytest.raises(ValueError, match=match):
        PaperHostedExecutionContext.from_metadata(_hosted_metadata(**{field_name: value}))


def test_hosted_execution_context_rejects_missing_and_extra_metadata() -> None:
    missing = _hosted_metadata()
    missing.pop("image_digest")
    with pytest.raises(ValueError, match="missing required fields: image_digest"):
        PaperHostedExecutionContext.from_metadata(missing)

    extra = _hosted_metadata(extra="not-supported")
    with pytest.raises(ValueError, match="unsupported fields: extra"):
        PaperHostedExecutionContext.from_metadata(extra)


def test_hosted_execution_context_derives_child_phase_identity() -> None:
    context = PaperHostedExecutionContext.from_metadata(_hosted_metadata())

    child = context.derive(phase="reconcile", suffix="reconcile:2026-06-19")

    assert child.phase == "reconcile"
    assert child.correlation_id == context.correlation_id
    assert child.execution_id == "exec-1:reconcile:2026-06-19"
    assert child.idempotency_key == "idem-1:reconcile:2026-06-19"


class _FakeRegistry:
    def record_deployment(self, context: PaperHostedExecutionContext) -> PaperDeploymentRecord:
        return PaperDeploymentRecord.from_context(context)

    def record_phase_run(self, context: PaperHostedExecutionContext) -> PaperPhaseRunRecord:
        return PaperPhaseRunRecord.from_context(context)


def test_deployment_registry_protocol_matches_fake_adapter() -> None:
    assert isinstance(_FakeRegistry(), PaperDeploymentRegistry)


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


def test_paper_request_objects_preserve_phase_inputs() -> None:
    decision_request = PaperDecisionRequest()
    approval_request = PaperApprovalRequest(
        proposal_id="proposal-1",
        decision="approve",
        actor="agent",
        fallback_used=True,
    )
    submission_request = PaperSubmissionRequest(retry_failed_submission=True)
    reconciliation_request = PaperReconciliationRequest()

    assert decision_request.now is None
    assert approval_request.proposal_id == "proposal-1"
    assert approval_request.decision == "approve"
    assert approval_request.actor == "agent"
    assert approval_request.fallback_used is True
    assert submission_request.retry_failed_submission is True
    assert reconciliation_request.broker is None


def test_paper_approval_client_decision_preserves_fallback_fields() -> None:
    decision = PaperApprovalClientDecision(
        decision="approve",
        rationale="Approved by contract test.",
        provider="openai",
        model="gpt-4o-mini",
        fallback_used=True,
        fallback_reason="openai backend failed: timeout",
    )

    assert decision.decision == "approve"
    assert decision.provider == "openai"
    assert decision.model == "gpt-4o-mini"
    assert decision.fallback_used is True
    assert decision.fallback_reason == "openai backend failed: timeout"


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
