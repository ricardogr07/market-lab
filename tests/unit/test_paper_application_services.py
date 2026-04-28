from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from tests._paper_fakes import (
    FakeAlpacaBroker,
    FakeAlpacaProvider,
    build_phase7_paper_config,
)

from marketlab.paper.application import ApprovalService, DecisionService
from marketlab.paper.contracts import PaperApprovalRequest, PaperDecisionRequest
from marketlab.paper.service import (
    PaperStateStore,
    decide_paper_proposal,
    run_paper_decision,
)
from marketlab.paper.state import _json_load


def _normalize_proposal(proposal: dict[str, object]) -> dict[str, object]:
    normalized = dict(proposal)
    normalized.pop("evidence_path", None)
    return normalized


def _normalize_status_paths(status: dict[str, object]) -> dict[str, object]:
    normalized = dict(status)
    for key in ("proposal_path", "evidence_path", "approval_path"):
        value = normalized.get(key)
        if value is not None:
            normalized[key] = Path(str(value)).name
    return normalized


def _load_trade_materials(config, proposal_id: str) -> tuple[dict[str, object], dict[str, object]]:
    store = PaperStateStore(config)
    proposal = store.load_proposal(proposal_id)
    evidence = store.load_evidence(str(proposal["effective_date"]))
    return proposal, evidence


def test_decision_service_matches_legacy_wrapper(tmp_path: Path) -> None:
    decision_now = datetime(2026, 4, 10, 20, 10, tzinfo=UTC)
    direct_config = build_phase7_paper_config(tmp_path / "direct")
    wrapper_config = build_phase7_paper_config(tmp_path / "wrapper")

    direct_result = DecisionService(direct_config).run(
        PaperDecisionRequest(
            now=decision_now,
            provider=FakeAlpacaProvider(symbol="VOO"),
            broker=FakeAlpacaBroker(symbol="VOO"),
        )
    )
    wrapper_result = run_paper_decision(
        wrapper_config,
        now=decision_now,
        provider=FakeAlpacaProvider(symbol="VOO"),
        broker=FakeAlpacaBroker(symbol="VOO"),
    )

    direct_proposal, direct_evidence = _load_trade_materials(direct_config, direct_result.proposal_id)
    wrapped_proposal, wrapped_evidence = _load_trade_materials(
        wrapper_config,
        str(wrapper_result["proposal_id"]),
    )

    assert set(wrapper_result) == set(direct_result.as_legacy_payload())
    assert str(wrapper_result["proposal_id"]) == direct_result.proposal_id
    assert Path(str(wrapper_result["proposal_path"])).name == Path(direct_result.proposal_path).name
    assert Path(str(wrapper_result["evidence_path"])).name == Path(direct_result.evidence_path).name
    assert str(wrapper_result["status_path"]).endswith("status.json")
    assert direct_result.status_path.endswith("status.json")
    assert _normalize_status_paths(wrapper_result["status"]) == _normalize_status_paths(
        direct_result.status
    )
    assert _normalize_proposal(wrapped_proposal) == _normalize_proposal(direct_proposal)
    assert wrapped_evidence == direct_evidence


def test_approval_service_matches_legacy_wrapper(tmp_path: Path) -> None:
    decision_now = datetime(2026, 4, 10, 20, 10, tzinfo=UTC)
    approval_now = datetime(2026, 4, 10, 20, 20, tzinfo=UTC)
    direct_config = build_phase7_paper_config(tmp_path / "direct")
    wrapper_config = build_phase7_paper_config(tmp_path / "wrapper")

    direct_decision = DecisionService(direct_config).run(
        PaperDecisionRequest(
            now=decision_now,
            provider=FakeAlpacaProvider(symbol="VOO"),
            broker=FakeAlpacaBroker(symbol="VOO"),
        )
    )
    wrapper_decision = run_paper_decision(
        wrapper_config,
        now=decision_now,
        provider=FakeAlpacaProvider(symbol="VOO"),
        broker=FakeAlpacaBroker(symbol="VOO"),
    )

    direct_result = ApprovalService(direct_config).run(
        PaperApprovalRequest(
            proposal_id=direct_decision.proposal_id,
            decision="approve",
            actor="agent",
            now=approval_now,
        )
    )
    wrapper_result = decide_paper_proposal(
        wrapper_config,
        proposal_id=str(wrapper_decision["proposal_id"]),
        decision="approve",
        actor="agent",
        now=approval_now,
    )

    direct_store = PaperStateStore(direct_config)
    wrapper_store = PaperStateStore(wrapper_config)
    direct_proposal = direct_store.load_proposal(direct_decision.proposal_id)
    wrapper_proposal = wrapper_store.load_proposal(str(wrapper_decision["proposal_id"]))
    direct_approval = _json_load(direct_store.trade_approval_path(str(direct_proposal["effective_date"])))
    wrapper_approval = _json_load(
        wrapper_store.trade_approval_path(str(wrapper_proposal["effective_date"]))
    )

    assert set(wrapper_result) == set(direct_result.as_legacy_payload())
    assert str(wrapper_result["proposal_id"]) == direct_result.proposal_id
    assert Path(str(wrapper_result["proposal_path"])).name == Path(direct_result.proposal_path).name
    assert Path(str(wrapper_result["approval_path"])).name == Path(direct_result.approval_path).name
    assert str(wrapper_result["status_path"]).endswith("status.json")
    assert direct_result.status_path.endswith("status.json")
    assert _normalize_status_paths(wrapper_result["status"]) == _normalize_status_paths(
        direct_result.status
    )
    assert _normalize_proposal(wrapper_proposal) == _normalize_proposal(direct_proposal)
    assert wrapper_approval == direct_approval
