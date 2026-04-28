from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from tests._paper_fakes import (
    FakeAlpacaBroker,
    FakeAlpacaProvider,
    build_phase7_paper_config,
)

from marketlab.paper.application import (
    ApprovalService,
    DecisionService,
    ReconciliationService,
    SubmissionService,
)
from marketlab.paper.contracts import (
    PaperApprovalRequest,
    PaperDecisionRequest,
    PaperReconciliationRequest,
    PaperSubmissionRequest,
)
from marketlab.paper.service import (
    PaperStateStore,
    decide_paper_proposal,
    reconcile_latest_submission_status,
    run_paper_decision,
    run_paper_submit,
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


def _normalize_payload_paths(payload: dict[str, object], *keys: str) -> dict[str, object]:
    normalized = dict(payload)
    for key in keys:
        value = normalized.get(key)
        if value is not None:
            normalized[key] = Path(str(value)).name
    return normalized


def _load_trade_materials(config, proposal_id: str) -> tuple[dict[str, object], dict[str, object]]:
    store = PaperStateStore(config)
    proposal = store.load_proposal(proposal_id)
    evidence = store.load_evidence(str(proposal["effective_date"]))
    return proposal, evidence


def _load_submission_materials(
    config,
    trade_date: str,
) -> tuple[dict[str, object], dict[str, object], dict[str, object], dict[str, object]]:
    store = PaperStateStore(config)
    submission = _json_load(store.trade_submission_path(trade_date))
    order_status = _json_load(store.trade_order_status_path(trade_date))
    order_preview = _json_load(store.trade_order_preview_path(trade_date))
    account_snapshot = _json_load(store.trade_account_snapshot_path(trade_date))
    return submission, order_status, order_preview, account_snapshot


def _legacy_submission_payload(result) -> dict[str, object]:
    payload = result.as_legacy_payload()
    payload.pop("proposal_id", None)
    if result.submission is not None:
        payload["submission"] = result.submission
    return payload


def _normalize_submission_payload(payload: dict[str, object]) -> dict[str, object]:
    normalized = dict(payload)
    if "submission_path" in normalized:
        normalized = _normalize_payload_paths(normalized, "submission_path", "status_path")
    status = normalized.get("status")
    if isinstance(status, dict):
        normalized["status"] = _normalize_payload_paths(status, "submission_path")
    submission = normalized.get("submission")
    if isinstance(submission, dict):
        normalized["submission"] = _normalize_payload_paths(
            submission,
            "order_preview_path",
            "account_snapshot_path",
            "order_status_path",
        )
    return normalized


def _seed_approved_long_proposal(config, *, broker: FakeAlpacaBroker) -> tuple[str, str]:
    decision_now = datetime(2026, 4, 10, 20, 10, tzinfo=UTC)
    approval_now = datetime(2026, 4, 10, 20, 20, tzinfo=UTC)
    proposal_result = run_paper_decision(
        config,
        now=decision_now,
        provider=FakeAlpacaProvider(symbol="QQQ"),
        broker=broker,
    )
    proposal_id = str(proposal_result["proposal_id"])
    store = PaperStateStore(config)
    proposal = store.load_proposal(proposal_id)
    proposal["decision"] = "long"
    proposal["target_weight"] = 1.0
    proposal["reference_price"] = 640.41
    store.update_proposal(proposal)
    decide_paper_proposal(
        config,
        proposal_id=proposal_id,
        decision="approve",
        actor="agent",
        now=approval_now,
    )
    return proposal_id, str(proposal["effective_date"])


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


def test_submission_service_matches_legacy_wrapper(tmp_path: Path) -> None:
    submission_now = datetime(2026, 4, 10, 23, 5, tzinfo=UTC)
    direct_config = build_phase7_paper_config(tmp_path / "direct", symbol="QQQ")
    wrapper_config = build_phase7_paper_config(tmp_path / "wrapper", symbol="QQQ")
    direct_broker = FakeAlpacaBroker(
        symbol="QQQ",
        equity=1000.0,
        buying_power=1000.0,
        cash=1000.0,
    )
    wrapper_broker = FakeAlpacaBroker(
        symbol="QQQ",
        equity=1000.0,
        buying_power=1000.0,
        cash=1000.0,
    )

    direct_proposal_id, direct_trade_date = _seed_approved_long_proposal(
        direct_config,
        broker=direct_broker,
    )
    wrapper_proposal_id, wrapper_trade_date = _seed_approved_long_proposal(
        wrapper_config,
        broker=wrapper_broker,
    )

    direct_result = SubmissionService(direct_config).run(
        PaperSubmissionRequest(
            now=submission_now,
            broker=direct_broker,
        )
    )
    wrapper_result = run_paper_submit(
        wrapper_config,
        now=submission_now,
        broker=wrapper_broker,
    )

    direct_submission, direct_order_status, direct_order_preview, direct_account_snapshot = (
        _load_submission_materials(direct_config, direct_trade_date)
    )
    wrapper_submission, wrapper_order_status, wrapper_order_preview, wrapper_account_snapshot = (
        _load_submission_materials(wrapper_config, wrapper_trade_date)
    )

    assert direct_proposal_id == wrapper_proposal_id
    assert direct_trade_date == wrapper_trade_date
    assert _normalize_submission_payload(wrapper_result) == _normalize_submission_payload(
        _legacy_submission_payload(direct_result)
    )
    assert _normalize_payload_paths(
        wrapper_submission,
        "order_preview_path",
        "account_snapshot_path",
        "order_status_path",
    ) == _normalize_payload_paths(
        direct_submission,
        "order_preview_path",
        "account_snapshot_path",
        "order_status_path",
    )
    assert wrapper_order_status == direct_order_status
    assert wrapper_order_preview == direct_order_preview
    assert wrapper_account_snapshot == direct_account_snapshot


def test_reconciliation_service_matches_legacy_wrapper(tmp_path: Path) -> None:
    submission_now = datetime(2026, 4, 10, 23, 5, tzinfo=UTC)
    reconciliation_now = datetime(2026, 4, 11, 14, 0, tzinfo=UTC)
    direct_config = build_phase7_paper_config(tmp_path / "direct", symbol="QQQ")
    wrapper_config = build_phase7_paper_config(tmp_path / "wrapper", symbol="QQQ")
    direct_broker = FakeAlpacaBroker(
        symbol="QQQ",
        equity=1000.0,
        buying_power=1000.0,
        cash=1000.0,
        order_status="accepted",
    )
    wrapper_broker = FakeAlpacaBroker(
        symbol="QQQ",
        equity=1000.0,
        buying_power=1000.0,
        cash=1000.0,
        order_status="accepted",
    )

    _, direct_trade_date = _seed_approved_long_proposal(direct_config, broker=direct_broker)
    _, wrapper_trade_date = _seed_approved_long_proposal(wrapper_config, broker=wrapper_broker)
    SubmissionService(direct_config).run(
        PaperSubmissionRequest(
            now=submission_now,
            broker=direct_broker,
        )
    )
    run_paper_submit(
        wrapper_config,
        now=submission_now,
        broker=wrapper_broker,
    )

    direct_broker.order_status = "rejected"
    wrapper_broker.order_status = "rejected"
    direct_result = ReconciliationService(direct_config).run(
        PaperReconciliationRequest(
            now=reconciliation_now,
            broker=direct_broker,
        )
    )
    wrapper_result = reconcile_latest_submission_status(
        wrapper_config,
        now=reconciliation_now,
        broker=wrapper_broker,
    )

    direct_submission = _json_load(PaperStateStore(direct_config).trade_submission_path(direct_trade_date))
    wrapper_submission = _json_load(PaperStateStore(wrapper_config).trade_submission_path(wrapper_trade_date))
    direct_order_status = _json_load(PaperStateStore(direct_config).trade_order_status_path(direct_trade_date))
    wrapper_order_status = _json_load(PaperStateStore(wrapper_config).trade_order_status_path(wrapper_trade_date))

    assert direct_result is not None
    assert wrapper_result is not None
    assert _normalize_payload_paths(
        wrapper_result,
        "submission_path",
        "order_status_path",
    ) == _normalize_payload_paths(
        direct_result.as_legacy_payload(),
        "submission_path",
        "order_status_path",
    )
    assert _normalize_payload_paths(
        wrapper_submission,
        "order_preview_path",
        "account_snapshot_path",
        "order_status_path",
    ) == _normalize_payload_paths(
        direct_submission,
        "order_preview_path",
        "account_snapshot_path",
        "order_status_path",
    )
    assert wrapper_order_status == direct_order_status
