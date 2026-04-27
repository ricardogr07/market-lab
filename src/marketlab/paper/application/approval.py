from __future__ import annotations

from marketlab.config import ExperimentConfig
from marketlab.paper.contracts import (
    PaperApprovalRequest,
    PaperApprovalResult,
)
from marketlab.paper.core import (
    APPROVAL_APPROVED,
    APPROVAL_REJECTED,
    _now_utc,
    validate_paper_trading_config,
)
from marketlab.paper.notifications import notify_paper_approval
from marketlab.paper.state import PaperStateStore, _json_dump


class ApprovalService:
    def __init__(self, config: ExperimentConfig) -> None:
        self._config = config

    def run(self, request: PaperApprovalRequest) -> PaperApprovalResult:
        config = self._config
        validate_paper_trading_config(config)
        if request.decision not in {"approve", "reject"}:
            raise RuntimeError("paper-approve requires decision to be either approve or reject.")
        if request.actor not in {"agent", "manual"}:
            raise RuntimeError("paper-approve requires actor to be either agent or manual.")

        store = PaperStateStore(config)
        proposal = store.load_proposal(request.proposal_id)
        trade_date = proposal["effective_date"]
        submission_path = store.trade_submission_path(trade_date)
        if submission_path.exists():
            raise RuntimeError("Cannot change approval after paper-submit has already persisted state.")

        required_actor = None
        if config.paper.execution_mode == "agent_approval":
            required_actor = "agent"
        elif config.paper.execution_mode == "manual_approval":
            required_actor = "manual"
        else:
            raise RuntimeError("paper-approve is not used when execution_mode='autonomous'.")

        if request.actor != required_actor:
            raise RuntimeError(
                f"paper-approve for execution_mode='{config.paper.execution_mode}' requires actor='{required_actor}'."
            )

        approval_status = (
            APPROVAL_APPROVED if request.decision == "approve" else APPROVAL_REJECTED
        )
        approval_timestamp = _now_utc(request.now).isoformat()
        proposal["approval_status"] = approval_status
        proposal["approval_actor"] = request.actor
        proposal["approval_decision"] = request.decision
        proposal["approval_timestamp"] = approval_timestamp
        if request.rationale is not None:
            proposal["approval_rationale"] = request.rationale
        if request.provider is not None:
            proposal["approval_backend"] = request.provider
        if request.model is not None:
            proposal["approval_model"] = request.model
        proposal["approval_fallback_used"] = bool(request.fallback_used)
        if request.fallback_reason:
            proposal["approval_fallback_reason"] = request.fallback_reason
        proposal_path = store.update_proposal(proposal)
        approval_record = {
            "proposal_id": request.proposal_id,
            "trade_date": trade_date,
            "decision": request.decision,
            "approval_status": approval_status,
            "actor": request.actor,
            "timestamp": approval_timestamp,
            "provider": request.provider,
            "model": request.model,
            "fallback_used": bool(request.fallback_used),
            "fallback_reason": request.fallback_reason,
            "rationale": request.rationale,
        }
        approval_path = _json_dump(store.trade_approval_path(trade_date), approval_record)
        status = {
            "event": "paper-approve",
            "status": approval_status,
            "proposal_id": request.proposal_id,
            "proposal_path": str(proposal_path),
            "approval_path": str(approval_path),
            "updated_at": approval_timestamp,
        }
        status_path = store.write_status(status)
        notify_paper_approval(
            config,
            store,
            proposal=proposal,
            approval_record=approval_record,
            now=request.now,
            transport=request.notification_transport,
        )
        return PaperApprovalResult(
            proposal_id=request.proposal_id,
            proposal_path=str(proposal_path),
            approval_path=str(approval_path),
            status_path=str(status_path),
            status=status,
            proposal=proposal,
            approval=approval_record,
        )
