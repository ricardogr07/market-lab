from __future__ import annotations

from marketlab.config import ExperimentConfig
from marketlab.paper.contracts import (
    PaperApprovalRequest,
    PaperApprovalResult,
    PaperUnitOfWorkFactory,
)
from marketlab.paper.core import (
    APPROVAL_APPROVED,
    APPROVAL_REJECTED,
    _now_utc,
    validate_paper_trading_config,
)
from marketlab.paper.persistence import build_filesystem_paper_uow_factory


class ApprovalService:
    def __init__(
        self,
        config: ExperimentConfig,
        *,
        uow_factory: PaperUnitOfWorkFactory | None = None,
    ) -> None:
        self._config = config
        self._uow_factory = uow_factory or build_filesystem_paper_uow_factory(config)

    def run(self, request: PaperApprovalRequest) -> PaperApprovalResult:
        config = self._config
        validate_paper_trading_config(config)
        if request.decision not in {"approve", "reject"}:
            raise RuntimeError("paper-approve requires decision to be either approve or reject.")
        if request.actor not in {"agent", "manual"}:
            raise RuntimeError("paper-approve requires actor to be either agent or manual.")

        with self._uow_factory() as uow:
            proposal = uow.trades.get_proposal(request.proposal_id)
            if proposal is None:
                raise FileNotFoundError(f"Unknown proposal_id: {request.proposal_id}")
            trade_date = str(proposal["effective_date"])
            submission = uow.trades.get_submission(trade_date)
            if submission is not None:
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
            proposal_path = uow.trades.save_proposal(proposal)
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
            approval_path = uow.trades.save_approval(
                trade_date=trade_date,
                approval=approval_record,
            )
            status = {
                "event": "paper-approve",
                "status": approval_status,
                "proposal_id": request.proposal_id,
                "proposal_path": str(proposal_path),
                "approval_path": str(approval_path),
                "updated_at": approval_timestamp,
            }
            status_path = uow.status.write_status(status)
            uow.commit()
            return PaperApprovalResult(
                proposal_id=request.proposal_id,
                proposal_path=str(proposal_path),
                approval_path=str(approval_path),
                status_path=str(status_path),
                status=status,
                proposal=proposal,
                approval=approval_record,
            )
