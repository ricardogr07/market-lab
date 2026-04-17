from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from marketlab.config import ExperimentConfig
from marketlab.env import load_env_file
from marketlab.paper.alpaca import AlpacaPaperBrokerClient
from marketlab.paper.notifications import (
    PaperLoopStageError,
    TelegramTransport,
    build_error_fingerprint,
    build_error_message,
)
from marketlab.paper.service import (
    APPROVAL_PENDING,
    PaperStateStore,
    _now_utc,
    _write_notification_record,
    decide_paper_proposal,
    read_paper_evidence,
    validate_paper_trading_config,
)


class AgentDecisionError(RuntimeError):
    pass


@dataclass(slots=True, frozen=True)
class AgentDecision:
    decision: str
    rationale: str
    provider: str
    model: str
    fallback_used: bool = False
    fallback_reason: str = ""


class AgentBackend:
    provider_name = "base"

    def evaluate(
        self,
        *,
        config: ExperimentConfig,
        proposal: dict[str, Any],
        evidence: dict[str, Any],
        status: dict[str, Any] | None,
        account_context: dict[str, Any],
    ) -> AgentDecision:
        raise NotImplementedError


def _json_dump(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return path


def _worker_state_path(config: ExperimentConfig) -> Path:
    return config.paper_state_dir / "agent_worker.json"


def _load_worker_state(config: ExperimentConfig) -> dict[str, Any]:
    path = _worker_state_path(config)
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _save_worker_state(config: ExperimentConfig, payload: dict[str, Any]) -> Path:
    return _json_dump(_worker_state_path(config), payload)


def _clear_worker_error_state(state: dict[str, Any]) -> None:
    for key in (
        "last_error_fingerprint",
        "last_error_stage",
        "last_error_type",
        "last_error_message",
        "last_error_proposal_id",
        "last_error_trade_date",
        "last_error_alert_at",
    ):
        state.pop(key, None)


def _notify_worker_error(
    config: ExperimentConfig,
    *,
    state: dict[str, Any],
    exc: Exception,
    now: datetime | None = None,
    transport: TelegramTransport | None = None,
) -> Path | None:
    if isinstance(exc, PaperLoopStageError):
        stage = exc.stage
        root_error = exc.cause
        proposal_id = exc.proposal_id
        trade_date = exc.trade_date
    else:
        stage = "paper-approve"
        root_error = exc
        proposal_id = ""
        trade_date = ""

    fingerprint = build_error_fingerprint(
        loop_name="agent",
        stage=stage,
        exc=root_error,
        proposal_id=proposal_id,
        trade_date=trade_date,
    )
    state["last_checked_at"] = _now_utc(now).isoformat()
    state["last_result"] = "error"
    if state.get("last_error_fingerprint") == fingerprint:
        return None

    state["last_error_fingerprint"] = fingerprint
    state["last_error_stage"] = stage
    state["last_error_type"] = type(root_error).__name__
    state["last_error_message"] = str(root_error)
    state["last_error_proposal_id"] = proposal_id
    state["last_error_trade_date"] = trade_date
    state["last_error_alert_at"] = _now_utc(now).isoformat()
    store = PaperStateStore(config)
    return _write_notification_record(
        config,
        store,
        stage="paper-error",
        outcome="error",
        message=build_error_message(
            config,
            loop_name="agent",
            stage=stage,
            exc=root_error,
            proposal_id=proposal_id,
            trade_date=trade_date,
        ),
        details={
            "experiment_name": config.experiment_name,
            "loop": "agent",
            "failed_stage": stage,
            "proposal_id": proposal_id,
            "trade_date": trade_date,
            "exception_type": type(root_error).__name__,
            "exception_message": str(root_error),
        },
        proposal_id=proposal_id,
        trade_date=trade_date,
        now=now,
        transport=transport,
    )


def _decision_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "decision": {
                "type": "string",
                "enum": ["approve", "reject"],
            },
            "rationale": {
                "type": "string",
            },
        },
        "required": ["decision", "rationale"],
        "additionalProperties": False,
    }


def _approval_policy_prompt() -> str:
    return (
        "Review the attached paper-trading proposal evidence and decide whether to "
        "approve or reject the existing proposal. You may only approve or reject the "
        "proposal as written. The consensus rule has already been applied by the system. "
        "If the persisted proposal and evidence are internally consistent for the same "
        "trade, approve it. Reject only when the persisted proposal or evidence is "
        "malformed, inconsistent, or refers to a different trade. Do not invent a "
        "different trade, symbol, quantity, side, target weight, threshold, or date. "
        "Return only the required structured output."
    )


def _coerce_agent_decision(payload: Any, *, provider: str, model: str) -> AgentDecision:
    if not isinstance(payload, dict):
        raise AgentDecisionError(f"{provider} returned a non-object structured response.")
    decision = str(payload.get("decision", "")).strip().lower()
    rationale = str(payload.get("rationale", "")).strip()
    if decision not in {"approve", "reject"}:
        raise AgentDecisionError(f"{provider} returned an invalid decision: {decision!r}")
    if rationale == "":
        raise AgentDecisionError(f"{provider} returned an empty rationale.")
    return AgentDecision(
        decision=decision,
        rationale=rationale,
        provider=provider,
        model=model,
    )


def _proposal_is_consistent(proposal: dict[str, Any], evidence: dict[str, Any]) -> tuple[bool, str]:
    if proposal.get("proposal_id") != evidence.get("proposal_id"):
        return False, "proposal_id mismatch"
    if proposal.get("symbol") != evidence.get("symbol"):
        return False, "symbol mismatch"
    if proposal.get("effective_date") != evidence.get("effective_date"):
        return False, "effective_date mismatch"
    if proposal.get("decision_policy") != "consensus_vote":
        return False, "unsupported decision policy"
    models = evidence.get("models", [])
    if not isinstance(models, list) or len(models) == 0:
        return False, "missing model evidence"
    consensus_rule = evidence.get("consensus_rule")
    if not isinstance(consensus_rule, dict):
        return False, "missing consensus rule"
    try:
        proposal_target_weight = float(proposal.get("target_weight", 0.0))
        evidence_target_weight = float(evidence.get("target_weight", 0.0))
        proposal_long_vote_count = int(proposal.get("long_vote_count", -1))
        evidence_long_vote_count = int(evidence.get("long_vote_count", -2))
        proposal_cash_vote_count = int(proposal.get("cash_vote_count", -1))
        evidence_cash_vote_count = int(evidence.get("cash_vote_count", -2))
        threshold = int(consensus_rule.get("min_long_votes", -1))
        model_count = int(consensus_rule.get("model_count", len(models)))
    except (TypeError, ValueError):
        return False, "invalid numeric proposal or evidence fields"
    if proposal.get("decision") != evidence.get("decision"):
        return False, "decision mismatch"
    if proposal_target_weight != evidence_target_weight:
        return False, "target_weight mismatch"
    if proposal_long_vote_count != evidence_long_vote_count:
        return False, "long_vote_count mismatch"
    if proposal_cash_vote_count != evidence_cash_vote_count:
        return False, "cash_vote_count mismatch"
    long_votes = sum(1 for row in models if row.get("vote") == "long")
    if long_votes != evidence_long_vote_count:
        return False, "model vote tally mismatch"
    cash_votes = len(models) - long_votes
    if cash_votes != evidence_cash_vote_count:
        return False, "cash vote tally mismatch"
    if model_count != len(models):
        return False, "consensus model_count mismatch"
    expected_target_weight = 1.0 if long_votes >= threshold else 0.0
    expected_decision = "long" if expected_target_weight > 0.0 else "cash"
    if proposal.get("decision") != expected_decision:
        return False, "consensus decision mismatch"
    if proposal_target_weight != expected_target_weight:
        return False, "consensus target_weight mismatch"
    return True, ""


class DeterministicConsensusBackend(AgentBackend):
    provider_name = "deterministic_consensus"

    def evaluate(
        self,
        *,
        config: ExperimentConfig,
        proposal: dict[str, Any],
        evidence: dict[str, Any],
        status: dict[str, Any] | None,
        account_context: dict[str, Any],
    ) -> AgentDecision:
        is_consistent, reason = _proposal_is_consistent(proposal, evidence)
        if not is_consistent:
            return AgentDecision(
                decision="reject",
                rationale=f"Rejected because the proposal evidence is inconsistent: {reason}.",
                provider=self.provider_name,
                model=self.provider_name,
            )

        long_vote_count = int(evidence["long_vote_count"])
        model_count = len(evidence["models"])
        threshold = int(evidence["consensus_rule"]["min_long_votes"])
        decision = str(proposal["decision"])
        return AgentDecision(
            decision="approve",
            rationale=(
                f"Approved because the proposal is internally consistent and the "
                f"{long_vote_count}/{model_count} consensus vote satisfies the "
                f"minimum-long-vote threshold of {threshold} for a {decision} action."
            ),
            provider=self.provider_name,
            model=self.provider_name,
        )


def _guardrail_primary_decision(
    *,
    config: ExperimentConfig,
    requested_backend: str,
    proposal: dict[str, Any],
    evidence: dict[str, Any],
    status: dict[str, Any] | None,
    account_context: dict[str, Any],
    primary_result: AgentDecision,
) -> AgentDecision:
    if requested_backend == "deterministic_consensus" or primary_result.decision != "reject":
        return primary_result

    deterministic_result = DeterministicConsensusBackend().evaluate(
        config=config,
        proposal=proposal,
        evidence=evidence,
        status=status,
        account_context=account_context,
    )
    if deterministic_result.decision != "approve":
        return primary_result

    return AgentDecision(
        decision=deterministic_result.decision,
        rationale=deterministic_result.rationale,
        provider=deterministic_result.provider,
        model=deterministic_result.model,
        fallback_used=True,
        fallback_reason=(
            f"{requested_backend} backend returned {primary_result.decision!r}, but "
            f"deterministic_consensus requires {deterministic_result.decision!r} for "
            "the persisted proposal evidence."
        ),
    )


def _extract_openai_text(response: Any) -> str:
    output_text = getattr(response, "output_text", None)
    if isinstance(output_text, str) and output_text.strip():
        return output_text

    parts: list[str] = []
    for output in getattr(response, "output", []) or []:
        if getattr(output, "type", None) != "message":
            continue
        for item in getattr(output, "content", []) or []:
            item_type = getattr(item, "type", None)
            if item_type == "refusal":
                raise AgentDecisionError(f"OpenAI refusal: {getattr(item, 'refusal', '')}")
            text_value = getattr(item, "text", None)
            if isinstance(text_value, str) and text_value.strip():
                parts.append(text_value)
    if parts:
        return "".join(parts)
    raise AgentDecisionError("OpenAI returned no parseable text output.")


class OpenAIAgentBackend(AgentBackend):
    provider_name = "openai"

    def evaluate(
        self,
        *,
        config: ExperimentConfig,
        proposal: dict[str, Any],
        evidence: dict[str, Any],
        status: dict[str, Any] | None,
        account_context: dict[str, Any],
    ) -> AgentDecision:
        load_env_file()
        api_key = os.environ.get("OPENAI_API_KEY", "").strip()
        if api_key == "":
            raise AgentDecisionError("OPENAI_API_KEY is not configured.")

        try:
            from openai import OpenAI
        except ImportError as exc:
            raise AgentDecisionError("The openai package is required for paper.agent_backend='openai'.") from exc

        client = OpenAI(api_key=api_key, timeout=config.paper.agent_timeout_seconds)
        response = client.responses.create(
            model=config.paper.agent_model,
            input=[
                {"role": "system", "content": _approval_policy_prompt()},
                {
                    "role": "user",
                    "content": json.dumps(
                        {
                            "proposal": proposal,
                            "evidence": evidence,
                            "latest_status": status,
                            "account_context": account_context,
                        },
                        sort_keys=True,
                    ),
                },
            ],
            text={
                "format": {
                    "type": "json_schema",
                    "name": "paper_agent_decision",
                    "strict": True,
                    "schema": _decision_schema(),
                }
            },
        )
        output_text = _extract_openai_text(response)
        return _coerce_agent_decision(
            json.loads(output_text),
            provider=self.provider_name,
            model=config.paper.agent_model,
        )


class ClaudeAgentBackend(AgentBackend):
    provider_name = "claude"

    def evaluate(
        self,
        *,
        config: ExperimentConfig,
        proposal: dict[str, Any],
        evidence: dict[str, Any],
        status: dict[str, Any] | None,
        account_context: dict[str, Any],
    ) -> AgentDecision:
        load_env_file()
        api_key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
        if api_key == "":
            raise AgentDecisionError("ANTHROPIC_API_KEY is not configured.")

        try:
            from anthropic import Anthropic
        except ImportError as exc:
            raise AgentDecisionError("The anthropic package is required for paper.agent_backend='claude'.") from exc

        client = Anthropic(api_key=api_key, timeout=config.paper.agent_timeout_seconds)
        response = client.messages.create(
            model=config.paper.agent_model,
            max_tokens=256,
            system=_approval_policy_prompt(),
            messages=[
                {
                    "role": "user",
                    "content": json.dumps(
                        {
                            "proposal": proposal,
                            "evidence": evidence,
                            "latest_status": status,
                            "account_context": account_context,
                        },
                        sort_keys=True,
                    ),
                }
            ],
            output_config={
                "format": {
                    "type": "json_schema",
                    "schema": _decision_schema(),
                }
            },
        )
        content = getattr(response, "content", []) or []
        if not content:
            raise AgentDecisionError("Claude returned no content.")
        first_item = content[0]
        text = getattr(first_item, "text", None)
        if not isinstance(text, str) or not text.strip():
            raise AgentDecisionError("Claude returned no structured JSON text.")
        return _coerce_agent_decision(
            json.loads(text),
            provider=self.provider_name,
            model=config.paper.agent_model,
        )


def _build_backend(config: ExperimentConfig, backend_name: str) -> AgentBackend:
    if backend_name == "deterministic_consensus":
        return DeterministicConsensusBackend()
    if backend_name == "openai":
        return OpenAIAgentBackend()
    if backend_name == "claude":
        return ClaudeAgentBackend()
    raise AgentDecisionError(f"Unsupported paper agent backend: {backend_name}")


def _evaluate_with_fallback(
    config: ExperimentConfig,
    *,
    proposal: dict[str, Any],
    evidence: dict[str, Any],
    status: dict[str, Any] | None,
    account_context: dict[str, Any],
) -> AgentDecision:
    requested_backend = config.paper.agent_backend
    primary = _build_backend(config, requested_backend)
    try:
        primary_result = primary.evaluate(
            config=config,
            proposal=proposal,
            evidence=evidence,
            status=status,
            account_context=account_context,
        )
        return _guardrail_primary_decision(
            config=config,
            requested_backend=requested_backend,
            proposal=proposal,
            evidence=evidence,
            status=status,
            account_context=account_context,
            primary_result=primary_result,
        )
    except Exception as exc:
        fallback_backend_name = config.paper.agent_fallback_backend
        if fallback_backend_name == requested_backend:
            raise
        fallback = _build_backend(config, fallback_backend_name)
        fallback_result = fallback.evaluate(
            config=config,
            proposal=proposal,
            evidence=evidence,
            status=status,
            account_context=account_context,
        )
        return AgentDecision(
            decision=fallback_result.decision,
            rationale=fallback_result.rationale,
            provider=fallback_result.provider,
            model=fallback_result.model,
            fallback_used=True,
            fallback_reason=f"{requested_backend} backend failed: {exc}",
        )


def _current_account_context(
    config: ExperimentConfig,
    *,
    broker: AlpacaPaperBrokerClient | None = None,
) -> dict[str, Any]:
    symbol = str(config.data.symbols[0])
    client = broker or AlpacaPaperBrokerClient()
    account = client.get_account()
    position = client.get_position(symbol)
    return {
        "account": account,
        "position": position,
    }


def run_agent_approval_iteration(
    config: ExperimentConfig,
    *,
    now: datetime | None = None,
    broker: AlpacaPaperBrokerClient | None = None,
    notification_transport: TelegramTransport | None = None,
) -> dict[str, Any]:
    validate_paper_trading_config(config)
    state = _load_worker_state(config)
    events: list[dict[str, Any]] = []

    if config.paper.execution_mode != "agent_approval":
        _clear_worker_error_state(state)
        state["last_checked_at"] = _now_utc(now).isoformat()
        state["last_result"] = "execution_mode_not_agent_approval"
        state_path = _save_worker_state(config, state)
        return {
            "agent_state_path": str(state_path),
            "events": [],
            "processed_count": 0,
        }

    store = PaperStateStore(config)
    proposals = sorted(
        [
            proposal
            for proposal in store.list_proposals()
            if proposal.get("approval_status", APPROVAL_PENDING) == APPROVAL_PENDING
            and not store.trade_submission_path(proposal["effective_date"]).exists()
        ],
        key=lambda proposal: (
            proposal.get("effective_date", ""),
            proposal.get("proposal_id", ""),
        ),
    )
    current_status = store.read_status()
    account_context = (
        _current_account_context(config, broker=broker)
        if proposals
        else {}
    )

    for proposal in proposals:
        try:
            evidence = read_paper_evidence(config, proposal_id=proposal["proposal_id"])
        except FileNotFoundError as exc:
            result = decide_paper_proposal(
                config,
                proposal_id=proposal["proposal_id"],
                decision="reject",
                actor="agent",
                rationale=(
                    "Rejected because the approval worker could not read the persisted "
                    f"proposal evidence: {exc}."
                ),
                fallback_reason=str(exc),
                now=now,
                notification_transport=notification_transport,
            )
            events.append(
                {
                    "proposal_id": proposal["proposal_id"],
                    "decision": "reject",
                    "provider": "",
                    "model": "",
                    "fallback_used": False,
                    "fallback_reason": str(exc),
                    "approval_path": result["approval_path"],
                }
            )
            continue
        try:
            decision = _evaluate_with_fallback(
                config,
                proposal=proposal,
                evidence=evidence,
                status=current_status,
                account_context=account_context,
            )
            result = decide_paper_proposal(
                config,
                proposal_id=proposal["proposal_id"],
                decision=decision.decision,
                actor="agent",
                rationale=decision.rationale,
                provider=decision.provider,
                model=decision.model,
                fallback_used=decision.fallback_used,
                fallback_reason=decision.fallback_reason,
                now=now,
                notification_transport=notification_transport,
            )
        except Exception as exc:
            raise PaperLoopStageError(
                loop_name="agent",
                stage="paper-approve",
                cause=exc,
                proposal_id=str(proposal["proposal_id"]),
                trade_date=str(proposal["effective_date"]),
            ) from exc
        events.append(
            {
                "proposal_id": proposal["proposal_id"],
                "decision": decision.decision,
                "provider": decision.provider,
                "model": decision.model,
                "fallback_used": decision.fallback_used,
                "fallback_reason": decision.fallback_reason,
                "approval_path": result["approval_path"],
            }
        )

    _clear_worker_error_state(state)
    state["last_checked_at"] = _now_utc(now).isoformat()
    state["last_processed_count"] = len(events)
    state["last_result"] = "processed" if events else "no_pending_proposals"
    state_path = _save_worker_state(config, state)
    return {
        "agent_state_path": str(state_path),
        "events": events,
        "processed_count": len(events),
    }


def run_agent_approval_loop(
    config: ExperimentConfig,
    *,
    once: bool = False,
    notification_transport: TelegramTransport | None = None,
) -> None:
    while True:
        loop_error: Exception | None = None
        try:
            summary = run_agent_approval_iteration(
                config,
                notification_transport=notification_transport,
            )
        except Exception as exc:
            loop_error = exc
            state = _load_worker_state(config)
            notification_path = _notify_worker_error(
                config,
                state=state,
                exc=exc,
                transport=notification_transport,
            )
            state_path = _save_worker_state(config, state)
            summary = {
                "agent_state_path": str(state_path),
                "events": [],
                "processed_count": 0,
                "error": {
                    "type": type(exc).__name__,
                    "message": str(exc),
                    "notification_path": str(notification_path) if notification_path else "",
                    "duplicate_suppressed": notification_path is None,
                },
            }
        print(json.dumps(summary, indent=2, sort_keys=True))
        if once:
            if loop_error is not None:
                raise loop_error
            return
        time.sleep(config.paper.poll_interval_seconds)
