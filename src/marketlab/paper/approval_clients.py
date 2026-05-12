from __future__ import annotations

import json
import os
from collections.abc import Mapping
from typing import Any

from marketlab.config import ExperimentConfig
from marketlab.env import load_env_file
from marketlab.paper.contracts import (
    PaperApprovalClient,
    PaperApprovalClientDecision,
    PaperApprovalEvaluationRequest,
)
from marketlab.strategies.tiered_allocation import nearest_tier


class PaperApprovalClientError(RuntimeError):
    pass


class _Backend:
    provider_name = "base"

    def evaluate(
        self,
        *,
        config: ExperimentConfig,
        request: PaperApprovalEvaluationRequest,
    ) -> PaperApprovalClientDecision:
        raise NotImplementedError


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


def _coerce_agent_decision(
    payload: Any,
    *,
    provider: str,
    model: str,
) -> PaperApprovalClientDecision:
    if not isinstance(payload, dict):
        raise PaperApprovalClientError(f"{provider} returned a non-object structured response.")
    decision = str(payload.get("decision", "")).strip().lower()
    rationale = str(payload.get("rationale", "")).strip()
    if decision not in {"approve", "reject"}:
        raise PaperApprovalClientError(f"{provider} returned an invalid decision: {decision!r}")
    if rationale == "":
        raise PaperApprovalClientError(f"{provider} returned an empty rationale.")
    return PaperApprovalClientDecision(
        decision=decision,
        rationale=rationale,
        provider=provider,
        model=model,
    )


def _proposal_is_consistent(
    proposal: Mapping[str, Any],
    evidence: Mapping[str, Any],
) -> tuple[bool, str]:
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
    allocation_policy = str(consensus_rule.get("allocation_policy", "binary_vote"))
    if allocation_policy == "average_model_weight_nearest_tier":
        try:
            model_target_weights = [float(row.get("target_weight", 0.0)) for row in models]
        except (TypeError, ValueError):
            return False, "invalid model target weights"
        expected_target_weight = nearest_tier(
            sum(model_target_weights) / len(model_target_weights)
        )
        expected_decision = (
            f"long_{int(expected_target_weight * 100)}"
            if expected_target_weight > 0.0
            else "cash"
        )
    else:
        expected_target_weight = 1.0 if long_votes >= threshold else 0.0
        expected_decision = "long" if expected_target_weight > 0.0 else "cash"
    if proposal.get("decision") != expected_decision:
        return False, "consensus decision mismatch"
    if proposal_target_weight != expected_target_weight:
        return False, "consensus target_weight mismatch"
    return True, ""


class _DeterministicConsensusBackend(_Backend):
    provider_name = "deterministic_consensus"

    def evaluate(
        self,
        *,
        config: ExperimentConfig,
        request: PaperApprovalEvaluationRequest,
    ) -> PaperApprovalClientDecision:
        is_consistent, reason = _proposal_is_consistent(request.proposal, request.evidence)
        if not is_consistent:
            return PaperApprovalClientDecision(
                decision="reject",
                rationale=f"Rejected because the proposal evidence is inconsistent: {reason}.",
                provider=self.provider_name,
                model=self.provider_name,
            )

        long_vote_count = int(request.evidence["long_vote_count"])
        model_count = len(request.evidence["models"])
        threshold = int(request.evidence["consensus_rule"]["min_long_votes"])
        decision = str(request.proposal["decision"])
        allocation_policy = str(
            request.evidence["consensus_rule"].get("allocation_policy", "binary_vote")
        )
        if allocation_policy == "average_model_weight_nearest_tier":
            return PaperApprovalClientDecision(
                decision="approve",
                rationale=(
                    "Approved because the tiered BTC allocation proposal is internally "
                    f"consistent and the persisted model weights round to {decision}."
                ),
                provider=self.provider_name,
                model=self.provider_name,
            )
        return PaperApprovalClientDecision(
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
    request: PaperApprovalEvaluationRequest,
    primary_result: PaperApprovalClientDecision,
) -> PaperApprovalClientDecision:
    if requested_backend == "deterministic_consensus" or primary_result.decision != "reject":
        return primary_result

    deterministic_result = _DeterministicConsensusBackend().evaluate(
        config=config,
        request=request,
    )
    if deterministic_result.decision != "approve":
        return primary_result

    return PaperApprovalClientDecision(
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
                raise PaperApprovalClientError(f"OpenAI refusal: {getattr(item, 'refusal', '')}")
            text_value = getattr(item, "text", None)
            if isinstance(text_value, str) and text_value.strip():
                parts.append(text_value)
    if parts:
        return "".join(parts)
    raise PaperApprovalClientError("OpenAI returned no parseable text output.")


class _OpenAIAgentBackend(_Backend):
    provider_name = "openai"

    def evaluate(
        self,
        *,
        config: ExperimentConfig,
        request: PaperApprovalEvaluationRequest,
    ) -> PaperApprovalClientDecision:
        load_env_file()
        api_key = os.environ.get("OPENAI_API_KEY", "").strip()
        if api_key == "":
            raise PaperApprovalClientError("OPENAI_API_KEY is not configured.")

        try:
            from openai import OpenAI
        except ImportError as exc:
            raise PaperApprovalClientError(
                "The openai package is required for paper.agent_backend='openai'."
            ) from exc

        client = OpenAI(api_key=api_key, timeout=config.paper.agent_timeout_seconds)
        response = client.responses.create(
            model=config.paper.agent_model,
            input=[
                {"role": "system", "content": _approval_policy_prompt()},
                {
                    "role": "user",
                    "content": json.dumps(
                        {
                            "proposal": dict(request.proposal),
                            "evidence": dict(request.evidence),
                            "latest_status": dict(request.status or {}),
                            "account_context": dict(request.account_context),
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


class _ClaudeAgentBackend(_Backend):
    provider_name = "claude"

    def evaluate(
        self,
        *,
        config: ExperimentConfig,
        request: PaperApprovalEvaluationRequest,
    ) -> PaperApprovalClientDecision:
        load_env_file()
        api_key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
        if api_key == "":
            raise PaperApprovalClientError("ANTHROPIC_API_KEY is not configured.")

        try:
            from anthropic import Anthropic
        except ImportError as exc:
            raise PaperApprovalClientError(
                "The anthropic package is required for paper.agent_backend='claude'."
            ) from exc

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
                            "proposal": dict(request.proposal),
                            "evidence": dict(request.evidence),
                            "latest_status": dict(request.status or {}),
                            "account_context": dict(request.account_context),
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
            raise PaperApprovalClientError("Claude returned no content.")
        first_item = content[0]
        text = getattr(first_item, "text", None)
        if not isinstance(text, str) or not text.strip():
            raise PaperApprovalClientError("Claude returned no structured JSON text.")
        return _coerce_agent_decision(
            json.loads(text),
            provider=self.provider_name,
            model=config.paper.agent_model,
        )


def _build_backend(backend_name: str) -> _Backend:
    if backend_name == "deterministic_consensus":
        return _DeterministicConsensusBackend()
    if backend_name == "openai":
        return _OpenAIAgentBackend()
    if backend_name == "claude":
        return _ClaudeAgentBackend()
    raise PaperApprovalClientError(f"Unsupported paper agent backend: {backend_name}")


class _ConfiguredPaperApprovalClient(PaperApprovalClient):
    def __init__(self, config: ExperimentConfig) -> None:
        self._config = config

    def evaluate(self, request: PaperApprovalEvaluationRequest) -> PaperApprovalClientDecision:
        requested_backend = self._config.paper.agent_backend
        primary = _build_backend(requested_backend)
        try:
            primary_result = primary.evaluate(config=self._config, request=request)
            return _guardrail_primary_decision(
                config=self._config,
                requested_backend=requested_backend,
                request=request,
                primary_result=primary_result,
            )
        except Exception as exc:
            fallback_backend_name = self._config.paper.agent_fallback_backend
            if fallback_backend_name == requested_backend:
                raise
            fallback = _build_backend(fallback_backend_name)
            fallback_result = fallback.evaluate(config=self._config, request=request)
            return PaperApprovalClientDecision(
                decision=fallback_result.decision,
                rationale=fallback_result.rationale,
                provider=fallback_result.provider,
                model=fallback_result.model,
                fallback_used=True,
                fallback_reason=f"{requested_backend} backend failed: {exc}",
            )


def build_default_paper_approval_client(
    config: ExperimentConfig,
) -> PaperApprovalClient:
    return _ConfiguredPaperApprovalClient(config)
