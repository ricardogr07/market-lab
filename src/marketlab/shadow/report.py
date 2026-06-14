from __future__ import annotations

import json
import math
from collections import Counter
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd

from marketlab.pipeline import ML_TUNED_STRATEGY_NAME, _strict_research_gate
from marketlab.reports.phase8_gates import (
    calculate_bull_participation_gate,
    calculate_signal_validity_gate,
)
from marketlab.shadow.contract import VerifiedShadowContract, verify_shadow_contract
from marketlab.shadow.evidence import (
    ShadowAttemptStore,
    ShadowDecisionEvidenceStore,
    ShadowLabelEvidenceStore,
)
from marketlab.shadow.journal import ShadowDecisionJournal
from marketlab.shadow.status import build_shadow_status


class ShadowReportError(RuntimeError):
    """Raised when a shadow evidence report cannot be generated safely."""


def build_shadow_report(
    config_path: str | Path,
    *,
    as_of: date,
    report_type: str = "monthly",
    journal: ShadowDecisionJournal | None = None,
    attempt_store: ShadowAttemptStore | None = None,
    decision_evidence_store: ShadowDecisionEvidenceStore | None = None,
    label_evidence_store: ShadowLabelEvidenceStore | None = None,
) -> dict[str, Any]:
    contract = verify_shadow_contract(config_path)
    if report_type == "final" and as_of < contract.earliest_final_evaluation:
        raise ShadowReportError(
            "Final shadow evidence cannot be generated before "
            f"{contract.earliest_final_evaluation.isoformat()}."
        )
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
    status = build_shadow_status(
        config_path,
        as_of=as_of,
        journal=selected_journal,
        attempt_store=selected_attempts,
        decision_evidence_store=selected_decision_evidence,
        label_evidence_store=selected_labels,
    )
    if status["integrity_errors"]:
        raise ShadowReportError("Canonical shadow evidence failed fingerprint validation.")

    decisions = _dated_records(status, selected_journal.read)
    attempts = [
        record
        for record in selected_attempts.list()
        if contract.protocol_start
        <= date.fromisoformat(str(record["effective_date"]))
        <= min(as_of, contract.protocol_end)
    ]
    decision_evidence = _dated_records(status, selected_decision_evidence.read)
    labels = _dated_records(status, selected_labels.read)
    integrity = _integrity_summary(
        contract=contract,
        decisions=decisions,
        attempts=attempts,
        decision_evidence=decision_evidence,
        labels=labels,
    )
    active_returns = {
        f"{int(cost_bps)}_bps": _active_return(labels, cost_bps=cost_bps)
        for cost_bps in (35.0, 50.0)
    }
    signal_metrics = _signal_metrics(decision_evidence, labels)
    signal_gate = calculate_signal_validity_gate(signal_metrics)
    bull_metrics = _bull_metrics(decision_evidence, labels)
    bull_gate = calculate_bull_participation_gate(bull_metrics)
    strict_gate = _shadow_strict_gate(contract, decision_evidence, labels)
    fallback_counts = Counter(
        str(record.get("selection_source")) for record in decision_evidence
    )
    classification_counts = status["counts"]
    completeness_passed = (
        int(classification_counts.get("missed", 0)) == 0
        and int(classification_counts.get("failed", 0)) == 0
        and int(classification_counts.get("pending", 0)) == 0
        and int(classification_counts.get("skipped", 0)) == 0
    )
    maturity_coverage = (
        len(labels) / len(decision_evidence) if decision_evidence else 0.0
    )
    strict_overall = _gate_overall(strict_gate)
    graduation_checks = {
        "scheduled_date_completeness": completeness_passed,
        "fingerprint_integrity": bool(integrity["passed"]),
        "frozen_contract_invariants": bool(integrity["invariants_match"]),
        "strict_research_gate": strict_overall,
        "active_return_35_bps": active_returns["35_bps"]["active_return"] > 0.0,
        "active_return_50_bps": active_returns["50_bps"]["active_return"] > 0.0,
        "signal_validity_gate": signal_gate.passed,
        "bull_participation_gate": bull_gate.passed,
        "zero_best_active_fallback": (
            int(fallback_counts.get("best_active_fallback", 0)) == 0
        ),
        "zero_regime_policy_fallback": (
            int(fallback_counts.get("regime_policy_fallback", 0)) == 0
        ),
        "full_maturity_coverage": (
            maturity_coverage == 1.0
            and as_of >= contract.earliest_final_evaluation
        ),
    }
    allocations = Counter(
        f"{float(record['selected_tier']):g}"
        for record in decision_evidence
        if record.get("selected_tier") is not None
    )
    return {
        "schema_version": 1,
        "report_type": report_type,
        "provisional": report_type != "final",
        "informational_only": True,
        "promotion_decision": None,
        "as_of": as_of.isoformat(),
        "candidate_id": contract.candidate_id,
        "behavior_version": contract.behavior_version,
        "config_hash": contract.config_hash,
        "behavior_hash": contract.behavior_hash,
        "code_lock": contract.code_lock,
        "protocol": {
            "start": contract.protocol_start.isoformat(),
            "end": contract.protocol_end.isoformat(),
            "earliest_final_evaluation": (
                contract.earliest_final_evaluation.isoformat()
            ),
            "maturity_lag_bars": contract.maturity_lag_bars,
        },
        "completeness": {
            "expected_dates": status["expected_dates"],
            "counts": classification_counts,
            "dates": status["dates"],
            "passed": completeness_passed,
        },
        "integrity": integrity,
        "active_returns": active_returns,
        "strict_research_gate": _frame_records(strict_gate),
        "signal_validity_gate": {
            "metrics": signal_metrics,
            "conditions": signal_gate.conditions,
            "passed": signal_gate.passed,
        },
        "bull_participation_gate": {
            "metrics": bull_metrics,
            "conditions": bull_gate.conditions,
            "passed": bull_gate.passed,
        },
        "fallback_counts": dict(sorted(fallback_counts.items())),
        "maturity": {
            "decision_evidence_count": len(decision_evidence),
            "label_evidence_count": len(labels),
            "coverage": maturity_coverage,
        },
        "operations": {
            "attempt_count": len(attempts),
            "allocation_distribution": dict(sorted(allocations.items())),
            "average_exposure": _mean(labels, "exposure"),
            "total_turnover": sum(float(record["turnover"]) for record in labels),
            "failure_count": int(classification_counts.get("failed", 0)),
            "skip_count": int(classification_counts.get("skipped", 0)),
            "missing_dates": [
                item["effective_date"]
                for item in status["dates"]
                if item["classification"] in {"missed", "pending"}
            ],
        },
        "graduation_checks": graduation_checks,
        "all_graduation_metrics_passed": all(graduation_checks.values()),
    }


def write_monthly_shadow_report(
    config_path: str | Path,
    *,
    as_of: date,
    output_root: str | Path | None = None,
    journal: ShadowDecisionJournal | None = None,
    attempt_store: ShadowAttemptStore | None = None,
    decision_evidence_store: ShadowDecisionEvidenceStore | None = None,
    label_evidence_store: ShadowLabelEvidenceStore | None = None,
) -> tuple[Path, Path, dict[str, Any]]:
    contract = verify_shadow_contract(config_path)
    report = build_shadow_report(
        config_path,
        as_of=as_of,
        report_type="monthly",
        journal=journal,
        attempt_store=attempt_store,
        decision_evidence_store=decision_evidence_store,
        label_evidence_store=label_evidence_store,
    )
    root = (
        Path(output_root)
        if output_root is not None
        else contract.artifact_root / "reports" / "monthly"
    )
    return _write_report(root / as_of.strftime("%Y-%m"), report)


def write_final_shadow_report(
    config_path: str | Path,
    *,
    as_of: date,
    output_root: str | Path | None = None,
    journal: ShadowDecisionJournal | None = None,
    attempt_store: ShadowAttemptStore | None = None,
    decision_evidence_store: ShadowDecisionEvidenceStore | None = None,
    label_evidence_store: ShadowLabelEvidenceStore | None = None,
) -> tuple[Path, Path, dict[str, Any]]:
    contract = verify_shadow_contract(config_path)
    report = build_shadow_report(
        config_path,
        as_of=as_of,
        report_type="final",
        journal=journal,
        attempt_store=attempt_store,
        decision_evidence_store=decision_evidence_store,
        label_evidence_store=label_evidence_store,
    )
    root = (
        Path(output_root)
        if output_root is not None
        else contract.artifact_root / "reports" / "final"
    )
    return _write_report(root / as_of.isoformat(), report)


def _write_report(
    directory: Path,
    report: dict[str, Any],
) -> tuple[Path, Path, dict[str, Any]]:
    directory.mkdir(parents=True, exist_ok=True)
    json_path = directory / "report.json"
    markdown_path = directory / "report.md"
    json_path.write_text(
        json.dumps(report, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    markdown_path.write_text(
        _markdown_report(report),
        encoding="utf-8",
        newline="\n",
    )
    return json_path, markdown_path, report


def _dated_records(status: dict[str, Any], reader) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for item in status["dates"]:
        record = reader(date.fromisoformat(str(item["effective_date"])))
        if record is not None:
            records.append(record)
    return records


def _integrity_summary(
    *,
    contract: VerifiedShadowContract,
    decisions: list[dict[str, Any]],
    attempts: list[dict[str, Any]],
    decision_evidence: list[dict[str, Any]],
    labels: list[dict[str, Any]],
) -> dict[str, object]:
    expected = {
        "candidate_id": contract.candidate_id,
        "behavior_version": contract.behavior_version,
        "config_hash": contract.config_hash,
        "behavior_hash": contract.behavior_hash,
        "code_lock": contract.code_lock,
    }
    invariant_errors: list[str] = []
    for record_type, records in (
        ("decision", decisions),
        ("attempt", attempts),
        ("decision_evidence", decision_evidence),
        ("label_evidence", labels),
    ):
        for record in records:
            for field, expected_value in expected.items():
                if record.get(field) != expected_value:
                    invariant_errors.append(
                        f"{record_type}:{record.get('effective_date')}:{field}"
                    )
    cross_link_errors: list[str] = []
    decisions_by_date = {
        str(record["effective_date"]): record for record in decisions
    }
    evidence_by_date = {
        str(record["effective_date"]): record for record in decision_evidence
    }
    for record in decision_evidence:
        decision = decisions_by_date.get(str(record["effective_date"]))
        if decision is None or record.get("decision_fingerprint") != decision.get(
            "output_fingerprint"
        ):
            cross_link_errors.append(
                f"decision_evidence:{record.get('effective_date')}"
            )
    for record in labels:
        effective_date = str(record["effective_date"])
        decision = decisions_by_date.get(effective_date)
        evidence = evidence_by_date.get(effective_date)
        if decision is None or record.get("decision_fingerprint") != decision.get(
            "output_fingerprint"
        ):
            cross_link_errors.append(f"label_decision:{effective_date}")
        if evidence is None or record.get(
            "decision_evidence_fingerprint"
        ) != evidence.get("output_fingerprint"):
            cross_link_errors.append(f"label_evidence:{effective_date}")
    return {
        "passed": not invariant_errors and not cross_link_errors,
        "invariants_match": not invariant_errors,
        "invariant_errors": invariant_errors,
        "cross_link_errors": cross_link_errors,
        "fingerprint_verified_records": (
            len(decisions) + len(attempts) + len(decision_evidence) + len(labels)
        ),
    }


def _active_return(labels: list[dict[str, Any]], *, cost_bps: float) -> dict[str, float]:
    strategy_returns = [
        float(record["exposure"]) * float(record["benchmark_return"])
        - float(record["turnover"]) * cost_bps / 10_000.0
        for record in labels
    ]
    benchmark_returns = [float(record["benchmark_return"]) for record in labels]
    strategy = _compound(strategy_returns)
    benchmark = _compound(benchmark_returns)
    return {
        "strategy_return": strategy,
        "benchmark_return": benchmark,
        "active_return": strategy - benchmark,
    }


def _signal_metrics(
    decision_evidence: list[dict[str, Any]],
    labels: list[dict[str, Any]],
) -> dict[str, object]:
    labels_by_date = {str(record["effective_date"]): record for record in labels}
    joined = [
        (record, labels_by_date[str(record["effective_date"])])
        for record in decision_evidence
        if str(record["effective_date"]) in labels_by_date
        and record.get("raw_score") is not None
    ]
    scores = [float(record["raw_score"]) for record, _ in joined]
    target_weights = [float(label["realized_target_weight"]) for _, label in joined]
    benchmark_returns = [float(label["benchmark_return"]) for _, label in joined]
    utilities = [float(label["realized_utility"]) for _, label in joined]
    tiers = [
        float(record["selected_tier"])
        for record in decision_evidence
        if record.get("selected_tier") is not None
    ]
    return {
        "score_target_weight_correlation": _correlation(scores, target_weights),
        "score_forward_return_correlation": _correlation(
            scores, benchmark_returns
        ),
        "score_realized_utility_correlation": _correlation(scores, utilities),
        "predicted_tier_100_fraction": (
            sum(abs(tier - 1.0) <= 1e-9 for tier in tiers) / len(tiers)
            if tiers
            else 0.0
        ),
        "any_selected_oos_predicted_tier_100": any(
            abs(tier - 1.0) <= 1e-9 for tier in tiers
        ),
    }


def _bull_metrics(
    decision_evidence: list[dict[str, Any]],
    labels: list[dict[str, Any]],
) -> dict[str, object]:
    labels_by_date = {str(record["effective_date"]): record for record in labels}
    bull = [
        labels_by_date[str(record["effective_date"])]
        for record in decision_evidence
        if bool(record.get("gate_bull"))
        and str(record["effective_date"]) in labels_by_date
    ]
    active = [
        float(record["strategy_return"]) - float(record["benchmark_return"])
        for record in bull
    ]
    missed_upside = [
        float(record["benchmark_return"])
        for record in bull
        if float(record["benchmark_return"]) > 0.0
        and float(record["exposure"]) < 1.0 - 1e-9
    ]
    selected_count = sum(
        str(record.get("selection_source")) != "none"
        for record in decision_evidence
    )
    return {
        "gate_bull_average_long_exposure": _mean(bull, "exposure"),
        "gate_bull_active_return_sum": sum(active) if bull else None,
        "gate_bull_underexposed_positive_benchmark_return_sum": sum(
            missed_upside
        ),
        "selected_fold_fraction": (
            selected_count / len(decision_evidence) if decision_evidence else 0.0
        ),
    }


def _shadow_strict_gate(
    contract: VerifiedShadowContract,
    decision_evidence: list[dict[str, Any]],
    labels: list[dict[str, Any]],
) -> pd.DataFrame:
    strategies = [
        (ML_TUNED_STRATEGY_NAME, None),
        ("buy_hold", 1.0),
        ("btc_static_25", 0.25),
        ("btc_static_50", 0.50),
        ("btc_static_75", 0.75),
        ("btc_rebalanced_25", 0.25),
        ("btc_rebalanced_50", 0.50),
        ("btc_rebalanced_75", 0.75),
    ]
    summary_rows: list[dict[str, object]] = []
    cost_rows: list[dict[str, object]] = []
    for strategy, fixed_exposure in strategies:
        returns_35, exposures, turnovers = _strategy_series(
            labels,
            fixed_exposure=fixed_exposure,
            cost_bps=35.0,
        )
        summary_rows.append(
            _strategy_summary_row(strategy, returns_35, exposures, turnovers)
        )
        for cost_bps in (35.0, 50.0):
            returns, _, _ = _strategy_series(
                labels,
                fixed_exposure=fixed_exposure,
                cost_bps=cost_bps,
            )
            cost_rows.append(
                {
                    "strategy": strategy,
                    "bps_per_trade": cost_bps,
                    "cumulative_return": _compound(returns),
                }
            )
    regime_rows = []
    labels_by_date = {str(record["effective_date"]): record for record in labels}
    for regime, records in _group_by(decision_evidence, "regime_classification").items():
        active_returns = [
            float(labels_by_date[str(record["effective_date"])]["strategy_return"])
            - float(labels_by_date[str(record["effective_date"])]["benchmark_return"])
            for record in records
            if str(record["effective_date"]) in labels_by_date
        ]
        regime_rows.append(
            {"slice_name": regime, "active_return": sum(active_returns)}
        )
    selections = pd.DataFrame(
        [
            {
                "fold_id": record["effective_date"],
                "selection_status": "selected",
                "selection_source": record["selection_source"],
            }
            for record in decision_evidence
        ]
    )
    target_diagnostics = _target_diagnostics(labels)
    probability_diagnostics = pd.DataFrame(
        [
            {
                "fold_id": record["effective_date"],
                "predicted_tier_weight": record["selected_tier"],
            }
            for record in decision_evidence
        ]
    )
    return _strict_research_gate(
        config=contract.config,
        strategy_summary=pd.DataFrame(summary_rows),
        cost_sensitivity=pd.DataFrame(cost_rows),
        regime_slices=pd.DataFrame(regime_rows),
        ml_strategy_tuning_selections=selections,
        allocation_target_diagnostics=target_diagnostics,
        allocation_probability_diagnostics=probability_diagnostics,
    )


def _target_diagnostics(labels: list[dict[str, Any]]) -> pd.DataFrame:
    rows = [
        {
            "scope": "global",
            "fold_id": "global",
            "target_weight": weight,
            "row_count": sum(
                abs(float(record["realized_target_weight"]) - weight) <= 1e-9
                for record in labels
            ),
        }
        for weight in (0.0, 0.25, 0.50, 1.0)
    ]
    rows.extend(
        {
            "scope": "train_validation",
            "fold_id": record["effective_date"],
            "target_weight": record["realized_target_weight"],
            "row_count": 1,
        }
        for record in labels
    )
    return pd.DataFrame(rows)


def _strategy_series(
    labels: list[dict[str, Any]],
    *,
    fixed_exposure: float | None,
    cost_bps: float,
) -> tuple[list[float], list[float], list[float]]:
    returns: list[float] = []
    exposures: list[float] = []
    turnovers: list[float] = []
    previous = 0.0
    for record in labels:
        exposure = (
            float(record["exposure"])
            if fixed_exposure is None
            else fixed_exposure
        )
        turnover = (
            float(record["turnover"])
            if fixed_exposure is None
            else abs(exposure - previous)
        )
        returns.append(
            exposure * float(record["benchmark_return"])
            - turnover * cost_bps / 10_000.0
        )
        exposures.append(exposure)
        turnovers.append(turnover)
        previous = exposure
    return returns, exposures, turnovers


def _strategy_summary_row(
    strategy: str,
    returns: list[float],
    exposures: list[float],
    turnovers: list[float],
) -> dict[str, object]:
    equity = pd.Series([1.0, *list(pd.Series(1.0 + pd.Series(returns)).cumprod())])
    drawdown = (equity / equity.cummax()) - 1.0
    standard_deviation = pd.Series(returns).std(ddof=0)
    sharpe = (
        float(pd.Series(returns).mean() / standard_deviation * math.sqrt(365.0))
        if returns and standard_deviation > 0.0
        else 0.0
    )
    return {
        "strategy": strategy,
        "cumulative_return": _compound(returns),
        "sharpe_like": sharpe,
        "max_drawdown": float(drawdown.min()) if not drawdown.empty else 0.0,
        "avg_gross_exposure": (
            sum(exposures) / len(exposures) if exposures else 0.0
        ),
        "avg_turnover": (
            sum(turnovers) / len(turnovers) if turnovers else 0.0
        ),
    }


def _gate_overall(gate: pd.DataFrame) -> bool:
    rows = gate.loc[gate["condition"].astype(str).eq("overall")]
    return bool(rows.iloc[0]["passed"]) if not rows.empty else False


def _compound(returns: list[float]) -> float:
    value = 1.0
    for period_return in returns:
        value *= 1.0 + period_return
    return value - 1.0


def _correlation(left: list[float], right: list[float]) -> float | None:
    if len(left) < 2 or len(left) != len(right):
        return None
    correlation = pd.Series(left).corr(pd.Series(right))
    return float(correlation) if pd.notna(correlation) else None


def _mean(records: list[dict[str, Any]], field: str) -> float | None:
    if not records:
        return None
    return sum(float(record[field]) for record in records) / len(records)


def _group_by(
    records: list[dict[str, Any]],
    field: str,
) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        grouped.setdefault(str(record.get(field, "missing")), []).append(record)
    return grouped


def _frame_records(frame: pd.DataFrame) -> list[dict[str, object]]:
    return [
        {str(key): _json_value(value) for key, value in row.items()}
        for row in frame.to_dict(orient="records")
    ]


def _json_value(value: object) -> object:
    if value is None or value is pd.NA:
        return None
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if hasattr(value, "item"):
        return value.item()
    return value


def _markdown_report(report: dict[str, Any]) -> str:
    provisional = "Yes" if report["provisional"] else "No"
    checks = report["graduation_checks"]
    lines = [
        "# Phase 9 BTC Shadow Evidence Report",
        "",
        f"- As of: `{report['as_of']}`",
        f"- Report type: `{report['report_type']}`",
        f"- Provisional: `{provisional}`",
        "- Informational only: `True`",
        "- Promotion decision: `None`",
        "",
        "## Graduation Metrics",
        "",
        "| Metric | Passed |",
        "| --- | --- |",
    ]
    lines.extend(
        f"| `{name}` | `{bool(passed)}` |"
        for name, passed in sorted(checks.items())
    )
    lines.extend(
        [
            "",
            "This report cannot enable paper execution, invoke approval services, "
            "or call a broker.",
            "",
        ]
    )
    return "\n".join(lines)
