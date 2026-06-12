from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path
from typing import Any

import yaml

from marketlab.config import ExperimentConfig, load_config

_DECLARED_HASH_FIELDS = {"behavior_hash", "config_hash"}


class ShadowContractError(RuntimeError):
    """Raised when a shadow candidate differs from its approved contract."""


@dataclass(frozen=True, slots=True)
class _ApprovedShadowCandidate:
    config_filename: str
    behavior_version: str
    protocol_start: date
    protocol_end: date
    earliest_final_evaluation: date
    maturity_lag_bars: int
    code_lock: str
    artifact_root: str
    config_hash: str
    behavior_hash: str


@dataclass(frozen=True, slots=True)
class VerifiedShadowContract:
    config: ExperimentConfig
    config_path: Path
    candidate_id: str
    behavior_version: str
    protocol_start: date
    protocol_end: date
    earliest_final_evaluation: date
    maturity_lag_bars: int
    code_lock: str
    artifact_root: Path
    config_hash: str
    behavior_hash: str


_APPROVED_SHADOW_CANDIDATES = {
    "btc-phase9-shadow-v1": _ApprovedShadowCandidate(
        config_filename="experiment.btc_phase9_shadow_daily.yaml",
        behavior_version="btc-phase8-guarded-gate-v1",
        protocol_start=date(2026, 6, 3),
        protocol_end=date(2027, 6, 2),
        earliest_final_evaluation=date(2027, 6, 16),
        maturity_lag_bars=14,
        code_lock="ce01124",
        artifact_root="artifacts/phase9-shadow",
        config_hash="d439acca79ca2108a4d907452b5d442ab67b319d430440d07f14f9adc1295f18",
        behavior_hash="71beba28529abba3482145094654c5eaf8f12355d92a93830fe746a241129550",
    )
}


def _canonical_hash(payload: object) -> str:
    encoded = json.dumps(
        payload,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _config_hash(payload: dict[str, Any]) -> str:
    canonical_payload = dict(payload)
    shadow_payload = canonical_payload.get("shadow")
    if isinstance(shadow_payload, dict):
        canonical_shadow = dict(shadow_payload)
        for field_name in _DECLARED_HASH_FIELDS:
            canonical_shadow.pop(field_name, None)
        canonical_payload["shadow"] = canonical_shadow
    return _canonical_hash(canonical_payload)


def _behavior_payload(
    config: ExperimentConfig,
    *,
    behavior_version: str,
) -> dict[str, object]:
    return {
        "behavior_version": behavior_version,
        "data": {
            "symbols": list(config.data.symbols),
            "start_date": config.data.start_date,
            "interval": config.data.interval,
        },
        "features": asdict(config.features),
        "target": asdict(config.target),
        "portfolio": asdict(config.portfolio),
        "baselines": asdict(config.baselines),
        "models": [asdict(model) for model in config.models],
        "evaluation": asdict(config.evaluation),
    }


def _behavior_hash(config: ExperimentConfig, *, behavior_version: str) -> str:
    return _canonical_hash(
        _behavior_payload(config, behavior_version=behavior_version)
    )


def _parse_protocol_date(label: str, value: object) -> date:
    if not isinstance(value, str):
        raise ShadowContractError(f"shadow.{label} must use YYYY-MM-DD format.")
    try:
        parsed = date.fromisoformat(value)
    except ValueError as exc:
        raise ShadowContractError(f"shadow.{label} must use YYYY-MM-DD format.") from exc
    if parsed.isoformat() != value:
        raise ShadowContractError(f"shadow.{label} must use YYYY-MM-DD format.")
    return parsed


def _load_raw_mapping(path: Path) -> dict[str, Any]:
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise ShadowContractError(f"Unable to read shadow config: {path}") from exc
    if not isinstance(payload, dict):
        raise ShadowContractError("Shadow config must contain a YAML mapping.")
    return payload


def _require_equal(label: str, actual: object, expected: object) -> None:
    if actual != expected:
        raise ShadowContractError(
            f"{label} differs from the approved shadow contract: "
            f"expected {expected!r}, received {actual!r}."
        )


def verify_shadow_contract(path: str | Path) -> VerifiedShadowContract:
    config_path = Path(path).resolve()
    raw_payload = _load_raw_mapping(config_path)
    try:
        config = load_config(config_path)
    except (AttributeError, OSError, TypeError, ValueError, yaml.YAMLError) as exc:
        raise ShadowContractError(f"Invalid shadow config: {config_path}") from exc

    shadow = config.shadow
    if shadow is None:
        raise ShadowContractError("Shadow config must define a top-level shadow section.")

    if not isinstance(shadow.candidate_id, str):
        raise ShadowContractError("shadow.candidate_id must be a string.")
    approved = _APPROVED_SHADOW_CANDIDATES.get(shadow.candidate_id)
    if approved is None:
        raise ShadowContractError(
            f"Unknown shadow candidate_id: {shadow.candidate_id!r}."
        )
    if (
        config_path.name != approved.config_filename
        or config_path.parent.name != "configs"
    ):
        raise ShadowContractError(
            "The approved shadow candidate must be loaded from "
            f"configs/{approved.config_filename}."
        )

    protocol_start = _parse_protocol_date("protocol_start", shadow.protocol_start)
    protocol_end = _parse_protocol_date("protocol_end", shadow.protocol_end)
    earliest_final = _parse_protocol_date(
        "earliest_final_evaluation",
        shadow.earliest_final_evaluation,
    )
    if not protocol_start <= protocol_end < earliest_final:
        raise ShadowContractError(
            "Shadow protocol dates must satisfy start <= end < earliest final evaluation."
        )

    _require_equal(
        "shadow.behavior_version",
        shadow.behavior_version,
        approved.behavior_version,
    )
    _require_equal("shadow.protocol_start", protocol_start, approved.protocol_start)
    _require_equal("shadow.protocol_end", protocol_end, approved.protocol_end)
    _require_equal(
        "shadow.earliest_final_evaluation",
        earliest_final,
        approved.earliest_final_evaluation,
    )
    _require_equal(
        "shadow.maturity_lag_bars",
        shadow.maturity_lag_bars,
        approved.maturity_lag_bars,
    )
    _require_equal("shadow.code_lock", shadow.code_lock, approved.code_lock)
    _require_equal("shadow.artifact_root", shadow.artifact_root, approved.artifact_root)
    _require_equal("shadow.config_hash", shadow.config_hash, approved.config_hash)
    _require_equal("shadow.behavior_hash", shadow.behavior_hash, approved.behavior_hash)

    if config.paper.enabled:
        raise ShadowContractError("paper.enabled must remain false for the shadow candidate.")
    _require_equal("data.symbols", config.data.symbols, ["BTC-USD"])
    _require_equal("data.interval", config.data.interval, "1d")
    _require_equal("data.end_date", config.data.end_date, shadow.protocol_end)
    _require_equal(
        "target.horizon_days",
        config.target.horizon_days,
        shadow.maturity_lag_bars,
    )
    _require_equal(
        "artifacts.output_dir",
        config.artifacts.output_dir,
        shadow.artifact_root,
    )

    try:
        actual_config_hash = _config_hash(raw_payload)
        actual_behavior_hash = _behavior_hash(
            config,
            behavior_version=shadow.behavior_version,
        )
    except (TypeError, ValueError) as exc:
        raise ShadowContractError(
            "Shadow config values must support deterministic JSON hashing."
        ) from exc
    _require_equal("config hash", actual_config_hash, approved.config_hash)
    _require_equal("behavior hash", actual_behavior_hash, approved.behavior_hash)

    return VerifiedShadowContract(
        config=config,
        config_path=config_path,
        candidate_id=shadow.candidate_id,
        behavior_version=shadow.behavior_version,
        protocol_start=protocol_start,
        protocol_end=protocol_end,
        earliest_final_evaluation=earliest_final,
        maturity_lag_bars=shadow.maturity_lag_bars,
        code_lock=shadow.code_lock,
        artifact_root=config.resolve_path(shadow.artifact_root),
        config_hash=actual_config_hash,
        behavior_hash=actual_behavior_hash,
    )
