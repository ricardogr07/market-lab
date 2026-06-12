from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
import yaml

from marketlab.config import load_config
from marketlab.shadow import ShadowContractError, verify_shadow_contract
from marketlab.shadow.contract import _behavior_hash, _config_hash

ROOT = Path(__file__).resolve().parents[2]
PHASE8_CONFIG = (
    ROOT
    / "configs"
    / "experiment.btc_phase8_guarded_gate_bull_risk_off_override_partial_support.yaml"
)
SHADOW_CONFIG = ROOT / "configs" / "experiment.btc_phase9_shadow_daily.yaml"
BEHAVIOR_VERSION = "btc-phase8-guarded-gate-v1"


def _payload() -> dict[str, Any]:
    payload = yaml.safe_load(SHADOW_CONFIG.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


def _write_mutation(
    tmp_path: Path,
    mutate: Callable[[dict[str, Any]], None],
) -> Path:
    payload = _payload()
    mutate(payload)
    config_dir = tmp_path / "configs"
    config_dir.mkdir()
    path = config_dir / SHADOW_CONFIG.name
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return path


def test_verified_shadow_contract_exposes_frozen_protocol() -> None:
    contract = verify_shadow_contract(SHADOW_CONFIG)

    assert contract.candidate_id == "btc-phase9-shadow-v1"
    assert contract.behavior_version == BEHAVIOR_VERSION
    assert contract.protocol_start.isoformat() == "2026-06-03"
    assert contract.protocol_end.isoformat() == "2027-06-02"
    assert contract.earliest_final_evaluation.isoformat() == "2027-06-16"
    assert contract.maturity_lag_bars == 14
    assert contract.code_lock == "ce01124"
    assert contract.config.paper.enabled is False
    assert contract.artifact_root == ROOT / "artifacts" / "phase9-shadow"


def test_shadow_behavior_exactly_matches_retained_phase8_candidate() -> None:
    phase8 = load_config(PHASE8_CONFIG)
    shadow = load_config(SHADOW_CONFIG)

    assert _behavior_hash(
        shadow,
        behavior_version=BEHAVIOR_VERSION,
    ) == _behavior_hash(
        phase8,
        behavior_version=BEHAVIOR_VERSION,
    )


def test_hashes_are_stable_across_yaml_key_order(tmp_path: Path) -> None:
    payload = _payload()
    reordered = dict(reversed(list(payload.items())))

    assert _config_hash(reordered) == _config_hash(payload)


def test_shadow_contract_rejects_noncanonical_config_location(tmp_path: Path) -> None:
    path = tmp_path / SHADOW_CONFIG.name
    path.write_text(SHADOW_CONFIG.read_text(encoding="utf-8"), encoding="utf-8")

    with pytest.raises(ShadowContractError, match="must be loaded from configs"):
        verify_shadow_contract(path)


def test_behavior_hash_ignores_operational_fields_but_config_hash_does_not(
    tmp_path: Path,
) -> None:
    original_payload = _payload()
    changed_payload = _payload()
    changed_payload["data"]["cache_dir"] = "artifacts/other-cache"

    original = load_config(SHADOW_CONFIG)
    changed_path = tmp_path / "operational-change.yaml"
    changed_path.write_text(
        yaml.safe_dump(changed_payload, sort_keys=False),
        encoding="utf-8",
    )
    changed = load_config(changed_path)

    assert _behavior_hash(original, behavior_version=BEHAVIOR_VERSION) == _behavior_hash(
        changed,
        behavior_version=BEHAVIOR_VERSION,
    )
    assert _config_hash(original_payload) != _config_hash(changed_payload)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda payload: payload["features"].update({"momentum_window": 8}), "config hash"),
        (lambda payload: payload["paper"].update({"enabled": True}), "paper.enabled"),
        (
            lambda payload: payload["shadow"].update({"candidate_id": "unknown"}),
            "Unknown shadow candidate_id",
        ),
        (
            lambda payload: payload["shadow"].update({"protocol_start": "06/03/2026"}),
            "YYYY-MM-DD",
        ),
        (
            lambda payload: payload["shadow"].update({"protocol_start": 20260603}),
            "YYYY-MM-DD",
        ),
        (
            lambda payload: payload["shadow"].update({"maturity_lag_bars": 15}),
            "maturity_lag_bars",
        ),
        (
            lambda payload: payload["shadow"].update({"code_lock": "deadbeef"}),
            "code_lock",
        ),
    ],
)
def test_shadow_contract_fails_closed_on_drift(
    tmp_path: Path,
    mutation: Callable[[dict[str, Any]], None],
    message: str,
) -> None:
    path = _write_mutation(tmp_path, mutation)

    with pytest.raises(ShadowContractError, match=message):
        verify_shadow_contract(path)


def test_changed_candidate_cannot_rebless_itself(tmp_path: Path) -> None:
    payload = _payload()
    payload["features"]["momentum_window"] = 8
    config_dir = tmp_path / "configs"
    config_dir.mkdir()
    changed_path = config_dir / SHADOW_CONFIG.name
    changed_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    changed_config = load_config(changed_path)
    payload["shadow"]["config_hash"] = _config_hash(payload)
    payload["shadow"]["behavior_hash"] = _behavior_hash(
        changed_config,
        behavior_version=BEHAVIOR_VERSION,
    )
    changed_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

    with pytest.raises(ShadowContractError, match="shadow.config_hash"):
        verify_shadow_contract(changed_path)
