from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from marketlab.config import load_config


def _write_config(
    path: Path,
    *,
    data: dict[str, object] | None = None,
    baselines: dict[str, object] | None = None,
) -> Path:
    payload = {
        "experiment_name": "config_fixture",
        "data": {
            "symbols": ["AAA", "BBB"],
            "start_date": "2024-01-01",
            "end_date": "2024-03-31",
            "cache_dir": "artifacts/test-cache",
            "prepared_panel_filename": "panel.csv",
        },
    }
    if data is not None:
        payload["data"].update(data)
    if baselines is not None:
        payload["baselines"] = baselines

    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return path


def test_load_config_preserves_backward_compatible_allocation_defaults(tmp_path: Path) -> None:
    config_path = _write_config(tmp_path / "config.yaml")

    config = load_config(config_path)

    assert config.data.symbol_groups == {}
    assert config.baselines.allocation.enabled is False
    assert config.baselines.allocation.mode == "equal"
    assert config.baselines.allocation.symbol_weights == {}
    assert config.baselines.allocation.group_weights == {}


def test_load_config_normalizes_nullable_mapping_sections(tmp_path: Path) -> None:
    config_path = _write_config(
        tmp_path / "config.yaml",
        data={"symbol_groups": None},
        baselines={
            "allocation": {
                "enabled": False,
                "symbol_weights": None,
                "group_weights": None,
            }
        },
    )

    config = load_config(config_path)

    assert config.data.symbol_groups == {}
    assert config.baselines.allocation.symbol_weights == {}
    assert config.baselines.allocation.group_weights == {}


def test_load_config_rejects_unknown_symbol_group_entries(tmp_path: Path) -> None:
    config_path = _write_config(
        tmp_path / "config.yaml",
        data={"symbol_groups": {"CCC": "growth"}},
    )

    with pytest.raises(ValueError, match="data.symbol_groups contains unknown symbols: CCC"):
        load_config(config_path)


def test_load_config_rejects_symbol_weights_that_do_not_match_symbols(tmp_path: Path) -> None:
    config_path = _write_config(
        tmp_path / "config.yaml",
        baselines={
            "allocation": {
                "enabled": True,
                "mode": "symbol_weights",
                "symbol_weights": {"AAA": 1.0},
            }
        },
    )

    with pytest.raises(
        ValueError,
        match="baselines.allocation.symbol_weights must match data.symbols exactly",
    ):
        load_config(config_path)


def test_load_config_accepts_valid_group_weight_allocations(tmp_path: Path) -> None:
    config_path = _write_config(
        tmp_path / "config.yaml",
        data={
            "symbols": ["AAA", "BBB", "CCC", "DDD"],
            "symbol_groups": {
                "AAA": "growth",
                "BBB": "growth",
                "CCC": "defensive",
                "DDD": "defensive",
            },
        },
        baselines={
            "allocation": {
                "enabled": True,
                "mode": "group_weights",
                "group_weights": {"growth": 0.75, "defensive": 0.25},
            }
        },
    )

    config = load_config(config_path)

    assert config.baselines.allocation.enabled is True
    assert config.baselines.allocation.mode == "group_weights"
    assert config.baselines.allocation.group_weights == {
        "growth": 0.75,
        "defensive": 0.25,
    }
