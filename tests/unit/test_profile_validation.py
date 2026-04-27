from __future__ import annotations

import configparser
import importlib.util
import sys
from pathlib import Path

SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "profile_validation.py"
TOX_INI_PATH = Path(__file__).resolve().parents[2] / "tox.ini"


def _load_module():
    module_name = "profile_validation"
    spec = importlib.util.spec_from_file_location(module_name, SCRIPT_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def test_resolve_envs_defaults_to_ci_matching_lanes() -> None:
    module = _load_module()

    assert module.resolve_envs(None) == ["lint", "docs", "typecheck", "py312", "package", "integration"]
    assert module.resolve_envs([]) == ["lint", "docs", "typecheck", "py312", "package", "integration"]


def test_resolve_envs_preserves_requested_order() -> None:
    module = _load_module()

    assert module.resolve_envs(["package", "integration"]) == ["package", "integration"]


def test_build_summary_table_formats_results() -> None:
    module = _load_module()
    results = [
        module.LaneResult(env="lint", exit_code=0, elapsed_seconds=1.25),
        module.LaneResult(env="docs", exit_code=3, elapsed_seconds=2.5),
    ]

    summary = module.build_summary_table(results)

    assert "Validation Summary" in summary
    assert "lint" in summary
    assert "docs" in summary
    assert "1.25" in summary
    assert "2.50" in summary
    assert "total" in summary
    assert "3.75" in summary


def test_run_validation_warns_when_not_on_python_312() -> None:
    module = _load_module()
    messages: list[str] = []

    exit_code = module.run_validation(
        ["lint"],
        runner=lambda command: 0,
        out=messages.append,
        version_info=(3, 14),
    )

    assert exit_code == 0
    assert any("current interpreter is Python 3.14" in message for message in messages)
    assert any(message.startswith("[start] lint:") for message in messages)
    assert any(message.startswith("[finish] lint:") for message in messages)
    assert any("Validation Summary" in message for message in messages)


def test_run_validation_returns_first_failure_exit_code_and_stops() -> None:
    module = _load_module()
    calls: list[str] = []
    exit_codes = {"lint": 0, "docs": 7, "py312": 0}

    def fake_runner(command: list[str]) -> int:
        env_name = command[-1]
        calls.append(env_name)
        return exit_codes[env_name]

    exit_code = module.run_validation(
        ["lint", "docs", "py312"],
        runner=fake_runner,
        out=lambda message: None,
        version_info=(3, 12),
    )

    assert exit_code == 7
    assert calls == ["lint", "docs"]


def test_parse_args_collects_repeated_env_flags() -> None:
    module = _load_module()

    args = module.parse_args(["--env", "lint", "--env", "package"])

    assert args.envs == ["lint", "package"]


def test_tox_preflight_tiers_match_expected_commands() -> None:
    parser = configparser.ConfigParser()
    parser.read(TOX_INI_PATH, encoding="utf-8")

    assert "preflight-fast" in parser["tox"]["env_list"]
    assert "preflight-slow" in parser["tox"]["env_list"]
    assert "typecheck" in parser["tox"]["env_list"]

    for env_name in ("lint", "docs", "typecheck", "package", "integration"):
        assert parser[f"testenv:{env_name}"]["basepython"] == "py312"
    assert parser["testenv:typecheck"]["commands"].strip() == (
        "{envpython} -m mypy src/marketlab/config.py src/marketlab/data/market.py "
        "src/marketlab/paper/alpaca.py src/marketlab/paper/notifications.py "
        "src/marketlab/paper/contracts.py src/marketlab/paper/agent.py "
        "src/marketlab/paper/scheduler.py"
    )

    assert parser["testenv:preflight-fast"]["basepython"] == "py312"
    assert parser["testenv:preflight-fast"]["commands"].strip() == "{envpython} -m tox -e lint,docs,typecheck,py312"

    assert parser["testenv:preflight-slow"]["basepython"] == "py312"
    assert parser["testenv:preflight-slow"]["commands"].strip() == "{envpython} -m tox -e package,integration"

    assert parser["testenv:preflight"]["basepython"] == "py312"
    assert parser["testenv:preflight"]["commands"].strip() == "{envpython} -m tox -e lint,docs,typecheck,package,py312,integration"

