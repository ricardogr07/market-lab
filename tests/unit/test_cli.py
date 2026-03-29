from __future__ import annotations

import shutil
from contextlib import contextmanager
from pathlib import Path

import pytest

from marketlab import cli
from marketlab.resources.templates import get_config_template_text


@contextmanager
def repo_scratch_dir(name: str):
    root = Path.cwd() / ".pytest_cli_tests" / name
    shutil.rmtree(root.parent, ignore_errors=True)
    root.mkdir(parents=True, exist_ok=True)
    try:
        yield root
    finally:
        shutil.rmtree(root.parent, ignore_errors=True)


@pytest.mark.parametrize("version_value", ["1.2.3", "0.0.0+local"])
def test_version_flag_reports_the_installed_version(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    version_value: str,
) -> None:
    monkeypatch.setattr(cli, "get_version", lambda: version_value)

    with pytest.raises(SystemExit) as excinfo:
        cli.main(["--version"])

    captured = capsys.readouterr()
    assert excinfo.value.code == 0
    assert captured.out.strip() == f"marketlab {version_value}"


def test_list_configs_prints_packaged_templates(capsys: pytest.CaptureFixture[str]) -> None:
    exit_code = cli.main(["list-configs"])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert captured.out.splitlines() == ["weekly_rank", "weekly_rank_smoke"]


def test_write_config_copies_the_selected_template(capsys: pytest.CaptureFixture[str]) -> None:
    with repo_scratch_dir("write_config") as root:
        output_path = root / "nested" / "weekly_rank.yaml"

        exit_code = cli.main(
            [
                "write-config",
                "--name",
                "weekly_rank",
                "--output",
                str(output_path),
            ]
        )

        captured = capsys.readouterr()
        expected_path = output_path.resolve()
        assert exit_code == 0
        assert expected_path.exists()
        assert expected_path.read_text(encoding="utf-8") == get_config_template_text(
            "weekly_rank"
        )
        assert captured.out.strip() == str(expected_path)


def test_write_config_refuses_to_overwrite_without_force() -> None:
    with repo_scratch_dir("write_config_overwrite") as root:
        output_path = root / "weekly_rank_smoke.yaml"
        output_path.write_text("existing", encoding="utf-8")

        with pytest.raises(SystemExit) as excinfo:
            cli.main(
                [
                    "write-config",
                    "--name",
                    "weekly_rank_smoke",
                    "--output",
                    str(output_path),
                ]
            )

        assert excinfo.value.code == 2
        assert output_path.read_text(encoding="utf-8") == "existing"


def test_write_config_with_force_overwrites_existing_file() -> None:
    with repo_scratch_dir("write_config_force") as root:
        output_path = root / "weekly_rank_smoke.yaml"
        output_path.write_text("existing", encoding="utf-8")

        exit_code = cli.main(
            [
                "write-config",
                "--name",
                "weekly_rank_smoke",
                "--output",
                str(output_path),
                "--force",
            ]
        )

        assert exit_code == 0
        assert output_path.read_text(encoding="utf-8") == get_config_template_text(
            "weekly_rank_smoke"
        )


def test_write_config_rejects_unknown_template_name() -> None:
    with repo_scratch_dir("write_config_unknown") as root:
        output_path = root / "unknown.yaml"

        with pytest.raises(SystemExit) as excinfo:
            cli.main(
                [
                    "write-config",
                    "--name",
                    "weekly_rank_unknown",
                    "--output",
                    str(output_path),
                ]
            )

        assert excinfo.value.code == 2
        assert not output_path.exists()
