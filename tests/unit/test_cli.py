from __future__ import annotations

import json
import shutil
from contextlib import contextmanager
from pathlib import Path

import pytest

from marketlab import cli
from marketlab.resources.templates import get_config_template_text


def _stderr_records(stderr: str) -> list[dict[str, object]]:
    return [
        json.loads(line)
        for line in stderr.splitlines()
        if line.strip() != ""
    ]


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
    assert captured.out.splitlines() == [
        "weekly_rank",
        "weekly_rank_smoke",
    ]


@pytest.mark.parametrize(
    ("template_name", "output_name"),
    [
        ("weekly_rank", "weekly_rank.yaml"),
        ("weekly_rank_smoke", "weekly_rank_smoke.yaml"),
    ],
)
def test_write_config_copies_the_selected_template(
    capsys: pytest.CaptureFixture[str],
    template_name: str,
    output_name: str,
) -> None:
    with repo_scratch_dir("write_config") as root:
        output_path = root / "nested" / output_name

        exit_code = cli.main(
            [
                "write-config",
                "--name",
                template_name,
                "--output",
                str(output_path),
            ]
        )

        captured = capsys.readouterr()
        expected_path = output_path.resolve()
        assert exit_code == 0
        assert expected_path.exists()
        assert expected_path.read_text(encoding="utf-8") == get_config_template_text(
            template_name
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


def test_paper_status_keeps_json_stdout_and_structured_logs_on_stderr(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(cli, "load_config", lambda path: object())
    monkeypatch.setattr(
        cli,
        "get_paper_status",
        lambda config: {
            "latest_proposal": {"proposal_id": "proposal-1"},
            "pending_proposal_count": 0,
            "status": {"status": "proposal_created"},
            "status_path": "status.json",
        },
    )

    exit_code = cli.main(["paper-status", "--config", "dummy.yaml"])

    captured = capsys.readouterr()
    records = _stderr_records(captured.err)

    assert exit_code == 0
    assert json.loads(captured.out)["latest_proposal"]["proposal_id"] == "proposal-1"
    assert [record["event"] for record in records] == [
        "paper.command.start",
        "paper.command.finish",
    ]
    assert all(record["deployment"] == "local_cli" for record in records)
    assert all(record["phase"] == "paper-status" for record in records)


def test_paper_decision_keeps_path_stdout_and_structured_logs_on_stderr(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(cli, "load_config", lambda path: object())
    monkeypatch.setattr(
        cli,
        "run_paper_decision",
        lambda config, **kwargs: {
            "proposal_path": "proposal.json",
            "status_path": "status.json",
            "status": {"status": "proposal_created"},
        },
    )

    exit_code = cli.main(["paper-decision", "--config", "dummy.yaml"])

    captured = capsys.readouterr()
    records = _stderr_records(captured.err)

    assert exit_code == 0
    assert captured.out.strip() == "proposal.json"
    assert [record["event"] for record in records] == [
        "paper.command.start",
        "paper.command.finish",
    ]
    assert records[-1]["outcome"] == "proposal_created"
