from __future__ import annotations

import json
import shutil
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest
from tests._paper_fakes import write_phase7_paper_config

from marketlab import cli
from marketlab.paper.contracts import (
    PaperDeploymentRegistryConflictError,
    PaperHostedExecutionContext,
)
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


def test_paper_db_migrate_runs_only_the_explicit_migration_command(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config = object()
    received: list[object] = []
    monkeypatch.setattr(cli, "load_config", lambda path: config)
    monkeypatch.setattr(
        cli,
        "migrate_paper_postgres_database",
        lambda received_config: received.append(received_config) or 2,
    )

    exit_code = cli.main(["paper-db-migrate", "--config", "postgres.yaml"])

    assert exit_code == 0
    assert received == [config]
    assert capsys.readouterr().out.strip() == "Applied paper database schema version: 2"


def test_paper_db_migrate_rejects_a_non_postgres_config(
) -> None:
    with repo_scratch_dir("postgres_migrate_filesystem_config") as root:
        config_path = write_phase7_paper_config(root / "config.yaml")

        with pytest.raises(SystemExit) as excinfo:
            cli.main(["paper-db-migrate", "--config", str(config_path)])

    assert excinfo.value.code == 2


def test_paper_state_import_defaults_to_dry_run_and_forwards_paths(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config = object()
    received: dict[str, object] = {}
    monkeypatch.setattr(cli, "load_config", lambda path: config)

    def _import(received_config, **kwargs):
        received["config"] = received_config
        received.update(kwargs)
        return {
            "mode": "dry-run",
            "counts": {"proposals": {"imported": 1}},
            "aggregate_checksum": "a" * 64,
        }

    monkeypatch.setattr(cli, "import_paper_state", _import)

    exit_code = cli.main(
        [
            "paper-state-import",
            "--config",
            "postgres.yaml",
            "--source-state-dir",
            "artifacts/paper/state",
            "--source-inbox-dir",
            "artifacts/paper/inbox",
            "--report-path",
            "artifacts/paper/state/imports/dry-run.json",
        ]
    )

    assert exit_code == 0
    assert received == {
        "config": config,
        "source_state_dir": "artifacts/paper/state",
        "source_inbox_dir": "artifacts/paper/inbox",
        "apply": False,
        "report_path": "artifacts/paper/state/imports/dry-run.json",
    }
    assert json.loads(capsys.readouterr().out)["mode"] == "dry-run"


def test_paper_state_import_apply_flag_enables_mutation(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(cli, "load_config", lambda path: object())
    received: dict[str, object] = {}

    def _import(received_config, **kwargs):
        del received_config
        received.update(kwargs)
        return {"mode": "apply", "counts": {}, "aggregate_checksum": "b" * 64}

    monkeypatch.setattr(cli, "import_paper_state", _import)

    exit_code = cli.main(
        [
            "paper-state-import",
            "--config",
            "postgres.yaml",
            "--source-state-dir",
            "state",
            "--source-inbox-dir",
            "inbox",
            "--apply",
        ]
    )

    assert exit_code == 0
    assert received["apply"] is True
    assert json.loads(capsys.readouterr().out)["mode"] == "apply"


def test_paper_state_import_rejects_a_non_postgres_config() -> None:
    with repo_scratch_dir("state_import_filesystem_config") as root:
        config_path = write_phase7_paper_config(root / "config.yaml")
        state_dir = root / "artifacts" / "paper" / "state"
        inbox_dir = root / "artifacts" / "paper" / "inbox"
        state_dir.mkdir(parents=True)
        inbox_dir.mkdir(parents=True)

        with pytest.raises(SystemExit) as excinfo:
            cli.main(
                [
                    "paper-state-import",
                    "--config",
                    str(config_path),
                    "--source-state-dir",
                    str(state_dir),
                    "--source-inbox-dir",
                    str(inbox_dir),
                ]
            )

    assert excinfo.value.code == 2


def test_paper_outbox_deliver_runs_only_approval_request_records(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config = object()
    publisher = object()
    received: dict[str, object] = {}
    monkeypatch.setattr(cli, "load_config", lambda path: config)
    monkeypatch.setattr(cli, "build_paper_outbox_publisher", lambda received_config: publisher)
    monkeypatch.setattr(cli, "build_paper_uow_factory", lambda received_config: "uow-factory")

    def _deliver(**kwargs):
        received.update(kwargs)
        return SimpleNamespace(delivered_message_ids=("approval-1",), failed_message_ids=())

    monkeypatch.setattr(cli, "deliver_pending_paper_outbox", _deliver)

    exit_code = cli.main(["paper-outbox-deliver", "--config", "dummy.yaml", "--limit", "7"])

    assert exit_code == 0
    assert received == {
        "uow_factory": "uow-factory",
        "publisher": publisher,
        "limit": 7,
        "event_types": frozenset(("paper.approval.requested",)),
    }
    assert json.loads(capsys.readouterr().out) == {
        "delivered_message_ids": ["approval-1"],
        "failed_message_ids": [],
    }


def test_paper_notifications_deliver_reports_failed_messages_with_nonzero_exit(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config = object()
    sink = object()
    received: dict[str, object] = {}
    monkeypatch.setattr(cli, "load_config", lambda path: config)
    monkeypatch.setattr(cli, "build_paper_uow_factory", lambda received_config: "uow-factory")
    monkeypatch.setattr(
        cli,
        "build_telegram_paper_notification_sink",
        lambda received_config: sink,
    )

    def _deliver(**kwargs):
        received.update(kwargs)
        return SimpleNamespace(delivered_message_ids=(), failed_message_ids=("notification-1",))

    monkeypatch.setattr(cli, "deliver_pending_paper_notifications", _deliver)

    exit_code = cli.main(["paper-notifications-deliver", "--config", "dummy.yaml"])

    assert exit_code == 1
    assert received == {"uow_factory": "uow-factory", "sink": sink, "limit": 100}
    assert json.loads(capsys.readouterr().out) == {
        "delivered_message_ids": [],
        "event_type": "paper.notification.requested",
        "failed_message_ids": ["notification-1"],
    }


def test_paper_blob_sync_and_service_bus_receive_are_bounded_worker_commands(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config = object()
    received: dict[str, object] = {}
    monkeypatch.setattr(cli, "load_config", lambda path: config)
    monkeypatch.setattr(
        cli,
        "sync_paper_review_artifacts",
        lambda received_config: (SimpleNamespace(blob_uri="https://blob/review.json"),),
    )
    monkeypatch.setattr(
        cli,
        "receive_paper_approval_requests",
        lambda received_config, **kwargs: received.update(kwargs)
        or SimpleNamespace(
            completed_message_ids=("approval-1",),
            abandoned_message_ids=(),
            failure_messages=(),
        ),
    )

    assert cli.main(["paper-blob-sync", "--config", "dummy.yaml"]) == 0
    assert json.loads(capsys.readouterr().out) == {
        "artifact_count": 1,
        "blob_uris": ["https://blob/review.json"],
    }

    assert (
        cli.main(
            [
                "paper-service-bus-receive",
                "--config",
                "dummy.yaml",
                "--max-messages",
                "4",
                "--max-wait-seconds",
                "2.5",
            ]
        )
        == 0
    )
    assert received == {"max_messages": 4, "max_wait_seconds": 2.5}
    assert json.loads(capsys.readouterr().out) == {
        "abandoned_message_ids": [],
        "completed_message_ids": ["approval-1"],
        "failure_messages": [],
    }


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


def test_paper_decision_hosted_flags_override_environment_defaults(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    captured_contexts: list[PaperHostedExecutionContext | None] = []
    monkeypatch.setattr(cli, "load_config", lambda path: object())
    monkeypatch.setenv("MARKETLAB_DEPLOYMENT_ID", "env-deployment")
    monkeypatch.setenv("MARKETLAB_ENVIRONMENT", "dev")
    monkeypatch.setenv("MARKETLAB_EXECUTION_ID", "env-exec")
    monkeypatch.setenv("MARKETLAB_CORRELATION_ID", "env-corr")
    monkeypatch.setenv("MARKETLAB_IDEMPOTENCY_KEY", "env-idem")
    monkeypatch.setenv("MARKETLAB_TRIGGER_SOURCE", "env")
    monkeypatch.setenv("MARKETLAB_REQUESTED_AT", "2026-06-19T12:00:00+00:00")
    monkeypatch.setenv("MARKETLAB_CONFIG_VERSION", "env-config")
    monkeypatch.setenv("MARKETLAB_IMAGE_DIGEST", "sha256:env")

    def _fake_decision(config, **kwargs):
        captured_contexts.append(kwargs["hosted_context"])
        return {
            "proposal_path": "proposal.json",
            "status_path": "status.json",
            "status": {"status": "proposal_created"},
        }

    monkeypatch.setattr(cli, "run_paper_decision", _fake_decision)

    exit_code = cli.main(
        [
            "paper-decision",
            "--config",
            "dummy.yaml",
            "--deployment-id",
            "flag-deployment",
            "--execution-id",
            "flag-exec",
            "--idempotency-key",
            "flag-idem",
        ]
    )

    records = _stderr_records(capsys.readouterr().err)
    assert exit_code == 0
    assert len(captured_contexts) == 1
    assert captured_contexts[0] is not None
    assert captured_contexts[0].deployment_id == "flag-deployment"
    assert captured_contexts[0].execution_id == "flag-exec"
    assert captured_contexts[0].idempotency_key == "flag-idem"
    assert captured_contexts[0].correlation_id == "env-corr"
    assert captured_contexts[0].phase == "decision"
    assert records[0]["deployment"] == "flag-deployment"
    assert records[0]["execution_id"] == "flag-exec"
    assert records[0]["correlation_id"] == "env-corr"


def test_paper_decision_rejects_partial_hosted_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    load_calls: list[str] = []
    monkeypatch.setenv("MARKETLAB_DEPLOYMENT_ID", "partial-deployment")
    monkeypatch.setattr(cli, "load_config", lambda path: load_calls.append(path))

    with pytest.raises(SystemExit) as excinfo:
        cli.main(["paper-decision", "--config", "dummy.yaml"])

    assert excinfo.value.code == 2
    assert load_calls == []


def test_paper_submit_cli_surfaces_duplicate_hosted_idempotency_conflict(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with repo_scratch_dir("hosted_duplicate_idempotency") as root:
        config_path = write_phase7_paper_config(root / "config.yaml")
        base_args = [
            "paper-submit",
            "--config",
            str(config_path),
            "--deployment-id",
            "qqq-paper-dev",
            "--environment",
            "dev",
            "--execution-id",
            "exec-1",
            "--correlation-id",
            "corr-1",
            "--idempotency-key",
            "idem-1",
            "--trigger-source",
            "cli",
            "--requested-at",
            "2026-06-19T12:00:00+00:00",
            "--config-version",
            "config-v1",
        ]

        assert cli.main([*base_args, "--image-digest", "sha256:first"]) == 0
        capsys.readouterr()

        with pytest.raises(PaperDeploymentRegistryConflictError):
            cli.main([*base_args, "--image-digest", "sha256:second"])
