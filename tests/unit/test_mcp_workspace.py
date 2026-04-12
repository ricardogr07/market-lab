from __future__ import annotations

from pathlib import Path

import pytest

from marketlab.mcp.workspace import WorkspaceSandbox


def test_create_config_from_template_normalizes_artifact_root(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    artifacts = tmp_path / "artifacts"
    sandbox = WorkspaceSandbox(workspace_root=workspace, artifact_root=artifacts)

    created_path = sandbox.create_config_from_template(
        template_name="weekly_rank_smoke",
        destination="configs/generated.yaml",
    )

    document = sandbox.read_config(created_path.relative_to(workspace))
    assert created_path.exists()
    assert document["artifacts"]["output_dir"] == str(artifacts.resolve())


def test_copy_repo_config_rewrites_output_dir_into_artifact_root(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir(parents=True)
    source_path = repo_root / "configs" / "repo_config.yaml"
    source_path.parent.mkdir(parents=True)
    source_path.write_text(
        "experiment_name: copied\nartifacts:\n  output_dir: artifacts/runs\n",
        encoding="utf-8",
    )

    sandbox = WorkspaceSandbox(
        workspace_root=tmp_path / "workspace",
        artifact_root=tmp_path / "artifacts",
        repo_root=repo_root,
    )

    copied_path = sandbox.copy_repo_config(
        repo_config_path="configs/repo_config.yaml",
        destination="configs/copied.yaml",
    )

    document = sandbox.read_config(copied_path.relative_to(sandbox.workspace_root))
    assert document["artifacts"]["output_dir"] == str(sandbox.artifact_root)


def test_patch_config_and_validate_config_use_existing_loader(tmp_path: Path) -> None:
    sandbox = WorkspaceSandbox(
        workspace_root=tmp_path / "workspace",
        artifact_root=tmp_path / "artifacts",
    )
    config_path = sandbox.create_config_from_template(
        template_name="weekly_rank_smoke",
        destination="configs/smoke.yaml",
    )

    sandbox.patch_config(
        config_path=config_path.relative_to(sandbox.workspace_root),
        patch={
            "data": {
                "cache_dir": str(sandbox.workspace_root / "cache"),
                "start_date": "2024-01-02",
                "end_date": "2024-03-29",
            },
            "artifacts": {"output_dir": str(sandbox.artifact_root)},
        },
    )

    validation = sandbox.validate_config(config_path.relative_to(sandbox.workspace_root))

    assert validation["summary"]["output_dir"] == str(sandbox.artifact_root)
    assert validation["summary"]["cache_dir"] == str((sandbox.workspace_root / "cache").resolve())


def test_validate_config_rejects_writes_outside_sandbox(tmp_path: Path) -> None:
    sandbox = WorkspaceSandbox(
        workspace_root=tmp_path / "workspace",
        artifact_root=tmp_path / "artifacts",
    )
    config_path = sandbox.create_config_from_template(
        template_name="weekly_rank_smoke",
        destination="configs/unsafe.yaml",
    )

    sandbox.patch_config(
        config_path=config_path.relative_to(sandbox.workspace_root),
        patch={"artifacts": {"output_dir": str(tmp_path.parent / "outside-runs")}},
    )

    with pytest.raises(ValueError, match="outside the allowed workspace or artifact roots"):
        sandbox.validate_config(config_path.relative_to(sandbox.workspace_root))
