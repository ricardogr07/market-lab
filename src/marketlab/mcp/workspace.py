from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

from marketlab.config import ExperimentConfig, load_config
from marketlab.resources.templates import (
    get_config_template_text,
    iter_config_template_names,
)


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def _safe_resolve(path: str | Path, root: Path) -> Path:
    candidate = Path(path)
    resolved = (root / candidate).resolve() if not candidate.is_absolute() else candidate.resolve()
    if not _is_relative_to(resolved, root):
        raise ValueError(f"Path {resolved} escapes the allowed root {root}.")
    return resolved


def _load_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if payload is None:
        return {}
    if not isinstance(payload, dict):
        raise ValueError(f"Config file {path} must contain a top-level mapping.")
    return payload


def _merge_patch(base: Any, patch: Any) -> Any:
    if not isinstance(base, dict) or not isinstance(patch, dict):
        return deepcopy(patch)

    merged = deepcopy(base)
    for key, value in patch.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _merge_patch(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def _config_summary(config: ExperimentConfig) -> dict[str, Any]:
    return {
        "experiment_name": config.experiment_name,
        "symbols": list(config.data.symbols),
        "cache_dir": str(config.cache_dir),
        "prepared_panel_path": str(config.prepared_panel_path),
        "output_dir": str(config.output_dir),
        "benchmark_strategy": config.evaluation.benchmark_strategy,
        "model_names": [model.name for model in config.models],
        "paper_enabled": config.paper.enabled,
        "paper_execution_mode": config.paper.execution_mode,
    }


class WorkspaceSandbox:
    def __init__(
        self,
        workspace_root: Path,
        artifact_root: Path,
        repo_root: Path | None = None,
    ) -> None:
        self.workspace_root = workspace_root.resolve()
        self.artifact_root = artifact_root.resolve()
        self.repo_root = repo_root.resolve() if repo_root is not None else None
        self.system_root = (self.workspace_root / ".marketlab-mcp").resolve()
        self.logs_root = (self.system_root / "logs").resolve()
        self.configs_root = (self.workspace_root / "configs").resolve()
        self._ensure_roots()

    def _ensure_roots(self) -> None:
        for root in (
            self.workspace_root,
            self.artifact_root,
            self.system_root,
            self.logs_root,
            self.configs_root,
        ):
            root.mkdir(parents=True, exist_ok=True)

    def resolve_workspace_path(self, path: str | Path) -> Path:
        return _safe_resolve(path, self.workspace_root)

    def resolve_artifact_path(self, path: str | Path) -> Path:
        return _safe_resolve(path, self.artifact_root)

    def resolve_repo_path(self, path: str | Path) -> Path:
        if self.repo_root is None:
            raise ValueError("Repo-root access is not enabled on this server.")
        return _safe_resolve(path, self.repo_root)

    def is_writable_path(self, path: Path) -> bool:
        resolved = path.resolve()
        return _is_relative_to(resolved, self.workspace_root) or _is_relative_to(
            resolved, self.artifact_root
        )

    def is_readable_path(self, path: Path) -> bool:
        resolved = path.resolve()
        return self.is_writable_path(resolved) or (
            self.repo_root is not None and _is_relative_to(resolved, self.repo_root)
        )

    def describe(self) -> dict[str, Any]:
        return {
            "workspace_root": str(self.workspace_root),
            "artifact_root": str(self.artifact_root),
            "repo_root": str(self.repo_root) if self.repo_root is not None else None,
            "logs_root": str(self.logs_root),
            "configs_root": str(self.configs_root),
        }

    def list_templates(self) -> list[str]:
        return list(iter_config_template_names())

    def create_config_from_template(
        self,
        *,
        template_name: str,
        destination: str | Path,
        force: bool = False,
    ) -> Path:
        destination_path = self.resolve_workspace_path(destination)
        if destination_path.exists() and not force:
            raise FileExistsError(f"Refusing to overwrite existing file: {destination_path}")

        payload = yaml.safe_load(get_config_template_text(template_name)) or {}
        payload.setdefault("artifacts", {})
        payload["artifacts"]["output_dir"] = str(self.artifact_root)
        destination_path.parent.mkdir(parents=True, exist_ok=True)
        destination_path.write_text(
            yaml.safe_dump(payload, sort_keys=False),
            encoding="utf-8",
        )
        return destination_path

    def copy_repo_config(
        self,
        *,
        repo_config_path: str | Path,
        destination: str | Path,
        force: bool = False,
    ) -> Path:
        source = self.resolve_repo_path(repo_config_path)
        destination_path = self.resolve_workspace_path(destination)
        if destination_path.exists() and not force:
            raise FileExistsError(f"Refusing to overwrite existing file: {destination_path}")
        payload = _load_yaml(source)
        payload.setdefault("artifacts", {})
        payload["artifacts"]["output_dir"] = str(self.artifact_root)
        destination_path.parent.mkdir(parents=True, exist_ok=True)
        destination_path.write_text(
            yaml.safe_dump(payload, sort_keys=False),
            encoding="utf-8",
        )
        return destination_path

    def read_config(self, config_path: str | Path) -> dict[str, Any]:
        path = self.resolve_workspace_path(config_path)
        return _load_yaml(path)

    def patch_config(
        self,
        *,
        config_path: str | Path,
        patch: dict[str, Any],
    ) -> Path:
        path = self.resolve_workspace_path(config_path)
        payload = _load_yaml(path)
        merged = _merge_patch(payload, patch)
        path.write_text(yaml.safe_dump(merged, sort_keys=False), encoding="utf-8")
        return path

    def validate_config(self, config_path: str | Path) -> dict[str, Any]:
        path = self.resolve_workspace_path(config_path)
        config = load_config(path)
        self.validate_execution_paths(config)
        return {
            "config_path": str(path),
            "summary": _config_summary(config),
        }

    def validate_execution_paths(self, config: ExperimentConfig) -> None:
        writable_paths = [config.cache_dir, config.output_dir]
        if config.paper.enabled:
            writable_paths.extend(
                [
                    config.paper_approval_inbox_dir,
                    config.paper_state_dir,
                ]
            )
        readable_paths = [
            config.prepared_panel_path,
            config.factor_model_path,
            config.optimized_external_covariance_path,
            config.optimized_external_expected_returns_path,
        ]
        for path in writable_paths:
            if not self.is_writable_path(path):
                raise ValueError(
                    f"Writable path {path} is outside the allowed workspace or artifact roots."
                )
        for path in readable_paths:
            if path is None:
                continue
            if not self.is_readable_path(path):
                raise ValueError(
                    f"Readable path {path} is outside the allowed workspace or repo roots."
                )
