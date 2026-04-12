from __future__ import annotations

import base64
from pathlib import Path
from typing import Any

import pandas as pd

from marketlab.mcp.workspace import WorkspaceSandbox


def register_artifact_tools(
    mcp: Any,
    *,
    sandbox: WorkspaceSandbox,
) -> None:
    @mcp.tool(
        name="marketlab_list_runs",
        description="List completed MarketLab run directories under the configured artifact root.",
        structured_output=True,
    )
    def marketlab_list_runs(limit: int = 20) -> dict[str, Any]:
        runs: list[dict[str, Any]] = []
        for metrics_path in sorted(
            sandbox.artifact_root.rglob("metrics.csv"),
            key=lambda path: path.stat().st_mtime_ns,
            reverse=True,
        ):
            run_dir = metrics_path.parent
            report_path = run_dir / "report.md"
            runs.append(
                {
                    "run_path": str(run_dir),
                    "experiment_name": run_dir.parent.name
                    if run_dir.parent != sandbox.artifact_root
                    else run_dir.name,
                    "timestamp_dir": run_dir.name,
                    "has_report": report_path.exists(),
                    "available_artifacts": sorted(path.name for path in run_dir.iterdir()),
                }
            )
            if len(runs) >= max(limit, 1):
                break
        return {"runs": runs}

    @mcp.tool(
        name="marketlab_get_run_summary",
        description="Read compact run-level summaries from a completed MarketLab run directory.",
        structured_output=True,
    )
    def marketlab_get_run_summary(run_path: str) -> dict[str, Any]:
        resolved_run = sandbox.resolve_artifact_path(run_path)
        summary: dict[str, Any] = {
            "run_path": str(resolved_run),
            "available_artifacts": sorted(path.name for path in resolved_run.iterdir()),
        }
        metrics_path = resolved_run / "metrics.csv"
        strategy_summary_path = resolved_run / "strategy_summary.csv"
        model_summary_path = resolved_run / "model_summary.csv"
        report_path = resolved_run / "report.md"
        if metrics_path.exists():
            summary["metrics_preview"] = _frame_preview(metrics_path, max_rows=20)
        if strategy_summary_path.exists():
            summary["strategy_summary_preview"] = _frame_preview(
                strategy_summary_path, max_rows=20
            )
        if model_summary_path.exists():
            summary["model_summary_preview"] = _frame_preview(model_summary_path, max_rows=20)
        if report_path.exists():
            summary["report_sections"] = _extract_markdown_sections(report_path)
        return summary

    @mcp.tool(
        name="marketlab_list_artifacts",
        description="List files inside one completed MarketLab run directory.",
        structured_output=True,
    )
    def marketlab_list_artifacts(run_path: str) -> dict[str, Any]:
        resolved_run = sandbox.resolve_artifact_path(run_path)
        artifacts = []
        for path in sorted(resolved_run.iterdir()):
            if path.is_file():
                artifacts.append(
                    {
                        "path": str(path),
                        "name": path.name,
                        "kind": _artifact_kind(path),
                        "byte_size": path.stat().st_size,
                    }
                )
        return {"run_path": str(resolved_run), "artifacts": artifacts}

    @mcp.tool(
        name="marketlab_read_table_artifact",
        description="Read a CSV artifact inside the artifact root and return a structured preview.",
        structured_output=True,
    )
    def marketlab_read_table_artifact(path: str, max_rows: int = 50) -> dict[str, Any]:
        resolved = sandbox.resolve_artifact_path(path)
        if resolved.suffix != ".csv":
            raise ValueError(f"{resolved} is not a CSV artifact.")
        return _frame_preview(resolved, max_rows=max_rows)

    @mcp.tool(
        name="marketlab_read_text_artifact",
        description="Read a Markdown, log, YAML, or text artifact under the workspace or artifact roots.",
        structured_output=True,
    )
    def marketlab_read_text_artifact(path: str, max_chars: int = 12000) -> dict[str, Any]:
        resolved = _resolve_readable_path(sandbox, path)
        if _artifact_kind(resolved) not in {"text", "file"}:
            raise ValueError(f"{resolved} is not a supported text artifact.")
        content = resolved.read_text(encoding="utf-8")
        return {
            "path": str(resolved),
            "char_count": len(content),
            "content": content[: max(max_chars, 1)],
        }

    @mcp.tool(
        name="marketlab_get_plot_artifact",
        description="Read an image artifact and return it as a base64-encoded payload for review.",
        structured_output=True,
    )
    def marketlab_get_plot_artifact(path: str) -> dict[str, Any]:
        resolved = sandbox.resolve_artifact_path(path)
        if _artifact_kind(resolved) != "image":
            raise ValueError(f"{resolved} is not a supported plot artifact.")
        encoded = base64.b64encode(resolved.read_bytes()).decode("ascii")
        mime_type = (
            "image/png"
            if resolved.suffix.lower() == ".png"
            else f"image/{resolved.suffix.lstrip('.')}"
        )
        return {
            "path": str(resolved),
            "mime_type": mime_type,
            "byte_size": resolved.stat().st_size,
            "base64": encoded,
        }

    @mcp.tool(
        name="marketlab_compare_runs",
        description="Compare summary-table artifacts across two completed MarketLab runs.",
        structured_output=True,
    )
    def marketlab_compare_runs(left_run_path: str, right_run_path: str) -> dict[str, Any]:
        left_run = sandbox.resolve_artifact_path(left_run_path)
        right_run = sandbox.resolve_artifact_path(right_run_path)
        comparison: dict[str, Any] = {
            "left_run_path": str(left_run),
            "right_run_path": str(right_run),
            "tables": {},
            "report_sections": {
                "left": _extract_markdown_sections(left_run / "report.md"),
                "right": _extract_markdown_sections(right_run / "report.md"),
            },
        }
        for file_name in ("metrics.csv", "strategy_summary.csv", "model_summary.csv"):
            left_path = left_run / file_name
            right_path = right_run / file_name
            if left_path.exists() and right_path.exists():
                comparison["tables"][file_name] = _merge_summary_frame(left_path, right_path)
        return comparison


def _extract_markdown_sections(path: Path) -> list[str]:
    if not path.exists():
        return []
    lines = path.read_text(encoding="utf-8").splitlines()
    return [line.strip("# ").strip() for line in lines if line.startswith("## ")]


def _frame_preview(path: Path, max_rows: int) -> dict[str, Any]:
    frame = pd.read_csv(path)
    preview = frame.head(max_rows)
    return {
        "path": str(path),
        "row_count": int(len(frame)),
        "columns": list(frame.columns),
        "rows": preview.to_dict(orient="records"),
    }


def _artifact_kind(path: Path) -> str:
    if path.suffix == ".csv":
        return "table"
    if path.suffix in {".md", ".log", ".txt", ".yaml", ".yml"}:
        return "text"
    if path.suffix.lower() in {".png", ".jpg", ".jpeg", ".gif"}:
        return "image"
    return "file"


def _merge_summary_frame(left: Path, right: Path) -> dict[str, Any]:
    left_frame = pd.read_csv(left)
    right_frame = pd.read_csv(right)
    key_candidates = ["strategy", "model_name", "fold_id"]
    key = next(
        (
            candidate
            for candidate in key_candidates
            if candidate in left_frame.columns and candidate in right_frame.columns
        ),
        None,
    )
    if key is None:
        return {
            "left_columns": list(left_frame.columns),
            "right_columns": list(right_frame.columns),
            "left_rows": left_frame.to_dict(orient="records"),
            "right_rows": right_frame.to_dict(orient="records"),
        }

    merged = left_frame.merge(
        right_frame,
        on=key,
        how="outer",
        suffixes=("_left", "_right"),
    )
    return {
        "key": key,
        "row_count": int(len(merged)),
        "columns": list(merged.columns),
        "rows": merged.to_dict(orient="records"),
    }


def _resolve_readable_path(sandbox: WorkspaceSandbox, path: str | Path) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        resolved = candidate.resolve()
        if not sandbox.is_readable_path(resolved):
            raise ValueError(f"Path {resolved} is outside the readable roots.")
        return resolved

    for resolver in (sandbox.resolve_artifact_path, sandbox.resolve_workspace_path):
        try:
            resolved = resolver(candidate)
            if resolved.exists():
                return resolved
        except ValueError:
            continue
    raise ValueError(
        f"Unable to resolve readable path {path!r} in the workspace or artifact roots."
    )
