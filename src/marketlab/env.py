from __future__ import annotations

import os
from pathlib import Path


def _strip_env_value(raw_value: str) -> str:
    value = raw_value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
        return value[1:-1]
    comment_index = value.find(" #")
    if comment_index >= 0:
        return value[:comment_index].rstrip()
    return value


def load_env_file(*, override: bool = False) -> Path | None:
    candidates: list[Path] = []
    configured = os.environ.get("MARKETLAB_ENV_FILE", "").strip()
    if configured:
        candidates.append(Path(configured).expanduser())
    candidates.append(Path.cwd() / ".env")

    seen: set[Path] = set()
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        if not resolved.exists() or not resolved.is_file():
            continue

        for raw_line in resolved.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("export "):
                line = line[len("export ") :].strip()
            if "=" not in line:
                continue
            key, raw_value = line.split("=", 1)
            env_key = key.strip()
            if env_key == "":
                continue
            env_value = _strip_env_value(raw_value)
            if override or env_key not in os.environ:
                os.environ[env_key] = env_value
        return resolved

    return None
