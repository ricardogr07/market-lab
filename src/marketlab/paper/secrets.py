from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from marketlab.env import load_env_file


@runtime_checkable
class PaperSecretProvider(Protocol):
    def get(self, name: str) -> str | None: ...


@dataclass(slots=True)
class EnvironmentPaperSecretProvider:
    load_env: bool = True

    def __post_init__(self) -> None:
        if self.load_env:
            load_env_file()

    def get(self, name: str) -> str | None:
        value = os.environ.get(name, "").strip()
        if value == "":
            return None
        return value


def resolve_paper_secret(
    name: str,
    *,
    secret_provider: PaperSecretProvider | None = None,
) -> str | None:
    if secret_provider is not None:
        value = secret_provider.get(name)
        if value is not None and str(value).strip() != "":
            return str(value).strip()
    value = os.environ.get(name, "").strip()
    if value == "":
        return None
    return value


def require_paper_secret(
    name: str,
    *,
    secret_provider: PaperSecretProvider | None = None,
) -> str:
    value = resolve_paper_secret(name, secret_provider=secret_provider)
    if value is None:
        raise RuntimeError(f"{name} is required.")
    return value
