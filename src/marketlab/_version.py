from __future__ import annotations

from importlib import metadata

FALLBACK_VERSION = "0.0.0+local"
DIST_NAME = "marketlab"


def get_version(dist_name: str = DIST_NAME, fallback: str = FALLBACK_VERSION) -> str:
    try:
        return metadata.version(dist_name)
    except metadata.PackageNotFoundError:
        return fallback


__version__ = get_version()