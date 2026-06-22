from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, Protocol, runtime_checkable
from urllib.parse import quote

from marketlab.config import ExperimentConfig
from marketlab.paper.contracts import PaperArtifactStore
from marketlab.paper.state import PaperStateStore


@runtime_checkable
class AzureBlobContainerClient(Protocol):
    """Subset of the Azure Blob container client used by the artifact adapter."""

    def upload_blob(
        self,
        name: str,
        data: bytes,
        *,
        overwrite: bool,
    ) -> Any: ...


@dataclass(slots=True, frozen=True)
class PaperBlobArtifactWrite:
    source_path: Path
    blob_uri: str


def _path_segment(value: str) -> str:
    normalized = value.strip()
    if normalized == "":
        raise ValueError("Azure Blob artifact path segments must not be empty.")
    return quote(normalized, safe="-_.")


def _trade_date_segment(trade_date: str) -> str:
    try:
        return date.fromisoformat(trade_date).isoformat()
    except ValueError as exc:
        raise ValueError("Paper artifact trade_date must be an ISO-8601 date.") from exc


def _build_azure_blob_container_client(
    *,
    account_url: str,
    container_name: str,
) -> AzureBlobContainerClient:
    try:
        from azure.identity import DefaultAzureCredential
        from azure.storage.blob import BlobServiceClient
    except ImportError as exc:  # pragma: no cover - depends on the optional Azure extra.
        raise RuntimeError(
            "Azure Blob artifact storage requires the 'azure' optional dependency. "
            "Install MarketLab with `.[azure]`."
        ) from exc

    credential = DefaultAzureCredential(exclude_interactive_browser_credential=True)
    service_client = BlobServiceClient(account_url=account_url, credential=credential)
    return service_client.get_container_client(container_name)


class AzureBlobPaperArtifactStore(PaperArtifactStore):
    """Store paper review artifacts in an environment- and deployment-scoped container.

    The adapter intentionally writes only the artifacts produced outside the
    transactional control store. PostgreSQL remains canonical for paper state;
    Blob is the review and debugging surface. Calls happen after broker reads
    and outside the unit-of-work transaction.
    """

    def __init__(
        self,
        config: ExperimentConfig,
        *,
        container_client: AzureBlobContainerClient | None = None,
    ) -> None:
        azure = config.paper.azure
        if azure.artifact_backend != "azure_blob":
            raise ValueError(
                "AzureBlobPaperArtifactStore requires "
                "paper.azure.artifact_backend='azure_blob'."
            )
        self._config = config
        self._account_url = azure.blob_account_url.rstrip("/")
        self._container_name = azure.blob_container_name
        self._environment = _path_segment(azure.artifact_environment)
        self._deployment_id = _path_segment(azure.artifact_deployment_id)
        self._container_client = container_client or _build_azure_blob_container_client(
            account_url=self._account_url,
            container_name=self._container_name,
        )

    def _blob_name(self, artifact_path: str) -> str:
        relative = Path(artifact_path)
        if relative.is_absolute() or ".." in relative.parts or len(relative.parts) == 0:
            raise ValueError("Paper Blob artifact_path must be a non-empty relative path.")
        return "/".join(
            (
                "paper",
                self._environment,
                self._deployment_id,
                *(_path_segment(part) for part in relative.parts),
            )
        )

    def _artifact_uri(self, blob_name: str) -> str:
        return "/".join(
            (
                self._account_url,
                _path_segment(self._container_name),
                blob_name,
            )
        )

    def write_artifact_bytes(
        self,
        *,
        artifact_path: str,
        data: bytes,
    ) -> str:
        blob_name = self._blob_name(artifact_path)
        self._container_client.upload_blob(
            blob_name,
            data,
            overwrite=True,
        )
        return self._artifact_uri(blob_name)

    def _write_json(
        self,
        *,
        trade_date: str,
        filename: str,
        payload: dict[str, Any],
    ) -> str:
        serialized = json.dumps(payload, indent=2, sort_keys=True).replace(
            "\n",
            os.linesep,
        ).encode("utf-8")
        return self.write_artifact_bytes(
            artifact_path=f"trades/{_trade_date_segment(trade_date)}/{filename}",
            data=serialized,
        )

    def sync_review_artifacts(self) -> tuple[PaperBlobArtifactWrite, ...]:
        """Copy the local paper review surface without a database transaction.

        The local filesystem remains the source for this bridge. Each upload
        overwrites the same versioned Blob key, preserving the file bytes and
        existing paper-artifact meanings exactly.
        """

        store = PaperStateStore(self._config)
        artifacts = _review_artifact_paths(store)
        writes = [
            PaperBlobArtifactWrite(
                source_path=source_path,
                blob_uri=self.write_artifact_bytes(
                    artifact_path=artifact_path,
                    data=source_path.read_bytes(),
                ),
            )
            for artifact_path, source_path in artifacts
        ]
        return tuple(writes)

    def write_trade_account_snapshot(
        self,
        *,
        trade_date: str,
        payload: dict[str, Any],
    ) -> Path | str:
        return self._write_json(
            trade_date=trade_date,
            filename="account_snapshot.json",
            payload=payload,
        )

    def write_trade_order_preview(
        self,
        *,
        trade_date: str,
        payload: dict[str, Any],
    ) -> Path | str:
        return self._write_json(
            trade_date=trade_date,
            filename="order_preview.json",
            payload=payload,
        )


def build_azure_blob_paper_artifact_store(
    config: ExperimentConfig,
) -> PaperArtifactStore:
    return AzureBlobPaperArtifactStore(config)


def sync_paper_review_artifacts(config: ExperimentConfig) -> tuple[PaperBlobArtifactWrite, ...]:
    """Synchronize all canonical local review artifacts to the configured Blob container."""

    return AzureBlobPaperArtifactStore(config).sync_review_artifacts()


def _review_artifact_paths(store: PaperStateStore) -> list[tuple[str, Path]]:
    roots = (
        ("trades", store.trades_root),
        ("notifications", store.notifications_root),
        ("reports", store.reports_root),
    )
    artifacts: list[tuple[str, Path]] = []
    for prefix, root in roots:
        for source_path in sorted(path for path in root.rglob("*") if path.is_file()):
            relative = source_path.relative_to(root).as_posix()
            artifacts.append((f"{prefix}/{relative}", source_path))
    for source_path in sorted(store.state_root.glob("*.json")):
        artifacts.append((f"state/{source_path.name}", source_path))
    return artifacts
