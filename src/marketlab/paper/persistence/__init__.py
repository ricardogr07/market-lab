from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING, Any

from marketlab.config import ExperimentConfig
from marketlab.paper.contracts import (
    PaperArtifactStore,
    PaperDeploymentRegistry,
    PaperUnitOfWorkFactory,
)

from .blob import (
    AzureBlobPaperArtifactStore,
    PaperBlobArtifactWrite,
    build_azure_blob_paper_artifact_store,
    sync_paper_review_artifacts,
)
from .filesystem import (
    FilesystemPaperArtifactStore,
    FilesystemPaperDeploymentRegistry,
    FilesystemPaperOutboxRepository,
    FilesystemPaperUnitOfWork,
    FilesystemPaperUnitOfWorkFactory,
    build_filesystem_paper_artifact_store,
    build_filesystem_paper_deployment_registry,
    build_filesystem_paper_uow_factory,
)

if TYPE_CHECKING:
    from .postgres import PostgreSQLMigration
from .sqlite import (
    SQLitePaperDeploymentRegistry,
    SQLitePaperOutboxRepository,
    SQLitePaperUnitOfWork,
    SQLitePaperUnitOfWorkFactory,
    build_sqlite_paper_deployment_registry,
    build_sqlite_paper_uow_factory,
)


def _postgres_module() -> Any:
    from . import postgres

    return postgres


def apply_postgres_migrations(
    *,
    dsn: str | None = None,
    migrations: Iterable[PostgreSQLMigration] | None = None,
) -> int:
    return _postgres_module().apply_postgres_migrations(dsn=dsn, migrations=migrations)


def build_postgres_paper_deployment_registry(
    config: ExperimentConfig,
) -> PaperDeploymentRegistry:
    return _postgres_module().build_postgres_paper_deployment_registry(config)


def build_postgres_paper_uow_factory(config: ExperimentConfig) -> PaperUnitOfWorkFactory:
    return _postgres_module().build_postgres_paper_uow_factory(config)


def build_paper_uow_factory(config: ExperimentConfig) -> PaperUnitOfWorkFactory:
    """Select the configured transactional paper-state adapter."""

    if config.paper.persistence_backend == "filesystem":
        return build_filesystem_paper_uow_factory(config)
    if config.paper.persistence_backend == "sqlite":
        return build_sqlite_paper_uow_factory(config)
    if config.paper.persistence_backend == "postgres":
        return build_postgres_paper_uow_factory(config)
    raise ValueError(f"Unsupported paper persistence backend: {config.paper.persistence_backend}")


def build_paper_artifact_store(config: ExperimentConfig) -> PaperArtifactStore:
    if config.paper.azure.artifact_backend == "filesystem":
        return build_filesystem_paper_artifact_store(config)
    if config.paper.azure.artifact_backend == "azure_blob":
        return build_azure_blob_paper_artifact_store(config)
    raise ValueError(
        f"Unsupported paper artifact backend: {config.paper.azure.artifact_backend}"
    )


def migrate_paper_postgres_database(config: ExperimentConfig) -> int:
    if config.paper.persistence_backend != "postgres":
        raise ValueError("paper-db-migrate requires paper.persistence_backend='postgres'.")
    return _postgres_module().migrate_paper_postgres_database(config)


def __getattr__(name: str) -> Any:
    if name in {
        "PostgreSQLPaperDeploymentRegistry",
        "PostgreSQLPaperOutboxRepository",
        "PostgreSQLPaperUnitOfWork",
        "PostgreSQLPaperUnitOfWorkFactory",
    }:
        return getattr(_postgres_module(), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "FilesystemPaperDeploymentRegistry",
    "FilesystemPaperOutboxRepository",
    "FilesystemPaperArtifactStore",
    "AzureBlobPaperArtifactStore",
    "PaperBlobArtifactWrite",
    "FilesystemPaperUnitOfWork",
    "FilesystemPaperUnitOfWorkFactory",
    "SQLitePaperDeploymentRegistry",
    "SQLitePaperOutboxRepository",
    "SQLitePaperUnitOfWork",
    "SQLitePaperUnitOfWorkFactory",
    "PostgreSQLPaperDeploymentRegistry",
    "PostgreSQLPaperOutboxRepository",
    "PostgreSQLPaperUnitOfWork",
    "PostgreSQLPaperUnitOfWorkFactory",
    "build_filesystem_paper_deployment_registry",
    "build_filesystem_paper_artifact_store",
    "build_azure_blob_paper_artifact_store",
    "sync_paper_review_artifacts",
    "build_filesystem_paper_uow_factory",
    "build_paper_uow_factory",
    "build_paper_artifact_store",
    "build_sqlite_paper_deployment_registry",
    "build_sqlite_paper_uow_factory",
    "apply_postgres_migrations",
    "build_postgres_paper_deployment_registry",
    "build_postgres_paper_uow_factory",
    "migrate_paper_postgres_database",
]
