from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING, Any

from marketlab.config import ExperimentConfig
from marketlab.paper.contracts import PaperDeploymentRegistry, PaperUnitOfWorkFactory

from .filesystem import (
    FilesystemPaperArtifactStore,
    FilesystemPaperDeploymentRegistry,
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


def migrate_paper_postgres_database(config: ExperimentConfig) -> int:
    if config.paper.persistence_backend != "postgres":
        raise ValueError("paper-db-migrate requires paper.persistence_backend='postgres'.")
    return _postgres_module().migrate_paper_postgres_database(config)


def __getattr__(name: str) -> Any:
    if name in {
        "PostgreSQLPaperDeploymentRegistry",
        "PostgreSQLPaperUnitOfWork",
        "PostgreSQLPaperUnitOfWorkFactory",
    }:
        return getattr(_postgres_module(), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "FilesystemPaperDeploymentRegistry",
    "FilesystemPaperArtifactStore",
    "FilesystemPaperUnitOfWork",
    "FilesystemPaperUnitOfWorkFactory",
    "SQLitePaperDeploymentRegistry",
    "SQLitePaperUnitOfWork",
    "SQLitePaperUnitOfWorkFactory",
    "PostgreSQLPaperDeploymentRegistry",
    "PostgreSQLPaperUnitOfWork",
    "PostgreSQLPaperUnitOfWorkFactory",
    "build_filesystem_paper_deployment_registry",
    "build_filesystem_paper_artifact_store",
    "build_filesystem_paper_uow_factory",
    "build_sqlite_paper_deployment_registry",
    "build_sqlite_paper_uow_factory",
    "apply_postgres_migrations",
    "build_postgres_paper_deployment_registry",
    "build_postgres_paper_uow_factory",
    "migrate_paper_postgres_database",
]
