# P9-08 Worker Plan: PostgreSQL Repositories And Forward-Only Migrations

- Branch: `feature/phase-9-postgres-persistence`
- Pull request: `feat(phase9): add PostgreSQL paper persistence`
- Dependency: P9-07 hosted execution context and registry

P9-08 adds an explicit PostgreSQL control-state adapter for the existing QQQ
paper workflow. Filesystem and SQLite behavior remain unchanged. PostgreSQL is
configured only by `MARKETLAB_PAPER_POSTGRES_DSN`; no DSN or database
credential is stored in tracked YAML or emitted to logs.
Its migration history is forward-only.

This packet does not change strategy, proposal, approval, broker, QQQ
configuration, Azure resources, Blob Storage, an outbox, Service Bus,
Terraform, or
live Azure operation. P9-09 owns Blob parity, a durable outbox, and Service
Bus duplicate handling.

## Persistence Contract

`paper.persistence_backend` accepts `filesystem`, `sqlite`, and `postgres`.
The PostgreSQL adapter implements the established public contracts:

- `PaperTradeRepository`
- `PaperStatusRepository`
- `PaperUnitOfWork`
- `PaperDeploymentRegistry`

All stored payloads are `jsonb`, with the existing proposal and trade identity
fields preserved. Proposal listings retain the established deterministic order:
`effective_date`, then `created_at`, then `proposal_id`, descending. The global
phase-run `idempotency_key` remains conflict-safe across every paper phase.

PostgreSQL is the transactional state store. The existing filesystem JSON
artifacts are still written as a temporary review mirror. A unit of work writes
the mirror before committing its database transaction. If any mirror write
fails, it restores the affected mirror paths and rolls the database transaction
back, preserving the filesystem and SQLite local semantics.

## Forward-Only Migration Contract

Packaged SQL migrations live in `marketlab.paper.persistence.migrations` and
are numbered and append-only. `marketlab_paper_schema_migrations` records each
version, filename, checksum, and applied timestamp. The migration runner takes
a PostgreSQL advisory lock, rejects edited historical migrations, and never
runs during normal paper commands.

The explicit command is:

```text
marketlab paper-db-migrate --config <postgres-config>
```

It requires `persistence_backend: postgres` and
`MARKETLAB_PAPER_POSTGRES_DSN`. It performs no provider, broker, scheduler, or
notification work and reports the applied schema version.

Rollback is a verified database restore followed by a forward corrective
migration. As a consequence, destructive down migrations are not supported.

## Local Validation

`docker/compose.postgres.yml` provides a disposable, pinned PostgreSQL image
for local tests. `tox -e postgres` starts the service, waits for its health
check, runs the contract and migration suite with a development-only DSN, and
tears the service down with its volume in a `finally` block.

The PostgreSQL suite proves clean installation, rerun no-op behavior, ordered
upgrade application, checksum tamper rejection, advisory-lock serialization,
fixture database restore, staged commit/rollback behavior, deterministic
listing, duplicate replay acceptance, conflicting replay rejection,
cross-phase idempotency conflicts, and database rollback after a failed
artifact mirror write.

## Acceptance

- filesystem, SQLite, and PostgreSQL adapters retain the same public paper
  repository behavior
- all PostgreSQL state is stored as `jsonb` with existing record identities
- migrations run only through `paper-db-migrate` under an advisory lock
- historical migration checksum changes fail closed
- failed artifact-mirror writes leave no committed PostgreSQL mutation
- no P9-09 or Azure scope is introduced
