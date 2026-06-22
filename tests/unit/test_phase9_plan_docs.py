from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PLAN = ROOT / "docs" / "PLAN.md"
P9_02_PLAN = ROOT / "docs" / "phase9" / "P9-02-WORKER-PLAN.md"
P9_02_EVIDENCE = ROOT / "docs" / "phase9" / "P9-02-BOOTSTRAP-EVIDENCE.md"
P9_03_PLAN = ROOT / "docs" / "phase9" / "P9-03-WORKER-PLAN.md"
P9_04_PLAN = ROOT / "docs" / "phase9" / "P9-04-WORKER-PLAN.md"
P9_05_PLAN = ROOT / "docs" / "phase9" / "P9-05-WORKER-PLAN.md"
P9_06_PLAN = ROOT / "docs" / "phase9" / "P9-06-WORKER-PLAN.md"
P9_07_PLAN = ROOT / "docs" / "phase9" / "P9-07-WORKER-PLAN.md"
P9_08_PLAN = ROOT / "docs" / "phase9" / "P9-08-WORKER-PLAN.md"
P9_09_PLAN = ROOT / "docs" / "phase9" / "P9-09-WORKER-PLAN.md"


def test_phase9_plan_replaces_obsolete_cloud_plan() -> None:
    content = PLAN.read_text(encoding="utf-8")

    assert not (ROOT / "CLOUD_MIGRATION_PLAN.md").exists()
    assert "# Phase 9 Roadmap" in content
    assert "Google Cloud" not in content
    assert "GCP" not in content


def test_phase9_plan_locks_three_track_scope() -> None:
    content = PLAN.read_text(encoding="utf-8")
    normalized = " ".join(content.split())

    required = [
        "## Track A: BTC Evidence Gate",
        "## Track B: Azure Foundation",
        "## Track C: QQQ Operations On Azure",
        "June 3, 2026",
        "June 16, 2027",
        "configs/experiment.btc_phase8_guarded_gate_bull_risk_off_override_partial_support.yaml",
        "configs/experiment.qqq_paper_daily.yaml",
        "Azure Container Apps Jobs",
        "Azure Database for PostgreSQL Flexible Server",
        "Azure Blob Storage",
        "Azure Service Bus",
        "infra/azure/phase9-shadow/",
        "infra/azure/qqq-paper/",
        "at least `10` consecutive NYSE trading days",
    ]

    assert all(term in normalized for term in required)


def test_phase9_plan_preserves_safety_boundaries() -> None:
    content = PLAN.read_text(encoding="utf-8")
    normalized = " ".join(content.split())

    required = [
        "signals-only BTC shadow confirmation",
        "cannot request approval, call a broker, submit an order",
        "Alpaca paper endpoints only",
        "MCP remains an inspection and approval surface",
        "VOO and both BTC operational stacks stay local during Phase 9",
        "live-money broker support",
    ]

    assert all(term in normalized for term in required)


def test_p9_02_worker_plan_requires_user_supervision() -> None:
    content = P9_02_PLAN.read_text(encoding="utf-8")
    normalized = " ".join(content.split())

    required = [
        "feature/phase-9-azure-bootstrap",
        "stop and obtain explicit approval",
        "running `az login`",
        "running a Terraform command that refreshes, plans against, imports, changes, or destroys real Azure resources",
        "committing the implementation",
        "pushing the branch",
        "USD 5",
        "Typically below `USD 1`",
        "terraform -chdir=infra/azure/bootstrap init -backend=false",
        "terraform -chdir=infra/azure/bootstrap plan -out bootstrap.tfplan",
        "-migrate-state",
        "no `id-token: write`",
        "no Azure login",
        "no `terraform plan`, `apply`, `destroy`, or import",
    ]

    assert all(term in normalized for term in required)


def test_p9_02_plan_is_linked_from_docs() -> None:
    roadmap = PLAN.read_text(encoding="utf-8")
    nav = (ROOT / "mkdocs.yml").read_text(encoding="utf-8")
    index = (ROOT / "docs" / "index.md").read_text(encoding="utf-8")

    assert "phase9/P9-02-WORKER-PLAN.md" in roadmap
    assert "Phase 9 P9-02 Worker Plan: phase9/P9-02-WORKER-PLAN.md" in nav
    assert "phase9/P9-02-WORKER-PLAN.md" in index
    assert "Phase 9 P9-02 Bootstrap Evidence: phase9/P9-02-BOOTSTRAP-EVIDENCE.md" in nav
    assert "phase9/P9-02-BOOTSTRAP-EVIDENCE.md" in index


def test_p9_02_evidence_records_completion_and_publication_gates() -> None:
    content = P9_02_EVIDENCE.read_text(encoding="utf-8")
    normalized = " ".join(content.split())

    required = [
        "June 7, 2026",
        "tfstate/bootstrap.tfstate",
        "five managed resources",
        "No changes",
        "The obsolete root-level local state copies were removed",
        "June 11, 2026",
        "`USD 5/month` budget with `50%`, `80%`, and `100%` alerts | Complete",
        "Azure Activity Log review | Complete",
        "Commit | Pending publication",
        "Push or pull request | Pending publication",
    ]

    assert all(term in normalized for term in required)


def test_p9_02_evidence_does_not_publish_live_azure_identifiers() -> None:
    content = P9_02_EVIDENCE.read_text(encoding="utf-8")

    assert "@" not in content
    assert "suffix `" not in content
    assert "| Subscription | Approved Phase 9 subscription; name and ID retained locally |" in content
    assert "| Resource suffix | Retained locally |" in content
    assert content.count("Name retained locally") == 2


def test_p9_03_plan_locks_shadow_scope_and_hashes() -> None:
    content = P9_03_PLAN.read_text(encoding="utf-8")
    normalized = " ".join(content.split())

    required = [
        "feature/phase-9-btc-shadow-lock",
        "ce01124",
        "2026-06-03",
        "2027-06-02",
        "2027-06-16",
        "btc-phase9-shadow-v1",
        "paper.enabled: true",
        "does not fetch market data, run models, schedule work",
        "cannot be silently reconstructed",
        "d439acca79ca2108a4d907452b5d442ab67b319d430440d07f14f9adc1295f18",
        "71beba28529abba3482145094654c5eaf8f12355d92a93830fe746a241129550",
        "verify_shadow_contract",
    ]

    assert all(term in normalized for term in required)


def test_p9_03_plan_is_linked_from_docs() -> None:
    roadmap = PLAN.read_text(encoding="utf-8")
    nav = (ROOT / "mkdocs.yml").read_text(encoding="utf-8")
    index = (ROOT / "docs" / "index.md").read_text(encoding="utf-8")

    assert "phase9/P9-03-WORKER-PLAN.md" in roadmap
    assert "Phase 9 P9-03 Worker Plan: phase9/P9-03-WORKER-PLAN.md" in nav
    assert "phase9/P9-03-WORKER-PLAN.md" in index


def test_p9_04_plan_locks_decision_and_journal_scope() -> None:
    content = P9_04_PLAN.read_text(encoding="utf-8")
    normalized = " ".join(content.split())

    required = [
        "feature/phase-9-btc-shadow-decision",
        "verify_shadow_contract",
        "ShadowDecisionEvaluation",
        "2026-05-27",
        "artifacts/phase9-shadow/decisions/<effective-date>.json",
        "ShadowJournalConflictError",
        "identical repeat",
        "does not schedule runs",
        "cannot reconstruct an earlier decision",
        "phase9-shadow-decision",
    ]

    assert all(term in normalized for term in required)


def test_p9_04_plan_is_linked_from_docs_and_cli_metadata() -> None:
    roadmap = PLAN.read_text(encoding="utf-8")
    nav = (ROOT / "mkdocs.yml").read_text(encoding="utf-8")
    index = (ROOT / "docs" / "index.md").read_text(encoding="utf-8")
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")

    assert "phase9/P9-04-WORKER-PLAN.md" in roadmap
    assert "Phase 9 P9-04 Worker Plan: phase9/P9-04-WORKER-PLAN.md" in nav
    assert "phase9/P9-04-WORKER-PLAN.md" in index
    assert 'phase9-shadow-decision = "marketlab.shadow.cli:main"' in pyproject


def test_p9_05_plan_locks_operations_and_reporting_scope() -> None:
    content = P9_05_PLAN.read_text(encoding="utf-8")
    normalized = " ".join(content.split())

    required = [
        "feature/phase-9-btc-shadow-operations",
        "ShadowDecisionEvaluator",
        "cannot reconstruct an earlier decision",
        "attempts/<effective-date>/<attempt-id>.json",
        "evidence/decisions/<effective-date>.json",
        "evidence/labels/<effective-date>.json",
        "state/status.json",
        "2027-06-16",
        "35 bps",
        "50 bps",
        "signal_validity_gate",
        "bull_participation_gate",
        "never approves trading",
        "P9-15",
    ]

    assert all(term in normalized for term in required)


def test_p9_05_plan_is_linked_from_docs_and_cli_metadata() -> None:
    roadmap = PLAN.read_text(encoding="utf-8")
    nav = (ROOT / "mkdocs.yml").read_text(encoding="utf-8")
    index = (ROOT / "docs" / "index.md").read_text(encoding="utf-8")
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")

    assert "phase9/P9-05-WORKER-PLAN.md" in roadmap
    assert "Phase 9 P9-05 Worker Plan: phase9/P9-05-WORKER-PLAN.md" in nav
    assert "phase9/P9-05-WORKER-PLAN.md" in index
    assert 'phase9-shadow-scheduler = "marketlab.shadow.cli:scheduler_main"' in pyproject
    assert 'phase9-shadow-status = "marketlab.shadow.cli:status_main"' in pyproject
    assert 'phase9-shadow-report = "marketlab.shadow.cli:report_main"' in pyproject


def test_p9_06_plan_locks_shadow_azure_launch_gate_scope() -> None:
    content = P9_06_PLAN.read_text(encoding="utf-8")
    normalized = " ".join(content.split())

    required = [
        "feature/phase-9-btc-shadow-azure",
        "feat(phase9): add BTC shadow Azure launch gate",
        "infra/azure/phase9-shadow/",
        "phase9-shadow.tfstate",
        "disabled scheduled Container Apps Job",
        "marketlab phase9-shadow-scheduler --config /app/configs/experiment.btc_phase9_shadow_daily.yaml --once",
        "enable_shadow_schedule = true",
        "launch_gate_evidence_uri",
        "append-only journal conflict handling is proven",
        "archive copy from Azure Files to Blob succeeds",
        "restore from a dated Blob snapshot into a scratch path succeeds",
        "no broker, approval, Alpaca, Telegram, Service Bus, or Key Vault secret",
        "must not become the P9-09 generic artifact store",
        "Terraform validation is format, backend-disabled init, and validate only",
        "Azure apply without separate explicit approval",
    ]

    assert all(term in normalized for term in required)


def test_p9_06_plan_is_linked_from_docs() -> None:
    roadmap = PLAN.read_text(encoding="utf-8")
    nav = (ROOT / "mkdocs.yml").read_text(encoding="utf-8")
    index = (ROOT / "docs" / "index.md").read_text(encoding="utf-8")

    assert "phase9/P9-06-WORKER-PLAN.md" in roadmap
    assert "Phase 9 P9-06 Worker Plan: phase9/P9-06-WORKER-PLAN.md" in nav
    assert "phase9/P9-06-WORKER-PLAN.md" in index


def test_p9_07_plan_locks_hosted_execution_registry_scope() -> None:
    content = P9_07_PLAN.read_text(encoding="utf-8")
    normalized = " ".join(content.split())

    required = [
        "feature/phase-9-hosted-execution-registry",
        "feat(phase9): add hosted paper execution registry",
        "deployment_id environment phase execution_id correlation_id idempotency_key trigger_source requested_at config_version image_digest",
        "Supported environments are `dev`, `uat`, and `paper-prod`",
        "Supported phases are `decision`, `agent_approve`, `submit`, and `reconcile`",
        "PaperHostedExecutionContext",
        "PaperDeploymentRegistry",
        "artifacts/paper/state/deployments/",
        "artifacts/paper/state/phase-runs/",
        "Duplicate `idempotency_key` values with identical metadata are accepted",
        "conflicting metadata raise before a paper phase creates providers, calls brokers, sends notifications",
        "does not add PostgreSQL, Blob Storage, Service Bus, QQQ Terraform, Azure SDK calls",
        "scheduler derives child phase contexts for `decision`, `submit`, and `reconcile`",
        "agent approval worker derives one `agent_approve` context per proposal",
        "Explicit flags win over environment values",
    ]

    assert all(term in normalized for term in required)


def test_p9_07_plan_is_linked_from_docs() -> None:
    roadmap = PLAN.read_text(encoding="utf-8")
    nav = (ROOT / "mkdocs.yml").read_text(encoding="utf-8")
    index = (ROOT / "docs" / "index.md").read_text(encoding="utf-8")

    assert "phase9/P9-07-WORKER-PLAN.md" in roadmap
    assert "Phase 9 P9-07 Worker Plan: phase9/P9-07-WORKER-PLAN.md" in nav
    assert "phase9/P9-07-WORKER-PLAN.md" in index


def test_p9_08_plan_locks_postgres_persistence_scope() -> None:
    content = P9_08_PLAN.read_text(encoding="utf-8")
    normalized = " ".join(content.split())

    required = [
        "feature/phase-9-postgres-persistence",
        "MARKETLAB_PAPER_POSTGRES_DSN",
        "PaperTradeRepository",
        "PaperStatusRepository",
        "PaperUnitOfWork",
        "PaperDeploymentRegistry",
        "jsonb",
        "paper-db-migrate",
        "PostgreSQL advisory lock",
        "checksum",
        "forward-only",
        "Blob Storage, an outbox, Service Bus",
        "destructive down migrations are not supported",
    ]

    assert all(term in normalized for term in required)


def test_p9_08_plan_is_linked_from_docs() -> None:
    roadmap = PLAN.read_text(encoding="utf-8")
    nav = (ROOT / "mkdocs.yml").read_text(encoding="utf-8")
    index = (ROOT / "docs" / "index.md").read_text(encoding="utf-8")

    assert "phase9/P9-08-WORKER-PLAN.md" in roadmap
    assert "Phase 9 P9-08 Worker Plan: phase9/P9-08-WORKER-PLAN.md" in nav
    assert "phase9/P9-08-WORKER-PLAN.md" in index


def test_p9_09_plan_defines_runtime_seams_blob_parity_and_outbox_delivery() -> None:
    content = P9_09_PLAN.read_text(encoding="utf-8")
    normalized = " ".join(content.split())

    required = [
        "feature/phase-9-paper-azure-seams",
        "feat(phase9): add paper Azure config and secret seams",
        "paper.azure config section",
        "EnvironmentPaperSecretProvider",
        "let Alpaca and Telegram resolve secrets through the new port",
        "route paper artifact-store construction through a backend selector",
        "AzureBlobPaperArtifactStore",
        "preserve the filesystem JSON serialization exactly in Blob payloads",
        "scope Blob addresses by environment, deployment, and ISO trade date",
        "idempotent outbox records to filesystem, SQLite, and PostgreSQL adapters",
        "forward-only PostgreSQL `003_paper_outbox.sql` migration",
        "Azure Service Bus publisher that preserves the outbox message ID",
        "failed delivery retries and delivered records are not republished",
        "persist Telegram notification intent atomically with decision, approval, and submission state",
        "validate and consume one `paper.approval.requested` envelope at a time",
        "Azure Service Bus receiver loop and message settlement",
        "Terraform or GitHub Actions changes",
    ]

    assert all(term in normalized for term in required)


def test_p9_09_plan_is_linked_from_docs() -> None:
    roadmap = PLAN.read_text(encoding="utf-8")
    nav = (ROOT / "mkdocs.yml").read_text(encoding="utf-8")
    index = (ROOT / "docs" / "index.md").read_text(encoding="utf-8")

    assert "phase9/P9-09-WORKER-PLAN.md" in roadmap
    assert "Phase 9 P9-09 Worker Plan: phase9/P9-09-WORKER-PLAN.md" in nav
    assert "phase9/P9-09-WORKER-PLAN.md" in index
