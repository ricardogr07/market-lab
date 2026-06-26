from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BOOTSTRAP = ROOT / "infra" / "azure" / "bootstrap"
PHASE9_SHADOW = ROOT / "infra" / "azure" / "phase9-shadow"
QQQ_PAPER = ROOT / "infra" / "azure" / "qqq-paper"
MAIN = BOOTSTRAP / "main.tf"
VERSIONS = BOOTSTRAP / "versions.tf"
LOCK = BOOTSTRAP / ".terraform.lock.hcl"
RUNBOOK = BOOTSTRAP / "README.md"
WORKFLOW = ROOT / ".github" / "workflows" / "terraform.yml"


def _normalized(path: Path) -> str:
    return " ".join(path.read_text(encoding="utf-8").split())


def test_bootstrap_resource_scope_is_locked() -> None:
    content = MAIN.read_text(encoding="utf-8")
    resource_types = re.findall(r'^resource\s+"([^"]+)"\s+"[^"]+"\s+\{', content, re.MULTILINE)
    data_types = re.findall(r'^data\s+"([^"]+)"\s+"[^"]+"\s+\{', content, re.MULTILINE)

    assert resource_types == [
        "azurerm_resource_group",
        "azurerm_storage_account",
        "azurerm_role_assignment",
        "azurerm_storage_container",
        "azurerm_management_lock",
    ]
    assert data_types == ["azurerm_client_config"]


def test_bootstrap_storage_and_state_security_controls_are_explicit() -> None:
    main = _normalized(MAIN)
    versions = _normalized(VERSIONS)

    required_main = [
        'account_replication_type = "ZRS"',
        'account_kind = "StorageV2"',
        'min_tls_version = "TLS1_2"',
        "https_traffic_only_enabled = true",
        "allow_nested_items_to_be_public = false",
        "shared_access_key_enabled = false",
        "default_to_oauth_authentication = true",
        "public_network_access_enabled = true",
        "local_user_enabled = false",
        "versioning_enabled = true",
        "delete_retention_policy { days = 30 }",
        "container_delete_retention_policy { days = 30 }",
        'container_access_type = "private"',
        'role_definition_name = "Storage Blob Data Contributor"',
        'principal_type = "User"',
        "depends_on = [azurerm_role_assignment.current_principal_state_access]",
        'lock_level = "CanNotDelete"',
    ]
    required_versions = [
        'required_version = "= 1.15.5"',
        'version = "~> 4.74.0"',
        'resource_provider_registrations = "none"',
        "storage_use_azuread = true",
    ]
    backend = _normalized(BOOTSTRAP / "backend.tf.example")

    assert all(clause in main for clause in required_main)
    assert all(clause in versions for clause in required_versions)
    assert 'backend "azurerm" { use_azuread_auth = true use_cli = true }' in backend
    assert "change_feed_enabled" not in main


def test_bootstrap_required_tags_cannot_be_overridden() -> None:
    main = _normalized(MAIN)
    variables = _normalized(BOOTSTRAP / "variables.tf")

    for tag in ["cost-center", "managed-by", "owner", "phase", "project", "purpose"]:
        assert f'"{tag}"' in main
        assert f'"{tag}"' in variables

    assert "tags = merge(var.additional_tags, local.required_tags)" in main
    assert "additional_tags cannot redefine a required MarketLab tag" in variables


def test_provider_lock_is_exact_and_cross_platform() -> None:
    content = LOCK.read_text(encoding="utf-8")
    normalized = " ".join(content.split())

    assert 'version = "4.74.0"' in normalized
    assert 'constraints = "~> 4.74.0"' in normalized
    assert 'provider "registry.terraform.io/hashicorp/azurerm"' in content
    assert content.count('"h1:') == 3
    assert content.count('"zh:') == 12


def test_terraform_ci_is_validation_only() -> None:
    workflow = WORKFLOW.read_text(encoding="utf-8")
    normalized = " ".join(workflow.lower().split())

    assert "contents: read" in workflow
    assert 'terraform_version: "1.15.5"' in workflow
    assert "terraform_wrapper: false" in workflow
    assert "uv run tox -e terraform" in workflow
    assert "persist-credentials: false" in workflow

    forbidden = [
        "azure/login",
        "id-token: write",
        "az login",
        "terraform plan",
        "terraform apply",
        "terraform destroy",
        "terraform import",
        "arm_subscription_id",
        "arm_tenant_id",
        "arm_client_id",
        "client-secret",
    ]
    assert all(term not in normalized for term in forbidden)


def test_terraform_tox_contract_disables_the_backend() -> None:
    tox = _normalized(ROOT / "tox.ini")

    assert "[testenv:terraform]" in tox
    assert "terraform fmt -check -recursive" in tox
    assert "init -backend=false -lockfile=readonly -input=false" in tox
    assert "validate -no-color" in tox
    assert "plan" not in tox.split("[testenv:terraform]", maxsplit=1)[1].split(
        "[testenv:typecheck]", maxsplit=1
    )[0]


def test_operator_values_and_terraform_state_are_ignored() -> None:
    ignored = (ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()

    assert "**/.terraform/" in ignored
    assert "*.tfstate" in ignored
    assert "*.tfstate.*" in ignored
    assert "**/backend.hcl" in ignored
    assert "**/*.backend.hcl" in ignored
    assert "**/backend-*.hcl" in ignored
    assert "**/*.tfbackend" in ignored
    assert "**/*.tfvars" in ignored
    assert "**/*.tfvars.json" in ignored
    assert "**/*.tfplan" in ignored
    assert "**/.terraformrc" in ignored
    assert "**/terraform.rc" in ignored
    assert "**/credentials.tfrc.json" in ignored
    assert "infra/azure/bootstrap/backend.tf" in ignored
    assert "infra/azure/phase9-shadow/backend.tf" in ignored
    assert "infra/azure/qqq-paper/backend.tf" in ignored
    assert (BOOTSTRAP / "backend.tf.example").exists()
    assert (BOOTSTRAP / "backend.hcl.example").exists()
    assert (BOOTSTRAP / "terraform.tfvars.example").exists()
    assert (PHASE9_SHADOW / "backend.tf.example").exists()
    assert (PHASE9_SHADOW / "backend.hcl.example").exists()
    assert (PHASE9_SHADOW / "terraform.tfvars.example").exists()
    assert (QQQ_PAPER / "backend.tf.example").exists()
    assert (QQQ_PAPER / "backend.hcl.example").exists()
    assert (QQQ_PAPER / "backend.paper-prod.hcl.example").exists()
    assert (QQQ_PAPER / "terraform.tfvars.example").exists()


def test_bootstrap_runbook_preserves_supervision_gates() -> None:
    runbook = _normalized(RUNBOOK)

    required = [
        "Stop and obtain explicit user approval",
        "`az login`",
        "resource-provider registration",
        "committing these files",
        "pushing the branch",
        "five approved resource types",
        "Azure RBAC propagation can take several minutes",
        "USD 5 monthly resource-group budget",
        "separately reviewed destroy plan",
    ]
    assert all(clause in runbook for clause in required)


def test_phase9_shadow_resource_scope_is_locked() -> None:
    content = (PHASE9_SHADOW / "main.tf").read_text(encoding="utf-8")
    resource_types = re.findall(r'^resource\s+"([^"]+)"\s+"[^"]+"\s+\{', content, re.MULTILINE)

    assert resource_types == [
        "azurerm_resource_group",
        "azurerm_user_assigned_identity",
        "azurerm_container_registry",
        "azurerm_log_analytics_workspace",
        "azurerm_application_insights",
        "azurerm_container_app_environment",
        "azurerm_storage_account",
        "azurerm_storage_share",
        "azurerm_storage_container",
        "azurerm_storage_management_policy",
        "azurerm_container_app_environment_storage",
        "azurerm_role_assignment",
        "azurerm_role_assignment",
        "azurerm_role_assignment",
        "azurerm_container_app_job",
        "azurerm_monitor_action_group",
        "azurerm_monitor_scheduled_query_rules_alert_v2",
        "azurerm_monitor_scheduled_query_rules_alert_v2",
    ]


def test_phase9_shadow_launch_gate_defaults_to_disabled_schedule() -> None:
    main = _normalized(PHASE9_SHADOW / "main.tf")
    variables = _normalized(PHASE9_SHADOW / "variables.tf")
    tfvars = _normalized(PHASE9_SHADOW / "terraform.tfvars.example")

    required = [
        "default = false",
        "create_shadow_job",
        "enable_shadow_schedule requires create_shadow_job",
        "for_each = var.enable_shadow_schedule ? [] : [1]",
        "for_each = var.enable_shadow_schedule ? [1] : []",
        "launch_gate_evidence_uri must be an https URI before enable_shadow_schedule can be true",
        '"15 1 * * *"',
        '"phase9-shadow-scheduler"',
        '"/app/configs/experiment.btc_phase9_shadow_daily.yaml"',
        '"--once"',
        "marketlab_image_digest must be an immutable sha256 digest",
    ]

    assert all(clause in variables or clause in main or clause in tfvars for clause in required)
    assert "create_shadow_job = false" in tfvars
    assert "enable_shadow_schedule = false" in tfvars
    assert "enable_shadow_alerts = false" in tfvars


def test_phase9_shadow_storage_archive_and_restore_controls_are_documented() -> None:
    main = _normalized(PHASE9_SHADOW / "main.tf")
    runbook = _normalized(PHASE9_SHADOW / "README.md")

    required_main = [
        'account_replication_type = "ZRS"',
        'account_kind = "StorageV2"',
        'min_tls_version = "TLS1_2"',
        "https_traffic_only_enabled = true",
        "allow_nested_items_to_be_public = false",
        "versioning_enabled = true",
        'name = "phase9-shadow-live"',
        'name = "phase9-shadow-archive"',
        'container_access_type = "private"',
        "azurerm_storage_management_policy",
        "delete_after_days_since_creation_greater_than = var.archive_retention_days",
        "delete_after_days_since_creation = var.archive_retention_days",
        'path = "/app/artifacts"',
        'storage_type = "AzureFile"',
    ]
    required_runbook = [
        "Archive Snapshot",
        "Restore Check",
        "az storage copy",
        "snapshots/$snapshotDate",
        "restore-check/$snapshotDate",
        "Never overwrite the live artifact tree",
        "`450` day lifecycle retention policy",
        "must not become the P9-09 generic Blob or Service Bus adapter layer",
    ]

    assert all(clause in main for clause in required_main)
    assert all(clause in runbook for clause in required_runbook)


def test_phase9_shadow_identity_alerts_and_no_public_ingress_are_locked() -> None:
    main_content = (PHASE9_SHADOW / "main.tf").read_text(encoding="utf-8")
    main = " ".join(main_content.split())

    required = [
        'type = "UserAssigned"',
        "identity_ids = [azurerm_user_assigned_identity.shadow.id]",
        "identity = azurerm_user_assigned_identity.shadow.id",
        'role_definition_name = "AcrPull"',
        'role_definition_name = "Storage Blob Data Contributor"',
        'role_definition_name = "Storage File Data SMB Share Contributor"',
        "azurerm_monitor_action_group",
        "azurerm_monitor_scheduled_query_rules_alert_v2",
        "job_failures",
        "missing_evidence",
        "ContainerAppSystemLogs_CL",
        "ContainerAppConsoleLogs_CL",
        "count = var.create_shadow_job ? 1 : 0",
    ]
    forbidden = [
        'resource "azurerm_container_app"',
        'resource "azurerm_key_vault"',
        'resource "azurerm_servicebus',
        'ingress {',
        "alpaca",
        "broker",
        "approval",
        "telegram",
    ]

    assert all(clause in main for clause in required)
    assert all(clause.lower() not in main_content.lower() for clause in forbidden)


def test_phase9_shadow_backend_provider_and_tox_validation_are_locked() -> None:
    backend = _normalized(PHASE9_SHADOW / "backend.tf.example")
    backend_hcl = _normalized(PHASE9_SHADOW / "backend.hcl.example")
    versions = _normalized(PHASE9_SHADOW / "versions.tf")
    lock = _normalized(PHASE9_SHADOW / ".terraform.lock.hcl")
    tox = _normalized(ROOT / "tox.ini")
    workflow = WORKFLOW.read_text(encoding="utf-8")

    assert 'backend "azurerm" { use_azuread_auth = true use_cli = true }' in backend
    assert "key = \"phase9-shadow.tfstate\"" in backend_hcl
    assert 'required_version = "= 1.15.5"' in versions
    assert 'version = "~> 4.74.0"' in versions
    assert 'resource_provider_registrations = "none"' in versions
    assert 'version = "4.74.0"' in lock
    assert "phase9-shadow init -backend=false -lockfile=readonly -input=false" in tox
    assert "phase9-shadow validate -no-color" in tox
    assert "infra/azure/phase9-shadow/.terraform.lock.hcl" in workflow


def test_qqq_paper_azure_resource_scope_is_locked() -> None:
    content = (QQQ_PAPER / "main.tf").read_text(encoding="utf-8")
    resource_types = re.findall(r'^resource\s+"([^"]+)"\s+"[^"]+"\s+\{', content, re.MULTILINE)
    data_types = re.findall(r'^data\s+"([^"]+)"\s+"[^"]+"\s+\{', content, re.MULTILINE)

    assert data_types == ["azurerm_client_config"]
    assert resource_types == [
        "azurerm_resource_group",
        "azurerm_user_assigned_identity",
        "azurerm_container_registry",
        "azurerm_log_analytics_workspace",
        "azurerm_application_insights",
        "azurerm_container_app_environment",
        "azurerm_storage_account",
        "azurerm_storage_share",
        "azurerm_storage_container",
        "azurerm_storage_management_policy",
        "azurerm_container_app_environment_storage",
        "azurerm_servicebus_namespace",
        "azurerm_servicebus_queue",
        "azurerm_key_vault",
        "azurerm_postgresql_flexible_server",
        "azurerm_postgresql_flexible_server_database",
        "azurerm_postgresql_flexible_server_firewall_rule",
        "azurerm_role_assignment",
        "azurerm_role_assignment",
        "azurerm_role_assignment",
        "azurerm_role_assignment",
        "azurerm_role_assignment",
        "azurerm_container_app_job",
        "azurerm_monitor_action_group",
    ]


def test_qqq_paper_azure_jobs_and_triggers_default_disabled() -> None:
    main = _normalized(QQQ_PAPER / "main.tf")
    variables = _normalized(QQQ_PAPER / "variables.tf")
    tfvars = _normalized(QQQ_PAPER / "terraform.tfvars.example")

    required = [
        "for_each = var.create_jobs ? local.jobs : {}",
        "paper-db-migrate",
        "paper-scheduler",
        "paper-agent-approve",
        "paper-outbox-deliver",
        "paper-notifications-deliver",
        "paper-blob-sync",
        "paper-service-bus-receive",
        "MARKETLAB_PAPER_RUNTIME_ENV_OVERRIDES",
        "MARKETLAB_PAPER_POSTGRES_DSN",
        "enable_scheduler_schedule requires create_jobs",
        "enable_service_bus_approval_trigger requires create_jobs",
        "event_trigger_config",
        'custom_rule_type = "azure-servicebus"',
        "paper-service-bus-receive",
        "enable_broker_secret_refs",
        "postgres_firewall_rules",
        "create_jobs requires at least one operator-approved PostgreSQL firewall rule",
        "MARKETLAB_PAPER_TELEGRAM_ENABLED",
        "AZURE_CLIENT_ID",
        'path = "/app/artifacts/paper/state"',
        'storage_type = "AzureFile"',
        "false",
        "terraform-disabled-template",
    ]

    assert all(clause in main or clause in variables or clause in tfvars for clause in required)
    assert "create_jobs = false" in tfvars
    assert "enable_scheduler_schedule = false" in tfvars
    assert "enable_service_bus_approval_trigger = false" in tfvars
    assert "enable_broker_secret_refs = false" in tfvars


def test_qqq_paper_azure_security_and_runtime_seams_are_locked() -> None:
    main_content = (QQQ_PAPER / "main.tf").read_text(encoding="utf-8")
    main = " ".join(main_content.split())
    variables = _normalized(QQQ_PAPER / "variables.tf")
    runbook = _normalized(QQQ_PAPER / "README.md")

    required = [
        'role_definition_name = "AcrPull"',
        'role_definition_name = "Storage Blob Data Contributor"',
        'role_definition_name = "Azure Service Bus Data Sender"',
        'role_definition_name = "Azure Service Bus Data Receiver"',
        'role_definition_name = "Key Vault Secrets User"',
        "shared_access_key_enabled = true",
        "default_to_oauth_authentication = true",
        "qqq-paper-state",
        "Container Apps Jobs mount the `qqq-paper-state` Azure Files share",
        "local_auth_enabled = false",
        "rbac_authorization_enabled = true",
        "requires_duplicate_detection = true",
        "dead_lettering_on_message_expiration = true",
        "configs/experiment.qqq_paper_daily.yaml",
        "passwordless PostgreSQL managed-identity authentication",
    ]
    forbidden = [
        'resource "azurerm_container_app"',
        "ingress {",
        "voo",
        "btc",
        "live-money",
        "terraform apply",
        "terraform destroy",
        "terraform import",
    ]

    assert all(clause in main or clause in variables or clause in runbook for clause in required)
    assert all(clause.lower() not in main_content.lower() for clause in forbidden)


def test_qqq_paper_backend_provider_and_tox_validation_are_locked() -> None:
    backend = _normalized(QQQ_PAPER / "backend.tf.example")
    backend_hcl = _normalized(QQQ_PAPER / "backend.hcl.example")
    backend_prod_hcl = _normalized(QQQ_PAPER / "backend.paper-prod.hcl.example")
    versions = _normalized(QQQ_PAPER / "versions.tf")
    lock = _normalized(QQQ_PAPER / ".terraform.lock.hcl")
    tox = _normalized(ROOT / "tox.ini")
    workflow = WORKFLOW.read_text(encoding="utf-8")

    assert 'backend "azurerm" { use_azuread_auth = true use_cli = true }' in backend
    assert "key = \"qqq-paper-dev.tfstate\"" in backend_hcl
    assert "key = \"qqq-paper-prod.tfstate\"" in backend_prod_hcl
    assert 'required_version = "= 1.15.5"' in versions
    assert 'version = "~> 4.74.0"' in versions
    assert 'resource_provider_registrations = "none"' in versions
    assert 'version = "4.74.0"' in lock
    assert "qqq-paper init -backend=false -lockfile=readonly -input=false" in tox
    assert "qqq-paper validate -no-color" in tox
    assert "infra/azure/qqq-paper/.terraform.lock.hcl" in workflow


def test_qqq_paper_prod_cutover_gates_are_locked() -> None:
    variables = _normalized(QQQ_PAPER / "variables.tf")
    tfvars = _normalized(QQQ_PAPER / "terraform.tfvars.example")
    runbook = _normalized(QQQ_PAPER / "README.md")

    required = [
        'contains(["dev", "paper-prod"], var.environment)',
        "environment must be either dev or paper-prod",
        "marketlab_image_digest must not be the placeholder digest when create_jobs is true",
        "enable_scheduler_schedule is allowed only for paper-prod after broker secret refs and all P9-13 evidence URIs are configured",
        "enable_service_bus_approval_trigger is allowed only for paper-prod after broker secret refs and all P9-13 evidence URIs are configured",
        "p9_12_parity_evidence_uri",
        "final_import_evidence_uri",
        "backup_restore_evidence_uri",
        "rollback_evidence_uri",
        "alert_evidence_uri",
        "alpaca_key_id_secret_id must be an https Key Vault secret ID when enable_broker_secret_refs is true",
        "anthropic_api_key_secret_id must be an https Key Vault secret ID when enable_broker_secret_refs is true",
        "telegram_bot_token_secret_id must be an https Key Vault secret ID when enable_broker_secret_refs is true",
        "environment = \"paper-prod\"",
        "qqq-paper-prod.tfstate",
        "Do not commit the real backend file, tfvars, Terraform plan, Terraform state, DSN, Key Vault secret IDs, or live Azure identifiers.",
    ]

    assert all(clause in variables or clause in tfvars or clause in runbook for clause in required)
