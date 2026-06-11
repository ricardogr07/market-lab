from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BOOTSTRAP = ROOT / "infra" / "azure" / "bootstrap"
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
    assert (BOOTSTRAP / "backend.tf.example").exists()
    assert (BOOTSTRAP / "backend.hcl.example").exists()
    assert (BOOTSTRAP / "terraform.tfvars.example").exists()


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
