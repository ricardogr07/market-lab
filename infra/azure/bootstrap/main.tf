data "azurerm_client_config" "current" {}

locals {
  required_tags = {
    "cost-center" = var.cost_center
    "managed-by"  = "terraform"
    "owner"       = var.owner
    "phase"       = "9"
    "project"     = "marketlab"
    "purpose"     = "terraform-state"
  }

  tags = merge(var.additional_tags, local.required_tags)
}

resource "azurerm_resource_group" "terraform_state" {
  name     = "rg-marketlab-terraform-${var.resource_suffix}"
  location = var.location
  tags     = local.tags
}

resource "azurerm_storage_account" "terraform_state" {
  name                            = "mltf${var.resource_suffix}"
  resource_group_name             = azurerm_resource_group.terraform_state.name
  location                        = azurerm_resource_group.terraform_state.location
  account_tier                    = "Standard"
  account_replication_type        = "ZRS"
  account_kind                    = "StorageV2"
  min_tls_version                 = "TLS1_2"
  https_traffic_only_enabled      = true
  allow_nested_items_to_be_public = false
  shared_access_key_enabled       = false
  default_to_oauth_authentication = true
  public_network_access_enabled   = true
  local_user_enabled              = false

  blob_properties {
    versioning_enabled = true

    delete_retention_policy {
      days = 30
    }

    container_delete_retention_policy {
      days = 30
    }
  }

  tags = local.tags
}

resource "azurerm_role_assignment" "current_principal_state_access" {
  scope                = azurerm_storage_account.terraform_state.id
  role_definition_name = "Storage Blob Data Contributor"
  principal_id         = data.azurerm_client_config.current.object_id
  principal_type       = "User"
}

resource "azurerm_storage_container" "terraform_state" {
  name                  = "tfstate"
  storage_account_id    = azurerm_storage_account.terraform_state.id
  container_access_type = "private"

  depends_on = [azurerm_role_assignment.current_principal_state_access]
}

resource "azurerm_management_lock" "terraform_state" {
  name       = "lock-marketlab-terraform-state"
  scope      = azurerm_storage_account.terraform_state.id
  lock_level = "CanNotDelete"
  notes      = "Protects the MarketLab Terraform state backend from accidental deletion."

  depends_on = [azurerm_storage_container.terraform_state]
}
