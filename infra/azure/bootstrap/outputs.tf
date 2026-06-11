output "backend_config" {
  description = "Non-secret values for the bootstrap root's backend.hcl file."
  value = {
    resource_group_name  = azurerm_resource_group.terraform_state.name
    storage_account_name = azurerm_storage_account.terraform_state.name
    container_name       = azurerm_storage_container.terraform_state.name
    key                  = "bootstrap.tfstate"
    use_azuread_auth     = true
    use_cli              = true
  }
}

output "state_keys" {
  description = "Reserved remote-state keys for the approved Phase 9 Terraform roots."
  value = {
    phase9_shadow  = "phase9-shadow.tfstate"
    qqq_paper_dev  = "qqq-paper-dev.tfstate"
    qqq_paper_uat  = "qqq-paper-uat.tfstate"
    qqq_paper_prod = "qqq-paper-prod.tfstate"
  }
}

output "state_storage_account_id" {
  description = "Resource ID used to verify the state storage account and delete lock."
  value       = azurerm_storage_account.terraform_state.id
}

output "operator_principal_id" {
  description = "Object ID receiving Storage Blob Data Contributor access."
  value       = data.azurerm_client_config.current.object_id
}
