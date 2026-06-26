output "resource_group_name" {
  description = "Resource group containing the disabled QQQ paper dev workload."
  value       = azurerm_resource_group.qqq.name
}

output "managed_identity_id" {
  description = "Managed identity assigned to QQQ paper Container Apps Jobs."
  value       = azurerm_user_assigned_identity.qqq.id
}

output "container_registry_login_server" {
  description = "ACR login server used with managed identity pull."
  value       = azurerm_container_registry.qqq.login_server
}

output "container_app_jobs_created" {
  description = "Whether Container Apps Job resources are included in this apply."
  value       = var.create_jobs
}

output "scheduler_schedule_enabled" {
  description = "Launch-gate state for the scheduler trigger."
  value       = var.enable_scheduler_schedule
}

output "service_bus_approval_trigger_enabled" {
  description = "Reserved launch-gate state for the Service Bus approval trigger."
  value       = var.enable_service_bus_approval_trigger
}

output "artifact_container_name" {
  description = "Private Blob container for QQQ paper artifacts."
  value       = azurerm_storage_container.artifacts.name
}

output "service_bus_queue_name" {
  description = "Service Bus queue for QQQ paper outbox and approval events."
  value       = azurerm_servicebus_queue.paper_events.name
}

output "postgres_database_name" {
  description = "PostgreSQL database name for QQQ paper control state."
  value       = azurerm_postgresql_flexible_server_database.qqq.name
}

output "key_vault_uri" {
  description = "Key Vault URI used by hosted runtime configuration."
  value       = azurerm_key_vault.qqq.vault_uri
}
