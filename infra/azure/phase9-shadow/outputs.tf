output "resource_group_name" {
  description = "Resource group containing the disabled BTC shadow workload."
  value       = azurerm_resource_group.shadow.name
}

output "managed_identity_id" {
  description = "Managed identity assigned to the Container Apps shadow job."
  value       = azurerm_user_assigned_identity.shadow.id
}

output "container_registry_login_server" {
  description = "ACR login server used with managed identity pull."
  value       = azurerm_container_registry.shadow.login_server
}

output "container_app_job_name" {
  description = "Container Apps Job name for the BTC shadow scheduler."
  value       = "caj-ml-p9-shadow-${var.resource_suffix}"
}

output "container_app_job_created" {
  description = "Whether the Container Apps Job resource is included in this apply."
  value       = var.create_shadow_job
}

output "schedule_enabled" {
  description = "Launch-gate state for the scheduled trigger."
  value       = var.enable_shadow_schedule
}

output "launch_gate_evidence_uri" {
  description = "Reviewed evidence URI required before enabling the schedule."
  value       = var.launch_gate_evidence_uri
}

output "live_artifact_share_name" {
  description = "Azure Files share mounted at /app/artifacts."
  value       = azurerm_storage_share.live_artifacts.name
}

output "archive_container_name" {
  description = "Versioned Blob container for dated BTC shadow archive snapshots."
  value       = azurerm_storage_container.archive.name
}
