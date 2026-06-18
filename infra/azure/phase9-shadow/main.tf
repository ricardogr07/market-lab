locals {
  required_tags = {
    "cost-center" = var.cost_center
    "managed-by"  = "terraform"
    "owner"       = var.owner
    "phase"       = "9"
    "project"     = "marketlab"
    "purpose"     = "btc-shadow"
    "workload"    = "phase9-shadow"
  }

  tags = merge(var.additional_tags, local.required_tags)

  container_image = "${azurerm_container_registry.shadow.login_server}/${var.container_repository}@${var.marketlab_image_digest}"
}

resource "azurerm_resource_group" "shadow" {
  name     = "rg-marketlab-phase9-shadow-${var.resource_suffix}"
  location = var.location
  tags     = local.tags
}

resource "azurerm_user_assigned_identity" "shadow" {
  name                = "id-ml-p9-shadow-${var.resource_suffix}"
  resource_group_name = azurerm_resource_group.shadow.name
  location            = azurerm_resource_group.shadow.location
  tags                = local.tags
}

resource "azurerm_container_registry" "shadow" {
  name                          = "mlp9sh${var.resource_suffix}acr"
  resource_group_name           = azurerm_resource_group.shadow.name
  location                      = azurerm_resource_group.shadow.location
  sku                           = "Basic"
  admin_enabled                 = false
  anonymous_pull_enabled        = false
  public_network_access_enabled = true
  export_policy_enabled         = false
  quarantine_policy_enabled     = false
  trust_policy_enabled          = false
  tags                          = local.tags
}

resource "azurerm_log_analytics_workspace" "shadow" {
  name                       = "log-ml-p9-shadow-${var.resource_suffix}"
  resource_group_name        = azurerm_resource_group.shadow.name
  location                   = azurerm_resource_group.shadow.location
  sku                        = "PerGB2018"
  retention_in_days          = 90
  daily_quota_gb             = 1
  internet_ingestion_enabled = true
  internet_query_enabled     = true
  tags                       = local.tags
}

resource "azurerm_application_insights" "shadow" {
  name                          = "appi-ml-p9-shadow-${var.resource_suffix}"
  resource_group_name           = azurerm_resource_group.shadow.name
  location                      = azurerm_resource_group.shadow.location
  application_type              = "other"
  workspace_id                  = azurerm_log_analytics_workspace.shadow.id
  local_authentication_disabled = true
  internet_ingestion_enabled    = true
  internet_query_enabled        = true
  tags                          = local.tags
}

resource "azurerm_container_app_environment" "shadow" {
  name                       = "cae-ml-p9-shadow-${var.resource_suffix}"
  resource_group_name        = azurerm_resource_group.shadow.name
  location                   = azurerm_resource_group.shadow.location
  log_analytics_workspace_id = azurerm_log_analytics_workspace.shadow.id
  public_network_access      = "Enabled"
  tags                       = local.tags
}

resource "azurerm_storage_account" "shadow" {
  name                            = "mlp9sh${var.resource_suffix}"
  resource_group_name             = azurerm_resource_group.shadow.name
  location                        = azurerm_resource_group.shadow.location
  account_tier                    = "Standard"
  account_replication_type        = "ZRS"
  account_kind                    = "StorageV2"
  access_tier                     = "Hot"
  min_tls_version                 = "TLS1_2"
  https_traffic_only_enabled      = true
  allow_nested_items_to_be_public = false
  shared_access_key_enabled       = true
  default_to_oauth_authentication = true
  public_network_access_enabled   = true
  local_user_enabled              = false

  blob_properties {
    versioning_enabled       = true
    last_access_time_enabled = true

    delete_retention_policy {
      days = 30
    }

    container_delete_retention_policy {
      days = 30
    }
  }

  share_properties {
    retention_policy {
      days = 30
    }

    smb {
      versions                        = ["SMB3.1.1"]
      authentication_types            = ["NTLMv2"]
      kerberos_ticket_encryption_type = ["AES-256"]
      channel_encryption_type         = ["AES-256-GCM"]
    }
  }

  tags = local.tags
}

resource "azurerm_storage_share" "live_artifacts" {
  name               = "phase9-shadow-live"
  storage_account_id = azurerm_storage_account.shadow.id
  quota              = var.live_share_quota_gb
  access_tier        = "Hot"
}

resource "azurerm_storage_container" "archive" {
  name                  = "phase9-shadow-archive"
  storage_account_id    = azurerm_storage_account.shadow.id
  container_access_type = "private"
}

resource "azurerm_storage_management_policy" "archive_retention" {
  storage_account_id = azurerm_storage_account.shadow.id

  rule {
    name    = "retain-shadow-archive"
    enabled = true

    filters {
      blob_types   = ["blockBlob"]
      prefix_match = ["${azurerm_storage_container.archive.name}/snapshots/"]
    }

    actions {
      base_blob {
        delete_after_days_since_creation_greater_than = var.archive_retention_days
      }

      snapshot {
        delete_after_days_since_creation_greater_than = var.archive_retention_days
      }

      version {
        delete_after_days_since_creation = var.archive_retention_days
      }
    }
  }

  depends_on = [azurerm_storage_container.archive]
}

resource "azurerm_container_app_environment_storage" "live_artifacts" {
  name                         = "phase9-shadow-live"
  container_app_environment_id = azurerm_container_app_environment.shadow.id
  account_name                 = azurerm_storage_account.shadow.name
  share_name                   = azurerm_storage_share.live_artifacts.name
  access_key                   = azurerm_storage_account.shadow.primary_access_key
  access_mode                  = "ReadWrite"
}

resource "azurerm_role_assignment" "shadow_acr_pull" {
  scope                = azurerm_container_registry.shadow.id
  role_definition_name = "AcrPull"
  principal_id         = azurerm_user_assigned_identity.shadow.principal_id
  principal_type       = "ServicePrincipal"
}

resource "azurerm_role_assignment" "shadow_blob_archive" {
  scope                = azurerm_storage_account.shadow.id
  role_definition_name = "Storage Blob Data Contributor"
  principal_id         = azurerm_user_assigned_identity.shadow.principal_id
  principal_type       = "ServicePrincipal"
}

resource "azurerm_role_assignment" "shadow_file_live_artifacts" {
  scope                = azurerm_storage_account.shadow.id
  role_definition_name = "Storage File Data SMB Share Contributor"
  principal_id         = azurerm_user_assigned_identity.shadow.principal_id
  principal_type       = "ServicePrincipal"
}

resource "azurerm_container_app_job" "shadow_scheduler" {
  name                         = "caj-ml-p9-shadow-${var.resource_suffix}"
  resource_group_name          = azurerm_resource_group.shadow.name
  location                     = azurerm_resource_group.shadow.location
  container_app_environment_id = azurerm_container_app_environment.shadow.id
  replica_timeout_in_seconds   = 3600
  replica_retry_limit          = 0
  tags                         = local.tags

  identity {
    type         = "UserAssigned"
    identity_ids = [azurerm_user_assigned_identity.shadow.id]
  }

  registry {
    server   = azurerm_container_registry.shadow.login_server
    identity = azurerm_user_assigned_identity.shadow.id
  }

  dynamic "manual_trigger_config" {
    for_each = var.enable_shadow_schedule ? [] : [1]

    content {
      parallelism              = 1
      replica_completion_count = 1
    }
  }

  dynamic "schedule_trigger_config" {
    for_each = var.enable_shadow_schedule ? [1] : []

    content {
      cron_expression          = var.shadow_cron_expression
      parallelism              = 1
      replica_completion_count = 1
    }
  }

  template {
    container {
      name   = "phase9-shadow-scheduler"
      image  = local.container_image
      cpu    = 0.5
      memory = "1Gi"
      args = [
        "phase9-shadow-scheduler",
        "--config",
        var.shadow_config_path,
        "--once",
      ]

      env {
        name  = "MARKETLAB_DEPLOYMENT_ID"
        value = "phase9-shadow"
      }

      env {
        name  = "MARKETLAB_SHADOW_LAUNCH_GATE"
        value = var.enable_shadow_schedule ? "enabled" : "disabled"
      }

      env {
        name  = "APPLICATIONINSIGHTS_CONNECTION_STRING"
        value = azurerm_application_insights.shadow.connection_string
      }

      volume_mounts {
        name = "live-artifacts"
        path = "/app/artifacts"
      }
    }

    volume {
      name         = "live-artifacts"
      storage_name = azurerm_container_app_environment_storage.live_artifacts.name
      storage_type = "AzureFile"
    }
  }

  depends_on = [
    azurerm_role_assignment.shadow_acr_pull,
    azurerm_role_assignment.shadow_blob_archive,
    azurerm_role_assignment.shadow_file_live_artifacts,
  ]
}

resource "azurerm_monitor_action_group" "shadow" {
  name                = "ag-ml-p9-shadow-${var.resource_suffix}"
  resource_group_name = azurerm_resource_group.shadow.name
  short_name          = "p9shadow"
  enabled             = var.enable_shadow_alerts
  tags                = local.tags

  email_receiver {
    name                    = "phase9-shadow-operator"
    email_address           = var.alert_email
    use_common_alert_schema = true
  }
}

resource "azurerm_monitor_scheduled_query_rules_alert_v2" "job_failures" {
  name                  = "alert-ml-p9-shadow-job-failures-${var.resource_suffix}"
  resource_group_name   = azurerm_resource_group.shadow.name
  location              = azurerm_resource_group.shadow.location
  display_name          = "MarketLab Phase 9 shadow job failures"
  description           = "Alerts only on BTC shadow Container Apps job failure evidence."
  enabled               = var.enable_shadow_alerts
  scopes                = [azurerm_log_analytics_workspace.shadow.id]
  severity              = 2
  evaluation_frequency  = "PT15M"
  window_duration       = "PT15M"
  skip_query_validation = true
  tags                  = local.tags

  criteria {
    query                   = <<-KQL
      ContainerAppSystemLogs_CL
      | where TimeGenerated > ago(15m)
      | where JobName_s == "${azurerm_container_app_job.shadow_scheduler.name}"
          or ContainerAppName_s == "${azurerm_container_app_job.shadow_scheduler.name}"
      | where Reason_s has_any ("Failed", "Error")
          or Log_s has_any ("failed", "error", "exception")
    KQL
    time_aggregation_method = "Count"
    operator                = "GreaterThan"
    threshold               = 0

    failing_periods {
      minimum_failing_periods_to_trigger_alert = 1
      number_of_evaluation_periods             = 1
    }
  }

  action {
    action_groups = [azurerm_monitor_action_group.shadow.id]
    email_subject = "MarketLab Phase 9 BTC shadow job failure"
  }
}

resource "azurerm_monitor_scheduled_query_rules_alert_v2" "missing_evidence" {
  name                  = "alert-ml-p9-shadow-missing-evidence-${var.resource_suffix}"
  resource_group_name   = azurerm_resource_group.shadow.name
  location              = azurerm_resource_group.shadow.location
  display_name          = "MarketLab Phase 9 shadow missing evidence"
  description           = "Alerts only when the scheduled BTC shadow lane misses expected decision evidence."
  enabled               = var.enable_shadow_schedule && var.enable_shadow_alerts
  scopes                = [azurerm_log_analytics_workspace.shadow.id]
  severity              = 2
  evaluation_frequency  = "PT1H"
  window_duration       = "P1D"
  skip_query_validation = true
  tags                  = local.tags

  criteria {
    query                   = <<-KQL
      ContainerAppConsoleLogs_CL
      | where TimeGenerated > ago(26h)
      | where ContainerAppName_s == "${azurerm_container_app_job.shadow_scheduler.name}"
      | where Log_s has_any ("decision_evidence", "label_evidence", "state/status.json")
    KQL
    time_aggregation_method = "Count"
    operator                = "LessThan"
    threshold               = 1

    failing_periods {
      minimum_failing_periods_to_trigger_alert = 1
      number_of_evaluation_periods             = 1
    }
  }

  action {
    action_groups = [azurerm_monitor_action_group.shadow.id]
    email_subject = "MarketLab Phase 9 BTC shadow evidence missing"
  }
}
