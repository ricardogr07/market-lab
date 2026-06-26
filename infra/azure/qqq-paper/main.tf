data "azurerm_client_config" "current" {}

locals {
  deployment_id       = "qqq-paper-${var.environment}"
  resource_name_env   = var.environment == "paper-prod" ? "prod" : var.environment
  key_vault_name      = "kv-ml-qqq-${local.resource_name_env}-${var.resource_suffix}"
  key_vault_name_safe = length(local.key_vault_name) <= 24

  required_tags = {
    "cost-center" = var.cost_center
    "environment" = var.environment
    "managed-by"  = "terraform"
    "owner"       = var.owner
    "phase"       = "9"
    "project"     = "marketlab"
    "purpose"     = "qqq-paper"
    "workload"    = "qqq-paper"
  }

  tags = merge(var.additional_tags, local.required_tags)

  container_image = "${azurerm_container_registry.qqq.login_server}/${var.container_repository}@${var.marketlab_image_digest}"

  common_env = {
    APPLICATIONINSIGHTS_CONNECTION_STRING        = azurerm_application_insights.qqq.connection_string
    AZURE_CLIENT_ID                              = azurerm_user_assigned_identity.qqq.client_id
    MARKETLAB_CONFIG_VERSION                     = local.deployment_id
    MARKETLAB_DEPLOYMENT_ID                      = local.deployment_id
    MARKETLAB_ENVIRONMENT                        = var.environment
    MARKETLAB_IMAGE_DIGEST                       = var.marketlab_image_digest
    MARKETLAB_PAPER_AZURE_ARTIFACT_BACKEND       = "azure_blob"
    MARKETLAB_PAPER_AZURE_ARTIFACT_DEPLOYMENT_ID = local.deployment_id
    MARKETLAB_PAPER_AZURE_ARTIFACT_ENVIRONMENT   = var.environment
    MARKETLAB_PAPER_AZURE_BLOB_ACCOUNT_URL       = azurerm_storage_account.qqq.primary_blob_endpoint
    MARKETLAB_PAPER_AZURE_BLOB_CONTAINER_NAME    = azurerm_storage_container.artifacts.name
    MARKETLAB_PAPER_AZURE_KEY_VAULT_URL          = azurerm_key_vault.qqq.vault_uri
    MARKETLAB_PAPER_AZURE_SECRET_BACKEND         = "key_vault"
    MARKETLAB_PAPER_AZURE_SERVICE_BUS_BACKEND    = "azure_service_bus"
    MARKETLAB_PAPER_AZURE_SERVICE_BUS_NAMESPACE  = "${azurerm_servicebus_namespace.qqq.name}.servicebus.windows.net"
    MARKETLAB_PAPER_AZURE_SERVICE_BUS_QUEUE_NAME = azurerm_servicebus_queue.paper_events.name
    MARKETLAB_PAPER_PERSISTENCE_BACKEND          = "postgres"
    MARKETLAB_PAPER_RUNTIME_ENV_OVERRIDES        = "1"
    MARKETLAB_PAPER_TELEGRAM_ALLOWED_EXPERIMENTS = "qqq_paper_daily"
    MARKETLAB_PAPER_TELEGRAM_ENABLED             = "false"
    MARKETLAB_TRIGGER_SOURCE                     = "azure-container-apps-job"
  }

  phase_env = {
    paper-db-migrate          = "decision"
    paper-scheduler           = "decision"
    paper-agent-approve       = "agent_approve"
    paper-outbox-deliver      = "agent_approve"
    paper-notifications       = "reconcile"
    paper-blob-sync           = "reconcile"
    paper-service-bus-receive = "agent_approve"
  }

  jobs = {
    paper-db-migrate = {
      name = "migrate"
      args = [
        "paper-db-migrate",
        "--config",
        var.qqq_config_path,
      ]
    }
    paper-scheduler = {
      name = "scheduler"
      args = [
        "paper-scheduler",
        "--config",
        var.qqq_config_path,
        "--once",
      ]
    }
    paper-agent-approve = {
      name = "agent-approve"
      args = [
        "paper-agent-approve",
        "--config",
        var.qqq_config_path,
        "--once",
      ]
    }
    paper-outbox-deliver = {
      name = "outbox"
      args = [
        "paper-outbox-deliver",
        "--config",
        var.qqq_config_path,
      ]
    }
    paper-notifications = {
      name = "notifications"
      args = [
        "paper-notifications-deliver",
        "--config",
        var.qqq_config_path,
      ]
    }
    paper-blob-sync = {
      name = "blob-sync"
      args = [
        "paper-blob-sync",
        "--config",
        var.qqq_config_path,
      ]
    }
    paper-service-bus-receive = {
      name = "service-bus-receive"
      args = [
        "paper-service-bus-receive",
        "--config",
        var.qqq_config_path,
        "--max-messages",
        "10",
        "--max-wait-seconds",
        "5",
      ]
    }
  }

  required_secret_refs = var.create_jobs ? {
    MARKETLAB_PAPER_POSTGRES_DSN = {
      secret_name         = "postgres-dsn"
      key_vault_secret_id = var.postgres_dsn_secret_id
    }
  } : {}

  broker_secret_refs = var.create_jobs && var.enable_broker_secret_refs ? {
    ALPACA_API_KEY_ID = {
      secret_name         = "alpaca-api-key-id"
      key_vault_secret_id = var.alpaca_key_id_secret_id
    }
    ALPACA_API_SECRET_KEY = {
      secret_name         = "alpaca-api-secret-key"
      key_vault_secret_id = var.alpaca_secret_key_secret_id
    }
    ANTHROPIC_API_KEY = {
      secret_name         = "anthropic-api-key"
      key_vault_secret_id = var.anthropic_api_key_secret_id
    }
    TELEGRAM_BOT_TOKEN = {
      secret_name         = "telegram-bot-token"
      key_vault_secret_id = var.telegram_bot_token_secret_id
    }
    TELEGRAM_CHAT_ID = {
      secret_name         = "telegram-chat-id"
      key_vault_secret_id = var.telegram_chat_id_secret_id
    }
  } : {}

  secret_refs = merge(local.required_secret_refs, local.broker_secret_refs)
}

resource "azurerm_resource_group" "qqq" {
  name     = "rg-marketlab-qqq-paper-${var.environment}-${var.resource_suffix}"
  location = var.location
  tags     = local.tags
}

resource "azurerm_user_assigned_identity" "qqq" {
  name                = "id-ml-qqq-paper-${var.environment}-${var.resource_suffix}"
  resource_group_name = azurerm_resource_group.qqq.name
  location            = azurerm_resource_group.qqq.location
  tags                = local.tags
}

resource "azurerm_container_registry" "qqq" {
  name                          = "mlp9qqq${var.resource_suffix}acr"
  resource_group_name           = azurerm_resource_group.qqq.name
  location                      = azurerm_resource_group.qqq.location
  sku                           = "Basic"
  admin_enabled                 = false
  anonymous_pull_enabled        = false
  public_network_access_enabled = true
  export_policy_enabled         = true
  quarantine_policy_enabled     = false
  trust_policy_enabled          = false
  tags                          = local.tags
}

resource "azurerm_log_analytics_workspace" "qqq" {
  name                       = "log-ml-qqq-paper-${var.environment}-${var.resource_suffix}"
  resource_group_name        = azurerm_resource_group.qqq.name
  location                   = azurerm_resource_group.qqq.location
  sku                        = "PerGB2018"
  retention_in_days          = 90
  daily_quota_gb             = 1
  internet_ingestion_enabled = true
  internet_query_enabled     = true
  tags                       = local.tags
}

resource "azurerm_application_insights" "qqq" {
  name                          = "appi-ml-qqq-paper-${var.environment}-${var.resource_suffix}"
  resource_group_name           = azurerm_resource_group.qqq.name
  location                      = azurerm_resource_group.qqq.location
  application_type              = "other"
  workspace_id                  = azurerm_log_analytics_workspace.qqq.id
  local_authentication_disabled = true
  internet_ingestion_enabled    = true
  internet_query_enabled        = true
  tags                          = local.tags
}

resource "azurerm_container_app_environment" "qqq" {
  name                       = "cae-ml-qqq-paper-${var.environment}-${var.resource_suffix}"
  resource_group_name        = azurerm_resource_group.qqq.name
  location                   = azurerm_resource_group.qqq.location
  log_analytics_workspace_id = azurerm_log_analytics_workspace.qqq.id
  public_network_access      = "Enabled"
  tags                       = local.tags
}

resource "azurerm_storage_account" "qqq" {
  name                            = "mlp9qqq${var.resource_suffix}"
  resource_group_name             = azurerm_resource_group.qqq.name
  location                        = azurerm_resource_group.qqq.location
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

resource "azurerm_storage_share" "paper_state" {
  name               = "qqq-paper-state"
  storage_account_id = azurerm_storage_account.qqq.id
  quota              = var.paper_state_share_quota_gb
  access_tier        = "Hot"
}

resource "azurerm_storage_container" "artifacts" {
  name                  = "qqq-paper-artifacts"
  storage_account_id    = azurerm_storage_account.qqq.id
  container_access_type = "private"
}

resource "azurerm_storage_management_policy" "artifact_retention" {
  storage_account_id = azurerm_storage_account.qqq.id

  rule {
    name    = "retain-qqq-paper-artifacts"
    enabled = true

    filters {
      blob_types   = ["blockBlob"]
      prefix_match = ["${azurerm_storage_container.artifacts.name}/paper/${var.environment}/${local.deployment_id}/"]
    }

    actions {
      base_blob {
        delete_after_days_since_creation_greater_than = var.artifact_retention_days
      }

      snapshot {
        delete_after_days_since_creation_greater_than = var.artifact_retention_days
      }

      version {
        delete_after_days_since_creation = var.artifact_retention_days
      }
    }
  }

  depends_on = [azurerm_storage_container.artifacts]
}

resource "azurerm_container_app_environment_storage" "paper_state" {
  name                         = "qqq-paper-state"
  container_app_environment_id = azurerm_container_app_environment.qqq.id
  account_name                 = azurerm_storage_account.qqq.name
  share_name                   = azurerm_storage_share.paper_state.name
  access_key                   = azurerm_storage_account.qqq.primary_access_key
  access_mode                  = "ReadWrite"
}

resource "azurerm_servicebus_namespace" "qqq" {
  name                = "sb-ml-qqq-paper-${var.environment}-${var.resource_suffix}"
  resource_group_name = azurerm_resource_group.qqq.name
  location            = azurerm_resource_group.qqq.location
  sku                 = "Standard"
  minimum_tls_version = "1.2"
  local_auth_enabled  = false
  tags                = local.tags
}

resource "azurerm_servicebus_queue" "paper_events" {
  name                                    = "qqq-paper-events"
  namespace_id                            = azurerm_servicebus_namespace.qqq.id
  requires_duplicate_detection            = true
  duplicate_detection_history_time_window = "PT24H"
  dead_lettering_on_message_expiration    = true
  max_delivery_count                      = 10
}

resource "azurerm_key_vault" "qqq" {
  name                          = local.key_vault_name
  resource_group_name           = azurerm_resource_group.qqq.name
  location                      = azurerm_resource_group.qqq.location
  tenant_id                     = data.azurerm_client_config.current.tenant_id
  sku_name                      = "standard"
  rbac_authorization_enabled    = true
  purge_protection_enabled      = false
  soft_delete_retention_days    = 30
  public_network_access_enabled = true
  tags                          = local.tags

  lifecycle {
    precondition {
      condition     = local.key_vault_name_safe
      error_message = "QQQ paper Key Vault name must be 24 characters or fewer."
    }
  }
}

resource "azurerm_postgresql_flexible_server" "qqq" {
  name                          = "psql-ml-qqq-paper-${var.environment}-${var.resource_suffix}"
  resource_group_name           = azurerm_resource_group.qqq.name
  location                      = azurerm_resource_group.qqq.location
  version                       = "16"
  administrator_login           = var.postgres_admin_login
  administrator_password        = var.postgres_admin_password
  sku_name                      = var.postgres_sku_name
  storage_mb                    = var.postgres_storage_mb
  backup_retention_days         = 7
  geo_redundant_backup_enabled  = false
  public_network_access_enabled = true
  tags                          = local.tags
}

resource "azurerm_postgresql_flexible_server_database" "qqq" {
  name      = "marketlab_qqq_paper"
  server_id = azurerm_postgresql_flexible_server.qqq.id
  charset   = "UTF8"
  collation = "en_US.utf8"
}

resource "azurerm_postgresql_flexible_server_firewall_rule" "job_access" {
  for_each = var.postgres_firewall_rules

  name             = "allow-${each.key}"
  server_id        = azurerm_postgresql_flexible_server.qqq.id
  start_ip_address = each.value.start_ip_address
  end_ip_address   = each.value.end_ip_address
}

resource "azurerm_role_assignment" "qqq_acr_pull" {
  scope                = azurerm_container_registry.qqq.id
  role_definition_name = "AcrPull"
  principal_id         = azurerm_user_assigned_identity.qqq.principal_id
  principal_type       = "ServicePrincipal"
}

resource "azurerm_role_assignment" "qqq_blob_artifacts" {
  scope                = azurerm_storage_account.qqq.id
  role_definition_name = "Storage Blob Data Contributor"
  principal_id         = azurerm_user_assigned_identity.qqq.principal_id
  principal_type       = "ServicePrincipal"
}

resource "azurerm_role_assignment" "qqq_servicebus_sender" {
  scope                = azurerm_servicebus_namespace.qqq.id
  role_definition_name = "Azure Service Bus Data Sender"
  principal_id         = azurerm_user_assigned_identity.qqq.principal_id
  principal_type       = "ServicePrincipal"
}

resource "azurerm_role_assignment" "qqq_servicebus_receiver" {
  scope                = azurerm_servicebus_namespace.qqq.id
  role_definition_name = "Azure Service Bus Data Receiver"
  principal_id         = azurerm_user_assigned_identity.qqq.principal_id
  principal_type       = "ServicePrincipal"
}

resource "azurerm_role_assignment" "qqq_key_vault_secrets" {
  scope                = azurerm_key_vault.qqq.id
  role_definition_name = "Key Vault Secrets User"
  principal_id         = azurerm_user_assigned_identity.qqq.principal_id
  principal_type       = "ServicePrincipal"
}

resource "azurerm_container_app_job" "qqq" {
  for_each = var.create_jobs ? local.jobs : {}

  name                         = "caj-ml-qqq-${var.environment}-${each.value.name}-${var.resource_suffix}"
  resource_group_name          = azurerm_resource_group.qqq.name
  location                     = azurerm_resource_group.qqq.location
  container_app_environment_id = azurerm_container_app_environment.qqq.id
  replica_timeout_in_seconds   = 3600
  replica_retry_limit          = 0
  tags                         = local.tags

  identity {
    type         = "UserAssigned"
    identity_ids = [azurerm_user_assigned_identity.qqq.id]
  }

  registry {
    server   = azurerm_container_registry.qqq.login_server
    identity = azurerm_user_assigned_identity.qqq.id
  }

  dynamic "secret" {
    for_each = local.secret_refs

    content {
      name                = secret.value.secret_name
      key_vault_secret_id = secret.value.key_vault_secret_id
      identity            = azurerm_user_assigned_identity.qqq.id
    }
  }

  dynamic "manual_trigger_config" {
    for_each = (
      (each.key == "paper-scheduler" && var.enable_scheduler_schedule)
      || (each.key == "paper-service-bus-receive" && var.enable_service_bus_approval_trigger)
    ) ? [] : [1]

    content {
      parallelism              = 1
      replica_completion_count = 1
    }
  }

  dynamic "schedule_trigger_config" {
    for_each = each.key == "paper-scheduler" && var.enable_scheduler_schedule ? [1] : []

    content {
      cron_expression          = var.scheduler_cron_expression
      parallelism              = 1
      replica_completion_count = 1
    }
  }

  dynamic "event_trigger_config" {
    for_each = each.key == "paper-service-bus-receive" && var.enable_service_bus_approval_trigger ? [1] : []

    content {
      parallelism              = 1
      replica_completion_count = 1

      scale {
        min_executions              = 0
        max_executions              = 1
        polling_interval_in_seconds = 30

        rules {
          name             = "qqq-paper-approval-queue"
          custom_rule_type = "azure-servicebus"
          identity_id      = azurerm_user_assigned_identity.qqq.id
          metadata = {
            messageCount = "1"
            namespace    = azurerm_servicebus_namespace.qqq.name
            queueName    = azurerm_servicebus_queue.paper_events.name
          }
        }
      }
    }
  }

  template {
    container {
      name   = each.key
      image  = local.container_image
      cpu    = 0.5
      memory = "1Gi"
      args   = each.value.args

      dynamic "env" {
        for_each = local.common_env

        content {
          name  = env.key
          value = env.value
        }
      }

      env {
        name  = "MARKETLAB_PHASE"
        value = local.phase_env[each.key]
      }

      env {
        name  = "MARKETLAB_REQUESTED_AT"
        value = "1970-01-01T00:00:00Z"
      }

      env {
        name  = "MARKETLAB_EXECUTION_ID"
        value = "terraform-disabled-template"
      }

      env {
        name  = "MARKETLAB_CORRELATION_ID"
        value = "terraform-disabled-template"
      }

      env {
        name  = "MARKETLAB_IDEMPOTENCY_KEY"
        value = "terraform-disabled-template:${each.key}"
      }

      dynamic "env" {
        for_each = local.secret_refs

        content {
          name        = env.key
          secret_name = env.value.secret_name
        }
      }

      volume_mounts {
        name = "paper-state"
        path = "/app/artifacts/paper/state"
      }
    }

    volume {
      name         = "paper-state"
      storage_name = azurerm_container_app_environment_storage.paper_state.name
      storage_type = "AzureFile"
    }
  }

  depends_on = [
    azurerm_container_app_environment_storage.paper_state,
    azurerm_role_assignment.qqq_acr_pull,
    azurerm_role_assignment.qqq_blob_artifacts,
    azurerm_role_assignment.qqq_key_vault_secrets,
    azurerm_role_assignment.qqq_servicebus_receiver,
    azurerm_role_assignment.qqq_servicebus_sender,
  ]
}

resource "azurerm_monitor_action_group" "qqq" {
  name                = "ag-ml-qqq-paper-${var.environment}-${var.resource_suffix}"
  resource_group_name = azurerm_resource_group.qqq.name
  short_name          = "p9qqq"
  enabled             = false
  tags                = local.tags

  email_receiver {
    name                    = "phase9-qqq-operator"
    email_address           = var.alert_email
    use_common_alert_schema = true
  }
}
