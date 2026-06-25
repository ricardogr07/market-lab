variable "resource_suffix" {
  description = "Globally unique lowercase alphanumeric suffix used by Azure resources."
  type        = string

  validation {
    condition     = can(regex("^[a-z0-9]{4,8}$", var.resource_suffix))
    error_message = "resource_suffix must contain 4-8 lowercase alphanumeric characters."
  }
}

variable "environment" {
  description = "Phase 9 QQQ paper environment. P9-10 creates only the dev source contract."
  type        = string
  default     = "dev"

  validation {
    condition     = var.environment == "dev"
    error_message = "P9-10 only supports the dev QQQ paper environment."
  }
}

variable "location" {
  description = "Azure region for the QQQ paper dev resources."
  type        = string
  default     = "eastus2"

  validation {
    condition     = can(regex("^[a-z0-9]+$", var.location))
    error_message = "location must be an Azure region name containing lowercase letters and numbers."
  }
}

variable "owner" {
  description = "User-approved owner tag for QQQ paper resources."
  type        = string

  validation {
    condition     = length(trimspace(var.owner)) > 0
    error_message = "owner must not be empty."
  }
}

variable "cost_center" {
  description = "User-approved cost-center tag for QQQ paper resources."
  type        = string

  validation {
    condition     = length(trimspace(var.cost_center)) > 0
    error_message = "cost_center must not be empty."
  }
}

variable "alert_email" {
  description = "User-approved operator email for Azure Monitor action-group receivers."
  type        = string

  validation {
    condition     = can(regex("^[^@\\s]+@[^@\\s]+\\.[^@\\s]+$", var.alert_email))
    error_message = "alert_email must be an email address approved for Phase 9 alerting."
  }
}

variable "marketlab_image_digest" {
  description = "Immutable MarketLab image digest already published to the QQQ paper ACR."
  type        = string
  default     = "sha256:0000000000000000000000000000000000000000000000000000000000000000"

  validation {
    condition = (
      !var.create_jobs
      || can(regex("^sha256:[0-9a-f]{64}$", var.marketlab_image_digest))
    )
    error_message = "marketlab_image_digest must be an immutable sha256 digest when create_jobs is true."
  }
}

variable "container_repository" {
  description = "ACR repository containing the MarketLab image."
  type        = string
  default     = "marketlab"

  validation {
    condition     = can(regex("^[a-z0-9][a-z0-9._/-]{0,127}$", var.container_repository))
    error_message = "container_repository must be a valid lowercase ACR repository name."
  }
}

variable "qqq_config_path" {
  description = "Container-local path to the canonical QQQ paper config."
  type        = string
  default     = "/app/configs/experiment.qqq_paper_daily.yaml"

  validation {
    condition     = var.qqq_config_path == "/app/configs/experiment.qqq_paper_daily.yaml"
    error_message = "qqq_config_path must remain on the canonical QQQ paper config."
  }
}

variable "scheduler_cron_expression" {
  description = "Frequent UTC tick; MarketLab preserves New York due-time enforcement internally."
  type        = string
  default     = "*/5 * * * *"

  validation {
    condition     = var.scheduler_cron_expression == "*/5 * * * *"
    error_message = "scheduler_cron_expression must remain the approved five-minute UTC tick."
  }
}

variable "create_jobs" {
  description = "Explicit job-creation switch. Keep false until the reviewed immutable image digest exists in the QQQ ACR."
  type        = bool
  default     = false
}

variable "enable_scheduler_schedule" {
  description = "Explicit launch-gate switch. False creates only a manual-trigger scheduler job."
  type        = bool
  default     = false

  validation {
    condition     = !var.enable_scheduler_schedule || var.create_jobs
    error_message = "enable_scheduler_schedule requires create_jobs because the scheduled trigger belongs to the Container Apps Job."
  }
}

variable "enable_service_bus_approval_trigger" {
  description = "Reserved approval-trigger switch. P9-10 keeps Service Bus-triggered jobs disabled."
  type        = bool
  default     = false

  validation {
    condition     = !var.enable_service_bus_approval_trigger
    error_message = "enable_service_bus_approval_trigger remains false in P9-10; P9-12 owns trigger activation evidence."
  }
}

variable "enable_broker_secret_refs" {
  description = "Explicit switch for broker/provider/notification secret references. Keep false for initial dev proof."
  type        = bool
  default     = false
}

variable "postgres_dsn_secret_id" {
  description = "Key Vault secret ID containing MARKETLAB_PAPER_POSTGRES_DSN for job execution."
  type        = string
  default     = ""
  sensitive   = true

  validation {
    condition     = !var.create_jobs || can(regex("^https://", var.postgres_dsn_secret_id))
    error_message = "postgres_dsn_secret_id must be an https Key Vault secret ID when create_jobs is true."
  }
}

variable "alpaca_key_id_secret_id" {
  description = "Optional Key Vault secret ID for ALPACA_API_KEY_ID, used only when enable_broker_secret_refs is true."
  type        = string
  default     = ""
  sensitive   = true
}

variable "alpaca_secret_key_secret_id" {
  description = "Optional Key Vault secret ID for ALPACA_API_SECRET_KEY, used only when enable_broker_secret_refs is true."
  type        = string
  default     = ""
  sensitive   = true
}

variable "anthropic_api_key_secret_id" {
  description = "Optional Key Vault secret ID for ANTHROPIC_API_KEY, used only when enable_broker_secret_refs is true."
  type        = string
  default     = ""
  sensitive   = true
}

variable "telegram_bot_token_secret_id" {
  description = "Optional Key Vault secret ID for TELEGRAM_BOT_TOKEN, used only when enable_broker_secret_refs is true."
  type        = string
  default     = ""
  sensitive   = true
}

variable "telegram_chat_id_secret_id" {
  description = "Optional Key Vault secret ID for TELEGRAM_CHAT_ID, used only when enable_broker_secret_refs is true."
  type        = string
  default     = ""
  sensitive   = true
}

variable "postgres_admin_login" {
  description = "PostgreSQL administrator login retained outside tracked tfvars."
  type        = string
  default     = "marketlabadmin"

  validation {
    condition     = can(regex("^[a-zA-Z][a-zA-Z0-9_]{2,31}$", var.postgres_admin_login))
    error_message = "postgres_admin_login must be a valid PostgreSQL administrator login."
  }
}

variable "postgres_admin_password" {
  description = "PostgreSQL administrator password retained outside tracked tfvars."
  type        = string
  sensitive   = true

  validation {
    condition     = length(var.postgres_admin_password) >= 16
    error_message = "postgres_admin_password must be at least 16 characters."
  }
}

variable "postgres_sku_name" {
  description = "PostgreSQL Flexible Server SKU for dev."
  type        = string
  default     = "B_Standard_B1ms"
}

variable "postgres_storage_mb" {
  description = "PostgreSQL storage in MB for dev."
  type        = number
  default     = 32768

  validation {
    condition     = var.postgres_storage_mb >= 32768
    error_message = "postgres_storage_mb must be at least 32768."
  }
}

variable "artifact_retention_days" {
  description = "Minimum lifecycle retention for QQQ paper Blob artifacts."
  type        = number
  default     = 90

  validation {
    condition     = var.artifact_retention_days >= 90
    error_message = "artifact_retention_days must preserve at least 90 days of dev paper artifacts."
  }
}

variable "additional_tags" {
  description = "Optional tags. Required MarketLab tags cannot be overridden."
  type        = map(string)
  default     = {}

  validation {
    condition = alltrue([
      for key in keys(var.additional_tags) : !contains(
        [
          "cost-center",
          "environment",
          "managed-by",
          "owner",
          "phase",
          "project",
          "purpose",
          "workload",
        ],
        lower(key),
      )
    ])
    error_message = "additional_tags cannot redefine a required MarketLab tag."
  }
}
