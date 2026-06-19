variable "resource_suffix" {
  description = "Globally unique lowercase alphanumeric suffix used by Azure resources."
  type        = string

  validation {
    condition     = can(regex("^[a-z0-9]{4,8}$", var.resource_suffix))
    error_message = "resource_suffix must contain 4-8 lowercase alphanumeric characters."
  }
}

variable "location" {
  description = "Azure region for the BTC shadow resources."
  type        = string
  default     = "eastus2"

  validation {
    condition     = can(regex("^[a-z0-9]+$", var.location))
    error_message = "location must be an Azure region name containing lowercase letters and numbers."
  }
}

variable "owner" {
  description = "User-approved owner tag for the BTC shadow resources."
  type        = string

  validation {
    condition     = length(trimspace(var.owner)) > 0
    error_message = "owner must not be empty."
  }
}

variable "cost_center" {
  description = "User-approved cost-center tag for the BTC shadow resources."
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
  description = "Immutable MarketLab image digest already published to the shadow ACR."
  type        = string

  validation {
    condition     = can(regex("^sha256:[0-9a-f]{64}$", var.marketlab_image_digest))
    error_message = "marketlab_image_digest must be an immutable sha256 digest."
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

variable "shadow_config_path" {
  description = "Container-local path to the frozen BTC shadow config."
  type        = string
  default     = "/app/configs/experiment.btc_phase9_shadow_daily.yaml"

  validation {
    condition     = var.shadow_config_path == "/app/configs/experiment.btc_phase9_shadow_daily.yaml"
    error_message = "shadow_config_path must remain on the frozen BTC Phase 9 shadow config."
  }
}

variable "shadow_cron_expression" {
  description = "UTC cron expression for the BTC shadow scheduled job."
  type        = string
  default     = "15 1 * * *"

  validation {
    condition     = var.shadow_cron_expression == "15 1 * * *"
    error_message = "shadow_cron_expression must remain the approved 01:15 UTC schedule."
  }
}

variable "enable_shadow_schedule" {
  description = "Explicit launch-gate switch. False creates only a manual-trigger job."
  type        = bool
  default     = false

  validation {
    condition     = !var.enable_shadow_schedule || var.create_shadow_job
    error_message = "enable_shadow_schedule requires create_shadow_job because the scheduled trigger belongs to the Container Apps Job."
  }
}

variable "create_shadow_job" {
  description = "Explicit job-creation switch. Keep false until the reviewed immutable image digest exists in the shadow ACR."
  type        = bool
  default     = false
}

variable "enable_shadow_alerts" {
  description = "Explicit alert switch used during supervised alert testing and launch."
  type        = bool
  default     = false
}

variable "launch_gate_evidence_uri" {
  description = "Reviewed evidence URI required before the scheduled job can be enabled."
  type        = string
  default     = ""

  validation {
    condition = (
      !var.enable_shadow_schedule
      || can(regex("^https://", var.launch_gate_evidence_uri))
    )
    error_message = "launch_gate_evidence_uri must be an https URI before enable_shadow_schedule can be true."
  }
}

variable "archive_retention_days" {
  description = "Minimum lifecycle retention for Blob archive snapshots and versions."
  type        = number
  default     = 450

  validation {
    condition     = var.archive_retention_days >= 450
    error_message = "archive_retention_days must preserve at least 450 days of BTC shadow evidence."
  }
}

variable "live_share_quota_gb" {
  description = "Quota for the Azure Files live artifact share."
  type        = number
  default     = 16

  validation {
    condition     = var.live_share_quota_gb >= 16 && var.live_share_quota_gb <= 5120
    error_message = "live_share_quota_gb must be between 16 and 5120."
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
