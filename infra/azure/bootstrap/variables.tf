variable "resource_suffix" {
  description = "Globally unique lowercase alphanumeric suffix used by Azure resources."
  type        = string

  validation {
    condition     = can(regex("^[a-z0-9]{4,8}$", var.resource_suffix))
    error_message = "resource_suffix must contain 4-8 lowercase alphanumeric characters."
  }
}

variable "location" {
  description = "Azure region for the Terraform state resources."
  type        = string
  default     = "eastus2"

  validation {
    condition     = can(regex("^[a-z0-9]+$", var.location))
    error_message = "location must be an Azure region name containing lowercase letters and numbers."
  }
}

variable "owner" {
  description = "User-approved owner tag for the Terraform state resources."
  type        = string

  validation {
    condition     = length(trimspace(var.owner)) > 0
    error_message = "owner must not be empty."
  }
}

variable "cost_center" {
  description = "User-approved cost-center tag for the Terraform state resources."
  type        = string

  validation {
    condition     = length(trimspace(var.cost_center)) > 0
    error_message = "cost_center must not be empty."
  }
}

variable "additional_tags" {
  description = "Optional tags. Required MarketLab tags cannot be overridden."
  type        = map(string)
  default     = {}

  validation {
    condition = alltrue([
      for key in keys(var.additional_tags) : !contains(
        ["cost-center", "managed-by", "owner", "phase", "project", "purpose"],
        lower(key),
      )
    ])
    error_message = "additional_tags cannot redefine a required MarketLab tag."
  }
}
