terraform {
  required_version = ">= 1.6, < 2.0"

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = ">= 5.50, < 6.0"
    }
    # Kubernetes/Helm providers for the in-cluster pieces this stack installs
    # directly (the AWS Load Balancer Controller). Bounded below 3.0 for helm
    # because 3.x replaces the `kubernetes {}` block used here with a nested
    # attribute.
    kubernetes = {
      source  = "hashicorp/kubernetes"
      version = ">= 2.30, < 3.0"
    }
    helm = {
      source  = "hashicorp/helm"
      version = ">= 2.13, < 3.0"
    }
  }

  # Point this at your own remote state backend (S3, HCP Terraform, etc).
  # Left unconfigured here so the module is backend-agnostic by default.
}
