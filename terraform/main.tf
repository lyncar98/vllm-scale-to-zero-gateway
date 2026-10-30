data "aws_caller_identity" "current" {}

locals {
  cluster_name = "inference-${var.environment}"

  default_tags = {
    Project     = "inference"
    ManagedBy   = "terraform"
    Environment = var.environment
  }
}
