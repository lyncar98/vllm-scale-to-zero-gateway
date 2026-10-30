output "cluster_name" {
  description = "EKS cluster name."
  value       = module.eks.cluster_name
}

output "cluster_endpoint" {
  description = "EKS API server endpoint."
  value       = module.eks.cluster_endpoint
}

output "cluster_certificate_authority_data" {
  description = "Base64 cluster CA data."
  value       = module.eks.cluster_certificate_authority_data
}

output "oidc_provider_arn" {
  description = "IRSA OIDC provider ARN."
  value       = module.eks.oidc_provider_arn
}

output "node_security_group_id" {
  description = "EKS managed node group security group ID."
  value       = module.eks.node_security_group_id
}

output "external_secrets_role_arn" {
  description = "IRSA role ARN for the External Secrets Operator."
  value       = aws_iam_role.external_secrets.arn
}

output "alb_controller_role_arn" {
  description = "IRSA role ARN for the AWS Load Balancer Controller (provisions the internal inference NLB)."
  value       = aws_iam_role.alb_controller.arn
}

output "model_server_role_arn" {
  description = "IRSA role ARN for the vLLM model server. Annotate the model-server ServiceAccount with this; grants S3 read only once model_read_bucket_arns is populated."
  value       = aws_iam_role.model_server.arn
}

output "node_ebs_kms_key_arn" {
  description = "CMK encrypting node root volumes and CSI-provisioned PVCs."
  value       = aws_kms_key.node_ebs.arn
}

output "gpu_node_group_name" {
  description = "Name of the accelerator node group, for runbooks and scale operations."
  value       = "gpu-inference"
}
