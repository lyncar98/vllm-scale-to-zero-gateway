provider "aws" {
  region = var.aws_region

  default_tags {
    tags = local.default_tags
  }
}

# Kubernetes + Helm providers for the AWS Load Balancer Controller
# (load-balancer-controller.tf).
#
# Authenticating with a short-lived EKS token from the AWS provider's own
# credentials rather than a kubeconfig keeps this usable from a CI runner with
# no local kubeconfig, as long as its AWS credentials have cluster-admin
# (e.g. via enable_cluster_creator_admin_permissions on the cluster module).
data "aws_eks_cluster_auth" "this" {
  name = module.eks.cluster_name
}

provider "kubernetes" {
  host                   = module.eks.cluster_endpoint
  cluster_ca_certificate = base64decode(module.eks.cluster_certificate_authority_data)
  token                  = data.aws_eks_cluster_auth.this.token
}

provider "helm" {
  kubernetes {
    host                   = module.eks.cluster_endpoint
    cluster_ca_certificate = base64decode(module.eks.cluster_certificate_authority_data)
    token                  = data.aws_eks_cluster_auth.this.token
  }
}
