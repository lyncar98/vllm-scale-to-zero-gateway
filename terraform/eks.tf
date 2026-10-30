# EKS cluster + managed node groups. The cluster is the only cloud-specific
# compute substrate; everything that runs on it (the vLLM Deployment, the
# NVIDIA device plugin, the gateway, autoscaling policy) is portable
# Helm/Kustomize, intended to be delivered by GitOps -- see ../k8s.
#
# Uses the upstream community module pinned to v20.x. v21 renames several
# input/output names and requires aws >= 6.0, outside this stack's
# `>= 5.50, < 6.0` provider constraint.
module "eks" {
  source  = "terraform-aws-modules/eks/aws"
  version = "~> 20.31"

  cluster_name    = local.cluster_name
  cluster_version = var.cluster_version
  vpc_id          = var.vpc_id
  subnet_ids      = var.private_subnet_ids

  # OIDC provider enabled for IRSA (workload identity).
  enable_irsa = true

  # API server reachable for GitOps/management; private path stays on for
  # in-VPC traffic. Grant the creating principal cluster-admin so day-2
  # access entries can be managed without a lockout.
  cluster_endpoint_public_access           = true
  cluster_endpoint_private_access          = true
  enable_cluster_creator_admin_permissions = true

  # Managed addons kept minimal and replaceable. aws-ebs-csi-driver is managed
  # as a standalone resource below so it can carry an IRSA role without
  # creating a dependency cycle (the role's trust references this module's
  # OIDC provider).
  #
  # The NVIDIA device plugin is deliberately NOT here: it's an in-cluster
  # concern, delivered by GitOps alongside everything else in ../k8s. Terraform
  # owns the cluster and the capacity; your GitOps tool owns what runs on it.
  cluster_addons = {
    coredns    = {}
    kube-proxy = {}
    vpc-cni    = {}
  }

  eks_managed_node_groups = {
    for name, ng in var.node_groups : name => {
      instance_types = ng.instance_types
      min_size       = ng.min_size
      max_size       = ng.max_size
      desired_size   = ng.desired_size
      ami_type       = ng.ami_type
      labels         = ng.labels
      # v20 expects taints as a map(object); the variable carries a list.
      taints = { for idx, t in ng.taints : "t${idx}" => t }

      # Encrypt node root volumes with a key this stack owns and grants to the
      # Auto Scaling service-linked role, rather than an account-default EBS
      # key (which commonly lacks that grant and breaks ASG launches with
      # InvalidKMSKey.InvalidState).
      block_device_mappings = {
        xvda = {
          device_name = "/dev/xvda"
          ebs = {
            volume_size           = ng.disk_size
            volume_type           = "gp3"
            encrypted             = true
            kms_key_id            = aws_kms_key.node_ebs.arn
            delete_on_termination = true
          }
        }
      }
    }
  }
}

# EBS CSI driver as a standalone addon so it can run under a dedicated IRSA
# role (see irsa.tf). depends_on the cluster module ensures node groups exist
# first, so the controller pods schedule immediately rather than blocking the
# addon.
resource "aws_eks_addon" "ebs_csi" {
  cluster_name                = module.eks.cluster_name
  addon_name                  = "aws-ebs-csi-driver"
  service_account_role_arn    = aws_iam_role.ebs_csi.arn
  resolve_conflicts_on_create = "OVERWRITE"
  resolve_conflicts_on_update = "OVERWRITE"

  tags = local.default_tags

  depends_on = [module.eks]
}
