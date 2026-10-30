# IRSA roles map Kubernetes service accounts to AWS IAM, so pods reach
# Secrets Manager / S3 / ECR without static keys.
#
# Native resources rather than a generic multi-role-factory module: hand-
# writing the web-identity trust against the EKS OIDC provider is the
# clearest, lowest-surprise option when you only need a handful of roles.
# `module.eks.oidc_provider` is the issuer URL without the leading https://
# (used for the sub/aud condition keys).

# ── External Secrets Operator -> read app secrets from AWS Secrets Manager ──
data "aws_iam_policy_document" "external_secrets_assume" {
  statement {
    effect  = "Allow"
    actions = ["sts:AssumeRoleWithWebIdentity"]
    principals {
      type        = "Federated"
      identifiers = [module.eks.oidc_provider_arn]
    }
    condition {
      test     = "StringEquals"
      variable = "${module.eks.oidc_provider}:sub"
      values   = ["system:serviceaccount:external-secrets:external-secrets"]
    }
    condition {
      test     = "StringEquals"
      variable = "${module.eks.oidc_provider}:aud"
      values   = ["sts.amazonaws.com"]
    }
  }
}

resource "aws_iam_role" "external_secrets" {
  name               = "${local.cluster_name}-external-secrets"
  assume_role_policy = data.aws_iam_policy_document.external_secrets_assume.json
  tags               = local.default_tags
}

resource "aws_iam_role_policy" "external_secrets" {
  name = "secrets-access"
  role = aws_iam_role.external_secrets.id
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Sid    = "ReadAppSecrets"
        Effect = "Allow"
        Action = ["secretsmanager:GetSecretValue", "secretsmanager:DescribeSecret"]
        # Scope this to your own secret naming convention. An explicit prefix
        # (not a wildcard across the whole account) keeps this role's blast
        # radius to only the secrets this cluster actually needs.
        Resource = "arn:aws:secretsmanager:${var.aws_region}:${data.aws_caller_identity.current.account_id}:secret:inference-${var.environment}/*"
      },
      # Secrets encrypted with a customer-managed KMS key (rather than the
      # aws/secretsmanager default) additionally require kms:Decrypt on that
      # key. Without this, GetSecretValue fails with "AccessDeniedException:
      # Access to KMS is not allowed". Scoped by kms:ViaService so this role
      # can only use keys through Secrets Manager, not to decrypt arbitrary
      # data.
      {
        Sid      = "DecryptSecretsManagerCmks"
        Effect   = "Allow"
        Action   = ["kms:Decrypt", "kms:DescribeKey"]
        Resource = "*"
        Condition = {
          StringEquals = {
            "kms:ViaService" = "secretsmanager.${var.aws_region}.amazonaws.com"
          }
        }
      },
    ]
  })
}

# ── EBS CSI driver controller -> provision/attach EBS volumes for PVCs ──────
data "aws_iam_policy_document" "ebs_csi_assume" {
  statement {
    effect  = "Allow"
    actions = ["sts:AssumeRoleWithWebIdentity"]
    principals {
      type        = "Federated"
      identifiers = [module.eks.oidc_provider_arn]
    }
    condition {
      test     = "StringEquals"
      variable = "${module.eks.oidc_provider}:sub"
      values   = ["system:serviceaccount:kube-system:ebs-csi-controller-sa"]
    }
    condition {
      test     = "StringEquals"
      variable = "${module.eks.oidc_provider}:aud"
      values   = ["sts.amazonaws.com"]
    }
  }
}

resource "aws_iam_role" "ebs_csi" {
  name               = "${local.cluster_name}-ebs-csi"
  assume_role_policy = data.aws_iam_policy_document.ebs_csi_assume.json
  tags               = local.default_tags
}

resource "aws_iam_role_policy_attachment" "ebs_csi" {
  role       = aws_iam_role.ebs_csi.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AmazonEBSCSIDriverPolicy"
}

# The CSI controller creates volumes encrypted with the node-EBS CMK, so it
# needs its own grant on that key -- AmazonEBSCSIDriverPolicy covers the EC2
# calls but not KMS on a customer-managed key.
resource "aws_iam_role_policy" "ebs_csi_kms" {
  name = "node-ebs-kms"
  role = aws_iam_role.ebs_csi.id
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Sid      = "UseNodeEbsKey"
        Effect   = "Allow"
        Action   = ["kms:Encrypt", "kms:Decrypt", "kms:ReEncrypt*", "kms:GenerateDataKey*", "kms:DescribeKey"]
        Resource = aws_kms_key.node_ebs.arn
      },
      {
        Sid      = "CreateGrantForAttach"
        Effect   = "Allow"
        Action   = ["kms:CreateGrant"]
        Resource = aws_kms_key.node_ebs.arn
        Condition = {
          Bool = { "kms:GrantIsForAWSResource" = "true" }
        }
      },
    ]
  })
}

# ── AWS Load Balancer Controller -> provision the internal NLB for inference ─
data "aws_iam_policy_document" "alb_controller_assume" {
  statement {
    effect  = "Allow"
    actions = ["sts:AssumeRoleWithWebIdentity"]
    principals {
      type        = "Federated"
      identifiers = [module.eks.oidc_provider_arn]
    }
    condition {
      test     = "StringEquals"
      variable = "${module.eks.oidc_provider}:sub"
      values   = ["system:serviceaccount:kube-system:aws-load-balancer-controller"]
    }
    condition {
      test     = "StringEquals"
      variable = "${module.eks.oidc_provider}:aud"
      values   = ["sts.amazonaws.com"]
    }
  }
}

resource "aws_iam_role" "alb_controller" {
  name               = "${local.cluster_name}-alb-controller"
  assume_role_policy = data.aws_iam_policy_document.alb_controller_assume.json
  tags               = local.default_tags
}

# The controller's permission set is large and AWS publishes it as a document
# rather than a managed policy. Kept as the upstream file (not a wildcard) so
# the surface is reviewable; this is the upstream recommended policy trimmed
# to what an internal-NLB-only path needs -- no WAF/Shield/Cognito statements
# if this cluster never fronts a public listener.
resource "aws_iam_role_policy" "alb_controller" {
  name   = "elb-management"
  role   = aws_iam_role.alb_controller.id
  policy = file("${path.module}/alb-controller-policy.json")
}

# ── Model server -> read promoted checkpoints from object storage ───────────
# Bakes weights into the image by default (see ../model-server/Dockerfile),
# so this role starts with no S3 access at all -- model_read_bucket_arns is
# empty. It exists now so the ServiceAccount annotation is stable before you
# add the first externally-stored checkpoint; populating the variable later
# is a tfvars change, not a redeploy.
variable "model_read_bucket_arns" {
  description = "S3 bucket ARNs the model server may read checkpoints from. Empty by default: weights are baked into the image."
  type        = list(string)
  default     = []
}

data "aws_iam_policy_document" "model_server_assume" {
  statement {
    effect  = "Allow"
    actions = ["sts:AssumeRoleWithWebIdentity"]
    principals {
      type        = "Federated"
      identifiers = [module.eks.oidc_provider_arn]
    }
    condition {
      test     = "StringEquals"
      variable = "${module.eks.oidc_provider}:sub"
      values   = ["system:serviceaccount:inference:model-server"]
    }
    condition {
      test     = "StringEquals"
      variable = "${module.eks.oidc_provider}:aud"
      values   = ["sts.amazonaws.com"]
    }
  }
}

resource "aws_iam_role" "model_server" {
  name               = "${local.cluster_name}-model-server"
  assume_role_policy = data.aws_iam_policy_document.model_server_assume.json
  tags               = local.default_tags
}

resource "aws_iam_role_policy" "model_server_s3" {
  # Only create the policy when there is something to grant. An empty
  # Resource list is invalid IAM, and a wildcard would defeat the point of an
  # explicit allow-list for what the model server can read.
  count = length(var.model_read_bucket_arns) > 0 ? 1 : 0

  name = "model-artifact-read"
  role = aws_iam_role.model_server.id
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Sid      = "ListModelBuckets"
        Effect   = "Allow"
        Action   = ["s3:ListBucket", "s3:GetBucketLocation"]
        Resource = var.model_read_bucket_arns
      },
      {
        Sid      = "ReadModelObjects"
        Effect   = "Allow"
        Action   = ["s3:GetObject", "s3:GetObjectVersion"]
        Resource = [for arn in var.model_read_bucket_arns : "${arn}/*"]
      },
      {
        Sid      = "DecryptModelObjects"
        Effect   = "Allow"
        Action   = ["kms:Decrypt", "kms:DescribeKey"]
        Resource = "*"
        Condition = {
          StringEquals = { "kms:ViaService" = "s3.${var.aws_region}.amazonaws.com" }
        }
      },
    ]
  })
}
