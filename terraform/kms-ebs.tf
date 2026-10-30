# EKS managed node groups launch their instances through an Auto Scaling
# group, so the EC2 Auto Scaling service-linked role must be allowed to use
# whatever KMS key encrypts the node root volumes. An account-default EBS key
# often does not grant that role, which makes every ASG launch fail with
# `Client.InvalidKMSKey.InvalidState` -> NodeCreationFailure. Owning a
# dedicated node-EBS key here, with the required grant, keeps node bring-up
# independent of whatever default key policy your account has.

data "aws_iam_policy_document" "node_ebs_kms" {
  statement {
    sid       = "EnableRootAccount"
    effect    = "Allow"
    actions   = ["kms:*"]
    resources = ["*"]
    principals {
      type        = "AWS"
      identifiers = ["arn:aws:iam::${data.aws_caller_identity.current.account_id}:root"]
    }
  }

  statement {
    sid    = "AllowAutoScalingUseOfKey"
    effect = "Allow"
    actions = [
      "kms:Encrypt",
      "kms:Decrypt",
      "kms:ReEncrypt*",
      "kms:GenerateDataKey*",
      "kms:DescribeKey",
    ]
    resources = ["*"]
    principals {
      type        = "AWS"
      identifiers = ["arn:aws:iam::${data.aws_caller_identity.current.account_id}:role/aws-service-role/autoscaling.amazonaws.com/AWSServiceRoleForAutoScaling"]
    }
  }

  statement {
    sid       = "AllowAutoScalingCreateGrant"
    effect    = "Allow"
    actions   = ["kms:CreateGrant"]
    resources = ["*"]
    principals {
      type        = "AWS"
      identifiers = ["arn:aws:iam::${data.aws_caller_identity.current.account_id}:role/aws-service-role/autoscaling.amazonaws.com/AWSServiceRoleForAutoScaling"]
    }
    condition {
      test     = "Bool"
      variable = "kms:GrantIsForAWSResource"
      values   = ["true"]
    }
  }
}

resource "aws_kms_key" "node_ebs" {
  description             = "${local.cluster_name} EKS node root-volume encryption"
  deletion_window_in_days = 30
  enable_key_rotation     = true
  policy                  = data.aws_iam_policy_document.node_ebs_kms.json
}

resource "aws_kms_alias" "node_ebs" {
  name          = "alias/${local.cluster_name}-node-ebs"
  target_key_id = aws_kms_key.node_ebs.key_id
}
