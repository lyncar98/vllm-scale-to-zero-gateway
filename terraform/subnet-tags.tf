# EKS subnet auto-discovery + the AWS Load Balancer Controller require these
# tags on your networking-owned subnets. If your subnets are managed by a
# separate stack (a shared networking module, Platform landing zone, etc),
# apply the cluster-specific tags here via aws_ec2_tag rather than mutating
# that stack.
#
# `internal-elb` on the private subnets is what makes the inference gateway's
# NLB land on a private address, matching the "private by default" design in
# the main README. The public-subnet `elb` tags exist only so a future
# direct-API / internet-facing product is a config change rather than a
# re-tag; nothing here requests a public LB by default.
resource "aws_ec2_tag" "private_internal_elb" {
  for_each    = toset(var.private_subnet_ids)
  resource_id = each.value
  key         = "kubernetes.io/role/internal-elb"
  value       = "1"
}

resource "aws_ec2_tag" "private_cluster" {
  for_each    = toset(var.private_subnet_ids)
  resource_id = each.value
  key         = "kubernetes.io/cluster/${local.cluster_name}"
  value       = "shared"
}

resource "aws_ec2_tag" "public_elb" {
  for_each    = toset(var.public_subnet_ids)
  resource_id = each.value
  key         = "kubernetes.io/role/elb"
  value       = "1"
}

resource "aws_ec2_tag" "public_cluster" {
  for_each    = toset(var.public_subnet_ids)
  resource_id = each.value
  key         = "kubernetes.io/cluster/${local.cluster_name}"
  value       = "shared"
}
