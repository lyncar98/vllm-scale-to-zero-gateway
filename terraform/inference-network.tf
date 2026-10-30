# The CIDRs allowed to reach the inference gateway's port. The NLB targets
# pod IPs directly (ip target type), so the node security group is where the
# rule goes. Scope this to the VPC(s) of whatever fronts the gateway (a BFF,
# an API layer, a VPN) -- never 0.0.0.0/0.
variable "inference_consumer_cidrs" {
  description = "CIDRs allowed to reach the inference gateway's port."
  type        = list(string)
}

resource "aws_vpc_security_group_ingress_rule" "inference_from_consumers" {
  for_each = toset(var.inference_consumer_cidrs)

  security_group_id = module.eks.node_security_group_id
  description       = "Inference gateway (8080) from ${each.value}"
  from_port         = 8080
  to_port           = 8080
  ip_protocol       = "tcp"
  cidr_ipv4         = each.value
}
