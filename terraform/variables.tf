variable "aws_region" {
  description = "AWS region for the EKS inference substrate."
  type        = string
  default     = "us-east-2"
}

variable "environment" {
  description = "Deployment environment."
  type        = string
  default     = "prod"
}

variable "vpc_id" {
  description = "VPC the cluster runs in. Bring your own -- this module deliberately doesn't create a VPC, so CIDR allocation and routing stay with whatever owns your networking."
  type        = string
}

variable "private_subnet_ids" {
  description = "Private subnet IDs for the EKS cluster and node groups."
  type        = list(string)
}

variable "public_subnet_ids" {
  description = "Public subnet IDs, tagged for a future internet-facing load balancer. Not used by the private inference path itself."
  type        = list(string)
  default     = []
}

variable "cluster_version" {
  description = "EKS control-plane Kubernetes version."
  type        = string
  default     = "1.31"
}

# ── Node groups ───────────────────────────────────────────────────────────────
# Two pools with deliberately different jobs:
#
#   system        — always-on, small, CPU. Runs cluster plumbing (CoreDNS, the
#                   AWS Load Balancer Controller, autoscalers) and the
#                   lightweight inference *gateway*. Never runs model servers.
#
#   gpu-inference — the accelerators. desired_size = 0 by default: the pool is
#                   defined so Karpenter / the cluster autoscaler can grow it
#                   on demand and return it to zero when idle. A 24/7
#                   g5.xlarge is on the order of $700/mo of mostly idle spend
#                   at low traffic, so scale-to-zero is the sane default and a
#                   warm pool is a one-line min_size change when traffic
#                   justifies it.
#
# The GPU pool is TAINTED. This is load-bearing, not hygiene: without the
# taint, ordinary pods schedule onto expensive accelerator nodes -- a failure
# mode that is invisible except on the bill. Only pods that explicitly
# tolerate the taint land there.
variable "node_groups" {
  description = "EKS managed node groups. See the comment above for why the GPU pool is tainted and defaults to zero desired capacity."
  type = map(object({
    instance_types = list(string)
    min_size       = number
    max_size       = number
    desired_size   = number
    ami_type       = optional(string, "AL2023_x86_64_STANDARD")
    disk_size      = optional(number, 50)
    labels         = map(string)
    taints = optional(list(object({
      key    = string
      value  = string
      effect = string
    })), [])
  }))

  default = {
    system = {
      instance_types = ["m6i.large"]
      min_size       = 2
      max_size       = 6
      desired_size   = 2
      labels         = { "inference.io/pool" = "system" }
    }

    gpu-inference = {
      # g5.xlarge = 1x NVIDIA A10G (24GB). Fits an 8B checkpoint at
      # bf16-with-quantization, or comfortably at 4-bit.
      # g6e.xlarge (1x L40S, 48GB) is the upgrade path for a larger model --
      # same pool, different instance type.
      instance_types = ["g5.xlarge"]
      min_size       = 0
      max_size       = 4
      # Scale-to-zero by default. Bump to 1 for a prototype that needs to
      # stay warm, or wire a cluster autoscaler / Karpenter (see
      # k8s/autoscaling/) to manage this dynamically instead.
      desired_size = 0

      # GPU-accelerated AMI: ships the NVIDIA driver so the device plugin has
      # something to advertise. A STANDARD AMI here would join the cluster
      # fine and then never expose nvidia.com/gpu, which looks like a
      # scheduling bug rather than an AMI mistake.
      ami_type = "AL2023_x86_64_NVIDIA"

      # Model weights are baked into the image rather than pulled at pod
      # start, so the node needs room for a multi-GB image layer cache.
      disk_size = 200

      labels = {
        "inference.io/pool"        = "gpu-inference"
        "inference.io/accelerator" = "nvidia-a10g"
      }

      taints = [{
        key    = "inference.io/accelerator"
        value  = "true"
        effect = "NO_SCHEDULE"
      }]
    }
  }
}
