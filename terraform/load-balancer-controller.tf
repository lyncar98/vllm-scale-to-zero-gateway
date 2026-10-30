# AWS Load Balancer Controller. The gateway's Service is type LoadBalancer
# with an internal NLB (ip targets, see ../k8s/gateway.yaml); nothing
# provisions it without this controller, so the Service would sit at
# <pending> with no useful events. Its IRSA role is in irsa.tf.
resource "helm_release" "aws_load_balancer_controller" {
  name       = "aws-load-balancer-controller"
  namespace  = "kube-system"
  repository = "https://aws.github.io/eks-charts"
  chart      = "aws-load-balancer-controller"
  version    = "3.6.0"

  values = [yamlencode({
    clusterName = module.eks.cluster_name
    region      = var.aws_region
    vpcId       = var.vpc_id
    serviceAccount = {
      create = true
      name   = "aws-load-balancer-controller"
      annotations = {
        "eks.amazonaws.com/role-arn" = aws_iam_role.alb_controller.arn
      }
    }
  })]

  depends_on = [module.eks]
}
