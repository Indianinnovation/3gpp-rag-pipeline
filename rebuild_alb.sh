#!/bin/bash
# rebuild_alb.sh — Recreate ALB infrastructure for ask3gpp
# Run this when you need to demo again. Takes ~5 minutes.
#
# Prerequisites:
#   - AWS CLI configured
#   - SSL certificate still valid (ACM)
#   - VPC/subnets still exist

REGION="us-east-1"
VPC_ID="vpc-06d3cdcf9f5f4489f"
SUBNETS="subnet-05313aeb6f2d2848e,subnet-0b6619f3cf1c7d6bf,subnet-0b6d4cb1434f51101"
SECURITY_GROUP="sg-05665bfefa3c0ef6f"
CERTIFICATE_ARN="arn:aws:acm:us-east-1:830088750022:certificate/9a27a5f8-24b2-4f7b-9ac9-b7d0140ad13a"

echo "Step 1: Creating Target Group..."
TG_ARN=$(aws elbv2 create-target-group \
  --name ask3gpp-tg \
  --protocol HTTP \
  --port 8000 \
  --vpc-id $VPC_ID \
  --target-type ip \
  --health-check-path /health \
  --health-check-interval-seconds 30 \
  --healthy-threshold-count 2 \
  --unhealthy-threshold-count 2 \
  --region $REGION \
  --query "TargetGroups[0].TargetGroupArn" \
  --output text)
echo "  Target Group: $TG_ARN"

echo "Step 2: Creating ALB..."
ALB_ARN=$(aws elbv2 create-load-balancer \
  --name ask3gpp-alb \
  --subnets $(echo $SUBNETS | tr ',' ' ') \
  --security-groups $SECURITY_GROUP \
  --scheme internet-facing \
  --type application \
  --ip-address-type ipv4 \
  --region $REGION \
  --query "LoadBalancers[0].LoadBalancerArn" \
  --output text)
echo "  ALB: $ALB_ARN"

echo "Step 3: Creating HTTPS Listener..."
aws elbv2 create-listener \
  --load-balancer-arn $ALB_ARN \
  --protocol HTTPS \
  --port 443 \
  --certificates CertificateArn=$CERTIFICATE_ARN \
  --default-actions Type=forward,TargetGroupArn=$TG_ARN \
  --region $REGION \
  --output text
echo "  HTTPS Listener created on port 443"

echo "Step 4: Get ALB DNS name..."
ALB_DNS=$(aws elbv2 describe-load-balancers \
  --load-balancer-arns $ALB_ARN \
  --region $REGION \
  --query "LoadBalancers[0].DNSName" \
  --output text)
echo "  ALB DNS: $ALB_DNS"

echo ""
echo "Step 5: Update DNS record for api.ask3gpp.com"
echo "  Point api.ask3gpp.com CNAME → $ALB_DNS"
echo "  (Do this in Route53 or your DNS provider)"
echo ""
echo "Step 6: Update ECS service with new target group"
echo "  aws ecs update-service --cluster ask3gpp --service ask3gpp-api --desired-count 1 --region $REGION"
echo ""
echo "✓ ALB rebuild complete! Total time: ~5 minutes"
