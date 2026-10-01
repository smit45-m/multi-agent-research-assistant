output "instance_id" {
  description = "The EC2 Instance ID (used for GitHub Actions SSM automated deployment)"
  value       = aws_instance.app_server.id
}

output "public_ip" {
  description = "Public IP address"
  value       = var.enable_elastic_ip ? (length(aws_eip.app_eip) > 0 ? aws_eip.app_eip[0].public_ip : "") : aws_instance.app_server.public_ip
}

output "application_access" {
  description = "Application access information via Cloudflare Tunnel (Free HTTPS, 0 domain cost) or direct IP"
  value = {
    cloudflare_tunnel = "Check instance /var/log/cloudflared.log or Cloudflare dashboard for public https://*.trycloudflare.com URL"
    direct_http_ip    = "http://${aws_instance.app_server.public_ip}"
  }
}

output "ecr_repository_url" {
  description = "Amazon ECR Repository URL for pushing Docker images"
  value       = aws_ecr_repository.app.repository_url
}

output "aws_region" {
  description = "Deployed AWS Region"
  value       = var.aws_region
}

output "monthly_cost_estimate" {
  description = "Estimated AWS monthly infrastructure cost breakdown (Ultra-Budget Prototype <$5.00/mo)"
  value = {
    compute_ec2_spot_t3_micro = "~$2.26 / month (Spot instance in ap-south-1, 2 vCPU + 1GB RAM)"
    ram_swap_boost_3gb        = "$0.00 (Free 3GB swap on NVMe SSD = 4GB effective RAM for PyTorch)"
    storage_ebs_12gb_gp3      = "~$0.96 / month (12GB gp3 SSD)"
    cloudflare_tunnel_https   = "$0.00 (Free public HTTPS endpoint with SSL & DDoS protection)"
    custom_domain_cost        = "$0.00 (Zero domain fees)"
    ecr_storage_lifecycle     = "~$0.15 / month"
    total_estimated_cost      = "~$3.37 / month (Strictly under $5.00/mo!)"
  }
}
