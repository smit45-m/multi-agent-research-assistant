# Query the latest Ubuntu 24.04 LTS AMI
data "aws_ami" "ubuntu" {
  most_recent = true
  owners      = ["099720109477"] # Canonical

  filter {
    name   = "name"
    values = ["ubuntu/images/hvm-ssd-gp3/ubuntu-noble-24.04-amd64-server-*"]
  }

  filter {
    name   = "virtualization-type"
    values = ["hvm"]
  }
}

# Cloud-Init User Data script to bootstrap Docker, SSM, and the application container
locals {
  user_data = <<-EOF
              #!/bin/bash
              set -ex

              # 1. Update and install prerequisites
              export DEBIAN_FRONTEND=noninteractive
              apt-get update -y
              apt-get install -y ca-certificates curl gnupg lsb-release unzip

              # Allocate 3GB NVMe/SSD Swapfile (Guarantees PyTorch + FAISS run with 5GB virtual RAM on budget instances)
              fallocate -l 3G /swapfile || dd if=/dev/zero of=/swapfile bs=1M count=3072
              chmod 600 /swapfile
              mkswap /swapfile
              swapon /swapfile
              echo '/swapfile none swap sw 0 0' >> /etc/fstab

              # 2. Install Docker
              install -m 0755 -d /etc/apt/keyrings
              curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
              chmod a+r /etc/apt/keyrings/docker.asc

              echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/ubuntu $(. /etc/os-release && echo "$VERSION_CODENAME") stable" | tee /etc/apt/sources.list.d/docker.list > /dev/null
              apt-get update -y
              apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin

              systemctl enable docker
              systemctl start docker

              # 3. Install AWS CLI v2
              curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
              unzip -q awscliv2.zip
              ./aws/install --update
              rm -rf aws awscliv2.zip

              # 4. Create persistent data directory for FAISS vectors and document store
              mkdir -p /opt/app/data
              chmod -R 777 /opt/app/data

              # 5. Write runtime environment configuration
              cat << 'ENVFILE' > /opt/app/.env
              API_HOST=0.0.0.0
              API_PORT=8000
              ENVIRONMENT=${var.environment}
              LOG_LEVEL=INFO
              MAX_CONCURRENT_REQUESTS=50
              GEMINI_API_KEY=${var.gemini_api_key}
              GROQ_API_KEY=${var.groq_api_key}
              JWT_SECRET_KEY=${var.jwt_secret_key}
              DATA_DIR=/app/data
              ENVFILE
              chmod 600 /opt/app/.env

              # 6. Create deployment helper script (used on startup and by CI/CD)
              cat << 'DEPLOY_SCRIPT' > /usr/local/bin/deploy-app
              #!/bin/bash
              set -e
              REGION="${var.aws_region}"
              REPO_URL="${aws_ecr_repository.app.repository_url}"

              echo "Authenticating with Amazon ECR..."
              aws ecr get-login-password --region $REGION | docker login --username AWS --password-stdin $REPO_URL || true

              echo "Checking if latest image is in ECR..."
              if docker pull $REPO_URL:latest; then
                IMAGE_TO_RUN="$REPO_URL:latest"
              else
                echo "ECR image not available yet. Running local fallback or waiting for first CI/CD push."
                exit 0
              fi

              echo "Stopping old container..."
              docker stop research-assistant || true
              docker rm -f research-assistant || true

              echo "Launching new container with persistent vector data..."
              docker run -d \
                --name research-assistant \
                --restart always \
                -p 80:8000 \
                -v /opt/app/data:/app/data \
                --env-file /opt/app/.env \
                $IMAGE_TO_RUN

              echo "Deployment complete. Container status:"
              docker ps --filter name=research-assistant
              DEPLOY_SCRIPT
              chmod +x /usr/local/bin/deploy-app

              # 7. Attempt initial deployment (will succeed once first image is pushed)
              /usr/local/bin/deploy-app || true

              # 8. Install and configure Cloudflare Tunnel (Free HTTPS URL, 0 domain cost, eliminates $3.65 IPv4 tax)
              curl -fsSL https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64.deb -o /tmp/cloudflared.deb
              dpkg -i /tmp/cloudflared.deb || true
              rm -f /tmp/cloudflared.deb

              if [ -n "${var.cloudflare_tunnel_token}" ]; then
                cloudflared service install "${var.cloudflare_tunnel_token}"
                systemctl start cloudflared
              else
                cat << 'CFTUNNEL' > /etc/systemd/system/cloudflared-tunnel.service
[Unit]
Description=Cloudflare Quick Tunnel (Free Public HTTPS)
After=network.target

[Service]
ExecStart=/usr/local/bin/cloudflared tunnel --url http://127.0.0.1:8000 --logfile /var/log/cloudflared.log
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
CFTUNNEL
                systemctl daemon-reload
                systemctl enable cloudflared-tunnel
                systemctl start cloudflared-tunnel
              fi
              EOF
}

# The EC2 Instance hosting the Docker Container
resource "aws_instance" "app_server" {
  ami                  = data.aws_ami.ubuntu.id
  instance_type        = var.instance_type
  subnet_id            = aws_subnet.public.id
  vpc_security_group_ids = [aws_security_group.app_sg.id]
  iam_instance_profile = aws_iam_instance_profile.ec2_profile.name

  dynamic "instance_market_options" {
    for_each = var.use_spot ? [1] : []
    content {
      market_type = "spot"
      spot_options {
        spot_instance_type             = "persistent"
        instance_interruption_behavior = "stop"
      }
    }
  }

  root_block_device {
    volume_size           = var.ebs_volume_size
    volume_type           = "gp3"
    delete_on_termination = false # Protects vector store and logs
    encrypted             = true

    tags = {
      Name = "research-assistant-disk"
    }
  }

  user_data                   = local.user_data
  user_data_replace_on_change = false

  tags = {
    Name = "research-assistant-server"
  }
}

# Optional Static Elastic IP (Disabled by default to save $3.65/mo and keep total AWS cost under $5/mo)
resource "aws_eip" "app_eip" {
  count    = var.enable_elastic_ip ? 1 : 0
  instance = aws_instance.app_server.id
  domain   = "vpc"

  tags = {
    Name = "research-assistant-eip"
  }
}
