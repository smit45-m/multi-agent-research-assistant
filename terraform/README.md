# Multi-Agent Research Assistant - Terraform AWS Deployment

This directory contains production-ready Infrastructure as Code (IaC) to deploy the Multi-Agent Research Assistant onto AWS using Terraform, Docker, and an Elastic IP for zero-domain-cost access.

---

## 🏗️ Architecture Overview

* **VPC & Subnet**: Dedicated lightweight VPC with Public Subnet and Internet Gateway (Free).
* **Security Group**: Allows HTTP (Port 80) and HTTPS (Port 443) ingress.
* **Storage**: 30GB gp3 EBS disk with `delete_on_termination = false` to guarantee persistent FAISS vector indices and uploaded user files.
* **Static Elastic IP**: Static public IP allocated directly to your instance (`$0 domain cost` — permanent fixed endpoint).
* **Amazon ECR**: Encrypted container repository with automated 14-day lifecycle policy to retain only the 3 latest images (<$0.20/mo).
* **IAM Instance Profile**: Grants EC2 rights to pull from ECR and enables AWS Systems Manager (SSM) for passwordless, SSH-free CI/CD deployments.
* **Cloud-Init Auto-Bootstrap**: Automatically provisions Docker, AWS CLI, and starts the container on host port 80.

---

## 💰 Monthly Cost & Efficiency Breakdown (Prototype Under $10/mo in ap-south-1 Mumbai)

| Component | Specification | Estimated Cost (USD) |
| :--- | :--- | :--- |
| **EC2 Compute** | `t3.small` Spot (2 vCPU, 2GB RAM) | **~$4.20 / month** |
| **RAM Boost (Swap)** | 3GB Swap on NVMe/SSD (= 5GB virtual RAM for PyTorch/FAISS) | **$0.00 / month** |
| **Persistent Storage** | 15 GB gp3 EBS Volume | **~$1.20 / month** |
| **Static IPv4 Address**| Elastic IP attached to running instance | **~$3.65 / month** ($0.005/hr) |
| **Container Registry** | Amazon ECR (Pruned to 2 images) | **~$0.15 / month** |
| **Custom Domain** | None (Using permanent Elastic IP URL) | **$0.00 / month** |
| **Total Estimated Cost** | **Full 24/7 Live Prototype** | **~$9.20 / month (Strictly <$10!)** |

*(Note: If your AWS account is in the 12-month Free Tier, `t3.micro` compute and storage are 100% free, dropping total monthly cost to **~$3.80/month**!)*

---

## 🚀 Quick Deployment Guide

### 1. Configure Variables
Copy `terraform.tfvars.example` to `terraform.tfvars`:
```bash
cp terraform.tfvars.example terraform.tfvars
```
Fill in your API keys in `terraform.tfvars`:
```hcl
gemini_api_key = "AIzaSy..."
groq_api_key   = "gsk_..."
jwt_secret_key = "your-custom-secret-key"
```

### 2. Plan and Deploy
```bash
terraform plan
terraform apply
```

### 3. Access Your Assistant
After `terraform apply` finishes, the CLI outputs will show:
```
application_url = "http://13.235.xxx.xxx"
```
Open that URL in your browser! Zero domain purchase required.
