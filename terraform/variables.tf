variable "aws_region" {
  description = "AWS region to deploy resources in"
  type        = string
  default     = "ap-south-1"
}

variable "environment" {
  description = "Environment name (e.g. dev, prod)"
  type        = string
  default     = "prod"
}

variable "instance_type" {
  description = "EC2 instance type (t3.micro: ~$2.26/mo Spot in ap-south-1, 2 vCPU 1GB RAM + 3GB SSD Swap)"
  type        = string
  default     = "t3.micro"
}

variable "use_spot" {
  description = "Use EC2 Spot instance for up to 70% cost savings"
  type        = bool
  default     = true
}

variable "enable_elastic_ip" {
  description = "Allocate an Elastic IP ($3.65/mo IPv4 fee). Set to false to use Free Cloudflare Tunnel and keep total AWS cost under $5/mo."
  type        = bool
  default     = false
}

variable "cloudflare_tunnel_token" {
  description = "Optional Cloudflare Zero Trust Tunnel token. If left blank, cloudflared runs a free Quick Tunnel giving a public https://*.trycloudflare.com URL."
  type        = string
  default     = ""
  sensitive   = true
}

variable "ebs_volume_size" {
  description = "Root EBS storage size in GB (gp3, 24GB is ~$1.92/mo, gives 17GB free space for PyTorch/Transformers container layers)"
  type        = number
  default     = 24
}

variable "app_port" {
  description = "Internal container port for FastAPI"
  type        = number
  default     = 8000
}

variable "host_port" {
  description = "External host port (80 for direct HTTP without custom domain)"
  type        = number
  default     = 80
}

variable "ecr_repository_name" {
  description = "Name of the Amazon ECR container repository"
  type        = string
  default     = "multi-agent-research-assistant"
}

variable "gemini_api_key" {
  description = "Google Gemini API key for research LLM"
  type        = string
  default     = ""
  sensitive   = true
}

variable "groq_api_key" {
  description = "Groq API key for fast inference fallback"
  type        = string
  default     = ""
  sensitive   = true
}

variable "jwt_secret_key" {
  description = "Secret key used for JWT authentication and session tokens"
  type        = string
  default     = "workbuddy-research-assistant-production-secret-token"
  sensitive   = true
}
