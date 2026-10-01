# IAM Role for EC2 Instance to pull ECR images and use AWS Systems Manager (SSM)
resource "aws_iam_role" "ec2_role" {
  name = "research_assistant_ec2_role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = "sts:AssumeRole"
        Effect = "Allow"
        Principal = {
          Service = "ec2.amazonaws.com"
        }
      }
    ]
  })
}

# Attach AmazonEC2ContainerRegistryReadOnly so the instance can pull from ECR
resource "aws_iam_role_policy_attachment" "ecr_read" {
  role       = aws_iam_role.ec2_role.name
  policy_arn = "arn:aws:iam::aws:policy/AmazonEC2ContainerRegistryReadOnly"
}

# Attach AmazonSSMManagedInstanceCore so GitHub Actions can deploy via SSM without opening port 22 or managing SSH keys
resource "aws_iam_role_policy_attachment" "ssm_core" {
  role       = aws_iam_role.ec2_role.name
  policy_arn = "arn:aws:iam::aws:policy/AmazonSSMManagedInstanceCore"
}

resource "aws_iam_instance_profile" "ec2_profile" {
  name = "research_assistant_instance_profile"
  role = aws_iam_role.ec2_role.name
}
