# IAM role and policies for the fleet manager

resource "aws_iam_role" "fleet_manager_role" {
  name = "fleet_manager_role"
  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect = "Allow"
      Principal = {
        Service = "ecs-tasks.amazonaws.com"
      }
      Action = "sts:AssumeRole"
    }]
  })
}

resource "aws_iam_policy" "fleet_manager_policy" {
  name        = "fleet_manager_policy"
  description = "Permissions for the fleet manager to manage EC2, S3, DynamoDB and CloudWatch"
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "ec2:DescribeInstances",
          "ec2:RunInstances",
          "ec2:TerminateInstances",
          "ec2:CreateTags",
          "ec2:DescribeInstanceTypeOfferings",
          "ec2:DescribeSpotInstanceRequests",
          "ec2:CancelSpotInstanceRequests",
          "ec2:ModifyInstanceAttribute",
          "s3:PutObject",
          "s3:GetObject",
          "s3:ListBucket",
          "dynamodb:PutItem",
          "dynamodb:GetItem",
          "dynamodb:UpdateItem",
          "dynamodb:Query",
          "cloudwatch:PutMetricData",
          "logs:CreateLogGroup",
          "logs:CreateLogStream",
          "logs:PutLogEvents"
        ]
        Resource = "*"
      }
    ]
  })
}

resource "aws_iam_role_policy_attachment" "attach_fleet_manager" {
  role       = aws_iam_role.fleet_manager_role.name
  policy_arn = aws_iam_policy.fleet_manager_policy.arn
}
