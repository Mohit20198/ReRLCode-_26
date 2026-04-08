resource "aws_ecs_cluster" "fleet_cluster" {
  name = "fleet-manager-cluster"
}

resource "aws_ecs_task_definition" "orchestrator" {
  family                   = "fleet-manager-orchestrator"
  network_mode             = "awsvpc"
  requires_compatibilities = ["FARGATE"]
  cpu                      = 512
  memory                   = 1024
  execution_role_arn       = aws_iam_role.fleet_manager_role.arn
  task_role_arn            = aws_iam_role.fleet_manager_role.arn

  container_definitions = jsonencode([{
    name  = "orchestrator"
    image = "fleet-manager-orchestrator:latest"
    essential = true
    portMappings = [{
      containerPort = 8000
      hostPort      = 8000
    }]
    environment = [
      { name = "AWS_REGION", value = var.aws_region },
      { name = "ENV", value = "prod" }
    ]
    logConfiguration = {
      logDriver = "awslogs"
      options = {
        "awslogs-group"         = "/ecs/fleet-manager-orchestrator"
        "awslogs-region"        = var.aws_region
        "awslogs-stream-prefix" = "ecs"
      }
    }
  }])
}

resource "aws_ecs_service" "orchestrator_service" {
  name            = "fleet-manager-service"
  cluster         = aws_ecs_cluster.fleet_cluster.id
  task_definition = aws_ecs_task_definition.orchestrator.arn
  desired_count   = 1
  launch_type     = "FARGATE"

  network_configuration {
    subnets          = [aws_subnet.fleet_subnet_a.id, aws_subnet.fleet_subnet_b.id]
    assign_public_ip = true
  }
}
