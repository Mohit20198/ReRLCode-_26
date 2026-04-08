output "vpc_id" {
  value = aws_vpc.fleet_vpc.id
}

output "subnet_ids" {
  value = [aws_subnet.fleet_subnet_a.id, aws_subnet.fleet_subnet_b.id]
}

output "s3_bucket_name" {
  value = aws_s3_bucket.fleet_checkpoints.bucket
}

output "dynamodb_table_jobs" {
  value = aws_dynamodb_table.fleet_jobs.name
}

output "ecs_cluster_name" {
  value = aws_ecs_cluster.fleet_cluster.name
}

output "orchestrator_service_name" {
  value = aws_ecs_service.orchestrator_service.name
}
