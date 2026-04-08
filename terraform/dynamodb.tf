resource "aws_dynamodb_table" "fleet_jobs" {
  name           = "fleet_manager_jobs"
  billing_mode   = "PAY_PER_REQUEST"
  hash_key       = "job_id"

  attribute {
    name = "job_id"
    type = "S"
  }

  tags = {
    Name        = "fleet-manager-jobs-table"
    Environment = "dev"
  }
}

resource "aws_dynamodb_table" "fleet_events" {
  name           = "fleet_manager_events"
  billing_mode   = "PAY_PER_REQUEST"
  hash_key       = "event_id"
  range_key      = "timestamp"

  attribute {
    name = "event_id"
    type = "S"
  }

  attribute {
    name = "timestamp"
    type = "N"
  }

  tags = {
    Name        = "fleet-manager-events-table"
    Environment = "dev"
  }
}
