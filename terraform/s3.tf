resource "aws_s3_bucket" "fleet_checkpoints" {
  bucket = "fleet-manager-checkpoints-${random_id.suffix.hex}"
  acl    = "private"

  versioning {
    enabled = true
  }

  tags = {
    Name        = "fleet-manager-checkpoints"
    Environment = "dev"
  }
}

resource "random_id" "suffix" {
  byte_length = 4
}
