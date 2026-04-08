resource "aws_vpc" "fleet_vpc" {
  cidr_block = "10.0.0.0/16"
  tags = {
    Name = "fleet-manager-vpc"
  }
}

resource "aws_subnet" "fleet_subnet_a" {
  vpc_id            = aws_vpc.fleet_vpc.id
  cidr_block        = "10.0.1.0/24"
  availability_zone = "us-east-1a"
  tags = {
    Name = "fleet-manager-subnet-a"
  }
}

resource "aws_subnet" "fleet_subnet_b" {
  vpc_id            = aws_vpc.fleet_vpc.id
  cidr_block        = "10.0.2.0/24"
  availability_zone = "us-east-1b"
  tags = {
    Name = "fleet-manager-subnet-b"
  }
}
