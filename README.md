# AWS Fleet Instance Manager — RL-Powered Spot Interruption Prevention

> **Prevent spot instance interruptions from derailing long-running workloads using a trained PPO Reinforcement Learning agent.**

---

## The Problem

Running a 12-hour ML training job on a spot instance? AWS can reclaim it at hour 10 with only **2 minutes warning**, wasting all your progress and money.

## The Solution

This system:
1. **Monitors** spot prices, AZ capacity, and interruption signals every 60 seconds
2. **Predicts** reclamation risk using a trained PPO + LSTM policy
3. **Proactively migrates** your workload to safer/cheaper instances *before* reclamation
4. **Checkpoints** job state to S3 so migration is seamless
5. **Minimizes total cost** — only falls back to on-demand as a last resort

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure AWS credentials

```bash
cp .env.example .env
# Fill in your AWS credentials and S3 bucket name
```

### 3. Train the RL agent (offline on historical data)

```bash
python -m training.train --job-hours 12 --budget 50 --timesteps 2000000

# This will generate 90-day synthetic price CSVs in data/historical_prices/
```

### 4. Run the fleet manager

```bash
# Start backend API + frontend dashboard
docker-compose up

# Or run individually:
uvicorn dashboard.backend.main:app --port 8000 &
cd dashboard/frontend && npm install && npm run dev
```

Dashboard: http://localhost:5173  
API Docs: http://localhost:8000/docs  
MLflow: http://localhost:5000

---

## Architecture

```
Telemetry Collector (60s) → Feature Vector (28-dim)
                              ↓
                      PPO Agent (LSTM)
                              ↓ Action
                  ┌───────────┼───────────┐
                STAY      MIGRATE      PAUSE
                           ↓
              Checkpoint → New Instance → Restore → Terminate Old
```

## Project Structure

```
agents/          # RL environment + PPO agent
orchestrator/    # Main control loop + migration engine + circuit breaker
telemetry/       # Spot price collector + feature builder
plugins/         # Workload adapters (PyTorch, TF, Spark, Generic)
training/        # Offline training + evaluation vs baselines
dashboard/       # FastAPI backend + React frontend
config/          # instance catalog (30 types) + settings
tests/           # Unit + integration tests
```

## Workload Plugins

| Plugin | Use Case | Checkpoint |
|--------|---------|-----------|
| `PyTorchPlugin` | ML training | model weights → S3 |
| `TensorFlowPlugin` | Keras/TF jobs | model.save() → S3 |
| `SparkPlugin` | Data pipelines | partition progress |
| `GenericPlugin` | Any process | EBS snapshot / tar |

## RL Design

- **Algorithm**: PPO with LSTM feature extractor (Stable-Baselines3)
- **State**: 28-dim vector (spot prices, AZ capacity, job progress, interruption risk)
- **Actions**: STAY | MIGRATE-to-X | PAUSE
- **Reward**: Cost-first (λ_cost=3.0), hard budget cap (λ_budget=200.0)
- **Training**: 10k simulated episodes on 90 days of historical spot data

## Safety Rails

- **Circuit breaker**: Blocks agent if >3 migrations in 10 minutes
- **Min stay time**: Cannot migrate a job migrated <5 minutes ago
- **Budget hard cap**: Episode terminates if cost exceeds cap
- **Human override**: Dashboard button forces on-demand migration

```bash
python -m pytest tests/ -v
```

---

## Infrastructure (IaC)

The `terraform/` directory contains complete resource definitions:
- **VPC** and subnets for isolated execution
- **IAM** roles for EC2, S3, and DynamoDB
- **S3** bucket for job checkpoints
- **DynamoDB** tables for job state and event logs
- **ECS** Fargate cluster for the orchestrator

To validate:
```bash
cd terraform
terraform init
terraform validate
```

---

## Testing

For a detailed walkthrough, see [WALKTHROUGH.md](file:///C:/Users/nehau/.gemini/antigravity/brain/bb6ddc18-b5a8-48ec-9bd1-7f15bcf45d63/walkthrough.md).
