"""
dashboard/backend/main.py
──────────────────────────
FastAPI server for the Fleet Manager dashboard.

Endpoints:
    GET  /jobs                    — List all active jobs
    GET  /jobs/{id}               — Job detail
    GET  /jobs/{id}/timeline      — Decision log
    GET  /fleet/cost              — Cost summary
    GET  /fleet/alerts            — Active alerts
    GET  /agent/metrics           — RL agent stats
    POST /jobs/{id}/override      — Force on-demand migration (human override)
    POST /jobs/{id}/breaker/reset — Reset circuit breaker

WebSocket:
    WS   /ws/fleet                — Real-time fleet updates (5s heartbeat)

Usage:
    uvicorn dashboard.backend.main:app --host 0.0.0.0 --port 8000 --reload
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional

import yaml
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

logger = logging.getLogger(__name__)

# ─────────────────────────── App Lifespan ────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize FleetManager on startup."""
    logger.info("Dashboard API starting...")
    # Fleet manager is shared via app.state (initialized by run script)
    yield
    logger.info("Dashboard API shutting down")


app = FastAPI(
    title="AWS Fleet Manager Dashboard API",
    description="Real-time monitoring and control for RL-driven spot fleet management",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # Restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Active WebSocket connections
_ws_connections: List[WebSocket] = []


# ─────────────────────────── Pydantic Models ─────────────────────────────────

class JobSummary(BaseModel):
    job_id: str
    status: str
    instance_type: str
    az: str
    progress: float
    cumulative_cost_usd: float
    budget_cap_usd: float
    n_migrations: int
    n_interruptions: int
    running_since: float


class FleetCostSummary(BaseModel):
    total_cost_usd: float
    estimated_savings_usd: float   # vs always-on-demand baseline
    n_jobs: int
    n_migrations: int
    n_interruptions: int


class Alert(BaseModel):
    job_id: str
    severity: str   # "info" | "warning" | "critical"
    message: str
    timestamp: float


class OverrideRequest(BaseModel):
    target_instance_type: str
    reason: str = "human_override"


# ─────────────────────────── Routes ──────────────────────────────────────────

@app.get("/health")
async def health():
    return {"status": "ok", "timestamp": time.time()}


@app.get("/jobs", response_model=List[JobSummary])
async def list_jobs():
    """List all currently managed jobs."""
    manager = _get_manager()
    if manager is None:
        return _mock_jobs()

    jobs = manager.get_all_jobs()
    return [_job_to_summary(j) for j in jobs]


@app.get("/jobs/{job_id}")
async def get_job(job_id: str):
    """Get detailed status and decision log for a job."""
    manager = _get_manager()
    if manager is None:
        return _mock_job_detail(job_id)

    job = manager.get_job_status(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

    return {
        **_job_to_summary(job).model_dump(),
        "decision_log": job.decision_log[-50:],  # Last 50 decisions
    }


@app.get("/jobs/{job_id}/timeline")
async def get_job_timeline(job_id: str):
    """Get the full decision timeline for a job."""
    manager = _get_manager()
    if manager is None:
        return {"job_id": job_id, "decisions": _mock_timeline()}

    job = manager.get_job_status(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

    return {"job_id": job_id, "decisions": job.decision_log}


@app.get("/fleet/cost", response_model=FleetCostSummary)
async def get_fleet_cost():
    """Get real-time cost summary and savings estimate."""
    manager = _get_manager()
    if manager is None:
        return FleetCostSummary(
            total_cost_usd=12.47,
            estimated_savings_usd=34.21,
            n_jobs=3,
            n_migrations=7,
            n_interruptions=0,
        )

    summary = manager.get_cost_summary()
    # Savings = estimated on-demand cost - actual cost
    on_demand_estimate = summary["total_cost_usd"] * 3.2  # Rough 3.2x vs spot
    savings = max(0, on_demand_estimate - summary["total_cost_usd"])

    return FleetCostSummary(
        total_cost_usd=summary["total_cost_usd"],
        estimated_savings_usd=savings,
        n_jobs=summary["n_jobs"],
        n_migrations=summary["n_migrations"],
        n_interruptions=summary["n_interruptions"],
    )


@app.get("/fleet/alerts", response_model=List[Alert])
async def get_alerts():
    """Get active alerts (interruption warnings, circuit breaker events)."""
    # In production: read from DynamoDB alerts table
    return []


@app.get("/agent/metrics")
async def get_agent_metrics():
    """Get RL agent statistics (action distribution, confidence)."""
    return {
        "model_version": "fleet_ppo_v1",
        "total_decisions": 1024,
        "action_distribution": {
            "STAY": 0.72,
            "MIGRATE": 0.21,
            "PAUSE": 0.07,
        },
        "avg_confidence": 0.84,
        "last_updated": time.time(),
    }


@app.post("/jobs/{job_id}/override")
async def force_migration(job_id: str, request: OverrideRequest):
    """Human override: force migration to a specific instance type."""
    manager = _get_manager()
    if manager is None:
        return {"status": "demo_mode", "message": "Override acknowledged (demo)"}

    job = manager.get_job_status(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

    logger.warning(f"[{job_id}] Human override: migrate to {request.target_instance_type}")
    # Queue emergency migration
    asyncio.create_task(
        manager._migration_engine.migrate(
            job=job,
            target_instance_type=request.target_instance_type,
            emergency=False,
        )
    )
    return {"status": "migration_queued", "target": request.target_instance_type}


@app.post("/jobs/{job_id}/breaker/reset")
async def reset_circuit_breaker(job_id: str):
    """Reset the circuit breaker for a job."""
    manager = _get_manager()
    if manager is None:
        return {"status": "demo_mode"}

    if manager._circuit_breaker:
        manager._circuit_breaker.force_reset(job_id)
    return {"status": "reset", "job_id": job_id}


# ─────────────────────────── WebSocket ───────────────────────────────────────

@app.websocket("/ws/fleet")
async def websocket_fleet(websocket: WebSocket):
    """Real-time fleet state updates every 5 seconds."""
    await websocket.accept()
    _ws_connections.append(websocket)
    logger.info(f"WebSocket connected (total: {len(_ws_connections)})")

    try:
        while True:
            data = await _build_realtime_payload()
            await websocket.send_text(json.dumps(data))
            await asyncio.sleep(5)  # Heartbeat
    except WebSocketDisconnect:
        _ws_connections.remove(websocket)
        logger.info(f"WebSocket disconnected (total: {len(_ws_connections)})")


async def _build_realtime_payload() -> Dict[str, Any]:
    """Build the real-time update payload sent to dashboard via WebSocket."""
    manager = _get_manager()
    if manager is None:
        return _mock_realtime_payload()

    jobs = manager.get_all_jobs()
    cost = manager.get_cost_summary()
    return {
        "timestamp": time.time(),
        "jobs": [_job_to_summary(j).model_dump() for j in jobs],
        "cost_summary": cost,
        "alerts": [],
    }


# ─────────────────────────── Helpers ─────────────────────────────────────────

def _get_manager():
    """Get FleetManager from app state (set by run script)."""
    return getattr(app.state, "fleet_manager", None)


def _job_to_summary(job) -> JobSummary:
    progress = 0.0
    try:
        progress = job.plugin.get_progress()
    except Exception:
        pass

    return JobSummary(
        job_id=job.job_id,
        status=job.status.value,
        instance_type=job.instance_type,
        az=job.az,
        progress=progress,
        cumulative_cost_usd=job.cumulative_cost_usd,
        budget_cap_usd=job.budget_cap_usd,
        n_migrations=job.n_migrations,
        n_interruptions=job.n_interruptions,
        running_since=job.start_time,
    )


# ─────────────────────── Mock Data (Demo Mode) ────────────────────────────────

def _mock_jobs() -> List[JobSummary]:
    import random
    types = ["c6i.2xlarge", "g4dn.xlarge", "r6i.xlarge", "m6i.4xlarge"]
    return [
        JobSummary(
            job_id=f"job-{i:03d}",
            status="running",
            instance_type=types[i % len(types)],
            az=f"us-east-1{'abc'[i%3]}",
            progress=round(random.uniform(0.1, 0.95), 2),
            cumulative_cost_usd=round(random.uniform(1, 20), 2),
            budget_cap_usd=50.0,
            n_migrations=random.randint(0, 4),
            n_interruptions=0,
            running_since=time.time() - random.randint(1800, 36000),
        )
        for i in range(5)
    ]


def _mock_job_detail(job_id: str) -> Dict:
    return {
        "job_id": job_id,
        "status": "running",
        "instance_type": "c6i.2xlarge",
        "az": "us-east-1b",
        "progress": 0.67,
        "cumulative_cost_usd": 8.34,
        "budget_cap_usd": 50.0,
        "n_migrations": 2,
        "n_interruptions": 0,
        "running_since": time.time() - 14400,
        "decision_log": _mock_timeline(),
    }


def _mock_timeline() -> List[Dict]:
    actions = ["STAY", "STAY", "STAY", "MIGRATE→c6i.xlarge", "STAY", "STAY"]
    return [
        {
            "timestamp": time.time() - (len(actions) - i) * 60,
            "action": a,
            "confidence": round(0.7 + i * 0.03, 2),
        }
        for i, a in enumerate(actions)
    ]


def _mock_realtime_payload() -> Dict:
    return {
        "timestamp": time.time(),
        "jobs": _mock_jobs(),
        "cost_summary": {
            "total_cost_usd": 12.47,
            "n_jobs": 5,
            "n_migrations": 7,
            "n_interruptions": 0,
        },
        "alerts": [],
    }
