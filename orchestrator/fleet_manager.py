"""
orchestrator/fleet_manager.py
──────────────────────────────
Main async orchestration loop for the AWS Fleet Instance Manager.

Manages up to MAX_CONCURRENT_JOBS=20 jobs simultaneously.
Every 60 seconds per job:
  1. Collect 28-dim observation from telemetry
  2. Query RL agent for action
  3. Execute: STAY | MIGRATE | PAUSE
  4. Update DynamoDB with decision log

Emergency path (2-min termination warning):
  Bypasses the 60s polling loop and immediately triggers checkpoint + migration.

Usage:
    manager = FleetManager(config, agent)
    await manager.start()                           # Main loop
    job_id = await manager.submit_job(job_config)  # Submit a job
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional

import yaml

from agents.ppo_agent import FleetPPOAgent
from orchestrator.circuit_breaker import CircuitBreaker
from orchestrator.migration import MigrationEngine
from orchestrator.warning_handler import TerminationWarningHandler
from plugins.base import WorkloadPlugin
from telemetry.feature_builder import FeatureBuilder, JobState
from telemetry.spot_collector import SpotCollector

logger = logging.getLogger(__name__)


class JobStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    MIGRATING = "migrating"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"


class Action(int, Enum):
    STAY = 0
    # MIGRATE_i = 1..N (dynamic, based on catalog size)
    PAUSE = -1  # Represented as N+1 in env


@dataclass
class ManagedJob:
    """Runtime state of one managed job."""
    job_id: str
    plugin: WorkloadPlugin
    instance_type: str
    instance_id: str
    az: str
    budget_cap_usd: float
    status: JobStatus = JobStatus.PENDING
    start_time: float = field(default_factory=time.time)
    cumulative_cost_usd: float = 0.0
    n_migrations: int = 0
    n_interruptions: int = 0
    last_action: int = 0
    last_action_time: float = 0.0
    decision_log: List[Dict] = field(default_factory=list)


@dataclass
class JobConfig:
    """Configuration for submitting a new job to the fleet manager."""
    plugin_type: str           # "pytorch" | "tensorflow" | "spark" | "generic"
    plugin_config: Dict
    instance_type: str         # Starting instance type
    az: str                    # Starting AZ
    budget_cap_usd: float      # Hard budget cap
    min_vcpus: int = 2         # Minimum vCPUs for migration candidates
    min_ram_gb: float = 4.0    # Minimum RAM for migration candidates


class FleetManager:
    """
    Async fleet manager: coordinates up to 20 concurrent spot jobs.

    Thread model:
        - Main asyncio loop runs _manage_job() coroutines for each job
        - Each job runs on its own 60-second polling cycle
        - Emergency warning handler runs on a faster 5-second cycle
        - All AWS API calls are dispatched to a thread pool executor
    """

    def __init__(self, config_path: str = "config/settings.yaml"):
        with open(config_path) as f:
            self.config = yaml.safe_load(f)

        self.max_jobs = self.config["fleet"]["max_concurrent_jobs"]
        self.min_stay_min = self.config["fleet"]["min_stay_minutes"]
        self.poll_interval = self.config["aws"]["polling_interval_sec"]

        # Core components (lazy initialized in start())
        self._agent: Optional[FleetPPOAgent] = None
        self._spot_collector: Optional[SpotCollector] = None
        self._feature_builder: Optional[FeatureBuilder] = None
        self._migration_engine: Optional[MigrationEngine] = None
        self._circuit_breaker: Optional[CircuitBreaker] = None

        # Job registry
        self._jobs: Dict[str, ManagedJob] = {}
        self._lock = asyncio.Lock()

        # Instance type list (matches environment action indices)
        import yaml as _yaml
        with open("config/instances.yaml") as f:
            catalog_raw = _yaml.safe_load(f)
        self._catalog = {i["type"]: i for i in catalog_raw["instances"]}
        self._instance_types = list(self._catalog.keys())

    # ─────────────────────────── Lifecycle ───────────────────────────────────

    async def start(self, model_path: Optional[str] = None):
        """Initialize all components and start the main management loop."""
        logger.info("FleetManager starting...")

        # Initialize telemetry
        self._spot_collector = SpotCollector(self.config)
        self._feature_builder = FeatureBuilder(self._catalog, self._spot_collector)

        # Initialize RL agent
        if model_path:
            from agents.environment import SpotFleetEnv
            env = SpotFleetEnv()
            self._agent = FleetPPOAgent.load(model_path, env=env)
            logger.info(f"RL agent loaded from {model_path}")
        else:
            logger.warning("No model path provided — using random policy (dev mode)")
            self._agent = None

        # Initialize orchestration components
        self._migration_engine = MigrationEngine(self.config, self._catalog)
        self._circuit_breaker = CircuitBreaker(
            max_migrations=self.config["fleet"]["circuit_breaker_migrations"],
            window_minutes=self.config["fleet"]["circuit_breaker_window_min"],
        )

        # Start background tasks
        asyncio.create_task(self._spot_collector.start())
        asyncio.create_task(self._termination_polling_loop())

        logger.info("FleetManager ready ✓")
        # Run job management loops forever
        await self._main_loop()

    async def _main_loop(self):
        """Main scheduling loop — dispatches per-job coroutines."""
        while True:
            async with self._lock:
                active_jobs = [
                    j for j in self._jobs.values()
                    if j.status in (JobStatus.RUNNING, JobStatus.PAUSED)
                ]

            tasks = [self._manage_job_step(job) for job in active_jobs]
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)

            await asyncio.sleep(self.poll_interval)

    # ─────────────────────────── Job Submission ───────────────────────────────

    async def submit_job(self, job_config: JobConfig) -> str:
        """
        Register a new job for management.

        Returns:
            job_id: UUID string identifying this job
        """
        async with self._lock:
            if len(self._jobs) >= self.max_jobs:
                raise RuntimeError(
                    f"Job cap reached ({self.max_jobs}). Cannot accept new job."
                )

        job_id = str(uuid.uuid4())[:8]

        # Instantiate plugin
        plugin = self._create_plugin(job_id, job_config)

        job = ManagedJob(
            job_id=job_id,
            plugin=plugin,
            instance_type=job_config.instance_type,
            instance_id=f"sim-{job_id}",  # Real instance ID set by provisioner
            az=job_config.az,
            budget_cap_usd=job_config.budget_cap_usd,
            status=JobStatus.RUNNING,
        )

        async with self._lock:
            self._jobs[job_id] = job

        logger.info(f"Job {job_id} submitted on {job_config.instance_type} ({job_config.az})")
        return job_id

    # ─────────────────────────── Per-Job Step ────────────────────────────────

    async def _manage_job_step(self, job: ManagedJob):
        """Execute one management step for a single job."""
        try:
            # Build observation
            job_state = self._build_job_state(job)
            obs = self._feature_builder.build(job_state)

            # Query RL agent
            if self._agent is not None:
                action, _ = self._agent.predict(obs, deterministic=True)
                action_probs = self._agent.get_action_probabilities(obs)
            else:
                # Fallback: random policy (dev/test mode)
                import random
                action = random.randint(0, len(self._instance_types))
                action_probs = None

            # Log decision
            decision = {
                "timestamp": time.time(),
                "action": action,
                "obs_snapshot": obs[:10].tolist(),  # First 10 features for logs
                "action_probs": action_probs.tolist() if action_probs is not None else [],
            }
            job.decision_log.append(decision)

            # Check circuit breaker
            if self._circuit_breaker.is_open(job.job_id):
                logger.warning(f"[{job.job_id}] Circuit breaker open — forcing STAY")
                action = 0  # STAY

            # Check minimum stay time
            if action != 0 and action != len(self._instance_types) + 1:
                min_stay_sec = self.min_stay_min * 60
                time_on_instance = time.time() - job.last_action_time
                if time_on_instance < min_stay_sec:
                    logger.debug(f"[{job.job_id}] Min stay time not met — converting MIGRATE to STAY")
                    action = 0

            # Execute action
            await self._execute_action(job, action, obs)

            job.last_action = action
            job.last_action_time = time.time()

        except Exception as e:
            logger.error(f"[{job.job_id}] Error in management step: {e}", exc_info=True)

    async def _execute_action(self, job: ManagedJob, action: int, obs):
        """Execute the chosen action for a job."""
        n_types = len(self._instance_types)

        if action == 0:
            # STAY — update cost only
            cost = self._estimate_step_cost(job.instance_type, self.poll_interval)
            job.cumulative_cost_usd += cost
            logger.debug(f"[{job.job_id}] STAY on {job.instance_type} (${cost:.4f})")

        elif 1 <= action <= n_types:
            # MIGRATE to instance_types[action-1]
            target_type = self._instance_types[action - 1]
            if target_type == job.instance_type:
                return  # No-op migration

            logger.info(f"[{job.job_id}] MIGRATE: {job.instance_type} → {target_type}")
            job.status = JobStatus.MIGRATING

            success = await self._migration_engine.migrate(
                job=job,
                target_instance_type=target_type,
                emergency=False,
            )

            if success:
                job.instance_type = target_type
                job.n_migrations += 1
                self._circuit_breaker.record_migration(job.job_id)
                logger.info(f"[{job.job_id}] Migration complete → {target_type}")
            else:
                logger.error(f"[{job.job_id}] Migration failed — remaining on {job.instance_type}")

            job.status = JobStatus.RUNNING

        elif action == n_types + 1:
            # PAUSE
            logger.info(f"[{job.job_id}] PAUSE — checkpointing and shutting down")
            job.status = JobStatus.PAUSED

    # ─────────────────────────── Emergency Path ───────────────────────────────

    async def _termination_polling_loop(self):
        """
        Fast-polling loop (every 5 seconds) that checks for AWS termination notices.
        When detected, triggers emergency checkpoint and migration immediately.
        """
        while True:
            await asyncio.sleep(5)
            async with self._lock:
                running_jobs = [
                    j for j in self._jobs.values()
                    if j.status == JobStatus.RUNNING
                ]

            for job in running_jobs:
                # Check termination notice via IMDS (from the spot instance)
                warning = SpotCollector.check_termination_notice()
                if warning:
                    logger.critical(
                        f"[{job.job_id}] ⚠️  TERMINATION WARNING — starting emergency migration"
                    )
                    asyncio.create_task(self._emergency_migrate(job))

    async def _emergency_migrate(self, job: ManagedJob):
        """Emergency migration triggered by 2-minute termination notice."""
        job.status = JobStatus.MIGRATING
        job.plugin.on_termination_warning()

        # Pick safest alternative (lowest interruption risk, meets requirements)
        target_type = self._migration_engine.pick_emergency_target(
            current_type=job.instance_type,
            catalog=self._catalog,
        )

        await self._migration_engine.migrate(
            job=job,
            target_instance_type=target_type,
            emergency=True,
        )
        job.status = JobStatus.RUNNING

    # ─────────────────────────── Helpers ─────────────────────────────────────

    def _build_job_state(self, job: ManagedJob) -> JobState:
        progress = job.plugin.get_progress()
        remaining = job.plugin.estimate_remaining_seconds()
        return JobState(
            job_id=job.job_id,
            instance_type=job.instance_type,
            instance_id=job.instance_id,
            az=job.az,
            uptime_seconds=time.time() - job.start_time,
            cumulative_cost_usd=job.cumulative_cost_usd,
            job_progress=progress,
            estimated_remaining_sec=remaining,
            budget_cap_usd=job.budget_cap_usd,
            n_interruptions=job.n_interruptions,
            n_migrations=job.n_migrations,
        )

    def _estimate_step_cost(self, instance_type: str, duration_sec: float) -> float:
        snap = self._spot_collector.get_latest(instance_type, "us-east-1a") if self._spot_collector else None
        price_per_hr = snap.price if snap else self._catalog.get(instance_type, {}).get("on_demand_price", 0.1)
        return (price_per_hr / 3600) * duration_sec

    def _create_plugin(self, job_id: str, config: JobConfig) -> WorkloadPlugin:
        from plugins.pytorch_plugin import PyTorchPlugin
        from plugins.generic_plugin import GenericPlugin
        plugins_map = {
            "pytorch": PyTorchPlugin,
            "generic": GenericPlugin,
        }
        plugin_cls = plugins_map.get(config.plugin_type, GenericPlugin)
        return plugin_cls(job_id=job_id, config=self.config)

    # ─────────────────────────── Public Status API ────────────────────────────

    def get_job_status(self, job_id: str) -> Optional[ManagedJob]:
        return self._jobs.get(job_id)

    def get_all_jobs(self) -> List[ManagedJob]:
        return list(self._jobs.values())

    def get_cost_summary(self) -> Dict:
        total = sum(j.cumulative_cost_usd for j in self._jobs.values())
        return {
            "total_cost_usd": total,
            "n_jobs": len(self._jobs),
            "n_migrations": sum(j.n_migrations for j in self._jobs.values()),
            "n_interruptions": sum(j.n_interruptions for j in self._jobs.values()),
        }
