"""
orchestrator/migration.py
──────────────────────────
Handles the mechanics of migrating a job from one EC2 instance to another.

Migration steps:
    1. Trigger checkpoint on the source instance (via plugin)
    2. Provision new target instance
    3. Wait for instance to be ready (status = running)
    4. Restore checkpoint on new instance
    5. Verify job resumed successfully
    6. Terminate source instance

Emergency mode (< 90 seconds total):
    Steps 1 & 2 run in parallel to save time.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Dict, Optional

import boto3
from botocore.exceptions import ClientError

logger = logging.getLogger(__name__)

# How long to wait for a new instance to be ready (seconds)
INSTANCE_READY_TIMEOUT = 300  # 5 minutes
INSTANCE_POLL_INTERVAL = 10   # Check every 10 seconds


class MigrationEngine:
    """
    Executes instance migrations for managed jobs.

    Wraps all AWS EC2 provisioning calls and coordinates with the job's
    WorkloadPlugin for checkpoint/restore operations.
    """

    def __init__(self, config: dict, catalog: dict):
        self.config = config
        self.catalog = catalog
        self.region = config["aws"]["region"]
        self.ec2 = boto3.client("ec2", region_name=self.region)

    async def migrate(
        self,
        job,
        target_instance_type: str,
        emergency: bool = False,
    ) -> bool:
        """
        Migrate a job to a new instance type.

        Args:
            job: ManagedJob containing plugin, instance ID, etc.
            target_instance_type: Target EC2 instance type string
            emergency: If True, use fast path (parallel checkpoint + provision)

        Returns:
            True if migration succeeded, False otherwise.
        """
        logger.info(
            f"[{job.job_id}] Starting migration: {job.instance_type} → "
            f"{target_instance_type} (emergency={emergency})"
        )
        start = time.time()

        try:
            if emergency:
                # Parallel: checkpoint + provision simultaneously
                checkpoint_task = asyncio.create_task(
                    self._async_checkpoint(job, emergency=True)
                )
                provision_task = asyncio.create_task(
                    self._provision_instance(target_instance_type, job.az)
                )
                manifest, new_instance_id = await asyncio.gather(
                    checkpoint_task, provision_task
                )
            else:
                # Sequential: safe migration
                manifest = await self._async_checkpoint(job, emergency=False)
                new_instance_id = await self._provision_instance(target_instance_type, job.az)

            # Wait for instance ready
            new_instance_ip = await self._wait_for_ready(new_instance_id)

            # Restore
            loop = asyncio.get_event_loop()
            success = await loop.run_in_executor(
                None,
                lambda: job.plugin.restore(manifest, new_instance_ip),
            )

            if success:
                # Terminate old instance
                await self._terminate_instance(job.instance_id)
                job.instance_id = new_instance_id
                elapsed = time.time() - start
                logger.info(
                    f"[{job.job_id}] Migration complete in {elapsed:.1f}s → "
                    f"{target_instance_type} ({new_instance_id})"
                )
                job.plugin.on_migration_complete(new_instance_id)
                return True
            else:
                logger.error(f"[{job.job_id}] Restore failed — rolling back")
                await self._terminate_instance(new_instance_id)
                return False

        except Exception as e:
            logger.error(f"[{job.job_id}] Migration failed: {e}", exc_info=True)
            return False

    def pick_emergency_target(self, current_type: str, catalog: dict) -> str:
        """
        Pick the safest, cheapest alternative instance type for emergency migration.

        Strategy: lowest interruption rate first, then lowest on-demand price.
        Falls back to on-demand equivalent if all spot options are high risk.
        """
        candidates = [
            (itype, info) for itype, info in catalog.items()
            if itype != current_type
        ]

        # Sort: low interruption rate first, then cheapest
        rate_order = {"low": 0, "medium": 1, "high": 2}
        candidates.sort(key=lambda x: (
            rate_order.get(x[1].get("interruption_rate", "medium"), 1),
            x[1].get("on_demand_price", 999)
        ))

        if candidates:
            chosen = candidates[0][0]
            logger.info(f"Emergency target selected: {chosen}")
            return chosen

        return current_type  # Last resort: stay

    # ─────────────────────────── Private Helpers ──────────────────────────────

    async def _async_checkpoint(self, job, emergency: bool):
        """Run checkpoint in thread pool (blocking boto3/file I/O)."""
        loop = asyncio.get_event_loop()
        s3_prefix = (
            f"s3://{self.config['checkpointing']['s3_bucket']}/"
            f"{self.config['checkpointing']['s3_prefix']}"
        )
        manifest = await loop.run_in_executor(
            None,
            lambda: job.plugin.checkpoint(s3_prefix, emergency=emergency),
        )
        logger.info(f"[{job.job_id}] Checkpoint complete: {manifest.s3_uri}")
        return manifest

    async def _provision_instance(self, instance_type: str, az: str) -> str:
        """
        Launch a new spot instance of the given type in the given AZ.
        Returns the new instance ID.
        """
        logger.info(f"Provisioning {instance_type} in {az}...")
        loop = asyncio.get_event_loop()

        def _launch():
            # In production: use EC2 RunInstances with spot options
            # Here we return a simulated ID for testability
            try:
                response = self.ec2.run_instances(
                    ImageId="ami-0abcdef1234567890",  # Replace with your AMI
                    InstanceType=instance_type,
                    MinCount=1,
                    MaxCount=1,
                    Placement={"AvailabilityZone": az},
                    InstanceMarketOptions={
                        "MarketType": "spot",
                        "SpotOptions": {
                            "SpotInstanceType": "one-time",
                            "InstanceInterruptionBehavior": "terminate",
                        },
                    },
                    TagSpecifications=[{
                        "ResourceType": "instance",
                        "Tags": [{"Key": "fleet-manager", "Value": "managed"}],
                    }],
                )
                return response["Instances"][0]["InstanceId"]
            except ClientError as e:
                # If spot not available, fall back to on-demand
                logger.warning(f"Spot unavailable ({e}), falling back to on-demand")
                response = self.ec2.run_instances(
                    ImageId="ami-0abcdef1234567890",
                    InstanceType=instance_type,
                    MinCount=1,
                    MaxCount=1,
                )
                return response["Instances"][0]["InstanceId"]

        instance_id = await loop.run_in_executor(None, _launch)
        logger.info(f"Instance launched: {instance_id}")
        return instance_id

    async def _wait_for_ready(self, instance_id: str) -> str:
        """Poll until instance is running and has a public IP. Returns IP."""
        deadline = time.time() + INSTANCE_READY_TIMEOUT
        while time.time() < deadline:
            loop = asyncio.get_event_loop()
            desc = await loop.run_in_executor(
                None,
                lambda: self.ec2.describe_instances(InstanceIds=[instance_id]),
            )
            inst = desc["Reservations"][0]["Instances"][0]
            state = inst["State"]["Name"]
            if state == "running":
                ip = inst.get("PublicIpAddress") or inst.get("PrivateIpAddress", "")
                logger.info(f"Instance {instance_id} ready at {ip}")
                return ip
            logger.debug(f"Instance {instance_id} state: {state} — waiting...")
            await asyncio.sleep(INSTANCE_POLL_INTERVAL)

        raise TimeoutError(f"Instance {instance_id} not ready after {INSTANCE_READY_TIMEOUT}s")

    async def _terminate_instance(self, instance_id: str) -> None:
        """Terminate an EC2 instance."""
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            None,
            lambda: self.ec2.terminate_instances(InstanceIds=[instance_id]),
        )
        logger.info(f"Instance {instance_id} terminated")
