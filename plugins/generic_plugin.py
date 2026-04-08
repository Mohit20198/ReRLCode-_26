"""
plugins/generic_plugin.py
──────────────────────────
Generic workload plugin for black-box/custom long-running processes.
Uses EBS snapshot for checkpoint (slower but universal).
Falls back gracefully when no progress tracking is available.
"""

from __future__ import annotations

import datetime
import logging
import os
import sys
import time
from typing import Any, Dict

import boto3

from plugins.base import (
    CheckpointError,
    CheckpointManifest,
    RestoreError,
    WorkloadPlugin,
)

logger = logging.getLogger(__name__)


class GenericPlugin(WorkloadPlugin):
    """
    Generic plugin: EBS snapshot-based checkpoint for any process.

    Limitations:
        - EBS snapshots take 5-10 minutes (not suitable for < 90s emergency)
        - Progress is estimated from elapsed time (inaccurate)
        - For emergency migration, uses a fast S3 tarball of /tmp/ instead

    Best for: custom scripts, legacy workloads, anything that can't export state.
    """

    PROGRESS_FILE = "/tmp/fleet_progress.json"

    def __init__(self, job_id: str, config: Dict[str, Any]):
        super().__init__(job_id, config)
        self.ec2 = boto3.client("ec2", region_name=config.get("aws", {}).get("region", "us-east-1"))
        self.s3 = boto3.client("s3")
        self.bucket = config.get("checkpointing", {}).get("s3_bucket", "fleet-checkpoints")
        self._start_time = time.time()
        self._estimated_duration_sec = 12 * 3600  # Default: 12 hours

    def checkpoint(self, destination_s3_uri: str, emergency: bool = False) -> CheckpointManifest:
        if emergency:
            # Fast path: tar /tmp/ and upload to S3 (best effort)
            return self._fast_checkpoint(destination_s3_uri)
        else:
            # Full EBS snapshot path (slower, more complete)
            return self._ebs_snapshot_checkpoint(destination_s3_uri)

    def restore(self, manifest: CheckpointManifest, target_instance_ip: str) -> bool:
        logger.info(f"[{self.job_id}] Restoring generic checkpoint from {manifest.s3_uri}")
        # For EBS snapshots: mount to new instance
        # For tar: download and extract
        if manifest.metadata.get("type") == "fast_tar":
            return self._restore_tar(manifest, target_instance_ip)
        else:
            logger.warning(f"[{self.job_id}] EBS restore requires manual AMI launch")
            return True  # Return True; orchestrator handles EBS restoration

    def get_progress(self) -> float:
        # Try progress file first
        try:
            if os.path.exists(self.PROGRESS_FILE):
                import json
                with open(self.PROGRESS_FILE) as f:
                    data = json.load(f)
                    return float(data.get("progress", 0.0))
        except Exception:
            pass
        # Fall back to time-based estimation
        elapsed = time.time() - self._start_time
        return min(elapsed / self._estimated_duration_sec, 0.99)

    def estimate_remaining_seconds(self) -> int:
        progress = self.get_progress()
        elapsed = time.time() - self._start_time
        if progress <= 0 or elapsed <= 0:
            return self._estimated_duration_sec
        rate = progress / elapsed
        remaining = (1.0 - progress) / rate if rate > 0 else sys.maxsize
        return int(remaining)

    def get_checkpoint_size_estimate_bytes(self) -> int:
        # EBS snapshots: estimate from volume size (default 100GB)
        return 100 * 1024 * 1024 * 1024

    # ─────────────────────────── Fast Checkpoint (Emergency) ─────────────────

    def _fast_checkpoint(self, destination_s3_uri: str) -> CheckpointManifest:
        """Tar /tmp/ working directory and upload to S3. < 30 seconds for small state."""
        import subprocess
        import tempfile

        logger.info(f"[{self.job_id}] Fast checkpoint (emergency) — tar /tmp/")
        tar_path = f"/tmp/fleet_emergency_{self.job_id}.tar.gz"

        try:
            subprocess.run(
                ["tar", "-czf", tar_path, "/tmp/", "--exclude=/tmp/fleet_emergency*"],
                timeout=60,
                check=True,
            )
        except subprocess.TimeoutExpired:
            raise CheckpointError(f"[{self.job_id}] Emergency tar timed out")
        except subprocess.CalledProcessError as e:
            raise CheckpointError(f"[{self.job_id}] tar failed: {e}")

        size = os.path.getsize(tar_path)
        s3_key = f"checkpoints/{self.job_id}/emergency_{int(time.time())}.tar.gz"
        self.s3.upload_file(tar_path, self.bucket, s3_key)
        os.remove(tar_path)

        return CheckpointManifest(
            job_id=self.job_id,
            plugin_type="generic",
            s3_uri=f"s3://{self.bucket}/{s3_key}",
            created_at=datetime.datetime.now(datetime.timezone.utc),
            size_bytes=size,
            metadata={"type": "fast_tar", "source": "/tmp/"},
        )

    def _ebs_snapshot_checkpoint(self, destination_s3_uri: str) -> CheckpointManifest:
        """Create EBS snapshot of the root volume (takes 5-10 minutes)."""
        logger.info(f"[{self.job_id}] EBS snapshot checkpoint starting...")
        # Get the instance's root volume
        instance_id = self._get_instance_id()
        if not instance_id:
            raise CheckpointError("Cannot determine instance ID — not running on EC2?")

        desc = self.ec2.describe_instances(InstanceIds=[instance_id])
        volumes = desc["Reservations"][0]["Instances"][0].get("BlockDeviceMappings", [])
        if not volumes:
            raise CheckpointError("No EBS volumes found")

        volume_id = volumes[0]["Ebs"]["VolumeId"]
        snapshot = self.ec2.create_snapshot(
            VolumeId=volume_id,
            Description=f"fleet-manager-checkpoint-{self.job_id}",
            TagSpecifications=[{
                "ResourceType": "snapshot",
                "Tags": [{"Key": "fleet-manager-job", "Value": self.job_id}],
            }],
        )

        return CheckpointManifest(
            job_id=self.job_id,
            plugin_type="generic",
            s3_uri=f"ebs-snapshot://{snapshot['SnapshotId']}",
            created_at=datetime.datetime.now(datetime.timezone.utc),
            size_bytes=0,  # Unknown until snapshot completes
            metadata={"type": "ebs_snapshot", "snapshot_id": snapshot["SnapshotId"]},
        )

    def _restore_tar(self, manifest: CheckpointManifest, target_ip: str) -> bool:
        import subprocess
        local_path = f"/tmp/fleet_restore_{self.job_id}.tar.gz"
        bucket, key = manifest.s3_uri.replace("s3://", "").split("/", 1)
        self.s3.download_file(bucket, key, local_path)
        subprocess.run(["tar", "-xzf", local_path, "-C", "/"], check=True)
        os.remove(local_path)
        return True

    @staticmethod
    def _get_instance_id() -> str:
        try:
            with urllib.request.urlopen(
                "http://169.254.169.254/latest/meta-data/instance-id", timeout=1
            ) as r:
                return r.read().decode()
        except Exception:
            return ""

import urllib.request  # noqa: E402 (needed for _get_instance_id)
