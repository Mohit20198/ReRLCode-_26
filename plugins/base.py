"""
plugins/base.py
───────────────
Abstract base class for all workload plugins.
Each workload type (PyTorch, TensorFlow, Spark, Generic) implements this
interface so the orchestrator can checkpoint/restore jobs uniformly.
"""

from __future__ import annotations

import abc
import dataclasses
import datetime
from typing import Any, Dict, Optional


@dataclasses.dataclass
class CheckpointManifest:
    """
    Metadata about a saved checkpoint.
    Stored alongside the checkpoint data so restore can reconstruct exactly.
    """
    job_id: str
    plugin_type: str              # "pytorch" | "tensorflow" | "spark" | "generic"
    s3_uri: str                   # s3://bucket/prefix/job_id/
    created_at: datetime.datetime
    size_bytes: int
    metadata: Dict[str, Any]      # Plugin-specific metadata (epoch, step, partition, etc.)
    is_incremental: bool = False  # True if only a diff was saved vs full state


class WorkloadPlugin(abc.ABC):
    """
    Abstract workload adapter.

    Subclasses must implement all @abstractmethod methods.
    The orchestrator calls these methods to checkpoint/restore jobs
    during migrations and emergency reclamation events.

    Emergency constraint:
        checkpoint() MUST complete in < 90 seconds when called with
        emergency=True (during a 2-minute termination warning).
    """

    def __init__(self, job_id: str, config: Dict[str, Any]):
        self.job_id = job_id
        self.config = config

    # ─────────────────────────── Required Methods ─────────────────────────────

    @abc.abstractmethod
    def checkpoint(
        self,
        destination_s3_uri: str,
        emergency: bool = False,
    ) -> CheckpointManifest:
        """
        Save the current job state to S3.

        Args:
            destination_s3_uri: S3 URI where checkpoint should be saved.
                Example: "s3://my-bucket/checkpoints/job-abc123/"
            emergency: If True, must complete within 90 seconds.
                Use faster/lossy strategies if needed (e.g., skip optimizer state).

        Returns:
            CheckpointManifest describing what was saved.

        Raises:
            CheckpointTimeoutError: If emergency=True and save exceeded 90s.
            CheckpointError: For any other failure.
        """
        ...

    @abc.abstractmethod
    def restore(
        self,
        manifest: CheckpointManifest,
        target_instance_ip: str,
    ) -> bool:
        """
        Restore job state from a checkpoint onto a new instance.

        Args:
            manifest: The manifest returned by a previous checkpoint() call.
            target_instance_ip: IP address of the new instance where job will resume.

        Returns:
            True if restore succeeded, False if partial restore (job may re-run some work).

        Raises:
            RestoreError: If restore failed completely.
        """
        ...

    @abc.abstractmethod
    def get_progress(self) -> float:
        """
        Return job progress as a float in [0.0, 1.0].

        0.0 = not started, 1.0 = complete.
        Used by the RL agent as an observation feature.
        """
        ...

    @abc.abstractmethod
    def estimate_remaining_seconds(self) -> int:
        """
        Estimate seconds until job completes at current rate.

        Used by the RL agent to assess urgency.
        Return sys.maxsize if unknown.
        """
        ...

    @abc.abstractmethod
    def get_checkpoint_size_estimate_bytes(self) -> int:
        """
        Estimate size of a full checkpoint in bytes.

        Used by the orchestrator to estimate migration time
        and decide whether emergency checkpoint will fit in 90 seconds.
        """
        ...

    # ─────────────────────────── Optional Hooks ───────────────────────────────

    def on_migration_start(self, target_instance_type: str) -> None:
        """Called just before migration begins. Override to pause work cleanly."""
        pass

    def on_migration_complete(self, new_instance_id: str) -> None:
        """Called after successful restore on new instance. Override to resume work."""
        pass

    def on_termination_warning(self) -> None:
        """Called immediately when a 2-minute AWS termination notice is detected."""
        pass


# ──────────────────────────── Custom Exceptions ───────────────────────────────

class CheckpointError(Exception):
    """Raised when a checkpoint operation fails."""


class CheckpointTimeoutError(CheckpointError):
    """Raised when emergency checkpoint exceeds the 90-second deadline."""


class RestoreError(Exception):
    """Raised when a restore operation fails completely."""
