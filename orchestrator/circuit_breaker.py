"""
orchestrator/circuit_breaker.py
────────────────────────────────
Safety rail that pauses RL agent automation if it causes too many migrations
in a short window. Prevents thrashing (rapid back-and-forth migrations) which
would waste money on overhead and destabilize jobs.

Configuration (from settings.yaml):
    circuit_breaker_migrations: 3    # Max migrations in window
    circuit_breaker_window_min: 10   # Window size in minutes

When the breaker opens:
  - All further automatic migrations for that job are blocked
  - The dashboard shows a CIRCUIT BROKEN alert
  - After the window passes, the breaker automatically resets (half-open → closed)
"""

from __future__ import annotations

import logging
import time
from collections import defaultdict, deque
from typing import Dict, Deque

logger = logging.getLogger(__name__)


class CircuitBreaker:
    """
    Per-job circuit breaker for migration rate limiting.

    States:
        CLOSED  — normal operation, agent can act freely
        OPEN    — too many migrations, agent is blocked (STAY forced)
        HALF_OPEN — window expired, testing if safe to resume
    """

    def __init__(self, max_migrations: int = 3, window_minutes: int = 10):
        self.max_migrations = max_migrations
        self.window_sec = window_minutes * 60
        # job_id → deque of migration timestamps
        self._migration_times: Dict[str, Deque[float]] = defaultdict(deque)
        self._open_until: Dict[str, float] = {}  # job_id → timestamp when breaker resets

    def is_open(self, job_id: str) -> bool:
        """Return True if circuit breaker is open (migrations blocked)."""
        now = time.time()

        # Check if breaker was manually opened
        if job_id in self._open_until:
            if now < self._open_until[job_id]:
                return True
            else:
                # Timeout expired - move to HALF_OPEN then CLOSED
                del self._open_until[job_id]
                self._migration_times[job_id].clear()
                logger.info(f"[{job_id}] Circuit breaker RESET → CLOSED")

        # Prune old timestamps outside the window
        timestamps = self._migration_times[job_id]
        while timestamps and (now - timestamps[0]) > self.window_sec:
            timestamps.popleft()

        if len(timestamps) >= self.max_migrations:
            # Trip the breaker
            self._open_until[job_id] = now + self.window_sec
            logger.warning(
                f"[{job_id}] 🔴 Circuit breaker OPEN — {len(timestamps)} migrations in "
                f"{self.window_sec/60:.0f} min window. Blocking for {self.window_sec/60:.0f} min."
            )
            return True

        return False

    def record_migration(self, job_id: str) -> None:
        """Record that a migration occurred for this job."""
        self._migration_times[job_id].append(time.time())
        count = len(self._migration_times[job_id])
        logger.debug(f"[{job_id}] Migration recorded ({count}/{self.max_migrations} in window)")

    def force_open(self, job_id: str, duration_minutes: float = 30.0) -> None:
        """Manually open the circuit breaker (e.g., from dashboard override)."""
        self._open_until[job_id] = time.time() + duration_minutes * 60
        logger.warning(f"[{job_id}] Circuit breaker FORCE OPENED for {duration_minutes} min")

    def force_reset(self, job_id: str) -> None:
        """Manually reset the circuit breaker."""
        self._open_until.pop(job_id, None)
        self._migration_times[job_id].clear()
        logger.info(f"[{job_id}] Circuit breaker FORCE RESET")

    def get_status(self, job_id: str) -> dict:
        """Get circuit breaker status for dashboard display."""
        now = time.time()
        timestamps = self._migration_times.get(job_id, deque())

        # Prune
        recent = [t for t in timestamps if (now - t) <= self.window_sec]

        is_open = job_id in self._open_until and now < self._open_until.get(job_id, 0)
        resets_in = max(0, self._open_until.get(job_id, 0) - now) if is_open else 0

        return {
            "job_id": job_id,
            "state": "OPEN" if is_open else "CLOSED",
            "migrations_in_window": len(recent),
            "max_migrations": self.max_migrations,
            "window_minutes": self.window_sec / 60,
            "resets_in_seconds": resets_in,
        }
