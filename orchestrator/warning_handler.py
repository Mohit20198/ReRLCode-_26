"""
orchestrator/warning_handler.py
────────────────────────────────
Polls EC2 Instance Metadata Service (IMDS) for spot termination notices.
Runs on a 5-second cycle (much faster than the 60-second main loop).
When a 2-minute warning is detected, triggers emergency checkpoint + migration.
"""

from __future__ import annotations

import asyncio
import logging
import time
import urllib.request
from typing import Callable, Optional

logger = logging.getLogger(__name__)

METADATA_BASE_URL = "http://169.254.169.254"
TERMINATION_PATH = "/latest/meta-data/spot/termination-time"
POLL_INTERVAL_FAST = 5   # seconds
METADATA_TIMEOUT = 1     # socket timeout in seconds


class TerminationWarningHandler:
    """
    Monitors EC2 IMDS for spot termination notices on the current instance.

    This must run FROM WITHIN the spot instance (not from the orchestrator).
    The orchestrator runs externally (e.g., as a Lambda or ECS Fargate task)
    and receives notifications via SQS/SNS from this agent running on each job instance.
    """

    def __init__(
        self,
        on_warning_detected: Callable[[], None],
        metadata_url: str = METADATA_BASE_URL,
    ):
        """
        Args:
            on_warning_detected: Callback to invoke when termination notice detected.
                Should trigger emergency checkpoint + migration.
            metadata_url: Override for testing (e.g., mocked IMDS).
        """
        self.on_warning = on_warning_detected
        self.metadata_url = metadata_url
        self._active = False
        self._warning_detected_at: Optional[float] = None

    async def start(self):
        """Run the termination warning polling loop forever."""
        self._active = True
        logger.info(f"TerminationWarningHandler started (polling every {POLL_INTERVAL_FAST}s)")

        while self._active:
            try:
                detected = await asyncio.get_event_loop().run_in_executor(
                    None, self._check_imds
                )
                if detected and self._warning_detected_at is None:
                    self._warning_detected_at = time.time()
                    logger.critical(
                        "⚠️  AWS SPOT TERMINATION NOTICE RECEIVED — "
                        f"~2 minutes to migrate. Triggering emergency protocol."
                    )
                    # Call the emergency callback
                    try:
                        if asyncio.iscoroutinefunction(self.on_warning):
                            await self.on_warning()
                        else:
                            self.on_warning()
                    except Exception as e:
                        logger.error(f"Error in warning callback: {e}", exc_info=True)

            except Exception as e:
                logger.debug(f"IMDS poll error (normal outside EC2): {e}")

            await asyncio.sleep(POLL_INTERVAL_FAST)

    def stop(self):
        self._active = False

    def _check_imds(self) -> bool:
        """
        Check IMDS for termination notice.
        Returns True if termination is scheduled, False otherwise.
        Must complete within METADATA_TIMEOUT seconds.
        """
        try:
            url = f"{self.metadata_url}{TERMINATION_PATH}"
            req = urllib.request.Request(url)
            with urllib.request.urlopen(req, timeout=METADATA_TIMEOUT) as resp:
                # HTTP 200 = termination notice present
                return resp.status == 200
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return False  # No notice
            return False
        except Exception:
            return False  # Not on EC2, or IMDS unavailable
