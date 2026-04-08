"""
telemetry/feature_builder.py
────────────────────────────
Assembles a 28-dimensional observation vector from:
  - SpotMarketSnapshot (current instance + top-8 alternatives)
  - CloudWatch instance metrics
  - Job progress & time remaining
  - Termination warning flag

This vector is fed directly into the RL agent's observation space.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

from telemetry.spot_collector import SpotCollector, SpotMarketSnapshot

logger = logging.getLogger(__name__)

# ─────────────────────────── Feature Index Map ───────────────────────────────
# Slot 0-4   : Current instance state (5 features)
# Slot 5-12  : Spot market signals for current instance (8 features)
# Slot 13-16 : AZ supply / capacity scores (4 features)
# Slot 17-24 : Alternative spot prices — top-8 candidates (8 features)
# Slot 25    : Termination warning flag (1 feature)
# Slot 26-27 : Interruption context (2 features)
# Total = 28
FEATURE_DIM = 28

# Encode instance families to integers for the RL observation
FAMILY_ENCODING = {"general": 0, "compute": 1, "memory": 2, "gpu": 3}
INTERRUPTION_RATE_ENCODING = {"low": 0, "medium": 1, "high": 2}


@dataclass
class JobState:
    """Runtime state of a managed job — updated by the orchestrator."""
    job_id: str
    instance_type: str
    instance_id: str
    az: str
    uptime_seconds: float         # Seconds since job started on current instance
    cumulative_cost_usd: float    # Total $ spent so far across all instances
    job_progress: float           # 0.0 → 1.0
    estimated_remaining_sec: int  # Seconds until job completes at current rate
    budget_cap_usd: float         # Hard budget cap for this job
    n_interruptions: int          # Interruptions suffered (real, not simulated)
    n_migrations: int             # Number of migrations performed


class FeatureBuilder:
    """
    Builds the 28-dim observation vector for the RL agent.

    Usage:
        builder = FeatureBuilder(catalog, spot_collector)
        obs = builder.build(job_state, termination_warning=False)
    """

    def __init__(
        self,
        catalog: Dict,
        spot_collector: SpotCollector,
        top_k_alternatives: int = 8,
    ):
        self.catalog = catalog           # {instance_type: config_dict}
        self.spot = spot_collector
        self.top_k = top_k_alternatives
        self._instance_types = list(catalog.keys())

    def build(
        self,
        job: JobState,
        termination_warning: bool = False,
        interruptions_last_24h: int = 0,
        migrations_last_24h: int = 0,
    ) -> np.ndarray:
        """
        Build a 28-dim float32 observation vector.

        Args:
            job: Current job state
            termination_warning: True if AWS 2-min termination notice is active
            interruptions_last_24h: Fleet-wide interruptions in last 24h for this type/AZ
            migrations_last_24h: Migrations performed today for this job

        Returns:
            np.ndarray of shape (28,), dtype float32
        """
        obs = np.zeros(FEATURE_DIM, dtype=np.float32)

        # ── Slot 0-4: Current Instance State ─────────────────────────────────
        cat = self.catalog.get(job.instance_type, {})
        family_id = FAMILY_ENCODING.get(cat.get("family", "general"), 0)
        obs[0] = float(family_id) / 3.0            # Normalized 0-1
        obs[1] = min(job.uptime_seconds / 86400.0, 1.0)  # Normalized to 1 day
        obs[2] = min(job.cumulative_cost_usd / job.budget_cap_usd, 2.0)  # Budget fraction
        obs[3] = np.clip(job.job_progress, 0.0, 1.0)
        obs[4] = min(job.estimated_remaining_sec / 86400.0, 1.0)

        # ── Slot 5-12: Spot Market Signals for Current Instance ───────────────
        snap = self.spot.get_latest(job.instance_type, job.az)
        on_demand = cat.get("on_demand_price", 1.0)

        if snap:
            obs[5] = snap.price / on_demand              # Relative to on-demand
            obs[6] = snap.price_5m_avg / on_demand
            obs[7] = snap.price_30m_avg / on_demand
            obs[8] = snap.price_60m_avg / on_demand
            obs[9] = np.clip(snap.pct_above_baseline / 100.0, -1.0, 2.0)
            obs[10] = snap.interruption_rate_num
            # Price trend: positive = rising, negative = falling
            obs[11] = np.clip((snap.price - snap.price_30m_avg) / (on_demand + 1e-9), -1.0, 1.0)
            obs[12] = float(INTERRUPTION_RATE_ENCODING.get(snap.interruption_rate, 1)) / 2.0
        else:
            # Fallback — no data yet, assume moderate conditions
            obs[5:13] = 0.5

        # ── Slot 13-16: AZ Capacity Scores ────────────────────────────────────
        # Simplified: derive from interruption rates per AZ for this instance type
        azs = cat.get("azs", [])
        for i, az in enumerate(azs[:4]):
            az_snap = self.spot.get_latest(job.instance_type, az)
            if az_snap:
                # Low interruption rate → high capacity score
                obs[13 + i] = 1.0 - az_snap.interruption_rate_num
            else:
                obs[13 + i] = 0.5

        # ── Slot 17-24: Alternative Instance Prices (top-K cheapest) ─────────
        alternatives = self._get_best_alternatives(
            current_type=job.instance_type,
            min_vcpus=cat.get("vcpus", 2),
            min_ram_gb=cat.get("ram_gb", 4),
        )
        for i, alt_snap in enumerate(alternatives[: self.top_k]):
            alt_on_demand = self.catalog.get(alt_snap.instance_type, {}).get(
                "on_demand_price", 1.0
            )
            obs[17 + i] = alt_snap.price / (alt_on_demand + 1e-9)

        # ── Slot 25: Termination Warning ─────────────────────────────────────
        obs[25] = 1.0 if termination_warning else 0.0

        # ── Slot 26-27: Historical Interruption Context ───────────────────────
        obs[26] = min(interruptions_last_24h / 10.0, 1.0)
        obs[27] = min(migrations_last_24h / 5.0, 1.0)

        assert obs.shape == (FEATURE_DIM,), f"Expected ({FEATURE_DIM},), got {obs.shape}"
        return obs

    def _get_best_alternatives(
        self,
        current_type: str,
        min_vcpus: int,
        min_ram_gb: float,
    ) -> List[SpotMarketSnapshot]:
        """
        Return cheapest alternative spot instances that meet minimum resource requirements,
        excluding the currently running type.
        """
        all_snaps = self.spot.get_all_latest()
        eligible = []
        seen_types = set()

        for snap in all_snaps:
            if snap.instance_type == current_type:
                continue
            if snap.instance_type in seen_types:
                continue
            cat = self.catalog.get(snap.instance_type, {})
            if cat.get("vcpus", 0) >= min_vcpus and cat.get("ram_gb", 0) >= min_ram_gb:
                eligible.append(snap)
                seen_types.add(snap.instance_type)

        # Sort by current spot price ascending (cheapest first)
        eligible.sort(key=lambda s: s.price)
        return eligible
