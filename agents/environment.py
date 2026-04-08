"""
agents/environment.py
──────────────────────
Defines the spot fleet management problem as a Gymnasium (gym) environment.
Used for offline RL training on historical spot price data.

The environment simulates:
  - A long-running job (e.g. 12-hour ML training)
  - AWS spot price dynamics for multiple instance types
  - Interruption events based on historical rates
  - Migration costs and downtime
  - Checkpoint save/restore overhead

Observation space : Box(28,) float32
Action space      : Discrete(N+2)  — STAY | MIGRATE_i | PAUSE
"""

from __future__ import annotations

import logging
import time
import pandas as pd
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import gymnasium as gym
import numpy as np
import yaml
from gymnasium import spaces

from agents.simulator import SpotPriceSimulator

logger = logging.getLogger(__name__)

# ──────────────────────────── Constants ──────────────────────────────────────
FEATURE_DIM = 28
POLL_INTERVAL_SEC = 60         # Each env step = 60s real time
MAX_JOB_HOURS = 24             # Maximum episode length
MAX_STEPS = (MAX_JOB_HOURS * 3600) // POLL_INTERVAL_SEC  # 1440 steps

# Migration overhead (seconds of downtime per migration)
MIGRATION_OVERHEAD_SEC = {
    "same_family": 120,
    "cross_family": 240,
    "to_on_demand": 180,
}

# On-demand price multiplier for action N+1
ON_DEMAND_MULTIPLIER = 3.0

# Interruption rate → probability per step (60s)
STEP_INTERRUPTION_PROB = {
    "low": 0.0005,     # ~5% over 30 days of continuous running
    "medium": 0.0015,
    "high": 0.004,
}


# ────────────────────────── Reward Weights ───────────────────────────────────
@dataclass
class RewardConfig:
    lambda_cost: float = 3.0
    lambda_migration: float = 1.5
    lambda_interruption: float = 80.0
    lambda_budget_cap: float = 200.0
    lambda_completion: float = 150.0
    lambda_time_efficiency: float = 0.5


# ─────────────────────────── Environment ─────────────────────────────────────

class SpotFleetEnv(gym.Env):
    """
    Gymnasium environment for spot fleet management.

    Actions:
        0        : STAY — do nothing, continue on current instance
        1..N     : MIGRATE to instance type (index in catalog)
        N+1      : PAUSE — checkpoint, terminate, wait for prices to drop

    Rewards:
        Cost-first design: heavily penalizes compute cost.
        Hard constraint: budget cap violation terminates episode.
    """

    metadata = {"render_modes": ["human", "ansi"]}

    def __init__(
        self,
        catalog_path: str = "config/instances.yaml",
        settings_path: str = "config/settings.yaml",
        reward_config: Optional[RewardConfig] = None,
        job_duration_hours: float = 12.0,
        budget_cap_usd: float = 50.0,
        render_mode: Optional[str] = None,
        simulator: Optional[SpotPriceSimulator] = None,
    ):
        super().__init__()

        self.render_mode = render_mode
        self.reward_cfg = reward_config or RewardConfig()

        # Load configs
        with open(catalog_path) as f:
            catalog_raw = yaml.safe_load(f)
        self.catalog = {i["type"]: i for i in catalog_raw["instances"]}
        self.instance_types = list(self.catalog.keys())
        self.n_types = len(self.instance_types)

        with open(settings_path) as f:
            self.settings = yaml.safe_load(f)

        # Job parameters
        self.job_duration_steps = int(job_duration_hours * 3600 / POLL_INTERVAL_SEC)
        self.budget_cap_usd = budget_cap_usd

        # Simulator for historical price replay (if provided)
        self.simulator = simulator

        # Define gym spaces
        self.observation_space = spaces.Box(
            low=0.0, high=10.0, shape=(FEATURE_DIM,), dtype=np.float32
        )
        # Actions: 0=STAY, 1..N=MIGRATE_i, N+1=PAUSE
        self.action_space = spaces.Discrete(self.n_types + 2)

        # Episode state (initialized in reset())
        self._step_count: int = 0
        self._current_type_idx: int = 0
        self._cumulative_cost: float = 0.0
        self._job_progress: float = 0.0
        self._n_migrations: int = 0
        self._n_interruptions: int = 0
        self._paused_steps: int = 0
        self._is_paused: bool = False
        self._termination_warning: bool = False
        self._spot_prices: np.ndarray = np.zeros(self.n_types)

    # ─────────────────────────── Gymnasium API ────────────────────────────────

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict] = None,
    ) -> Tuple[np.ndarray, Dict]:
        super().reset(seed=seed)

        # Start on cheapest available spot instance
        self._step_count = 0
        self._current_type_idx = 0
        self._cumulative_cost = 0.0
        self._job_progress = 0.0
        self._n_migrations = 0
        self._n_interruptions = 0
        self._paused_steps = 0
        self._is_paused = False
        self._termination_warning = False

        # Initialize simulated spot prices
        if self.simulator:
            sim_obs, _ = self.simulator.reset()
            self._spot_prices = self._get_prices_from_sim_obs(sim_obs)
        else:
            self._spot_prices = self._init_spot_prices()

        obs = self._build_observation()
        info = self._get_info()
        return obs, info

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        assert self.action_space.contains(action), f"Invalid action: {action}"

        reward = 0.0
        terminated = False
        truncated = False

        # ── Update simulated spot prices ──
        if self.simulator:
            sim_obs, _, _, _, sim_info = self.simulator.step(action)
            self._spot_prices = self._get_prices_from_sim_obs(sim_obs)
            interrupted = sim_info.get("interrupted", False)
        else:
            self._spot_prices = self._update_spot_prices(self._spot_prices)

            # ── Check for random interruption event ──
            current_type = self.instance_types[self._current_type_idx]
            intr_rate = self.catalog[current_type].get("interruption_rate", "medium")
            intr_prob = STEP_INTERRUPTION_PROB[intr_rate]
            interrupted = (not self._is_paused) and (self.np_random.random() < intr_prob)

        # ── Termination warning: occurs 1 step before interruption ──
        self._termination_warning = interrupted  # Simplified: warning = interruption

        # ── Process action ──
        migration_overhead_sec = 0.0
        action_cost = 0.0

        if action == 0:
            # STAY
            pass

        elif 1 <= action <= self.n_types:
            # MIGRATE to instance type (action - 1)
            target_idx = action - 1
            if target_idx != self._current_type_idx:
                # Compute migration cost (downtime during transfer)
                src = self.instance_types[self._current_type_idx]
                tgt = self.instance_types[target_idx]
                overhead = self._compute_migration_overhead(src, tgt)
                migration_overhead_sec = overhead
                action_cost = (self._spot_prices[target_idx] / 3600) * overhead
                self._current_type_idx = target_idx
                self._n_migrations += 1
                reward -= self.reward_cfg.lambda_migration * (action_cost / self.budget_cap_usd)

        elif action == self.n_types + 1:
            # PAUSE — checkpoint and wait
            self._is_paused = True
            self._paused_steps += 1

        # ── Compute step cost ──
        if not self._is_paused:
            step_price = self._spot_prices[self._current_type_idx]
            step_cost = (step_price / 3600) * POLL_INTERVAL_SEC
            self._cumulative_cost += step_cost + action_cost

            # Cost reward component (normalized to budget)
            reward -= self.reward_cfg.lambda_cost * (step_cost / self.budget_cap_usd)

            # Advance job progress (proportional to time on instance)
            effective_seconds = max(0, POLL_INTERVAL_SEC - migration_overhead_sec)
            progress_step = effective_seconds / (self.job_duration_steps * POLL_INTERVAL_SEC)
            self._job_progress = min(1.0, self._job_progress + progress_step)
        else:
            # Auto-resume if prices have dropped enough
            if self._paused_steps >= 5:  # Wait minimum 5 steps (~5 min)
                cheapest_idx = int(np.argmin(self._spot_prices))
                cheapest_price = self._spot_prices[cheapest_idx]
                ref_price = self.catalog[self.instance_types[cheapest_idx]].get(
                    "on_demand_price", 1.0
                )
                if cheapest_price < ref_price * 0.7:  # Resume if price < 70% of on-demand
                    self._is_paused = False
                    self._current_type_idx = cheapest_idx
                    self._paused_steps = 0

        # ── Handle interruption ──
        if interrupted and not self._is_paused:
            self._n_interruptions += 1
            reward -= self.reward_cfg.lambda_interruption
            # Partial progress loss (last 10% of work since checkpoint)
            self._job_progress = max(0.0, self._job_progress - 0.10)

        # ── Budget cap check ──
        if self._cumulative_cost > self.budget_cap_usd:
            reward -= self.reward_cfg.lambda_budget_cap
            terminated = True
            logger.debug(f"Episode terminated: budget cap exceeded (${self._cumulative_cost:.2f})")

        # ── Job completion ──
        if self._job_progress >= 1.0:
            budget_remaining = self.budget_cap_usd - self._cumulative_cost
            reward += self.reward_cfg.lambda_completion
            reward += self.reward_cfg.lambda_time_efficiency * (budget_remaining / self.budget_cap_usd)
            terminated = True
            logger.debug(f"Episode complete: ${self._cumulative_cost:.2f} spent")

        # ── Max steps ──
        self._step_count += 1
        if self._step_count >= MAX_STEPS:
            truncated = True

        obs = self._build_observation()
        info = self._get_info()

        return obs, float(reward), terminated, truncated, info

    def render(self):
        if self.render_mode == "human" or self.render_mode == "ansi":
            current_type = self.instance_types[self._current_type_idx]
            price = self._spot_prices[self._current_type_idx]
            status = "PAUSED" if self._is_paused else "RUNNING"
            print(
                f"Step {self._step_count:4d} | {status:7s} | {current_type:16s} "
                f"| ${price:.4f}/hr | Progress: {self._job_progress*100:.1f}% "
                f"| Cost: ${self._cumulative_cost:.3f} | Migrations: {self._n_migrations}"
            )

    # ────────────────────────── Observation Builder ───────────────────────────

    def _build_observation(self) -> np.ndarray:
        obs = np.zeros(FEATURE_DIM, dtype=np.float32)
        ci = self._current_type_idx
        current_type = self.instance_types[ci]
        cat = self.catalog[current_type]

        # Current instance state (0-4)
        obs[0] = ci / max(self.n_types - 1, 1)
        obs[1] = self._step_count / MAX_STEPS
        obs[2] = min(self._cumulative_cost / self.budget_cap_usd, 2.0)
        obs[3] = self._job_progress
        obs[4] = 1.0 - self._job_progress  # Remaining fraction

        # Spot market signals for current instance (5-12)
        price = self._spot_prices[ci]
        on_demand = cat.get("on_demand_price", 1.0)
        obs[5] = price / (on_demand + 1e-9)
        obs[6] = obs[5]   # Simplified: no historical rolling avg in simulation
        obs[7] = obs[5]
        obs[8] = obs[5]
        obs[9] = (price - on_demand * 0.3) / (on_demand + 1e-9)  # % above typical spot
        intr_rate = cat.get("interruption_rate", "medium")
        obs[10] = {"low": 0.05, "medium": 0.15, "high": 0.35}.get(intr_rate, 0.15)
        obs[11] = 0.0    # Price trend (simplified)
        obs[12] = {"low": 0.0, "medium": 0.5, "high": 1.0}.get(intr_rate, 0.5)

        # AZ signals (13-16): use interruption rate as proxy
        for i in range(4):
            obs[13 + i] = 1.0 - obs[10]

        # Alternative instance prices (17-24)
        sorted_indices = np.argsort(self._spot_prices)
        k = 0
        for idx in sorted_indices:
            if idx != ci and k < 8:
                alt_on_demand = self.catalog[self.instance_types[idx]].get("on_demand_price", 1.0)
                obs[17 + k] = self._spot_prices[idx] / (alt_on_demand + 1e-9)
                k += 1

        # Termination warning (25)
        obs[25] = 1.0 if self._termination_warning else 0.0

        # Historical context (26-27)
        obs[26] = min(self._n_interruptions / 10.0, 1.0)
        obs[27] = min(self._n_migrations / 5.0, 1.0)

        return obs

    # ─────────────────────────── Helpers ─────────────────────────────────────

    def _init_spot_prices(self) -> np.ndarray:
        """Initialize spot prices as ~30% of on-demand with noise."""
        prices = np.array([
            self.catalog[t].get("on_demand_price", 1.0) * (0.25 + self.np_random.random() * 0.15)
            for t in self.instance_types
        ], dtype=np.float32)
        return prices

    def _update_spot_prices(self, prices: np.ndarray) -> np.ndarray:
        """
        Simulate spot price dynamics: random walk with mean reversion.
        Prices tend to stay near 30% of on-demand but can spike.
        """
        new_prices = prices.copy()
        for i, itype in enumerate(self.instance_types):
            on_demand = self.catalog[itype].get("on_demand_price", 1.0)
            target = on_demand * 0.30
            # Mean-reverting random walk
            noise = self.np_random.normal(0, 0.003) * on_demand
            reversion = 0.01 * (target - prices[i])
            new_prices[i] = max(0.01, prices[i] + noise + reversion)
        return new_prices

    def _compute_migration_overhead(self, src_type: str, tgt_type: str) -> float:
        """Estimate migration overhead in seconds based on families."""
        src_fam = self.catalog[src_type].get("family", "general")
        tgt_fam = self.catalog[tgt_type].get("family", "general")
        if src_fam == tgt_fam:
            return MIGRATION_OVERHEAD_SEC["same_family"]
        return MIGRATION_OVERHEAD_SEC["cross_family"]

    def _get_info(self) -> Dict[str, Any]:
        return {
            "step": self._step_count,
            "current_instance": self.instance_types[self._current_type_idx],
            "cumulative_cost": self._cumulative_cost,
            "job_progress": self._job_progress,
            "n_migrations": self._n_migrations,
            "n_interruptions": self._n_interruptions,
            "is_paused": self._is_paused,
        }

    def _get_prices_from_sim_obs(self, sim_obs: pd.Series) -> np.ndarray:
        """Extract prices for all instance types from simulator observation."""
        # For simplicity, assume simulator only gives current instance price
        # In a real setup, it might give a vector for all N types.
        # Here we update only the current instance in the price cache.
        ci = self._current_type_idx
        prices = self._spot_prices.copy()
        prices[ci] = sim_obs["spot_price"]
        # In this simplified implementation, other prices remain as they were
        # or follow the internal random walk.
        return prices
