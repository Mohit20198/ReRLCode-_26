import os
import pandas as pd
import numpy as np
import yaml
from typing import Tuple, Dict, Any

# ---------------------------------------------------------------------------
# Data generation utilities (run once on import if data folder is empty)
# ---------------------------------------------------------------------------
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "historical_prices")
os.makedirs(DATA_DIR, exist_ok=True)

INSTANCE_CATALOG_PATH = os.path.join(os.path.dirname(__file__), "..", "config", "instances.yaml")

def _load_instance_catalog():
    with open(INSTANCE_CATALOG_PATH, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg.get("instances", [])

_INSTANCES = _load_instance_catalog()

def _generate_synthetic_price_series(instance_type: str, days: int = 90, seed: int = 42) -> pd.DataFrame:
    """Create a synthetic hourly price series for *instance_type*.
    Columns: timestamp, spot_price, on_demand_price, risk_factor.
    """
    rng = np.random.default_rng(seed + hash(instance_type) % 2**32)
    hours = days * 24
    timestamps = pd.date_range(start="2023-01-01", periods=hours, freq="h")
    # Pull base prices from the instance catalog (fallback defaults)
    base = next((i for i in _INSTANCES if i["instance_type"] == instance_type), None)
    if not base:
        on_demand = 0.5
        spot = 0.3
        risk = 0.1
    else:
        on_demand = float(base.get("on_demand_price", 0.5))
        spot = float(base.get("spot_price", on_demand * 0.6))
        risk = float(base.get("risk_factor", 0.1))
    # Add sinusoidal daily pattern + random noise
    daily_cycle = np.sin(np.linspace(0, 2 * np.pi, hours)) * 0.05
    noise = rng.normal(0, 0.02, size=hours)
    spot_series = np.clip(spot * (1 + daily_cycle + noise), 0.01, None)
    # Risk factor varies slowly over time (simulating market volatility)
    risk_series = np.clip(risk + rng.normal(0, 0.02, size=hours), 0, 1)
    df = pd.DataFrame({
        "timestamp": timestamps,
        "spot_price": spot_series,
        "on_demand_price": on_demand,
        "risk_factor": risk_series,
    })
    return df

def _ensure_price_csv(instance_type: str) -> str:
    """Make sure a CSV exists for *instance_type* and return its path."""
    csv_path = os.path.join(DATA_DIR, f"{instance_type}.csv")
    if not os.path.isfile(csv_path):
        df = _generate_synthetic_price_series(instance_type)
        df.to_csv(csv_path, index=False)
    return csv_path

# ---------------------------------------------------------------------------
# Simulator class – compatible with gymnasium Env (but lightweight, no inheritance)
# ---------------------------------------------------------------------------
class SpotPriceSimulator:
    """Replay hourly spot‑price data for a given instance type.

    The simulator is deterministic when a ``seed`` is supplied. It provides
    ``reset`` and ``step`` methods that mimic the Gym API, allowing ``SpotFleetEnv``
    to plug it in without modification.
    """

    def __init__(self, instance_type: str, seed: int | None = None):
        self.instance_type = instance_type
        csv_path = _ensure_price_csv(instance_type)
        self.df = pd.read_csv(csv_path, parse_dates=["timestamp"])
        self.rng = np.random.default_rng(seed)
        self.current_idx = 0
        self.max_idx = len(self.df) - 1

    def reset(self, start_idx: int | None = None) -> Tuple[pd.Series, Dict[str, Any]]:
        """Reset the simulator to a random start hour (or ``start_idx``).
        Returns the first observation dict and an empty info dict.
        """
        if start_idx is None:
            self.current_idx = self.rng.integers(0, self.max_idx // 2)
        else:
            self.current_idx = max(0, min(start_idx, self.max_idx))
        obs = self._current_observation()
        return obs, {}

    def step(self, action: int) -> Tuple[pd.Series, float, bool, bool, Dict[str, Any]]:
        """Advance one hour.
        ``action`` is ignored – the simulator only provides price dynamics.
        Returns ``(obs, reward, terminated, truncated, info)``.
        """
        # Advance time
        self.current_idx = min(self.current_idx + 1, self.max_idx)
        obs = self._current_observation()
        # Determine if an interruption occurs based on risk_factor
        risk = float(self.df.iloc[self.current_idx]["risk_factor"])
        interruption_prob = risk * 0.05  # scale factor – higher risk → higher chance
        interrupted = self.rng.random() < interruption_prob
        reward = 0.0  # reward is computed by the environment, not the simulator
        terminated = False
        truncated = False
        info = {"interrupted": interrupted, "risk_factor": risk}
        return obs, reward, terminated, truncated, info

    def _current_observation(self) -> pd.Series:
        row = self.df.iloc[self.current_idx]
        # Return a Series that mimics the 28‑dim vector layout used elsewhere.
        # For simplicity we expose the raw columns; the environment will
        # transform them into the full vector.
        return pd.Series({
            "spot_price": row["spot_price"],
            "on_demand_price": row["on_demand_price"],
            "risk_factor": row["risk_factor"],
            "timestamp": row["timestamp"],
        })

# ---------------------------------------------------------------------------
# Convenience factory used by SpotFleetEnv (if it wants a simulator)
# ---------------------------------------------------------------------------
def make_simulator(instance_type: str, seed: int | None = None) -> SpotPriceSimulator:
    return SpotPriceSimulator(instance_type, seed)
