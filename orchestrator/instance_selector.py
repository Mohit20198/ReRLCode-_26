import yaml
import os
from typing import Tuple, List, Dict

# Load instance catalog once at import time
INSTANCE_CATALOG_PATH = os.path.join(os.path.dirname(__file__), "..", "config", "instances.yaml")

def _load_instances() -> List[Dict]:
    """Load the instance catalog from ``config/instances.yaml``.
    The YAML file is expected to contain a list of mappings with the keys:
        - instance_type (str)
        - on_demand_price (float)
        - spot_price (float)
        - risk_factor (float)  # 0.0 (no risk) → 1.0 (high risk)
    """
    with open(INSTANCE_CATALOG_PATH, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data.get("instances", [])

_INSTANCES = _load_instances()

def select_instance(
    budget_cap_usd: float,
    current_cost_usd: float,
    remaining_hours: float,
    risk_cap: float = 0.2,
) -> Tuple[str, str]:
    """Select the cheapest *spot* instance that satisfies a hard risk cap.

    Parameters
    ----------
    budget_cap_usd: float
        Total budget allocated for the job.
    current_cost_usd: float
        Cost incurred so far.
    remaining_hours: float
        Estimated hours left for the job.
    risk_cap: float, optional
        Hard upper bound for ``risk_factor``. Defaults to ``0.2`` (hard cap).

    Returns
    -------
    Tuple[str, str]
        ``(instance_type, availability_zone)``. If no spot instance meets the
        constraints, the function falls back to the cheapest *on‑demand*
        instance (risk ignored).
    """
    # Filter by risk cap first
    candidates = [inst for inst in _INSTANCES if inst.get("risk_factor", 1.0) <= risk_cap]

    # Compute projected cost for each candidate
    viable = []
    for inst in candidates:
        projected = current_cost_usd + (inst["spot_price"] * remaining_hours)
        if projected <= budget_cap_usd:
            # Simple cost‑efficiency score: spot_price * risk_factor (lower is better)
            score = inst["spot_price"] * inst.get("risk_factor", 1.0)
            viable.append((score, inst))

    if viable:
        # Pick the lowest score
        _, best = min(viable, key=lambda x: x[0])
        # Choose a deterministic AZ (first in a static list for demo)
        az = "us-east-1a"
        return best["instance_type"], az

    # Fallback: cheapest on‑demand instance (ignore risk)
    on_demand = min(_INSTANCES, key=lambda i: i["on_demand_price"])
    return on_demand["instance_type"], "us-east-1a"

# ---------------------------------------------------------------------------
# Helper for unit‑testing – expose the raw catalog (read‑only)
def get_instance_catalog() -> List[Dict]:
    """Return a copy of the loaded instance catalog for inspection/testing."""
    return [inst.copy() for inst in _INSTANCES]
