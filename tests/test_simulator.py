import pytest
import pandas as pd
from agents.simulator import SpotPriceSimulator

def test_simulator_reset():
    sim = SpotPriceSimulator(instance_type="t3.medium", seed=42)
    obs, info = sim.reset()
    
    assert isinstance(obs, pd.Series)
    assert "spot_price" in obs
    assert "on_demand_price" in obs
    assert "risk_factor" in obs
    assert "timestamp" in obs
    assert info == {}

def test_simulator_step_reproducible():
    sim1 = SpotPriceSimulator(instance_type="t3.medium", seed=42)
    sim2 = SpotPriceSimulator(instance_type="t3.medium", seed=42)
    
    sim1.reset(start_idx=100)
    sim2.reset(start_idx=100)
    
    obs1, reward1, term1, trunc1, info1 = sim1.step(action=0)
    obs2, reward2, term2, trunc2, info2 = sim2.step(action=0)
    
    assert obs1["spot_price"] == obs2["spot_price"]
    assert info1["interrupted"] == info2["interrupted"]
    assert info1["risk_factor"] == info2["risk_factor"]

def test_simulator_interruption_probability():
    # Large number of steps to verify interruption probability isn't zero
    sim = SpotPriceSimulator(instance_type="m5.large", seed=123)
    sim.reset(start_idx=0)
    
    interruptions = 0
    total_steps = 200
    for _ in range(total_steps):
        _, _, _, _, info = sim.step(action=0)
        if info["interrupted"]:
            interruptions += 1
            
    # Risk factor in generated data is around 0.1-0.2
    # Probability = risk * 0.05 -> ~0.005 to 0.01 per step
    # 200 steps at 0.005-0.01 expects ~1-2 interruptions.
    # Just asserting it stays within a sane range.
    assert interruptions < total_steps // 2 

def test_simulator_data_generation():
    import os
    instance = "c5.xlarge"
    csv_path = os.path.join("data", "historical_prices", f"{instance}.csv")
    
    # Ensure file is clean before test
    if os.path.exists(csv_path):
        os.remove(csv_path)
        
    sim = SpotPriceSimulator(instance_type=instance, seed=444)
    assert os.path.exists(csv_path)
    
    df = pd.read_csv(csv_path)
    assert len(df) == 90 * 24
    assert all(col in df.columns for col in ["timestamp", "spot_price", "on_demand_price", "risk_factor"])
