import pytest
from orchestrator.instance_selector import select_instance, get_instance_catalog

def test_select_instance_within_budget():
    # Job with 10 hr remaining, $2.0 budget cap, and 0 cost so far.
    # Should pick cheapest instance that fits the $2.0 cap and 0.2 risk.
    instance_type, az = select_instance(
        budget_cap_usd=2.0,
        current_cost_usd=0.0,
        remaining_hours=10.0,
        risk_cap=0.2
    )
    assert instance_type is not None
    assert az == "us-east-1a"
    
    # Verify the selected instance actually meets the criteria
    catalog = get_instance_catalog()
    selected = next(inst for inst in catalog if inst["instance_type"] == instance_type)
    assert selected["risk_factor"] <= 0.2
    assert (selected["spot_price"] * 10.0) <= 2.0

def test_select_instance_risk_cap():
    # High risk cap (1.0) should allow riskier instances
    instance_type, az = select_instance(
        budget_cap_usd=10.0,
        current_cost_usd=0.0,
        remaining_hours=1.0,
        risk_cap=1.0
    )
    assert instance_type is not None
    
    # Low risk cap (0.05) should be very restrictive
    instance_type_safe, _ = select_instance(
        budget_cap_usd=10.0,
        current_cost_usd=0.0,
        remaining_hours=1.0,
        risk_cap=0.05
    )
    assert instance_type_safe is not None

def test_select_instance_fallback_on_demand():
    # Budget so low that no spot instance can fulfill it.
    # Or risk cap so low that no spot instance qualifies.
    # Should fallback to cheapest on-demand.
    instance_type, az = select_instance(
        budget_cap_usd=0.0001,
        current_cost_usd=0.0,
        remaining_hours=100.0,
        risk_cap=0.01
    )
    assert instance_type is not None
    
    # Check that fallback happened (price will likely exceed budget)
    catalog = get_instance_catalog()
    on_demand_prices = [inst["on_demand_price"] for inst in catalog]
    min_on_demand = min(on_demand_prices)
    selected = next(inst for inst in catalog if inst["instance_type"] == instance_type)
    assert selected["on_demand_price"] == min_on_demand
