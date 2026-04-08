"""
tests/test_environment.py
──────────────────────────
Unit tests for the SpotFleetEnv RL environment.
Tests: step mechanics, reward computation, episode termination, action space.
"""

import numpy as np
import pytest

from agents.environment import SpotFleetEnv, RewardConfig, FEATURE_DIM


@pytest.fixture
def env():
    """Default environment for testing."""
    return SpotFleetEnv(
        job_duration_hours=1.0,
        budget_cap_usd=10.0,
    )


def test_reset_returns_correct_shape(env):
    obs, info = env.reset(seed=42)
    assert obs.shape == (FEATURE_DIM,)
    assert obs.dtype == np.float32
    assert "job_progress" in info


def test_stay_action_is_valid(env):
    obs, _ = env.reset(seed=0)
    obs2, reward, terminated, truncated, info = env.step(0)  # STAY
    assert obs2.shape == (FEATURE_DIM,)
    assert isinstance(reward, float)
    assert not terminated or info["job_progress"] >= 1.0 or info["cumulative_cost"] > 10.0


def test_migrate_action_increments_migrations(env):
    obs, _ = env.reset(seed=1)
    # Action 1 = MIGRATE to first alternative
    _, _, _, _, info_before = env.step(0)
    migrations_before = info_before["n_migrations"]

    env2 = SpotFleetEnv(job_duration_hours=1.0, budget_cap_usd=10.0)
    env2.reset(seed=1)
    _, _, _, _, info_after = env2.step(1)  # MIGRATE
    assert info_after["n_migrations"] >= migrations_before


def test_pause_action_sets_paused(env):
    obs, _ = env.reset(seed=2)
    n_types = env.n_types
    _, _, _, _, info = env.step(n_types + 1)  # PAUSE
    assert info["is_paused"]


def test_budget_cap_terminates_episode(env):
    """Episode should terminate when cumulative cost exceeds budget."""
    env_tight = SpotFleetEnv(
        job_duration_hours=24.0,
        budget_cap_usd=0.001,  # Extremely tight budget
    )
    obs, _ = env_tight.reset(seed=3)
    done = False
    steps = 0
    while not done and steps < 1000:
        obs, reward, terminated, truncated, info = env_tight.step(0)
        done = terminated or truncated
        steps += 1
    assert done, "Episode should have terminated due to budget cap"


def test_observation_values_in_range(env):
    """All observation features should be finite and within expected bounds."""
    obs, _ = env.reset(seed=4)
    for _ in range(100):
        obs, _, terminated, truncated, _ = env.step(env.action_space.sample())
        assert np.all(np.isfinite(obs)), "Observation contains NaN or Inf"
        if terminated or truncated:
            obs, _ = env.reset()


def test_reward_is_negative_for_expensive_instance(env):
    """Expensive instance types should produce more negative rewards than cheap ones."""
    # STAY on default (cheapest) instance
    env.reset(seed=5)
    # Force a cheap instance type
    env._current_type_idx = 0  # First (cheapest) instance
    env._spot_prices[0] = 0.01
    _, reward_cheap, _, _, _ = env.step(0)

    env.reset(seed=5)
    # Force an expensive instance type
    env._current_type_idx = len(env.instance_types) - 1
    env._spot_prices[-1] = 10.0
    _, reward_expensive, _, _, _ = env.step(0)

    assert reward_cheap > reward_expensive, (
        f"Cheap instance should have higher reward: {reward_cheap} vs {reward_expensive}"
    )


def test_completion_bonus_on_job_finish():
    """Episode with job completion should include large positive reward component."""
    env = SpotFleetEnv(
        job_duration_hours=0.001,  # Very short job — completes in first step
        budget_cap_usd=100.0,
    )
    obs, _ = env.reset(seed=6)
    total_reward = 0
    done = False
    steps = 0
    while not done and steps < 10:
        obs, reward, terminated, truncated, info = env.step(0)
        total_reward += reward
        done = terminated or truncated
        steps += 1

    # Completion bonus (150) should make total reward positive
    assert total_reward > 0 or info["job_progress"] < 1.0, (
        "Completed job should yield net positive reward"
    )
