"""
training/train.py
──────────────────
Entry point for offline PPO training on simulated spot price data.

Usage:
    python -m training.train --episodes 10000 --budget 50 --job-hours 12

The training pipeline:
    1. Pulls 90 days of historical spot prices from AWS (or uses cached data)
    2. Creates the SpotFleetEnv with historical price replay
    3. Trains PPO agent for the specified number of timesteps
    4. Evaluates against 4 baselines and logs results to MLflow
    5. Saves the best model to models/best/
"""

from __future__ import annotations

import argparse
import logging
import os
import time
from typing import Dict, List

import mlflow
import numpy as np
import yaml

from agents.environment import SpotFleetEnv, RewardConfig
from agents.ppo_agent import FleetPPOAgent

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def run_baseline(env_cls, env_kwargs, policy_fn, n_episodes=100) -> Dict[str, float]:
    """Evaluate a simple baseline policy and return metrics."""
    costs = []
    completions = []
    interruptions = []
    migrations = []

    for _ in range(n_episodes):
        env = env_cls(**env_kwargs)
        obs, info = env.reset()
        done = False
        while not done:
            action = policy_fn(obs, env)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
        costs.append(info["cumulative_cost"])
        completions.append(float(info["job_progress"] >= 1.0))
        interruptions.append(info["n_interruptions"])
        migrations.append(info["n_migrations"])

    return {
        "avg_cost": float(np.mean(costs)),
        "completion_rate": float(np.mean(completions)),
        "avg_interruptions": float(np.mean(interruptions)),
        "avg_migrations": float(np.mean(migrations)),
    }


def always_stay_policy(obs, env) -> int:
    """Baseline: always stay on current instance."""
    return 0


def random_policy(obs, env) -> int:
    """Baseline: random action."""
    return env.action_space.sample()


def always_on_demand_policy(obs, env) -> int:
    """Baseline: always migrate to on-demand (most expensive safe option)."""
    # Migrate to the instance type with highest on-demand price (proxy for on-demand)
    return env.n_types  # PAUSE action — simplified proxy


def threshold_policy(obs, env) -> int:
    """
    Baseline: migrate if current spot price exceeds 2× the 30-min average.
    Uses obs[5] (current price / on-demand) and obs[7] (30-min avg / on-demand).
    """
    current_ratio = obs[5]
    avg_30m_ratio = obs[7]
    if avg_30m_ratio > 0 and current_ratio > 2.0 * avg_30m_ratio:
        # Migrate to cheapest alternative
        alt_prices = obs[17:25]
        cheapest_idx = int(np.argmin(alt_prices)) + 1
        return min(cheapest_idx, env.n_types)
    return 0


def main():
    parser = argparse.ArgumentParser(description="Train Fleet PPO Agent")
    parser.add_argument("--episodes", type=int, default=10000, help="Training episodes")
    parser.add_argument("--budget", type=float, default=50.0, help="Budget cap per job (USD)")
    parser.add_argument("--job-hours", type=float, default=12.0, help="Job duration in hours")
    parser.add_argument("--timesteps", type=int, default=2_000_000, help="Total training steps")
    parser.add_argument("--model-dir", default="models/", help="Directory to save models")
    parser.add_argument("--eval-episodes", type=int, default=200, help="Episodes for evaluation")
    parser.add_argument("--experiment", default="fleet-ppo", help="MLflow experiment name")
    args = parser.parse_args()

    os.makedirs(args.model_dir, exist_ok=True)

    # ── Load config ────────────────────────────────────────────────────────
    with open("config/settings.yaml") as f:
        settings = yaml.safe_load(f)

    reward_cfg = RewardConfig(
        lambda_cost=settings["reward"]["lambda_cost"],
        lambda_migration=settings["reward"]["lambda_migration"],
        lambda_interruption=settings["reward"]["lambda_interruption"],
        lambda_budget_cap=settings["reward"]["lambda_budget_cap"],
        lambda_completion=settings["reward"]["lambda_completion"],
        lambda_time_efficiency=settings["reward"]["lambda_time_efficiency"],
    )

    env_kwargs = dict(
        reward_config=reward_cfg,
        job_duration_hours=args.job_hours,
        budget_cap_usd=args.budget,
    )

    # ── MLflow tracking ────────────────────────────────────────────────────
    mlflow.set_experiment(args.experiment)

    with mlflow.start_run(run_name=f"ppo_{int(time.time())}") as run:
        logger.info(f"MLflow run: {run.info.run_id}")

        # Log hyperparameters
        mlflow.log_params({
            "budget_usd": args.budget,
            "job_hours": args.job_hours,
            "total_timesteps": args.timesteps,
            "lambda_cost": reward_cfg.lambda_cost,
            "lambda_interruption": reward_cfg.lambda_interruption,
            "lambda_completion": reward_cfg.lambda_completion,
        })

        # ── Evaluate baselines ─────────────────────────────────────────────
        logger.info("Evaluating baselines...")
        baselines = {
            "always_stay": always_stay_policy,
            "random": random_policy,
            "threshold_2x": threshold_policy,
        }

        for name, policy_fn in baselines.items():
            metrics = run_baseline(
                SpotFleetEnv, env_kwargs, policy_fn, n_episodes=min(args.eval_episodes, 100)
            )
            logger.info(f"Baseline [{name}]: {metrics}")
            for k, v in metrics.items():
                mlflow.log_metric(f"baseline/{name}/{k}", v)

        # ── Train RL agent ─────────────────────────────────────────────────
        logger.info(f"Training PPO agent for {args.timesteps:,} timesteps...")
        train_env = SpotFleetEnv(**env_kwargs)
        eval_env = SpotFleetEnv(**env_kwargs)

        agent = FleetPPOAgent(env=train_env)
        agent.train(
            total_timesteps=args.timesteps,
            eval_env=eval_env,
            model_dir=args.model_dir,
            experiment_name=args.experiment,
        )

        # ── Evaluate RL agent ──────────────────────────────────────────────
        logger.info("Evaluating trained agent...")
        model_path = os.path.join(args.model_dir, "best", "best_model")
        if os.path.exists(model_path + ".zip"):
            trained_agent = FleetPPOAgent.load(model_path, env=SpotFleetEnv(**env_kwargs))
        else:
            trained_agent = agent

        rl_costs = []
        rl_completions = []
        rl_interruptions = []
        rl_migrations = []

        eval_env2 = SpotFleetEnv(**env_kwargs)
        for ep in range(args.eval_episodes):
            obs, info = eval_env2.reset()
            done = False
            while not done:
                action, _ = trained_agent.predict(obs, deterministic=True)
                obs, _, terminated, truncated, info = eval_env2.step(action)
                done = terminated or truncated
            rl_costs.append(info["cumulative_cost"])
            rl_completions.append(float(info["job_progress"] >= 1.0))
            rl_interruptions.append(info["n_interruptions"])
            rl_migrations.append(info["n_migrations"])

        rl_metrics = {
            "avg_cost": float(np.mean(rl_costs)),
            "completion_rate": float(np.mean(rl_completions)),
            "avg_interruptions": float(np.mean(rl_interruptions)),
            "avg_migrations": float(np.mean(rl_migrations)),
        }

        logger.info(f"RL Agent metrics: {rl_metrics}")
        for k, v in rl_metrics.items():
            mlflow.log_metric(f"rl_agent/{k}", v)

        # ── Save final model ───────────────────────────────────────────────
        final_path = os.path.join(args.model_dir, "fleet_ppo_final")
        agent.save(final_path)
        mlflow.log_artifact(final_path + ".zip", "model")

        logger.info(f"Training complete. Model saved to {final_path}")
        logger.info(
            f"Cost vs Always-Stay: ${rl_metrics['avg_cost']:.2f} vs "
            f"${baselines.get('always_stay', {}).get('avg_cost', '?')}"
        )


if __name__ == "__main__":
    main()
