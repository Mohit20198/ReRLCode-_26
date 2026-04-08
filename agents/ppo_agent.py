"""
agents/ppo_agent.py
───────────────────
PPO agent with LSTM policy network for spot fleet management.
Uses Stable-Baselines3 with a custom LSTM feature extractor.

The LSTM captures temporal dependencies in spot price trends:
  - Price that has been rising for 10+ minutes → higher interruption risk
  - Sustained price below baseline → stable, safe to stay

Training:
    from agents.ppo_agent import FleetPPOAgent
    agent = FleetPPOAgent(env)
    agent.train(total_timesteps=2_000_000)
    agent.save("models/fleet_ppo_v1")

Inference:
    agent = FleetPPOAgent.load("models/fleet_ppo_v1", env)
    action, state = agent.predict(obs, deterministic=True)
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, Optional, Tuple, Type

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import (
    BaseCallback,
    CheckpointCallback,
    EvalCallback,
)
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

logger = logging.getLogger(__name__)

FEATURE_DIM = 28


# ─────────────────────────── LSTM Feature Extractor ──────────────────────────

class LSTMFleetExtractor(BaseFeaturesExtractor):
    """
    Custom LSTM-based feature extractor for the fleet observation space.

    Architecture:
        Input (28,) → Linear projection → LSTM (128 hidden) → tanh → (128,)

    The LSTM maintains hidden state across steps, capturing temporal patterns
    in spot price trends that are invisible to a feedforward network.
    """

    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 128):
        super().__init__(observation_space, features_dim)

        self.projection = nn.Linear(FEATURE_DIM, 64)
        self.lstm = nn.LSTM(64, features_dim, batch_first=True)
        self.norm = nn.LayerNorm(features_dim)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        # observations: (batch, 28)
        x = torch.relu(self.projection(observations))
        # Add sequence dimension: (batch, 1, 64)
        x = x.unsqueeze(1)
        out, _ = self.lstm(x)
        # Take last timestep: (batch, 128)
        out = self.norm(out[:, -1, :])
        return torch.tanh(out)


# ─────────────────────────── Agent Wrapper ───────────────────────────────────

class FleetPPOAgent:
    """
    Wrapper around Stable-Baselines3 PPO for fleet management.

    Provides:
        - Training with MLflow logging
        - Evaluation vs baselines
        - Model persistence (save/load)
        - Inference API matching the orchestrator's interface
    """

    # PPO hyperparameters tuned for fleet management
    DEFAULT_HYPERPARAMS: Dict[str, Any] = {
        "learning_rate": 3e-4,
        "n_steps": 2048,          # Steps per env per update
        "batch_size": 64,
        "n_epochs": 10,
        "gamma": 0.995,           # High discount — agent cares about long-term cost
        "gae_lambda": 0.95,
        "clip_range": 0.2,
        "ent_coef": 0.01,         # Exploration entropy
        "vf_coef": 0.5,
        "max_grad_norm": 0.5,
        "verbose": 1,
    }

    def __init__(
        self,
        env: Optional[gym.Env] = None,
        hyperparams: Optional[Dict] = None,
        device: str = "auto",
    ):
        self.device = device
        self.hyperparams = {**self.DEFAULT_HYPERPARAMS, **(hyperparams or {})}
        self._model: Optional[PPO] = None

        if env is not None:
            self._setup(env)

    def _setup(self, env: gym.Env):
        """Initialize PPO with custom LSTM policy network."""
        vec_env = DummyVecEnv([lambda: env])
        self._vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True)

        policy_kwargs = {
            "features_extractor_class": LSTMFleetExtractor,
            "features_extractor_kwargs": {"features_dim": 128},
            "net_arch": [dict(pi=[64, 64], vf=[64, 64])],
            "activation_fn": nn.Tanh,
        }

        self._model = PPO(
            policy="MlpPolicy",
            env=self._vec_env,
            policy_kwargs=policy_kwargs,
            device=self.device,
            **self.hyperparams,
        )
        logger.info(f"FleetPPOAgent initialized — {self._model.num_timesteps} timesteps so far")

    def train(
        self,
        total_timesteps: int = 2_000_000,
        eval_env: Optional[gym.Env] = None,
        model_dir: str = "models/",
        experiment_name: str = "fleet-ppo",
    ) -> None:
        """
        Train the PPO agent.

        Args:
            total_timesteps: Total environment steps for training (default 2M ~ 10k episodes)
            eval_env: Separate env for periodic evaluation
            model_dir: Directory to save model checkpoints
            experiment_name: MLflow experiment name
        """
        os.makedirs(model_dir, exist_ok=True)

        callbacks = [
            CheckpointCallback(
                save_freq=50_000,
                save_path=model_dir,
                name_prefix="fleet_ppo",
            ),
            FleetMetricsCallback(),
        ]

        if eval_env is not None:
            eval_vec = DummyVecEnv([lambda: eval_env])
            callbacks.append(
                EvalCallback(
                    eval_vec,
                    best_model_save_path=os.path.join(model_dir, "best"),
                    log_path=os.path.join(model_dir, "eval_logs"),
                    eval_freq=50_000,
                    n_eval_episodes=20,
                    deterministic=True,
                )
            )

        logger.info(f"Starting training: {total_timesteps:,} timesteps")
        self._model.learn(
            total_timesteps=total_timesteps,
            callback=callbacks,
            tb_log_name=experiment_name,
            reset_num_timesteps=False,
        )
        logger.info("Training complete")

    def predict(
        self,
        observation: np.ndarray,
        deterministic: bool = True,
    ) -> Tuple[int, Optional[np.ndarray]]:
        """
        Choose an action given the current observation.

        Args:
            observation: 28-dim feature vector from FeatureBuilder
            deterministic: If True, picks the mode action (no exploration)

        Returns:
            (action_int, state) — state is always None for MLP policies
        """
        if self._model is None:
            raise RuntimeError("Agent not initialized. Call train() or load() first.")

        obs = np.array(observation, dtype=np.float32)
        if self._vec_env is not None:
            obs = self._vec_env.normalize_obs(obs)

        action, state = self._model.predict(obs, deterministic=deterministic)
        return int(action), state

    def get_action_probabilities(self, observation: np.ndarray) -> np.ndarray:
        """Return softmax action probabilities (for dashboard confidence display)."""
        if self._model is None:
            raise RuntimeError("Agent not initialized.")

        obs_tensor = torch.FloatTensor(observation).unsqueeze(0).to(self._model.device)
        with torch.no_grad():
            distribution = self._model.policy.get_distribution(obs_tensor)
            probs = distribution.distribution.probs.cpu().numpy().squeeze()
        return probs

    def save(self, path: str) -> None:
        """Save trained model to disk."""
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        self._model.save(path)
        if self._vec_env is not None:
            self._vec_env.save(path + "_vecnorm.pkl")
        logger.info(f"Model saved to {path}")

    @classmethod
    def load(cls, path: str, env: Optional[gym.Env] = None) -> "FleetPPOAgent":
        """Load a previously trained model."""
        agent = cls.__new__(cls)
        agent.device = "auto"

        if env is not None:
            vec_env = DummyVecEnv([lambda: env])
            vecnorm_path = path + "_vecnorm.pkl"
            if os.path.exists(vecnorm_path):
                agent._vec_env = VecNormalize.load(vecnorm_path, vec_env)
            else:
                agent._vec_env = VecNormalize(vec_env)
        else:
            agent._vec_env = None

        agent._model = PPO.load(path, env=agent._vec_env)
        logger.info(f"Model loaded from {path}")
        return agent


# ─────────────────────────── Training Callbacks ───────────────────────────────

class FleetMetricsCallback(BaseCallback):
    """
    Logs fleet-specific metrics during training:
     - Average cost per episode
     - Migration frequency
     - Interruption rate
     - Budget utilization
    """

    def __init__(self):
        super().__init__()
        self._episode_costs = []
        self._episode_migrations = []
        self._episode_interruptions = []

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            if "episode" in info:
                # SB3 stores episode stats here on done
                pass
            # Track from env info
            cost = info.get("cumulative_cost", 0)
            migrations = info.get("n_migrations", 0)
            interruptions = info.get("n_interruptions", 0)
            self._episode_costs.append(cost)
            self._episode_migrations.append(migrations)
            self._episode_interruptions.append(interruptions)

        if self.n_calls % 10000 == 0 and self._episode_costs:
            avg_cost = np.mean(self._episode_costs[-100:])
            avg_mig = np.mean(self._episode_migrations[-100:])
            avg_intr = np.mean(self._episode_interruptions[-100:])
            self.logger.record("fleet/avg_cost_usd", avg_cost)
            self.logger.record("fleet/avg_migrations", avg_mig)
            self.logger.record("fleet/avg_interruptions", avg_intr)

        return True  # Continue training
