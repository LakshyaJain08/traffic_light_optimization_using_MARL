"""
MARL Training Module – Ray RLlib PPO (centralized training, decentralized
execution).

Trains one shared policy across all intersection agents.
"""

from __future__ import annotations

import csv
from datetime import datetime
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from src.sumo_env import EPISODE_METRICS_BUFFER

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Legacy stub – kept so .callbacks() call compiles without error.
# Actual metrics come from src.sumo_env.EPISODE_METRICS_BUFFER, not from this callback.
# ---------------------------------------------------------------------------

def _make_metrics_callback():
    """
    Return an RLlib DefaultCallbacks subclass that harvests waiting_time_raw,
    co2_raw and fuel_raw from each agent's last info dict and exposes them as
    episode-level custom metrics (RLlib automatically appends _mean / _min /
    _max suffixes when aggregating across episodes in the result dict).
    """
    from ray.rllib.algorithms.callbacks import DefaultCallbacks

    class TrafficMetricsCallback(DefaultCallbacks):
        """No-op stub – metrics are captured via the real SUMO env."""
        pass

    return TrafficMetricsCallback


# ---------------------------------------------------------------------------
# Helper: extract a scalar from the nested RLlib result dict
# ---------------------------------------------------------------------------

def _get_result(result: dict, key: str, default: float = 0.0) -> float:
    """
    Try common result-dict locations used across Ray 2.x versions for a given
    metric key.  Checks top-level, then 'env_runners', then 'sampler_results'.
    """
    if key in result and result[key] is not None:
        v = result[key]
        try:
            return float(v)
        except (TypeError, ValueError):
            pass
    for sub in ("env_runners", "sampler_results"):
        sub_dict = result.get(sub, {}) or {}
        if key in sub_dict and sub_dict[key] is not None:
            try:
                return float(sub_dict[key])
            except (TypeError, ValueError):
                pass
    return default


# ---------------------------------------------------------------------------
# RLlib env wrapper (MultiAgentEnv adapter)
# ---------------------------------------------------------------------------

def _make_rllib_env_cls(cfg: dict):
    """
    Dynamically create an RLlib-compatible MultiAgentEnv class that wraps
    MultiAgentSumoEnv with the given config.
    """
    import ray
    from ray.rllib.env.multi_agent_env import MultiAgentEnv
    from src.sumo_env import MultiAgentSumoEnv

    class SumoRLLibEnv(MultiAgentEnv):
        """RLlib MultiAgentEnv wrapper around MultiAgentSumoEnv."""

        def __init__(self, env_config: dict = None):
            super().__init__()
            _cfg = env_config or cfg
            self._env = MultiAgentSumoEnv(_cfg)
            # Initialise spaces without consuming a dataset episode.
            self._env.ensure_initialized()
            self._agents = list(self._env.get_agent_ids())
            self._observation_space = self._env.observation_space
            self._action_space = self._env.action_space
            self._last_obs = None

        # ---- Required properties ----
        @property
        def observation_space(self):
            return self._observation_space

        @property
        def action_space(self):
            return self._action_space

        def reset(self, *, seed=None, options=None):
            obs, info = self._env.reset(seed=seed, options=options)
            self._last_obs = obs
            return obs, info

        def step(self, action_dict):
            return self._env.step(action_dict)

        def close(self):
            self._env.close()

        def get_agent_ids(self):
            return set(self._agents)

    return SumoRLLibEnv


# ---------------------------------------------------------------------------
# MARLTrainer
# ---------------------------------------------------------------------------

class MARLTrainer:
    """
    Manages RLlib PPO training for the multi-agent traffic signal system.

    Parameters
    ----------
    config : dict
        Full project config loaded from config/config.yaml.
    """

    def __init__(self, config: dict) -> None:
        self.cfg = config
        self.train_cfg = config["training"]
        self.log_cfg = config["logging"]

        self._algo = None
        self._results_dir = Path(self.log_cfg["results_dir"])
        self._results_dir.mkdir(parents=True, exist_ok=True)

        self._training_log_path = Path(self.log_cfg["training_log"])
        self._episode_rewards: List[float] = []
        self._episode_waits: List[float] = []
        self._episode_co2s: List[float] = []
        self._episode_fuels: List[float] = []

    # ------------------------------------------------------------------
    def setup(self) -> None:
        """Initialise Ray and build the PPO algorithm."""
        import ray
        from ray.rllib.algorithms.ppo import PPOConfig
        from ray import tune

        if not ray.is_initialized():
            ray.init(
                ignore_reinit_error=True,
                local_mode=True,  # single-process mode – avoids raylet startup on Windows
                include_dashboard=False,
            )
            logger.info("[Ray] Initialized (local mode).")

        env_cls = _make_rllib_env_cls(self.cfg)
        tune.register_env("sumo_marl", lambda cfg: env_cls(cfg))

        tl_ids = self.cfg["intersections"]["ids"]

        # Policy mapping: shared or individual
        if self.train_cfg.get("shared_policy", True):
            policy_mapping = {tid: "shared_policy" for tid in tl_ids}
            policies = {"shared_policy"}
            mapping_fn = lambda agent_id, episode, worker, **kw: "shared_policy"
        else:
            policy_mapping = {tid: tid for tid in tl_ids}
            policies = set(tl_ids)
            mapping_fn = lambda agent_id, episode, worker, **kw: agent_id

        algo_cfg = (
            PPOConfig()
            .api_stack(
                enable_rl_module_and_learner=False,
                enable_env_runner_and_connector_v2=False,
            )
            .environment(
                env="sumo_marl",
                env_config=self.cfg,
                disable_env_checking=True,
            )
            .framework("torch")
            .multi_agent(
                policies=policies,
                policy_mapping_fn=mapping_fn,
            )
            .training(
                lr=self.train_cfg.get("learning_rate", 3e-4),
                gamma=self.train_cfg.get("gamma", 0.99),
                lambda_=self.train_cfg.get("lambda_", 1.0),
                clip_param=self.train_cfg.get("clip_param", 0.2),
                vf_clip_param=self.train_cfg.get("vf_clip_param", 10.0),
                grad_clip=self.train_cfg.get("grad_clip", None),
                train_batch_size=self.train_cfg.get("train_batch_size", 4096),
                num_epochs=self.train_cfg.get("num_sgd_iter", 10),
                minibatch_size=self.train_cfg.get("sgd_minibatch_size", 512),
                entropy_coeff=self.train_cfg.get("entropy_coeff", 0.01),
            )
            .callbacks(_make_metrics_callback())
            .env_runners(
                num_env_runners=0,
            )
            .resources(num_gpus=0)
        )

        self._algo = algo_cfg.build_algo()
        logger.info("[Trainer] PPO algorithm built.")

    # ------------------------------------------------------------------
    def train(self) -> None:
        """Run the full training loop and log results."""
        assert self._algo is not None, "Call setup() first."

        n_episodes = self.train_cfg.get("num_episodes", 200)
        ckpt_freq = self.train_cfg.get("checkpoint_freq", 20)
        ckpt_dir = Path(self.train_cfg.get("checkpoint_dir", "results/checkpoints"))
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        # Archive any existing log so each run starts with a clean CSV.
        self._training_log_path.parent.mkdir(parents=True, exist_ok=True)
        if self._training_log_path.exists() and self._training_log_path.stat().st_size > 0:
            archive_dir = self._results_dir / "archive"
            archive_dir.mkdir(parents=True, exist_ok=True)
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            archived = archive_dir / f"{self._training_log_path.stem}_{stamp}{self._training_log_path.suffix}"
            self._training_log_path.replace(archived)
            logger.info("[Trainer] Archived previous training log to %s", archived)

        # CSV log header
        with open(self._training_log_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "episode", "mean_reward",
                "mean_waiting_time", "mean_co2", "mean_fuel",
                "training_iteration", "episodes_this_iter", "env_steps_this_iter",
            ])

        logger.info("[Trainer] Starting training for %d iterations …", n_episodes)

        for ep in range(1, n_episodes + 1):
            result = self._algo.train()

            # Extract reward from RLlib result dict (try multiple key paths)
            ep_reward = _get_result(result, "episode_reward_mean")

            # Drain episode-level metrics pushed by src.sumo_env.MultiAgentSumoEnv
            ep_metrics = list(EPISODE_METRICS_BUFFER)
            EPISODE_METRICS_BUFFER.clear()
            if ep_metrics:
                wait = float(np.mean([m["waiting_time_raw"] for m in ep_metrics]))
                co2  = float(np.mean([m["co2_raw"]          for m in ep_metrics]))
                fuel = float(np.mean([m["fuel_raw"]         for m in ep_metrics]))
            else:
                wait = co2 = fuel = 0.0
                logger.warning("[Trainer] No episode metrics were collected in iteration %d.", ep)

            self._episode_rewards.append(ep_reward)
            self._episode_waits.append(wait)
            self._episode_co2s.append(co2)
            self._episode_fuels.append(fuel)

            # Log
            log_every = self.cfg["logging"].get("log_every", 1)
            if ep % log_every == 0:
                logger.info(
                    "Ep %4d/%d | reward=%.4f | wait=%.1f | CO2=%.1f | fuel=%.1f",
                    ep, n_episodes, ep_reward, wait, co2, fuel,
                )
                with open(self._training_log_path, "a", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        ep,
                        ep_reward,
                        wait,
                        co2,
                        fuel,
                        result.get("training_iteration", ep),
                        (result.get("env_runners", {}) or {}).get("episodes_this_iter", 0),
                        result.get("num_env_steps_sampled_this_iter", 0),
                    ])

            # Checkpoint
            if ep % ckpt_freq == 0:
                path = self._algo.save(str(ckpt_dir))
                logger.info("[Trainer] Checkpoint saved: %s", path)

        # Final save
        final_path = self._algo.save(str(ckpt_dir))
        logger.info("[Trainer] Training complete. Final checkpoint: %s", final_path)

    # ------------------------------------------------------------------
    def plot_training(self) -> None:
        """Generate reward and metric plots from training history."""
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            sns.set_theme(style="darkgrid")
        except ImportError:
            logger.warning("matplotlib/seaborn not found – skipping plots.")
            return

        plot_path = Path(self.log_cfg.get("reward_plot", "results/reward_plot.png"))

        fig, axes = plt.subplots(2, 2, figsize=(14, 8))
        fig.suptitle("MARL Training Progress", fontsize=14)

        eps = list(range(1, len(self._episode_rewards) + 1))

        axes[0, 0].plot(eps, self._episode_rewards, color="steelblue")
        axes[0, 0].set_title("Mean Episode Reward")
        axes[0, 0].set_xlabel("Episode")

        axes[0, 1].plot(eps, self._episode_waits, color="tomato")
        axes[0, 1].set_title("Mean Waiting Time (s)")
        axes[0, 1].set_xlabel("Episode")

        axes[1, 0].plot(eps, self._episode_co2s, color="seagreen")
        axes[1, 0].set_title("Mean CO2 Emission (mg)")
        axes[1, 0].set_xlabel("Episode")

        axes[1, 1].plot(eps, self._episode_fuels, color="darkorange")
        axes[1, 1].set_title("Mean Fuel Consumption (ml)")
        axes[1, 1].set_xlabel("Episode")

        plt.tight_layout()
        plt.savefig(plot_path, dpi=150)
        plt.close()
        logger.info("[Trainer] Reward plot saved: %s", plot_path)

    # ------------------------------------------------------------------
    def shutdown(self) -> None:
        """Shutdown Ray."""
        import ray
        if self._algo is not None:
            try:
                self._algo.stop()
            except Exception:
                pass
            self._algo = None
        if ray.is_initialized():
            ray.shutdown()
            logger.info("[Ray] Shutdown.")
