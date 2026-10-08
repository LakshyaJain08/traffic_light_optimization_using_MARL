"""
Evaluation module.

Loads a trained RLlib checkpoint and runs evaluation episodes,
logging per-step and per-episode metrics to CSV.  Also generates
comparison plots against the fixed-time baseline.
"""

from __future__ import annotations

import csv
from datetime import datetime
import logging
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------

class Evaluator:
    """
    Runs evaluation episodes using a trained RLlib policy and compares
    results against a fixed-time baseline.

    Parameters
    ----------
    config : dict
        Full project config.
    checkpoint_path : str | Path | None
        Path to the RLlib checkpoint.  If None the latest checkpoint in
        config.training.checkpoint_dir is used.
    n_eval_episodes : int
        Number of evaluation episodes.
    """

    def __init__(
        self,
        config: dict,
        checkpoint_path: Optional[str] = None,
        n_eval_episodes: int = 5,
    ) -> None:
        self.cfg = config
        self.n_eval = n_eval_episodes
        self._ckpt = checkpoint_path
        self._log_cfg = config["logging"]

    # ------------------------------------------------------------------
    def _resolve_checkpoint(self) -> str:
        if self._ckpt:
            return str(Path(self._ckpt).resolve())
        ckpt_dir = Path(self.cfg["training"]["checkpoint_dir"]).resolve()
        if not ckpt_dir.exists():
            raise FileNotFoundError(
                f"No checkpoints found in {ckpt_dir}.  Train first."
            )

        checkpoints = sorted(ckpt_dir.glob("checkpoint_*"))
        if checkpoints:
            return str(checkpoints[-1].resolve())

        # RLlib on this setup saves checkpoint contents directly into the
        # configured directory (algorithm_state.pkl + rllib_checkpoint.json).
        if (ckpt_dir / "algorithm_state.pkl").exists() and (ckpt_dir / "rllib_checkpoint.json").exists():
            return str(ckpt_dir.resolve())

        raise FileNotFoundError(
            f"No checkpoints found in {ckpt_dir}.  Train first."
        )

    # ------------------------------------------------------------------
    def evaluate_baseline(self) -> Dict[str, float]:
        """Run fixed-time baseline over the same number of episodes as RL eval."""
        from src.baseline import FixedTimeController

        controller = FixedTimeController(self.cfg)
        max_steps = self.cfg["training"].get("max_steps_per_episode")
        base_seed = int(self.cfg["sumo"].get("seed", 42))

        waits: List[float] = []
        co2s: List[float] = []
        fuels: List[float] = []
        details: List[Dict[str, float]] = []

        for ep in range(self.n_eval):
            metrics = controller.run(
                max_steps=max_steps,
                episode_index=ep,
                sumo_seed=base_seed + ep,
            )
            waits.append(float(metrics.get("avg_waiting_per_step", 0.0)))
            co2s.append(float(metrics.get("avg_co2_per_step", 0.0)))
            fuels.append(float(metrics.get("avg_fuel_per_step", 0.0)))
            details.append({
                "episode": ep + 1,
                "avg_waiting_per_step": waits[-1],
                "avg_co2_per_step": co2s[-1],
                "avg_fuel_per_step": fuels[-1],
            })
            logger.info(
                "[Eval Baseline] ep=%d wait=%.1f co2=%.1f fuel=%.1f",
                ep + 1,
                waits[-1],
                co2s[-1],
                fuels[-1],
            )

        return {
            "avg_waiting_per_step": float(np.mean(waits)) if waits else 0.0,
            "avg_co2_per_step": float(np.mean(co2s)) if co2s else 0.0,
            "avg_fuel_per_step": float(np.mean(fuels)) if fuels else 0.0,
            "episode_details": details,
        }

    # ------------------------------------------------------------------
    def evaluate_rl(
        self,
        explore: bool = False,
        force_phase_switch_interval: Optional[int] = 10,
    ) -> Dict[str, float]:
        """Run n eval episodes with the trained RL policy, return averages."""
        from ray.rllib.algorithms.ppo import PPOConfig
        from src.marl_trainer import _make_rllib_env_cls

        ckpt = self._resolve_checkpoint()
        logger.info("[Eval] Loading checkpoint: %s", ckpt)

        env_cls = _make_rllib_env_cls(self.cfg)

        algo_cfg = (
            PPOConfig()
            .api_stack(
                enable_rl_module_and_learner=False,
                enable_env_runner_and_connector_v2=False,
            )
            .environment(env=env_cls, env_config=self.cfg, disable_env_checking=True)
            .framework("torch")
            .env_runners(num_env_runners=0)
        )
        algo = algo_cfg.build_algo()
        algo.restore(ckpt)
        policy = algo.get_policy("shared_policy")
        base_seed = int(self.cfg["sumo"].get("seed", 42))

        all_waits, all_co2s, all_fuels, all_rewards = [], [], [], []
        episode_rows = []

        for ep in range(self.n_eval):
            env = env_cls(self.cfg)
            obs, _ = env.reset(options={"episode_index": ep, "sumo_seed": base_seed + ep})
            done = {"__all__": False}
            ep_reward = 0.0
            ep_wait = ep_co2 = ep_fuel = 0.0
            ep_steps = 0

            while not done["__all__"]:
                actions = {}
                for tid, ob in obs.items():
                    action, _, _ = policy.compute_single_action(ob, explore=explore)
                    actions[tid] = action

                # Demo-only fallback: nudge a phase switch periodically so
                # the GUI does not appear frozen if policy sticks to "maintain".
                if force_phase_switch_interval and force_phase_switch_interval > 0:
                    if ep_steps > 0 and ep_steps % force_phase_switch_interval == 0:
                        first_tid = next(iter(actions.keys()))
                        actions[first_tid] = 1

                obs, rewards, dones, truncated, infos = env.step(actions)
                done = dones
                ep_reward += sum(rewards.values()) / len(rewards)
                info = next(iter(infos.values()))
                ep_wait += info.get("waiting_time_raw", 0.0)
                ep_co2 += info.get("co2_raw", 0.0)
                ep_fuel += info.get("fuel_raw", 0.0)
                ep_steps += 1

            env.close()
            all_rewards.append(ep_reward)
            all_waits.append(ep_wait / max(ep_steps, 1))
            all_co2s.append(ep_co2 / max(ep_steps, 1))
            all_fuels.append(ep_fuel / max(ep_steps, 1))
            episode_rows.append({
                "episode": ep + 1,
                "avg_reward_per_episode": ep_reward,
                "avg_waiting_per_step": all_waits[-1],
                "avg_co2_per_step": all_co2s[-1],
                "avg_fuel_per_step": all_fuels[-1],
                "steps": ep_steps,
            })
            logger.info("[Eval RL] ep=%d reward=%.3f wait=%.1f co2=%.1f fuel=%.1f steps=%d",
                        ep + 1, ep_reward, all_waits[-1], all_co2s[-1], all_fuels[-1], ep_steps)

        algo.stop()
        return {
            "avg_reward": float(np.mean(all_rewards)),
            "avg_waiting": float(np.mean(all_waits)),
            "avg_co2": float(np.mean(all_co2s)),
            "avg_fuel": float(np.mean(all_fuels)),
            "episode_details": episode_rows,
        }

    # ------------------------------------------------------------------
    def compare_and_save(
        self,
        baseline_metrics: Dict[str, float],
        rl_metrics: Dict[str, float],
    ) -> None:
        """
        Write evaluation_metrics.csv and comparison_plot.png.

        Parameters
        ----------
        baseline_metrics : dict
            Output of FixedTimeController.run().
        rl_metrics : dict
            Output of evaluate_rl().
        """
        results_dir = Path(self._log_cfg["results_dir"])
        results_dir.mkdir(parents=True, exist_ok=True)

        # --- CSV ---
        csv_path = Path(self._log_cfg["eval_log"])
        if csv_path.exists() and csv_path.stat().st_size > 0:
            archive_dir = results_dir / "archive"
            archive_dir.mkdir(parents=True, exist_ok=True)
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            archived = archive_dir / f"{csv_path.stem}_{stamp}{csv_path.suffix}"
            shutil.copy2(csv_path, archived)
            logger.info("[Eval] Archived previous evaluation log to %s", archived)

        def pct_reduction(base, rl):
            if base == 0:
                return 0.0
            return 100.0 * (base - rl) / abs(base)

        b_wait = baseline_metrics.get("avg_waiting_per_step", 0.0)
        b_co2 = baseline_metrics.get("avg_co2_per_step", 0.0)
        b_fuel = baseline_metrics.get("avg_fuel_per_step", 0.0)
        r_wait = rl_metrics.get("avg_waiting", 0.0)
        r_co2 = rl_metrics.get("avg_co2", 0.0)
        r_fuel = rl_metrics.get("avg_fuel", 0.0)

        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["metric", "baseline", "rl_agent", "pct_reduction"])
            writer.writerow(["waiting_time_per_step", b_wait, r_wait,
                             pct_reduction(b_wait, r_wait)])
            writer.writerow(["co2_per_step", b_co2, r_co2,
                             pct_reduction(b_co2, r_co2)])
            writer.writerow(["fuel_per_step", b_fuel, r_fuel,
                             pct_reduction(b_fuel, r_fuel)])
            writer.writerow([])
            writer.writerow(["episode", "baseline_wait", "rl_wait", "baseline_co2", "rl_co2", "baseline_fuel", "rl_fuel"])
            b_details = baseline_metrics.get("episode_details", []) or []
            r_details = rl_metrics.get("episode_details", []) or []
            for idx in range(max(len(b_details), len(r_details))):
                b = b_details[idx] if idx < len(b_details) else {}
                r = r_details[idx] if idx < len(r_details) else {}
                writer.writerow([
                    idx + 1,
                    b.get("avg_waiting_per_step", 0.0),
                    r.get("avg_waiting_per_step", 0.0),
                    b.get("avg_co2_per_step", 0.0),
                    r.get("avg_co2_per_step", 0.0),
                    b.get("avg_fuel_per_step", 0.0),
                    r.get("avg_fuel_per_step", 0.0),
                ])
        logger.info("[Eval] Metrics saved: %s", csv_path)

        # Validation checks (SRS §9)
        wait_red = pct_reduction(b_wait, r_wait)
        co2_red = pct_reduction(b_co2, r_co2)
        fuel_red = pct_reduction(b_fuel, r_fuel)
        logger.info("[Validation] Wait reduction: %.1f%% (need >10%%)", wait_red)
        logger.info("[Validation] CO2  reduction: %.1f%% (need >5%%)", co2_red)
        logger.info("[Validation] Fuel reduction: %.1f%% (need >5%%)", fuel_red)

        # --- Plot ---
        self._comparison_plot(
            baseline_metrics={"Waiting": b_wait, "CO2": b_co2, "Fuel": b_fuel},
            rl_metrics={"Waiting": r_wait, "CO2": r_co2, "Fuel": r_fuel},
        )

    # ------------------------------------------------------------------
    def _comparison_plot(
        self,
        baseline_metrics: Dict[str, float],
        rl_metrics: Dict[str, float],
    ) -> None:
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
        except ImportError:
            logger.warning("matplotlib not found – skipping comparison plot.")
            return

        sns.set_theme(style="whitegrid")
        labels = list(baseline_metrics.keys())
        base_vals = [float(baseline_metrics[k]) for k in labels]
        rl_vals = [float(rl_metrics[k]) for k in labels]

        fig, axes = plt.subplots(1, len(labels), figsize=(12, 4.8))
        if len(labels) == 1:
            axes = [axes]

        colors = ["#4878D0", "#EE854A"]
        for idx, (ax, label, bv, rv) in enumerate(zip(axes, labels, base_vals, rl_vals)):
            bars = ax.bar(["Baseline", "MARL"], [bv, rv], color=colors, width=0.55)
            ax.set_title(label)
            ax.set_ylabel("Per-step value")

            ymax = max(bv, rv, 1e-9)
            ax.set_ylim(0, ymax * 1.25)

            for bar in bars:
                val = bar.get_height()
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    val + ymax * 0.03,
                    f"{val:.1f}",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )

            if bv > 0:
                red = 100.0 * (bv - rv) / bv
                ax.text(
                    0.5,
                    0.93,
                    f"Reduction: {red:.1f}%",
                    transform=ax.transAxes,
                    ha="center",
                    va="top",
                    fontsize=9,
                    color="darkgreen",
                )

        fig.suptitle("Baseline vs MARL Agent - Per-Step Averages", fontsize=13)

        plt.tight_layout()
        plot_path = Path(self._log_cfg["comparison_plot"])
        plt.savefig(plot_path, dpi=150)
        plt.close()
        logger.info("[Eval] Comparison plot saved: %s", plot_path)
