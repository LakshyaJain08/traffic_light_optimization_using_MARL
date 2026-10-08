"""
Experiment runner.

Supports all 5 experiment modes defined in SRS §5:
  1. fixed_time      – baseline fixed-time control only
  2. wait_only       – MARL optimising waiting time only (w1=1, w2=0, w3=0)
  3. co2_only        – MARL optimising CO2 only (w1=0, w2=1, w3=0)
  4. fuel_only       – MARL optimising fuel only (w1=0, w2=0, w3=1)
  5. multi_objective – full MARL with default weights (SRS §3.5)

Usage:
    python experiments/run_experiments.py --mode all
    python experiments/run_experiments.py --mode fixed_time
    python experiments/run_experiments.py --mode multi_objective
"""

from __future__ import annotations

import argparse
import copy
import csv
import logging
import os
import sys
from pathlib import Path

# make project root importable
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s – %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("experiments")


# ---------------------------------------------------------------------------
# Experiment definitions
# ---------------------------------------------------------------------------

EXPERIMENTS = {
    "fixed_time": {
        "description": "Fixed-time control baseline (no RL)",
        "reward_override": None,
    },
    "wait_only": {
        "description": "MARL – waiting time optimisation only",
        "reward_override": {"w1": 1.0, "w2": 0.0, "w3": 0.0},
    },
    "co2_only": {
        "description": "MARL – CO2 optimisation only",
        "reward_override": {"w1": 0.0, "w2": 1.0, "w3": 0.0},
    },
    "fuel_only": {
        "description": "MARL – fuel optimisation only",
        "reward_override": {"w1": 0.0, "w2": 0.0, "w3": 1.0},
    },
    "multi_objective": {
        "description": "Full multi-objective MARL (w1=0.5, w2=0.3, w3=0.2)",
        "reward_override": None,   # use defaults from config
    },
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_config(cfg_path: str = "config/config.yaml") -> dict:
    with open(cfg_path, "r") as f:
        return yaml.safe_load(f)


def apply_reward_override(cfg: dict, override: dict) -> dict:
    cfg = copy.deepcopy(cfg)
    cfg["reward"].update(override)
    return cfg


# ---------------------------------------------------------------------------
# Runners
# ---------------------------------------------------------------------------

def run_baseline(cfg: dict) -> dict:
    """Run fixed-time baseline and return metrics."""
    from src.baseline import FixedTimeController
    logger.info("=" * 60)
    logger.info("Running FIXED-TIME baseline …")
    ctrl = FixedTimeController(cfg)
    metrics = ctrl.run()
    logger.info("Baseline done.  %s", metrics)
    return metrics


def run_marl(cfg: dict, label: str) -> dict:
    """Train and evaluate one MARL variant.  Returns evaluation metrics."""
    import copy as _copy

    exp_cfg = _copy.deepcopy(cfg)

    # Redirect checkpoints to prevent collisions between experiments
    exp_cfg["training"]["checkpoint_dir"] = f"results/checkpoints/{label}"
    exp_cfg["logging"]["training_log"] = f"results/training_logs_{label}.csv"
    exp_cfg["logging"]["reward_plot"] = f"results/reward_plot_{label}.png"
    exp_cfg["logging"]["eval_log"] = f"results/evaluation_metrics_{label}.csv"
    exp_cfg["logging"]["comparison_plot"] = f"results/comparison_plot_{label}.png"

    from src.marl_trainer import MARLTrainer
    trainer = MARLTrainer(exp_cfg)
    trainer.setup()
    trainer.train()
    trainer.plot_training()

    from src.evaluation import Evaluator
    evaluator = Evaluator(exp_cfg, n_eval_episodes=3)
    rl_metrics = evaluator.evaluate_rl()

    trainer.shutdown()
    return rl_metrics


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Run MARL traffic signal experiments")
    parser.add_argument(
        "--mode",
        choices=list(EXPERIMENTS.keys()) + ["all"],
        default="multi_objective",
        help="Experiment mode to run.",
    )
    parser.add_argument(
        "--config",
        default="config/config.yaml",
        help="Path to config YAML.",
    )
    parser.add_argument(
        "--skip-baseline",
        action="store_true",
        help="Skip the fixed-time baseline run (load from file if available).",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    Path("results").mkdir(exist_ok=True)

    modes = list(EXPERIMENTS.keys()) if args.mode == "all" else [args.mode]

    # --- Baseline ---
    baseline_metrics: dict = {}
    baseline_csv = Path("results/baseline_metrics.csv")

    if "fixed_time" in modes or not args.skip_baseline:
        baseline_metrics = run_baseline(cfg)
        with open(baseline_csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(list(baseline_metrics.keys()))
            writer.writerow(list(baseline_metrics.values()))
    elif baseline_csv.exists():
        import pandas as pd
        df = pd.read_csv(baseline_csv)
        baseline_metrics = df.iloc[0].to_dict()
        logger.info("Loaded baseline metrics from %s", baseline_csv)

    # --- MARL variants ---
    summary_rows = []
    for mode in modes:
        if mode == "fixed_time":
            continue   # already ran above

        exp = EXPERIMENTS[mode]
        logger.info("=" * 60)
        logger.info("Experiment: %s – %s", mode, exp["description"])
        logger.info("=" * 60)

        exp_cfg = cfg
        if exp["reward_override"]:
            exp_cfg = apply_reward_override(cfg, exp["reward_override"])

        rl_metrics = run_marl(exp_cfg, label=mode)

        # Comparison + CSV
        from src.evaluation import Evaluator
        evaluator = Evaluator(exp_cfg)
        evaluator.compare_and_save(baseline_metrics, rl_metrics)

        summary_rows.append({
            "experiment": mode,
            "avg_reward": rl_metrics.get("avg_reward", 0),
            "avg_waiting": rl_metrics.get("avg_waiting", 0),
            "avg_co2": rl_metrics.get("avg_co2", 0),
            "avg_fuel": rl_metrics.get("avg_fuel", 0),
        })

    # --- Summary ---
    if summary_rows:
        summary_path = Path("results/experiment_summary.csv")
        import pandas as pd
        df = pd.DataFrame(summary_rows)
        df.to_csv(summary_path, index=False)
        logger.info("Summary saved: %s", summary_path)
        print("\n" + df.to_string(index=False))

    logger.info("All experiments complete.")


if __name__ == "__main__":
    main()
