"""
main.py – Entry point for the Multi-Objective MARL Traffic Signal
          Optimization system.

Quick-start commands:
  # 1. Generate SUMO network files (once):
  python sumo_files/generate_sumo_files.py

  # 2. Run full multi-objective training + evaluation:
  python main.py

  # 3. Run only baseline:
  python main.py --mode baseline

  # 4. Run specific experiment:
  python main.py --mode eval --checkpoint results/checkpoints/multi_objective/...

    # 5. Run faculty demo (single GUI rollout):
    python main.py --mode demo --checkpoint results/checkpoints

    # 6. Run baseline-only demo (no RL model):
    python main.py --mode baseline_demo
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

import yaml

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s – %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("main")


# ---------------------------------------------------------------------------
# Config loader
# ---------------------------------------------------------------------------

def load_config(path: str = "config/config.yaml") -> dict:
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    return cfg


# ---------------------------------------------------------------------------
# Modes
# ---------------------------------------------------------------------------

def mode_generate(cfg: dict) -> None:
    """Generate SUMO network, route and config files."""
    logger.info("Generating SUMO files …")
    import subprocess
    result = subprocess.run(
        [sys.executable, "sumo_files/generate_sumo_files.py"],
        check=True,
    )
    logger.info("SUMO files generated.")


def mode_baseline(cfg: dict) -> None:
    """Run fixed-time baseline controller."""
    from src.baseline import FixedTimeController
    ctrl = FixedTimeController(cfg)
    metrics = ctrl.run()
    print("\n── Baseline Results ──────────────────────────────────────")
    for k, v in metrics.items():
        print(f"  {k:35s}: {v:.4f}")
    print("──────────────────────────────────────────────────────────\n")


def mode_baseline_demo(cfg: dict) -> None:
    """Run a single GUI demo episode with fixed-time control (no RL model)."""
    from src.baseline import FixedTimeController

    demo_cfg = cfg.setdefault("baseline_demo", {})
    cfg.setdefault("sumo", {})
    cfg.setdefault("training", {})

    cfg["sumo"]["use_gui"] = True
    cfg["sumo"]["randomize"] = False
    cfg["sumo"]["gui_delay_ms"] = int(demo_cfg.get("gui_delay_ms", 250))

    max_steps = int(
        demo_cfg.get(
            "max_steps",
            cfg["training"].get("max_steps_per_episode", 240),
        )
    )

    ctrl = FixedTimeController(cfg)
    logger.info("Running baseline-only demo (fixed-time, no RL model) ...")
    metrics = ctrl.run(
        max_steps=max_steps,
        episode_index=0,
        sumo_seed=int(cfg["sumo"].get("seed", 42)),
    )

    print("\n-- Baseline Demo (No Model) --------------------------")
    print(f"  Steps                   : {metrics.get('n_steps', 0)}")
    print(f"  Avg waiting time / step : {metrics.get('avg_waiting_per_step', 0.0):.4f}")
    print(f"  Avg CO2 / step          : {metrics.get('avg_co2_per_step', 0.0):.4f}")
    print(f"  Avg fuel / step         : {metrics.get('avg_fuel_per_step', 0.0):.4f}")
    print("------------------------------------------------------\n")


def mode_train(cfg: dict) -> None:
    """Train MARL agents with default multi-objective reward."""
    from src.marl_trainer import MARLTrainer
    trainer = MARLTrainer(cfg)
    trainer.setup()
    try:
        trainer.train()
        trainer.plot_training()
    finally:
        trainer.shutdown()


def mode_eval(cfg: dict, checkpoint: str | None = None) -> None:
    """Evaluate trained agents vs baseline."""
    from src.evaluation import Evaluator

    evaluator = Evaluator(
        cfg,
        checkpoint_path=checkpoint,
        n_eval_episodes=cfg.get("evaluation", {}).get("n_eval_episodes", 10),
    )
    logger.info("Running baseline for comparison …")
    baseline = evaluator.evaluate_baseline()
    rl = evaluator.evaluate_rl()

    # Compare & save
    evaluator.compare_and_save(baseline, rl)

    print("\n── Evaluation Results ────────────────────────────────────")
    print(f"  {'Metric':<30}  {'Baseline':>12}  {'MARL':>12}  {'Δ%':>8}")
    print("  " + "-" * 70)
    for m, bk, rk in [
        ("Waiting time / step", "avg_waiting_per_step", "avg_waiting"),
        ("CO2 / step",          "avg_co2_per_step",     "avg_co2"),
        ("Fuel / step",         "avg_fuel_per_step",    "avg_fuel"),
    ]:
        bv = baseline.get(bk, 0)
        rv = rl.get(rk, 0)
        red = 100.0 * (bv - rv) / max(abs(bv), 1e-9)
        print(f"  {m:<30}  {bv:>12.4f}  {rv:>12.4f}  {red:>+7.1f}%")
    print("──────────────────────────────────────────────────────────\n")


def mode_demo(cfg: dict, checkpoint: str | None = None) -> None:
    """Run a single-episode GUI demo rollout with camera-friendly settings."""
    from src.evaluation import Evaluator

    demo_cfg = cfg.setdefault("demo", {})
    cfg.setdefault("evaluation", {})
    cfg.setdefault("training", {})

    # Demo-safe overrides for a clean live presentation.
    cfg["sumo"]["use_gui"] = True
    cfg["sumo"]["randomize"] = False
    cfg["sumo"]["gui_delay_ms"] = int(demo_cfg.get("gui_delay_ms", 250))

    cfg["evaluation"]["n_eval_episodes"] = int(demo_cfg.get("n_eval_episodes", 1))
    cfg["training"]["max_steps_per_episode"] = int(
        demo_cfg.get("max_steps_per_episode", cfg["training"].get("max_steps_per_episode", 300))
    )
    demo_explore = bool(demo_cfg.get("rl_explore", True))
    force_switch_interval = int(demo_cfg.get("force_phase_switch_interval", 20))

    evaluator = Evaluator(
        cfg,
        checkpoint_path=checkpoint,
        n_eval_episodes=cfg["evaluation"]["n_eval_episodes"],
    )

    logger.info("Running faculty demo rollout (RL only, no baseline) ...")
    rl = evaluator.evaluate_rl(
        explore=demo_explore,
        force_phase_switch_interval=force_switch_interval,
    )

    print("\n-- Faculty Demo Rollout -------------------------------")
    print(f"  Episodes                 : {cfg['evaluation']['n_eval_episodes']}")
    print(f"  Avg waiting time / step : {rl.get('avg_waiting', 0.0):.4f}")
    print(f"  Avg CO2 / step          : {rl.get('avg_co2', 0.0):.4f}")
    print(f"  Avg fuel / step         : {rl.get('avg_fuel', 0.0):.4f}")
    print("------------------------------------------------------\n")


def mode_full(cfg: dict) -> None:
    """Generate files → train → evaluate (end-to-end pipeline)."""
    sumo_cfg_path = Path(cfg["sumo"]["config_file"])
    if not sumo_cfg_path.exists():
        logger.info("SUMO config not found – generating files first …")
        mode_generate(cfg)

    mode_train(cfg)
    mode_eval(cfg)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Multi-Objective MARL Traffic Signal Optimization",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--mode",
        choices=["full", "generate", "baseline", "baseline_demo", "train", "eval", "demo"],
        default="full",
        help=(
            "Execution mode:\n"
            "  full      – end-to-end pipeline (default)\n"
            "  generate  – create SUMO network files only\n"
            "  baseline  – run fixed-time baseline only\n"
            "  baseline_demo – fixed-time SUMO-GUI demo (no model)\n"
            "  train     – train MARL agents only\n"
            "  eval      – evaluate trained agents vs baseline\n"
            "  demo      – one-episode faculty demo in SUMO-GUI\n"
        ),
    )
    parser.add_argument(
        "--config",
        default="config/config.yaml",
        help="Path to config YAML (default: config/config.yaml).",
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Path to RLlib checkpoint for eval mode.",
    )
    parser.add_argument(
        "--gui",
        action="store_true",
        help="Open SUMO-GUI during simulation.",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    cfg = load_config(args.config)

    if args.gui:
        cfg["sumo"]["use_gui"] = True

    Path("results").mkdir(exist_ok=True)

    dispatch = {
        "full": lambda: mode_full(cfg),
        "generate": lambda: mode_generate(cfg),
        "baseline": lambda: mode_baseline(cfg),
        "baseline_demo": lambda: mode_baseline_demo(cfg),
        "train": lambda: mode_train(cfg),
        "eval": lambda: mode_eval(cfg, checkpoint=args.checkpoint),
        "demo": lambda: mode_demo(cfg, checkpoint=args.checkpoint),
    }

    logger.info("Mode: %s", args.mode)
    dispatch[args.mode]()


if __name__ == "__main__":
    main()
