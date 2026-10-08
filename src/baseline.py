"""
Fixed-time (rule-based) baseline traffic signal controller.

Runs SUMO directly via TraCI without any RL policy.  Used as the
performance baseline that the MARL system must beat.
"""

from __future__ import annotations

import logging
import os
import sys
import time
import uuid
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from src.dataset_loader import DatasetLoader
from src.vehicle_profiles import VehicleEmissionModel

logger = logging.getLogger(__name__)


def _import_traci():
    sumo_home = os.environ.get("SUMO_HOME", "")
    if sumo_home:
        tools = os.path.join(sumo_home, "tools")
        if tools not in sys.path:
            sys.path.append(tools)
    import traci
    return traci


# ---------------------------------------------------------------------------
# FixedTimeController
# ---------------------------------------------------------------------------

class FixedTimeController:
    """
    Baseline: all intersections run fixed green/yellow cycles.

    Parameters
    ----------
    config : dict
        Project config.
    green_duration : int | None
        Override config baseline green_duration.
    yellow_duration : int | None
        Override config baseline yellow_duration.
    """

    def __init__(
        self,
        config: dict,
        green_duration: Optional[int] = None,
        yellow_duration: Optional[int] = None,
    ) -> None:
        self.cfg = config
        baseline_cfg = config.get("baseline", {})
        self.green_dur = green_duration or baseline_cfg.get("green_duration", 30)
        self.yellow_dur = yellow_duration or baseline_cfg.get("yellow_duration", 4)
        self.tl_ids: List[str] = config["intersections"]["ids"]
        self._dataset: Optional[DatasetLoader] = None
        self._vehicle_profiles: Optional[VehicleEmissionModel] = None

        ds_cfg = config.get("dataset", {})
        if ds_cfg.get("enabled", False):
            data_dir = Path(ds_cfg["data_dir"])
            if data_dir.exists():
                lane_map_raw = ds_cfg.get("lane_map", {})
                lane_map = {k: list(v) for k, v in lane_map_raw.items()} if lane_map_raw else None
                self._dataset = DatasetLoader(
                    data_dir=data_dir,
                    tl_ids=self.tl_ids,
                    lane_map=lane_map,
                )

        vehicle_profiles_path = ds_cfg.get("vehicle_profiles_file")
        if vehicle_profiles_path:
            candidate = Path(vehicle_profiles_path)
            if candidate.exists():
                self._vehicle_profiles = VehicleEmissionModel(candidate)
                logger.info(
                    "[Baseline] Vehicle profiles loaded – %d profiles from %s.",
                    self._vehicle_profiles.n_profiles,
                    candidate,
                )

    # ------------------------------------------------------------------
    def run(
        self,
        max_steps: Optional[int] = None,
        episode_index: int = 0,
        sumo_seed: Optional[int] = None,
    ) -> Dict[str, float]:
        """
        Run one full episode with fixed-time control.

        Returns
        -------
        metrics : dict
            total_waiting, total_co2, total_fuel, n_steps.
        """
        traci = _import_traci()

        ds_cfg = self.cfg.get("dataset", {})
        if self._dataset is not None and ds_cfg.get("patch_routes", True):
            routes_path = Path(self.cfg["sumo"]["route_file"])
            if routes_path.exists():
                self._dataset.patch_routes_xml(
                    routes_xml_path=routes_path,
                    episode_index=episode_index,
                    window_start=((episode_index + 1) * 300) % 50000,
                    window_steps=300,
                )

        cfg_file = str(Path(self.cfg["sumo"]["config_file"]).resolve())
        sumo_bin = "sumo-gui" if self.cfg["sumo"].get("use_gui", False) else "sumo"

        cmd = [
            sumo_bin,
            "-c", cfg_file,
            "--no-step-log", "true",
            "--no-warnings", "true",
        ]
        if self.cfg["sumo"].get("use_gui", False) and self.cfg["sumo"].get("gui_auto_start", True):
            cmd.append("--start")
        if self.cfg["sumo"].get("randomize", False):
            cmd.append("--random")
        else:
            seed = int(sumo_seed if sumo_seed is not None else self.cfg["sumo"].get("seed", 42))
            cmd.extend(["--seed", str(seed)])
        label = f"baseline-{uuid.uuid4().hex}"
        traci.start(cmd, label=label)
        conn = traci.getConnection(label)

        try:
            # Set fixed programs for every TL
            for tid in self.tl_ids:
                try:
                    logics = conn.trafficlight.getAllProgramLogics(tid)
                    if logics:
                        logic = logics[0]
                        phases = list(logic.phases)
                        for i, ph in enumerate(phases):
                            if "G" in ph.state or "g" in ph.state:
                                phases[i] = type(ph)(duration=self.green_dur,
                                                     state=ph.state)
                            else:
                                phases[i] = type(ph)(duration=self.yellow_dur,
                                                     state=ph.state)
                        conn.trafficlight.setProgramLogic(tid, logic)
                except Exception as exc:
                    logger.debug("Could not set TL logic for %s: %s", tid, exc)

            sim_end = self.cfg["sumo"]["simulation_end"]
            step_limit = max_steps or sim_end

            total_wait = 0.0
            total_co2 = 0.0
            total_fuel = 0.0
            step = 0

            while step < step_limit:
                if conn.simulation.getMinExpectedNumber() <= 0:
                    break
                if conn.simulation.getTime() >= sim_end:
                    break

                conn.simulationStep()
                if self.cfg["sumo"].get("use_gui", False):
                    gui_delay_ms = int(self.cfg["sumo"].get("gui_delay_ms", 0))
                    if gui_delay_ms > 0:
                        time.sleep(gui_delay_ms / 1000.0)
                step += 1

                if self._vehicle_profiles is not None:
                    metrics = self._vehicle_profiles.aggregate_step_metrics(conn)
                    total_wait += metrics["waiting_time"]
                    total_co2 += metrics["co2_weighted"]
                    total_fuel += metrics["fuel_weighted"]
                else:
                    for vid in conn.vehicle.getIDList():
                        total_wait += conn.vehicle.getWaitingTime(vid)
                        total_co2 += conn.vehicle.getCO2Emission(vid)
                        total_fuel += conn.vehicle.getFuelConsumption(vid)
            metrics = {
                "total_waiting": total_wait,
                "total_co2": total_co2,
                "total_fuel": total_fuel,
                "n_steps": step,
                "avg_waiting_per_step": total_wait / max(step, 1),
                "avg_co2_per_step": total_co2 / max(step, 1),
                "avg_fuel_per_step": total_fuel / max(step, 1),
            }
        finally:
            try:
                conn.close()
            except Exception:
                pass

        logger.info("[Baseline] %s", metrics)
        return metrics
