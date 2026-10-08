"""
Multi-Agent SUMO Traffic Signal Environment.

Implements a Gymnasium-compatible multi-agent environment where each
signalized intersection is one independent agent.  Compatible with
RLlib's MultiAgentEnv API.

State (per agent):
  [queue_length × n_lanes,  waiting_time × n_lanes,
   current_phase_one_hot × n_phases,  ARIMA_forecast × k]

Action (discrete, per agent):
  0 – maintain current phase
  1 – switch to next phase
  2 – extend green by N seconds
  3 – reduce green by N seconds
"""

from __future__ import annotations

import logging
import os
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# ---- Gymnasium ----
import gymnasium as gym
from gymnasium import spaces

# ---- Project modules ----
from src.arima_predictor import MultiIntersectionARIMA
from src.dataset_loader import DatasetLoader
from src.reward import RewardCalculator
from src.vehicle_profiles import VehicleEmissionModel

logger = logging.getLogger(__name__)


# Episode metrics produced by the real SUMO env.
# The trainer drains this after each RLlib training iteration.
EPISODE_METRICS_BUFFER: List[Dict[str, float]] = []

# ---------------------------------------------------------------------------
# TraCI import helper  (SUMO must be installed and SUMO_HOME set)
# ---------------------------------------------------------------------------

def _import_traci():
    sumo_home = os.environ.get("SUMO_HOME", "")
    if sumo_home:
        tools = os.path.join(sumo_home, "tools")
        if tools not in sys.path:
            sys.path.append(tools)
    try:
        import traci
        return traci
    except ImportError as exc:
        raise ImportError(
            "traci not found.  Set the SUMO_HOME environment variable and ensure "
            "SUMO >= 1.15 is installed.  See https://sumo.dlr.de/docs/Downloads.php"
        ) from exc


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ACTIONS = {
    0: "maintain",
    1: "next_phase",
    2: "extend_green",
    3: "reduce_green",
}

N_PHASES = 4        # number of TL phases in the logic (2 green + 2 yellow)
N_GREEN_PHASES = 2  # phases that are considered "green" (index 0 and 2)


# ---------------------------------------------------------------------------
# Helper: get lanes belonging to a junction
# ---------------------------------------------------------------------------

def _get_controlled_lanes(traci, tl_id: str) -> List[str]:
    """Return unique ordered list of lanes controlled by a traffic light."""
    links = traci.trafficlight.getControlledLinks(tl_id)
    lanes: List[str] = []
    seen = set()
    for link_group in links:
        for link in link_group:
            lane = link[0]   # incoming lane
            if lane not in seen:
                lanes.append(lane)
                seen.add(lane)
    return lanes


# ---------------------------------------------------------------------------
# MultiAgentSumoEnv
# ---------------------------------------------------------------------------

class MultiAgentSumoEnv:
    """
    Gymnasium-style multi-agent environment wrapping SUMO via TraCI.

    Parameters
    ----------
    config : dict
        Loaded from config/config.yaml.  Required sub-keys:
        sumo, intersections, state, actions, reward, arima.
    use_gui : bool | None
        Override config.sumo.use_gui (useful for evaluation).
    """

    # RLlib metadata
    metadata: Dict[str, Any] = {}

    def __init__(self, config: dict, use_gui: Optional[bool] = None) -> None:
        self.cfg = config
        self.sumo_cfg = config["sumo"]
        self.inter_cfg = config["intersections"]
        self.state_cfg = config["state"]
        self.act_cfg = config["actions"]
        self.rew_cfg = config["reward"]
        self.arima_cfg = config["arima"]

        self.tl_ids: List[str] = self.inter_cfg["ids"]
        self.n_agents = len(self.tl_ids)
        self.use_gui = use_gui if use_gui is not None else self.sumo_cfg.get("use_gui", False)

        # ARIMA
        self.arima = MultiIntersectionARIMA(
            intersection_ids=self.tl_ids,
            order=(self.arima_cfg["p"], self.arima_cfg["d"], self.arima_cfg["q"]),
            steps_ahead=self.arima_cfg["steps_ahead"],
            min_history=self.arima_cfg["min_history"],
        )

        # Reward
        self.reward_calc = RewardCalculator(
            w1=self.rew_cfg["w1"],
            w2=self.rew_cfg["w2"],
            w3=self.rew_cfg["w3"],
            clip_min=self.rew_cfg.get("clip_min", -1000.0),
            normalize=self.rew_cfg.get("normalize", True),
            delta_reward_weight=self.rew_cfg.get("delta_reward_weight", 0.5),
            co2_log_scale=self.rew_cfg.get("co2_log_scale", 1e5),
            fuel_log_scale=self.rew_cfg.get("fuel_log_scale", 1e5),
        )

        # Per-agent lane lists (populated on first reset)
        self._controlled_lanes: Dict[str, List[str]] = {}
        self._n_lanes: int = 0   # max lanes across all agents

        # Spaces – defined lazily after first TraCI connection
        self._obs_space: Optional[spaces.Box] = None
        self._act_space: Optional[spaces.Discrete] = None

        # Runtime state
        self._traci = None
        self._traci_label: Optional[str] = None
        self._sumo_process_started = False
        self._step_count = 0
        self._episode_count = 0
        self._current_phases: Dict[str, int] = {}
        self._phase_timers: Dict[str, int] = {}   # steps since last change
        self._yellow_remaining: Dict[str, int] = {}
        self._episode_wait_sum = 0.0
        self._episode_co2_sum = 0.0
        self._episode_fuel_sum = 0.0
        self._current_sumo_seed: Optional[int] = None
        self._gui_vehicle_tracked: bool = False

        # Dataset integration
        self._dataset: Optional[DatasetLoader] = None
        self._episode_index: int = 0
        ds_cfg = self.cfg.get("dataset", {})

        # Vehicle emission model (CO2 / fuel profiles per vehicle type)
        self._vehicle_profiles: Optional[VehicleEmissionModel] = None
        vehicle_profiles_path = ds_cfg.get("vehicle_profiles_file")
        if vehicle_profiles_path:
            candidate = Path(vehicle_profiles_path)
            if candidate.exists():
                self._vehicle_profiles = VehicleEmissionModel(candidate)
                logger.info(
                    "[Env] Vehicle profiles loaded – %d profiles from %s.",
                    self._vehicle_profiles.n_profiles,
                    candidate,
                )
            else:
                logger.warning("[Env] Vehicle profile CSV not found: %s", candidate)

        # Inflow tracking for ARIMA
        self._prev_arrived: Dict[str, int] = {tid: 0 for tid in self.tl_ids}
        if ds_cfg.get("enabled", False):
            data_dir = Path(ds_cfg["data_dir"])
            if data_dir.exists():
                lane_map_raw = ds_cfg.get("lane_map", {})
                # convert keys/values from yaml (may be str keys, list values)
                lane_map = {k: list(v) for k, v in lane_map_raw.items()} if lane_map_raw else None
                self._dataset = DatasetLoader(
                    data_dir=data_dir,
                    tl_ids=self.tl_ids,
                    lane_map=lane_map,
                )
                logger.info("[Env] Dataset loaded – %d episodes available.",
                            self._dataset.n_episodes)
            else:
                logger.warning("[Env] Dataset path not found: %s – running without real data.", data_dir)

    # ------------------------------------------------------------------
    # One-time initialization
    # ------------------------------------------------------------------

    def ensure_initialized(self) -> None:
        """Start/stop SUMO once to discover lanes and build spaces."""
        if self._obs_space is None or self._act_space is None:
            self._start_sumo()
            self._stop_sumo()

    # ------------------------------------------------------------------
    # Spaces (lazy, after first connect)
    # ------------------------------------------------------------------

    def _build_spaces(self):
        n_l = self._n_lanes
        k = self.arima_cfg["steps_ahead"]
        # queue + wait per lane + one-hot phase + arima
        obs_dim = n_l + n_l + N_PHASES + k
        self._obs_dim = obs_dim
        self._obs_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )
        self._act_space = spaces.Discrete(self.act_cfg["n_actions"])

    @property
    def observation_space(self):
        if self._obs_space is None:
            raise RuntimeError("Call reset() before accessing observation_space.")
        return self._obs_space

    @property
    def action_space(self):
        if self._act_space is None:
            raise RuntimeError("Call reset() before accessing action_space.")
        return self._act_space

    # ------------------------------------------------------------------
    # SUMO lifecycle
    # ------------------------------------------------------------------

    def _start_sumo(self):
        traci = _import_traci()

        sumo_bin = "sumo-gui" if self.use_gui else "sumo"
        cfg_file = str(Path(self.sumo_cfg["config_file"]).resolve())

        cmd = [
            sumo_bin,
            "-c", cfg_file,
            "--no-step-log", "true",
            "--no-warnings", "true",
        ]
        if self.use_gui and self.sumo_cfg.get("gui_auto_start", True):
            # Required for SUMO-GUI to advance immediately under TraCI.
            cmd.append("--start")
        if self.sumo_cfg.get("randomize", False):
            cmd.append("--random")
        else:
            seed = self._current_sumo_seed
            if seed is None:
                seed = int(self.sumo_cfg.get("seed", 42))
            cmd.extend(["--seed", str(seed)])
        logger.debug("Starting SUMO: %s", " ".join(cmd))
        self._traci_label = f"sumo-env-{uuid.uuid4().hex}"
        traci.start(cmd, label=self._traci_label)
        self._traci = traci.getConnection(self._traci_label)
        self._sumo_process_started = True

        # Discover controlled lanes
        max_l = 0
        for tid in self.tl_ids:
            lanes = _get_controlled_lanes(self._traci, tid)
            self._controlled_lanes[tid] = lanes
            max_l = max(max_l, len(lanes))
        self._n_lanes = max_l

        # Build spaces once
        self._build_spaces()

        if self.use_gui:
            self._configure_gui_view()

    def _stop_sumo(self):
        if self._traci is not None and self._sumo_process_started:
            try:
                self._traci.close()
            except Exception:
                pass
            self._sumo_process_started = False
            self._traci = None
            self._traci_label = None

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def reset(
        self, *, seed: Optional[int] = None, options: Optional[dict] = None
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        """Reset the simulation and return initial observations."""
        options = options or {}
        self._stop_sumo()
        self.arima.reset()
        self.reward_calc.reset()
        self._step_count = 0
        self._episode_count += 1
        self._current_phases = {tid: 0 for tid in self.tl_ids}
        self._phase_timers = {tid: 0 for tid in self.tl_ids}
        self._yellow_remaining = {tid: 0 for tid in self.tl_ids}
        self._prev_arrived = {tid: 0 for tid in self.tl_ids}
        self._episode_wait_sum = 0.0
        self._episode_co2_sum = 0.0
        self._episode_fuel_sum = 0.0
        self._gui_vehicle_tracked = False
        self._current_sumo_seed = int(options.get("sumo_seed", seed if seed is not None else self.sumo_cfg.get("seed", 42)))

        # Dataset integration ------------------------------------------------
        ds_cfg = self.cfg.get("dataset", {})
        if self._dataset is not None:
            # Cycle through daily episodes
            forced_episode_index = options.get("episode_index")
            if forced_episode_index is not None:
                self._episode_index = int(forced_episode_index) % self._dataset.n_episodes
            elif ds_cfg.get("cycle_episodes", True):
                self._episode_index = (self._episode_count - 1) % self._dataset.n_episodes

            # Patch routes.rou.xml with real demand level
            if ds_cfg.get("patch_routes", True):
                routes_path = Path(self.sumo_cfg["route_file"])
                if routes_path.exists():
                    self._dataset.patch_routes_xml(
                        routes_xml_path=routes_path,
                        episode_index=self._episode_index,
                        window_start=(self._episode_count * 300) % 50000,
                        window_steps=300,
                    )

            # Pre-warm ARIMA with real historical data
            warmup_steps = ds_cfg.get("arima_warmup_steps", 60)
            warmup = self._dataset.get_arima_warmup(
                episode_index=self._episode_index,
                n_steps=warmup_steps,
            )
            for tid, history in warmup.items():
                for val in history:
                    self.arima.predictors[tid].update(val)
            logger.debug("[Env] ARIMA pre-warmed with %d real steps (ep=%d).",
                         warmup_steps, self._episode_index)
        # --------------------------------------------------------------------

        self._start_sumo()

        obs = self._get_observations()
        infos = {tid: {} for tid in self.tl_ids}
        return obs, infos

    # ------------------------------------------------------------------
    # Step
    # ------------------------------------------------------------------

    def step(
        self, action_dict: Dict[str, int]
    ) -> Tuple[
        Dict[str, np.ndarray],
        Dict[str, float],
        Dict[str, bool],
        Dict[str, bool],
        Dict[str, Any],
    ]:
        """
        Apply actions, advance simulation, return (obs, rewards, terminated,
        truncated, infos).
        """
        assert self._traci is not None, "Call reset() first."

        # --- Apply actions ---
        for tid, action in action_dict.items():
            self._apply_action(tid, action)

        # --- Advance simulation ---
        self._traci.simulationStep()
        if self.use_gui:
            self._update_gui_tracking()
        if self.use_gui:
            gui_delay_ms = int(self.sumo_cfg.get("gui_delay_ms", 350))
            if gui_delay_ms > 0:
                time.sleep(gui_delay_ms / 1000.0)
        self._step_count += 1
        for tid in self.tl_ids:
            self._phase_timers[tid] += 1
            if self._yellow_remaining[tid] > 0:
                self._yellow_remaining[tid] -= 1
                if self._yellow_remaining[tid] == 0:
                    try:
                        self._traci.trafficlight.setPhase(tid, self._current_phases[tid])
                    except Exception:
                        pass

        # --- Collect metrics ---
        if self._vehicle_profiles is not None:
            metrics = self._vehicle_profiles.aggregate_step_metrics(self._traci)
            total_wait = metrics["waiting_time"]
            total_co2 = metrics["co2_weighted"]
            total_fuel = metrics["fuel_weighted"]
            total_co2_raw = metrics["co2_raw"]
            total_fuel_raw = metrics["fuel_raw"]
        else:
            total_wait = 0.0
            total_co2 = 0.0
            total_fuel = 0.0
            total_co2_raw = 0.0
            total_fuel_raw = 0.0
            for vid in self._traci.vehicle.getIDList():
                total_wait += self._traci.vehicle.getWaitingTime(vid)
                co2 = self._traci.vehicle.getCO2Emission(vid)
                fuel = self._traci.vehicle.getFuelConsumption(vid)
                total_co2 += co2
                total_fuel += fuel
                total_co2_raw += co2
                total_fuel_raw += fuel

        self._episode_wait_sum += total_wait
        self._episode_co2_sum += total_co2
        self._episode_fuel_sum += total_fuel

        # --- Update ARIMA histories ---
        inflows = self._compute_inflows()
        self.arima.update(inflows)

        # --- Compute reward ---
        reward, rew_info = self.reward_calc.compute(total_wait, total_co2, total_fuel)
        rewards = {tid: reward for tid in self.tl_ids}

        # --- Observations ---
        obs = self._get_observations()

        # --- Termination ---
        max_steps = self.cfg["training"]["max_steps_per_episode"]
        sim_end = self.sumo_cfg["simulation_end"]
        done = (
            self._step_count >= max_steps
            or self._traci.simulation.getTime() >= sim_end
            or self._traci.simulation.getMinExpectedNumber() <= 0
        )

        terminated = {tid: done for tid in self.tl_ids}
        terminated["__all__"] = done
        truncated = {tid: False for tid in self.tl_ids}
        truncated["__all__"] = False

        infos = {
            tid: {
                **rew_info,
                "step": self._step_count,
            }
            for tid in self.tl_ids
        }

        if done:
            denom = float(max(self._step_count, 1))
            EPISODE_METRICS_BUFFER.append({
                "waiting_time_raw": self._episode_wait_sum / denom,
                "co2_raw": self._episode_co2_sum / denom,
                "fuel_raw": self._episode_fuel_sum / denom,
                "co2_sim_raw": total_co2_raw / denom,
                "fuel_sim_raw": total_fuel_raw / denom,
                "reward": reward,
                "episode_steps": float(self._step_count),
            })

        return obs, rewards, terminated, truncated, infos

    def _configure_gui_view(self) -> None:
        """Apply camera-friendly GUI settings for demonstrations."""
        if self._traci is None:
            return
        view_id = self.sumo_cfg.get("gui_view_id", "View #0")
        zoom = float(self.sumo_cfg.get("gui_zoom", 3500.0))
        offset_x = float(self.sumo_cfg.get("gui_offset_x", 0.0))
        offset_y = float(self.sumo_cfg.get("gui_offset_y", 0.0))
        scheme = self.sumo_cfg.get("gui_scheme")

        try:
            self._traci.gui.setZoom(view_id, zoom)
        except Exception:
            pass
        try:
            self._traci.gui.setOffset(view_id, offset_x, offset_y)
        except Exception:
            pass
        if scheme:
            try:
                self._traci.gui.setSchema(view_id, scheme)
            except Exception:
                pass

    def _update_gui_tracking(self) -> None:
        """Track the first available vehicle to keep the demo visually active."""
        if self._traci is None or self._gui_vehicle_tracked:
            return
        if not self.sumo_cfg.get("gui_track_first_vehicle", True):
            return

        vehicle_ids = self._traci.vehicle.getIDList()
        if not vehicle_ids:
            return

        view_id = self.sumo_cfg.get("gui_view_id", "View #0")
        try:
            self._traci.gui.trackVehicle(view_id, vehicle_ids[0])
            self._gui_vehicle_tracked = True
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Action application
    # ------------------------------------------------------------------

    def _apply_action(self, tl_id: str, action: int) -> None:
        """Translate discrete action → TraCI commands."""
        traci = self._traci
        if self._yellow_remaining[tl_id] > 0:
            return   # still in yellow transition – ignore action

        current = traci.trafficlight.getPhase(tl_id)
        n_phases = len(traci.trafficlight.getAllProgramLogics(tl_id)[0].phases)
        yellow_dur = self.sumo_cfg.get("yellow_duration", 4)
        extend_s = self.act_cfg.get("extend_seconds", 5)
        reduce_s = self.act_cfg.get("reduce_seconds", 5)
        min_green = self.sumo_cfg.get("min_green", 10)
        max_green = self.sumo_cfg.get("max_green", 60)

        if self._phase_timers[tl_id] < min_green and action in (1, 3):
            return

        if action == 0:
            # Maintain – do nothing
            pass

        elif action == 1:
            # Switch to next green phase via an intermediate yellow phase.
            # For the generated SUMO logic, phases alternate as:
            # green_0, yellow_0, green_1, yellow_1
            yellow_phase = (current + 1) % n_phases
            next_phase = (current + 2) % n_phases
            traci.trafficlight.setPhase(tl_id, yellow_phase)
            traci.trafficlight.setPhaseDuration(tl_id, yellow_dur)
            self._yellow_remaining[tl_id] = yellow_dur
            # Store the pending green phase to be applied once yellow ends.
            self._current_phases[tl_id] = next_phase
            self._phase_timers[tl_id] = 0

        elif action == 2:
            # Extend green duration
            logic = traci.trafficlight.getAllProgramLogics(tl_id)[0]
            phases = list(logic.phases)
            new_dur = min(phases[current].duration + extend_s, max_green)
            traci.trafficlight.setPhaseDuration(tl_id, new_dur)

        elif action == 3:
            # Reduce green duration
            logic = traci.trafficlight.getAllProgramLogics(tl_id)[0]
            phases = list(logic.phases)
            new_dur = max(phases[current].duration - reduce_s, min_green)
            traci.trafficlight.setPhaseDuration(tl_id, new_dur)

    # ------------------------------------------------------------------
    # Observation construction
    # ------------------------------------------------------------------

    def _get_observations(self) -> Dict[str, np.ndarray]:
        forecasts = self.arima.predict_all()
        obs = {}
        for tid in self.tl_ids:
            obs[tid] = self._build_obs(tid, forecasts[tid])
        return obs

    def _build_obs(self, tl_id: str, arima_forecast: np.ndarray) -> np.ndarray:
        lanes = self._controlled_lanes.get(tl_id, [])
        n_l = self._n_lanes
        k = self.arima_cfg["steps_ahead"]

        queue = np.zeros(n_l, dtype=np.float32)
        wait = np.zeros(n_l, dtype=np.float32)

        for i, lane in enumerate(lanes[:n_l]):
            try:
                queue[i] = self._traci.lane.getLastStepHaltingNumber(lane)
                wait[i] = self._traci.lane.getWaitingTime(lane)
            except Exception:
                pass

        # One-hot phase
        phase_oh = np.zeros(N_PHASES, dtype=np.float32)
        try:
            ph = self._traci.trafficlight.getPhase(tl_id)
            phase_oh[ph % N_PHASES] = 1.0
        except Exception:
            pass

        # Normalise ARIMA to same length k
        arima_vec = arima_forecast[:k] if len(arima_forecast) >= k else np.pad(
            arima_forecast, (0, k - len(arima_forecast))
        )

        obs = np.concatenate([queue, wait, phase_oh, arima_vec.astype(np.float32)])
        return obs.astype(np.float32)

    # ------------------------------------------------------------------
    # Inflow counting (for ARIMA updates)
    # ------------------------------------------------------------------

    def _compute_inflows(self) -> Dict[str, float]:
        """Approximate inflow = vehicles currently on approaches to each TL."""
        inflows: Dict[str, float] = {}
        for tid in self.tl_ids:
            lanes = self._controlled_lanes.get(tid, [])
            count = 0
            for lane in lanes:
                try:
                    count += self._traci.lane.getLastStepVehicleNumber(lane)
                except Exception:
                    pass
            inflows[tid] = float(count)
        return inflows

    # ------------------------------------------------------------------
    # Misc
    # ------------------------------------------------------------------

    def close(self):
        self._stop_sumo()

    def get_agent_ids(self) -> List[str]:
        return list(self.tl_ids)

    def seed(self, seed: Optional[int] = None):
        pass

    # ------------------------------------------------------------------
    # RLlib compatibility shim
    # ------------------------------------------------------------------

    def observation_space_sample(self, agent_ids=None):
        ids = agent_ids or self.tl_ids
        return {tid: self._obs_space.sample() for tid in ids}

    def action_space_sample(self, agent_ids=None):
        ids = agent_ids or self.tl_ids
        return {tid: self._act_space.sample() for tid in ids}
