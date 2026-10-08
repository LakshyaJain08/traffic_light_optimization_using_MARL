"""
DatasetLoader – loads the Delhi Traffic Density Dataset and provides:

  1. Vehicle flow rates (veh/hour) for SUMO route calibration per episode.
  2. Historical density timeseries for ARIMA pre-warming.
  3. Per-step density tensors that can substitute or augment SUMO observations.

Dataset columns
---------------
EpochTime       – Unix timestamp (1-second steps)
QueueDensity1–6 – normalized (0–1) queue density per monitoring lane
StopDensity1–6  – normalized (0–1) stop/halted density per monitoring lane

The 6 lane channels are mapped to the 4 SUMO intersections as follows
(configurable via LANE_TO_TL_MAP):

  Lane 1,2  → TL00   (top-left)
  Lane 3,4  → TL01   (top-right)
  Lane 5    → TL10   (bottom-left)
  Lane 6    → TL11   (bottom-right)
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lane → intersection mapping  (6 data channels → 4 TL agents)
# ---------------------------------------------------------------------------
LANE_TO_TL_MAP: Dict[str, List[int]] = {
    "TL00": [1, 2],   # dataset lane indices (1-based)
    "TL01": [3, 4],
    "TL10": [5],
    "TL11": [6],
}

# Assumed road capacity for density→flow conversion (vehicles per hour per lane)
ROAD_CAPACITY_VPH = 1800   # typical urban arterial capacity


# ---------------------------------------------------------------------------
# DatasetLoader
# ---------------------------------------------------------------------------

class DatasetLoader:
    """
    Loads all CSV files in the Delhi Traffic Density Dataset folder and
    exposes them as indexed episodes.

    Parameters
    ----------
    data_dir : str | Path
        Path to the ``DelhiTrafficDensityDataset`` folder.
    tl_ids : list[str]
        Traffic-light IDs that match LANE_TO_TL_MAP keys.
    lane_map : dict | None
        Override LANE_TO_TL_MAP if your intersection IDs differ.
    """

    def __init__(
        self,
        data_dir: str | Path,
        tl_ids: Optional[List[str]] = None,
        lane_map: Optional[Dict[str, List[int]]] = None,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.tl_ids = tl_ids or list(LANE_TO_TL_MAP.keys())
        self.lane_map = lane_map or LANE_TO_TL_MAP

        self._files: List[Path] = sorted(self.data_dir.glob("*.csv"))
        if not self._files:
            raise FileNotFoundError(f"No CSV files found in {self.data_dir}")

        self._loaded: Dict[str, pd.DataFrame] = {}   # filename stem → DataFrame
        logger.info("[DatasetLoader] Found %d daily CSV files.", len(self._files))

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def n_episodes(self) -> int:
        return len(self._files)

    @property
    def episode_names(self) -> List[str]:
        return [f.stem for f in self._files]

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def load_episode(self, index: int) -> pd.DataFrame:
        """
        Load (and cache) one day's CSV by index.

        Returns DataFrame with columns:
          EpochTime, QueueDensity1..6, StopDensity1..6
        """
        path = self._files[index % len(self._files)]
        key = path.stem
        if key not in self._loaded:
            df = pd.read_csv(path)
            df = df.sort_values("EpochTime").reset_index(drop=True)
            self._loaded[key] = df
            logger.debug("[DatasetLoader] Loaded %s (%d rows)", path.name, len(df))
        return self._loaded[key]

    def load_all(self) -> None:
        """Pre-load all CSV files into memory."""
        for i in range(len(self._files)):
            self.load_episode(i)
        logger.info("[DatasetLoader] All %d episodes loaded.", len(self._files))

    # ------------------------------------------------------------------
    # ARIMA pre-warming
    # ------------------------------------------------------------------

    def get_arima_warmup(
        self,
        episode_index: int,
        n_steps: int = 60,
        offset: int = 0,
    ) -> Dict[str, List[float]]:
        """
        Return the first ``n_steps`` inflow observations from a day's data
        for each intersection, suitable for pre-loading into ARIMAPredictor.

        Parameters
        ----------
        episode_index : int
            Which daily CSV to use (0-based).
        n_steps : int
            How many historical steps to return (≥ 20 recommended).
        offset : int
            Start reading from this row index (0 = start of day).

        Returns
        -------
        dict[tl_id → list[float]]
            Per-intersection inflow histories (density × capacity).
        """
        df = self.load_episode(episode_index)
        slice_df = df.iloc[offset: offset + n_steps]

        histories: Dict[str, List[float]] = {tid: [] for tid in self.tl_ids}
        for _, row in slice_df.iterrows():
            for tid, lane_indices in self.lane_map.items():
                if tid not in self.tl_ids:
                    continue
                # Average queue density across lanes assigned to this TL
                densities = [
                    row.get(f"QueueDensity{li}", 0.0) for li in lane_indices
                ]
                avg_density = float(np.mean(densities))
                # Convert density to estimated vehicle inflow (vehicles/step)
                inflow = avg_density * ROAD_CAPACITY_VPH / 3600.0
                histories[tid].append(inflow)

        return histories

    # ------------------------------------------------------------------
    # SUMO demand calibration
    # ------------------------------------------------------------------

    def get_flow_rates(
        self,
        episode_index: int,
        window_start: int = 0,
        window_steps: int = 300,
    ) -> Dict[str, float]:
        """
        Compute mean vehicle flow rates (veh/hour) for a time window in a
        daily episode.  Use these to populate ``vehsPerHour`` in routes.rou.xml.

        Parameters
        ----------
        episode_index : int
        window_start : int
            Row offset (0 = start of day, 3600 = 1 hour into day, etc.)
        window_steps : int
            Number of 1-second rows to average over.

        Returns
        -------
        dict[tl_id → float]
            Estimated veh/hour entering each intersection.
        """
        df = self.load_episode(episode_index)
        window = df.iloc[window_start: window_start + window_steps]

        flows: Dict[str, float] = {}
        for tid, lane_indices in self.lane_map.items():
            if tid not in self.tl_ids:
                continue
            densities = []
            for li in lane_indices:
                col = f"QueueDensity{li}"
                if col in window.columns:
                    densities.extend(window[col].tolist())
            avg_density = float(np.mean(densities)) if densities else 0.0
            flows[tid] = avg_density * ROAD_CAPACITY_VPH
        return flows

    # ------------------------------------------------------------------
    # Per-step density access
    # ------------------------------------------------------------------

    def get_step_densities(
        self,
        episode_index: int,
        step: int,
    ) -> Dict[str, Dict[str, float]]:
        """
        Return queue and stop densities for a single simulation step.

        Useful for replacing or augmenting SUMO's TraCI observations.

        Returns
        -------
        dict[tl_id → {"queue": float, "stop": float}]
        """
        df = self.load_episode(episode_index)
        row_idx = min(step, len(df) - 1)
        row = df.iloc[row_idx]

        result: Dict[str, Dict[str, float]] = {}
        for tid, lane_indices in self.lane_map.items():
            if tid not in self.tl_ids:
                continue
            q_vals = [row.get(f"QueueDensity{li}", 0.0) for li in lane_indices]
            s_vals = [row.get(f"StopDensity{li}", 0.0) for li in lane_indices]
            result[tid] = {
                "queue": float(np.mean(q_vals)),
                "stop": float(np.mean(s_vals)),
            }
        return result

    # ------------------------------------------------------------------
    # Route XML patcher
    # ------------------------------------------------------------------

    def patch_routes_xml(
        self,
        routes_xml_path: str | Path,
        episode_index: int,
        window_start: int = 0,
        window_steps: int = 300,
        output_path: Optional[str | Path] = None,
    ) -> Path:
        """
        Rewrite the ``vehsPerHour`` attribute in routes.rou.xml to match
        real Delhi traffic demand for a specific episode window.

        Parameters
        ----------
        routes_xml_path : str | Path
        episode_index : int
        window_start : int
            Row offset into the daily CSV.
        window_steps : int
            Rows to average for flow estimation.
        output_path : str | Path | None
            Where to write patched XML.  Defaults to overwriting the original.

        Returns
        -------
        Path  – path to the written file.
        """
        import xml.etree.ElementTree as ET

        flows = self.get_flow_rates(episode_index, window_start, window_steps)
        avg_flow = float(np.mean(list(flows.values()))) if flows else 200.0

        tree = ET.parse(str(routes_xml_path))
        root = tree.getroot()

        for flow_elem in root.findall("flow"):
            flow_elem.set("vehsPerHour", str(int(round(avg_flow))))

        out = Path(output_path or routes_xml_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        tmp_out = out.with_suffix(out.suffix + ".tmp")
        tree.write(str(tmp_out), xml_declaration=True, encoding="utf-8")
        tmp_out.replace(out)
        logger.info(
            "[DatasetLoader] Patched %s with flow=%.0f veh/h (ep=%d, rows %d–%d)",
            out.name, avg_flow, episode_index, window_start,
            window_start + window_steps,
        )
        return out
