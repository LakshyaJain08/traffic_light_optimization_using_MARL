"""
Vehicle emission and fuel-efficiency profiles.

Loads the CO2 / fuel efficiency CSV and exposes:

  - SUMO vehicle-type definitions for a mixed fleet.
  - Per-type scaling factors for CO2 and fuel consumption.
  - Step-level aggregation helpers for TraCI connections.

The CSV is treated as a catalogue of representative vehicle profiles.
Each row becomes one sampled vehicle type in the route mix.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


_VEHICLE_DYNAMICS: Dict[str, Dict[str, Any]] = {
    "car": {
        "accel": 2.6,
        "decel": 4.5,
        "sigma": 0.5,
        "length": 5.0,
        "minGap": 2.5,
        "maxSpeed": 13.89,
        "guiShape": "passenger",
    },
    "suv": {
        "accel": 2.3,
        "decel": 4.2,
        "sigma": 0.5,
        "length": 5.4,
        "minGap": 2.5,
        "maxSpeed": 13.0,
        "guiShape": "passenger",
    },
    "2w": {
        "accel": 3.2,
        "decel": 5.0,
        "sigma": 0.4,
        "length": 2.0,
        "minGap": 1.0,
        "maxSpeed": 16.67,
        "guiShape": "passenger",
    },
    "bus": {
        "accel": 1.2,
        "decel": 4.0,
        "sigma": 0.6,
        "length": 12.0,
        "minGap": 3.0,
        "maxSpeed": 11.11,
        "guiShape": "bus",
    },
    "truck": {
        "accel": 1.0,
        "decel": 4.0,
        "sigma": 0.6,
        "length": 8.0,
        "minGap": 2.5,
        "maxSpeed": 11.11,
        "guiShape": "truck",
    },
    "auto_rickshaw": {
        "accel": 2.0,
        "decel": 4.5,
        "sigma": 0.5,
        "length": 3.0,
        "minGap": 1.0,
        "maxSpeed": 12.5,
        "guiShape": "taxi",
    },
    "electric": {
        "accel": 2.7,
        "decel": 4.5,
        "sigma": 0.4,
        "length": 5.0,
        "minGap": 2.5,
        "maxSpeed": 13.89,
        "guiShape": "passenger",
    },
}


class VehicleEmissionModel:
    """Represent the CSV as a set of vehicle profiles and scaling factors."""

    required_columns = {
        "vehicle_type",
        "fuel_type",
        "engine_category",
        "fuel_efficiency_kmpl",
        "co2_emission_gpkm",
    }

    def __init__(self, csv_path: str | Path) -> None:
        self.csv_path = Path(csv_path)
        if not self.csv_path.exists():
            raise FileNotFoundError(f"Vehicle profile CSV not found: {self.csv_path}")

        df = pd.read_csv(self.csv_path)
        missing = self.required_columns.difference(df.columns)
        if missing:
            raise ValueError(
                f"Vehicle profile CSV is missing columns: {sorted(missing)}"
            )

        df = df.copy()
        for column in ("vehicle_type", "fuel_type", "engine_category"):
            df[column] = df[column].astype(str).str.strip().str.lower()

        df["fuel_efficiency_kmpl"] = pd.to_numeric(df["fuel_efficiency_kmpl"], errors="coerce")
        df["co2_emission_gpkm"] = pd.to_numeric(df["co2_emission_gpkm"], errors="coerce")
        df = df.dropna(subset=["fuel_efficiency_kmpl", "co2_emission_gpkm"])
        if df.empty:
            raise ValueError(f"Vehicle profile CSV has no usable rows: {self.csv_path}")

        df = df.reset_index(drop=True)
        df["profile_id"] = [self._make_profile_id(row, idx) for idx, row in df.iterrows()]
        df["vehicle_type"] = df["vehicle_type"].replace({"two_wheeler": "2w"})

        self._profiles = df
        self._baseline_profile = self._select_baseline_profile()
        baseline_co2 = max(float(self._baseline_profile["co2_emission_gpkm"]), 1e-6)
        baseline_eff = max(float(self._baseline_profile["fuel_efficiency_kmpl"]), 1e-6)
        self._profiles["co2_factor"] = self._profiles["co2_emission_gpkm"] / baseline_co2
        self._profiles["fuel_factor"] = baseline_eff / self._profiles["fuel_efficiency_kmpl"].clip(lower=1e-6)

        logger.info(
            "[VehicleEmissionModel] Loaded %d profiles from %s",
            len(self._profiles),
            self.csv_path,
        )

    # ------------------------------------------------------------------
    @staticmethod
    def _make_profile_id(row: pd.Series, index: int) -> str:
        parts = [
            str(row["vehicle_type"]),
            str(row["fuel_type"]),
            str(row["engine_category"]),
        ]
        profile_id = "_".join(parts)
        profile_id = profile_id.replace(" ", "_").replace("/", "_")
        return f"{profile_id}_{index}"

    # ------------------------------------------------------------------
    def _select_baseline_profile(self) -> pd.Series:
        car_rows = self._profiles[self._profiles["vehicle_type"] == "car"]
        if not car_rows.empty:
            return car_rows.iloc[0]
        return self._profiles.iloc[0]

    # ------------------------------------------------------------------
    @property
    def profiles(self) -> pd.DataFrame:
        return self._profiles.copy()

    # ------------------------------------------------------------------
    @property
    def n_profiles(self) -> int:
        return len(self._profiles)

    # ------------------------------------------------------------------
    def get_profile(self, vehicle_type: str) -> pd.Series:
        normalized = str(vehicle_type).strip().lower()
        exact = self._profiles[self._profiles["profile_id"] == normalized]
        if not exact.empty:
            return exact.iloc[0]

        category = self._profiles[self._profiles["vehicle_type"] == normalized]
        if not category.empty:
            return category.iloc[0]

        for _, row in self._profiles.iterrows():
            if row["profile_id"].startswith(normalized + "_"):
                return row

        return self._baseline_profile

    # ------------------------------------------------------------------
    def co2_factor(self, vehicle_type: str) -> float:
        return float(self.get_profile(vehicle_type)["co2_factor"])

    # ------------------------------------------------------------------
    def fuel_factor(self, vehicle_type: str) -> float:
        return float(self.get_profile(vehicle_type)["fuel_factor"])

    # ------------------------------------------------------------------
    def build_sumo_vtypes(self) -> List[Dict[str, Any]]:
        vtypes: List[Dict[str, Any]] = []
        for _, row in self._profiles.iterrows():
            dynamics = _VEHICLE_DYNAMICS.get(str(row["vehicle_type"]), _VEHICLE_DYNAMICS["car"])
            vtypes.append({
                "id": row["profile_id"],
                **dynamics,
            })
        return vtypes

    # ------------------------------------------------------------------
    def build_type_distribution(self) -> List[Dict[str, Any]]:
        probability = 1.0 / max(len(self._profiles), 1)
        return [
            {
                "id": row["profile_id"],
                "probability": probability,
            }
            for _, row in self._profiles.iterrows()
        ]

    # ------------------------------------------------------------------
    def aggregate_step_metrics(self, traci_conn: Any) -> Dict[str, float]:
        """Aggregate waiting time, raw emissions, and profile-weighted emissions."""
        total_wait = 0.0
        total_co2_raw = 0.0
        total_fuel_raw = 0.0
        total_co2_weighted = 0.0
        total_fuel_weighted = 0.0

        for vid in traci_conn.vehicle.getIDList():
            wait = float(traci_conn.vehicle.getWaitingTime(vid))
            co2 = float(traci_conn.vehicle.getCO2Emission(vid))
            fuel = float(traci_conn.vehicle.getFuelConsumption(vid))
            vtype = traci_conn.vehicle.getTypeID(vid)

            profile = self.get_profile(vtype)
            total_wait += wait
            total_co2_raw += co2
            total_fuel_raw += fuel
            total_co2_weighted += co2 * float(profile["co2_factor"])
            total_fuel_weighted += fuel * float(profile["fuel_factor"])

        return {
            "waiting_time": total_wait,
            "co2_raw": total_co2_raw,
            "fuel_raw": total_fuel_raw,
            "co2_weighted": total_co2_weighted,
            "fuel_weighted": total_fuel_weighted,
            "vehicle_count": float(len(traci_conn.vehicle.getIDList())),
        }
