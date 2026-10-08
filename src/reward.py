"""
Multi-objective reward computation.

R = -(w1 * W_t  +  w2 * C_t  +  w3 * F_t)

where:
  W_t … total waiting time   (sum over all vehicles)
  C_t … total CO2 emissions  (mg/step)
  F_t … total fuel consumed  (ml/step)
"""

from __future__ import annotations

import logging
from typing import Dict, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Running normalisers (online mean/std)
# ---------------------------------------------------------------------------

class _RunningNorm:
    """Welford online mean/variance estimator for scalar streams."""

    def __init__(self, eps: float = 1e-8) -> None:
        self.n = 0
        self.mean = 0.0
        self.M2 = 0.0
        self.eps = eps

    def update(self, x: float) -> None:
        self.n += 1
        delta = x - self.mean
        self.mean += delta / self.n
        self.M2 += delta * (x - self.mean)

    @property
    def std(self) -> float:
        if self.n < 2:
            return 1.0
        return max(float(np.sqrt(self.M2 / self.n)), self.eps)

    def normalize(self, x: float) -> float:
        return (x - self.mean) / self.std


# ---------------------------------------------------------------------------
# RewardCalculator
# ---------------------------------------------------------------------------

class RewardCalculator:
    """
    Computes the scalar multi-objective reward for one simulation step.

    Parameters
    ----------
    w1, w2, w3 : float
        Weights for waiting time, CO2, and fuel respectively.
    clip_min : float
        Minimum allowed reward (prevents extreme negative values).
    normalize : bool
        If True, each component is z-score normalised online before weighting.
    """

    def __init__(
        self,
        w1: float = 0.5,
        w2: float = 0.3,
        w3: float = 0.2,
        clip_min: float = -1000.0,
        normalize: bool = True,
        delta_reward_weight: float = 0.5,
        co2_log_scale: float = 1e5,
        fuel_log_scale: float = 1e5,
    ) -> None:
        self.w1 = w1
        self.w2 = w2
        self.w3 = w3
        self.clip_min = clip_min
        self.normalize = normalize
        self.delta_reward_weight = delta_reward_weight
        self.co2_log_scale = co2_log_scale
        self.fuel_log_scale = fuel_log_scale

        self._norm_w = _RunningNorm()
        self._norm_c = _RunningNorm()
        self._norm_f = _RunningNorm()
        self._prev_waiting: Optional[float] = None
        self._prev_co2: Optional[float] = None
        self._prev_fuel: Optional[float] = None

    # ------------------------------------------------------------------
    def compute(
        self,
        total_waiting: float,
        total_co2: float,
        total_fuel: float,
    ) -> Tuple[float, Dict[str, float]]:
        """
        Parameters
        ----------
        total_waiting : float
            Sum of vehicle waiting times this step (seconds).
        total_co2 : float
            Sum of CO2 emissions this step (mg).
        total_fuel : float
            Sum of fuel consumption this step (ml).

        Returns
        -------
        reward : float
            Scalar reward.
        info : dict
            Raw and normalised component values for logging.
        """
        # Guard against NaN
        total_waiting = float(np.nan_to_num(total_waiting))
        total_co2 = float(np.nan_to_num(total_co2))
        total_fuel = float(np.nan_to_num(total_fuel))

        raw_co2 = total_co2
        raw_fuel = total_fuel

        # Aggressive log scaling to keep emissions on a comparable scale.
        total_co2 = float(np.log1p(total_co2 / max(self.co2_log_scale, 1.0)))
        total_fuel = float(np.log1p(total_fuel / max(self.fuel_log_scale, 1.0)))

        # Normalise BEFORE updating so the current sample is compared against
        # history, not against itself (updating first always yields z-score=0
        # for the very first observation of each incremental batch).
        if self.normalize:
            w_n = self._norm_w.normalize(total_waiting)
            c_n = self._norm_c.normalize(total_co2)
            f_n = self._norm_f.normalize(total_fuel)
        else:
            w_n = total_waiting
            c_n = total_co2
            f_n = total_fuel

        base_reward = -(self.w1 * w_n + self.w2 * c_n + self.w3 * f_n)

        # Reward relative improvement from one step to the next.
        # If congestion/emissions/fuel drop, the deltas become negative and
        # the shaped term becomes positive.
        if self._prev_waiting is None:
            delta_reward = 0.0
        else:
            dw = (total_waiting - self._prev_waiting) / max(abs(self._prev_waiting), 1.0)
            dc = (total_co2 - self._prev_co2) / max(abs(self._prev_co2), 1.0)
            df = (total_fuel - self._prev_fuel) / max(abs(self._prev_fuel), 1.0)
            delta_reward = -(
                self.w1 * float(np.tanh(dw))
                + self.w2 * float(np.tanh(dc))
                + self.w3 * float(np.tanh(df))
            )

        reward = base_reward + self.delta_reward_weight * delta_reward

        # Update running statistics after computing the reward
        self._norm_w.update(total_waiting)
        self._norm_c.update(total_co2)
        self._norm_f.update(total_fuel)
        self._prev_waiting = total_waiting
        self._prev_co2 = total_co2
        self._prev_fuel = total_fuel

        reward = float(np.clip(reward, self.clip_min, 1.0))

        info = {
            "waiting_time_raw": total_waiting,
            "co2_raw": raw_co2,
            "fuel_raw": raw_fuel,
            "co2_scaled": total_co2,
            "fuel_scaled": total_fuel,
            "base_reward": base_reward,
            "delta_reward": delta_reward,
            "reward": reward,
        }
        return reward, info

    # ------------------------------------------------------------------
    def reset(self) -> None:
        """Reset running statistics (call at environment reset if desired)."""
        self._norm_w = _RunningNorm()
        self._norm_c = _RunningNorm()
        self._norm_f = _RunningNorm()
        self._prev_waiting = None
        self._prev_co2 = None
        self._prev_fuel = None
