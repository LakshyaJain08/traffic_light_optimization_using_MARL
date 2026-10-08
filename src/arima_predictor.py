"""
ARIMA-based short-term traffic inflow predictor.

For each signalized intersection an ARIMAPredictor instance maintains a
rolling history of observed vehicle inflows and forecasts the next k steps.
"""

from __future__ import annotations

import logging
import warnings
from collections import deque
from typing import Deque, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

# Suppress statsmodels convergence warnings during experimentation
warnings.filterwarnings("ignore", category=UserWarning, module="statsmodels")
warnings.filterwarnings("ignore", category=FutureWarning)


class ARIMAPredictor:
    """
    Maintains a sliding window of vehicle inflow observations for a single
    intersection and returns ARIMA forecasts.

    Parameters
    ----------
    order : tuple
        (p, d, q) ARIMA order.  Default (5, 1, 0) per SRS §3.3.3.
    steps_ahead : int
        Number of future steps to forecast (k).  Default 5.
    min_history : int
        Minimum observations required before fitting ARIMA.
        Returns zeros if history is shorter.
    window : int
        Maximum rolling-window length kept in memory.
    """

    def __init__(
        self,
        order: tuple = (5, 1, 0),
        steps_ahead: int = 5,
        min_history: int = 20,
        window: int = 200,
    ) -> None:
        self.order = order
        self.steps_ahead = steps_ahead
        self.min_history = min_history
        self.history: Deque[float] = deque(maxlen=window)
        self._last_forecast: np.ndarray = np.zeros(steps_ahead, dtype=np.float32)

    # ------------------------------------------------------------------
    def update(self, observation: float) -> None:
        """Append a new inflow observation to history."""
        self.history.append(float(observation))

    # ------------------------------------------------------------------
    def predict(self) -> np.ndarray:
        """
        Fit ARIMA on current history and return a forecast of length
        ``steps_ahead``.  Returns the last valid forecast (or zeros) on
        failure.
        """
        if len(self.history) < self.min_history:
            return self._last_forecast.copy()

        try:
            from statsmodels.tsa.arima.model import ARIMA  # lazy import

            series = np.array(self.history, dtype=float)
            model = ARIMA(series, order=self.order)
            result = model.fit(method_kwargs={"warn_convergence": False})
            forecast = result.forecast(steps=self.steps_ahead)
            forecast = np.clip(forecast, 0, None).astype(np.float32)
            self._last_forecast = forecast
        except Exception as exc:
            logger.debug("ARIMA fit failed: %s – reusing last forecast.", exc)

        return self._last_forecast.copy()

    # ------------------------------------------------------------------
    def reset(self) -> None:
        """Clear history (call at environment reset)."""
        self.history.clear()
        self._last_forecast = np.zeros(self.steps_ahead, dtype=np.float32)


# ---------------------------------------------------------------------------
# Multi-intersection manager
# ---------------------------------------------------------------------------

class MultiIntersectionARIMA:
    """
    Manages one ARIMAPredictor per intersection and provides a dict interface.

    Parameters
    ----------
    intersection_ids : list[str]
        SUMO traffic-light IDs.
    order : tuple
        ARIMA (p, d, q) order.
    steps_ahead : int
        Forecast horizon k.
    min_history : int
        Minimum history length before fitting.
    """

    def __init__(
        self,
        intersection_ids: List[str],
        order: tuple = (5, 1, 0),
        steps_ahead: int = 5,
        min_history: int = 20,
    ) -> None:
        self.ids = intersection_ids
        self.predictors: Dict[str, ARIMAPredictor] = {
            tid: ARIMAPredictor(
                order=order,
                steps_ahead=steps_ahead,
                min_history=min_history,
            )
            for tid in intersection_ids
        }

    # ------------------------------------------------------------------
    def update(self, inflows: Dict[str, float]) -> None:
        """Update history for each intersection with latest inflow counts."""
        for tid, val in inflows.items():
            if tid in self.predictors:
                self.predictors[tid].update(val)

    # ------------------------------------------------------------------
    def predict_all(self) -> Dict[str, np.ndarray]:
        """Return ARIMA forecasts for every intersection."""
        return {tid: pred.predict() for tid, pred in self.predictors.items()}

    # ------------------------------------------------------------------
    def reset(self) -> None:
        """Reset all predictors."""
        for pred in self.predictors.values():
            pred.reset()
