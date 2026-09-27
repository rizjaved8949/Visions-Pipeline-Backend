"""Lightweight constant-velocity Kalman filter for bounding-box tracking.

The pipeline already has ByteTrack (which uses its own internal Kalman); this
module is not a replacement for that. It exists so the GuardSelector can
extrapolate the selected guard's box for a bounded number of frames after a
detection loss, so the operator overlay does not freeze on a stale rectangle.

State: [cx, cy, w, h, vcx, vcy, vw, vh]  (position + size + first-order deltas)
Observation: [cx, cy, w, h]  (bounding-box center + size)

Everything is float64 numpy - no external dependency beyond numpy. The filter
is deliberately conservative: on a missed observation it *predicts* but does
not consume measurement noise, and its predicted box is capped by
``max_predict_seconds`` so it can never wander indefinitely.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from ..types import BBox


class BBoxKalman:
    """Constant-velocity Kalman filter over (center_x, center_y, width, height).

    Parameters mirror the vocabulary of `filterpy.kalman.KalmanFilter` but the
    implementation is inlined so guard_monitoring keeps zero additional
    dependencies (filterpy is not in requirements.txt).
    """

    _DIM_X = 8
    _DIM_Z = 4

    def __init__(
        self,
        process_noise: float = 1.0,
        measurement_noise: float = 5.0,
        initial_uncertainty: float = 100.0,
        max_predict_seconds: float = 1.5,
    ):
        self.max_predict_seconds = float(max_predict_seconds)
        self._q = float(process_noise)
        self._r = float(measurement_noise)
        self._p0 = float(initial_uncertainty)
        self.reset()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def reset(self) -> None:
        self.x = np.zeros(self._DIM_X, dtype=float)
        self.P = np.eye(self._DIM_X, dtype=float) * self._p0
        self.last_time: Optional[float] = None
        self.seconds_since_measurement: float = 0.0
        self._initialised = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def initialise(self, bbox: BBox, now: float) -> None:
        cx, cy, w, h = self._bbox_to_state(bbox)
        self.x[:] = [cx, cy, w, h, 0.0, 0.0, 0.0, 0.0]
        self.P = np.eye(self._DIM_X, dtype=float) * self._p0
        self.last_time = float(now)
        self.seconds_since_measurement = 0.0
        self._initialised = True

    @property
    def initialised(self) -> bool:
        return self._initialised

    def update(self, bbox: BBox, now: float) -> None:
        """Correct the state with a real detection at time ``now``."""
        if not self._initialised:
            self.initialise(bbox, now)
            return
        self._predict_to(now)
        z = np.asarray(self._bbox_to_state(bbox), dtype=float)
        H = np.zeros((self._DIM_Z, self._DIM_X), dtype=float)
        H[0, 0] = H[1, 1] = H[2, 2] = H[3, 3] = 1.0
        R = np.eye(self._DIM_Z, dtype=float) * self._r
        y = z - H @ self.x
        S = H @ self.P @ H.T + R
        try:
            K = self.P @ H.T @ np.linalg.inv(S)
        except np.linalg.LinAlgError:
            return
        self.x = self.x + K @ y
        I = np.eye(self._DIM_X, dtype=float)
        self.P = (I - K @ H) @ self.P
        self.last_time = float(now)
        self.seconds_since_measurement = 0.0

    def predict(self, now: float) -> Optional[BBox]:
        """Return the predicted bbox at ``now``, or None if too stale/uninit."""
        if not self._initialised or self.last_time is None:
            return None
        dt = float(now) - self.last_time
        if dt < 0:
            # Backwards timestamps are a caller bug; return the current state.
            dt = 0.0
        if self.seconds_since_measurement + dt > self.max_predict_seconds:
            return None
        # Predict without mutating internal state - callers may request
        # multiple predictions before a real update lands.
        F = self._transition_matrix(dt)
        predicted = F @ self.x
        return self._state_to_bbox(predicted[:4])

    def advance_without_measurement(self, now: float) -> Optional[BBox]:
        """Roll the filter forward one step and *keep* the predicted state.

        Used every frame where no measurement is available, so a following
        predict() call gets a fresher prior. Returns the predicted bbox or
        None if the prediction is no longer trustworthy.
        """
        if not self._initialised or self.last_time is None:
            return None
        dt = float(now) - self.last_time
        if dt <= 0:
            return self.predict(now)
        if self.seconds_since_measurement + dt > self.max_predict_seconds:
            return None
        F = self._transition_matrix(dt)
        self.x = F @ self.x
        Q = self._process_noise_matrix(dt)
        self.P = F @ self.P @ F.T + Q
        self.last_time = float(now)
        self.seconds_since_measurement += dt
        return self._state_to_bbox(self.x[:4])

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _predict_to(self, now: float) -> None:
        if self.last_time is None:
            self.last_time = float(now)
            return
        dt = float(now) - self.last_time
        if dt <= 0:
            return
        F = self._transition_matrix(dt)
        self.x = F @ self.x
        Q = self._process_noise_matrix(dt)
        self.P = F @ self.P @ F.T + Q
        self.last_time = float(now)

    def _transition_matrix(self, dt: float) -> np.ndarray:
        F = np.eye(self._DIM_X, dtype=float)
        for i in range(4):
            F[i, i + 4] = dt
        return F

    def _process_noise_matrix(self, dt: float) -> np.ndarray:
        # Simple diagonal - position noise grows with dt^2, velocity with dt.
        q = self._q
        Q = np.zeros((self._DIM_X, self._DIM_X), dtype=float)
        for i in range(4):
            Q[i, i] = q * dt * dt
            Q[i + 4, i + 4] = q * dt
        return Q

    @staticmethod
    def _bbox_to_state(bbox: BBox) -> tuple[float, float, float, float]:
        x1, y1, x2, y2 = bbox
        w = max(1.0, float(x2 - x1))
        h = max(1.0, float(y2 - y1))
        cx = float(x1) + w / 2.0
        cy = float(y1) + h / 2.0
        return cx, cy, w, h

    @staticmethod
    def _state_to_bbox(state) -> BBox:
        cx, cy, w, h = float(state[0]), float(state[1]), float(state[2]), float(state[3])
        w = max(1.0, w)
        h = max(1.0, h)
        return (cx - w / 2.0, cy - h / 2.0, cx + w / 2.0, cy + h / 2.0)
