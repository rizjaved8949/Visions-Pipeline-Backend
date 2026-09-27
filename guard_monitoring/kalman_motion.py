import numpy as np
from typing import Tuple, Optional
from .types import BBox


class KalmanBBoxPredictor:
    """
    Constant-velocity Kalman Filter for tracking bounding box center and scale
    during transient detection dropouts.
    """

    def __init__(self):
        self.active = False
        # State: [cx, cy, w, h, v_cx, v_cy]
        self.state = np.zeros(6, dtype=np.float32)
        self.covariance = np.eye(6, dtype=np.float32) * 10.0

        # Process noise
        self.Q = np.diag([1.0, 1.0, 1.0, 1.0, 4.0, 4.0]).astype(np.float32)
        # Measurement noise
        self.R = np.diag([2.0, 2.0, 4.0, 4.0]).astype(np.float32)

    def reset(self, bbox: BBox) -> None:
        cx, cy = bbox.center
        w, h = bbox.width, bbox.height
        self.state = np.array([cx, cy, w, h, 0.0, 0.0], dtype=np.float32)
        self.covariance = np.eye(6, dtype=np.float32) * 10.0
        self.active = True

    def update(self, bbox: BBox, dt: float = 1.0 / 30.0) -> None:
        if not self.active:
            self.reset(bbox)
            return

        # Predict step
        F = np.eye(6, dtype=np.float32)
        F[0, 4] = dt
        F[1, 5] = dt

        self.state = F @ self.state
        self.covariance = F @ self.covariance @ F.T + self.Q * dt

        # Measurement update
        cx, cy = bbox.center
        w, h = bbox.width, bbox.height
        z = np.array([cx, cy, w, h], dtype=np.float32)

        H = np.zeros((4, 6), dtype=np.float32)
        H[0, 0] = H[1, 1] = H[2, 2] = H[3, 3] = 1.0

        y = z - H @ self.state
        S = H @ self.covariance @ H.T + self.R
        K = self.covariance @ H.T @ np.linalg.inv(S)

        self.state = self.state + K @ y
        self.covariance = (np.eye(6, dtype=np.float32) - K @ H) @ self.covariance

    def predict(self, dt: float = 1.0 / 30.0) -> BBox:
        if not self.active:
            return BBox(0, 0, 0, 0)

        F = np.eye(6, dtype=np.float32)
        F[0, 4] = dt
        F[1, 5] = dt

        self.state = F @ self.state
        self.covariance = F @ self.covariance @ F.T + self.Q * dt

        cx, cy, w, h = self.state[:4]
        w = max(10.0, float(w))
        h = max(10.0, float(h))

        x1 = float(cx - w / 2.0)
        y1 = float(cy - h / 2.0)
        x2 = float(cx + w / 2.0)
        y2 = float(cy + h / 2.0)

        return BBox(x1, y1, x2, y2)