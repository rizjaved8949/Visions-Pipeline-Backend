from __future__ import annotations

from collections import deque

import numpy as np

from ..geometry import bbox_diagonal, bbox_footpoint, point_in_polygon
from ..types import MovementState


class MovementMonitor:
    def __init__(
        self,
        history_seconds: float,
        stationary_radius_ratio: float,
        minimum_history_seconds: float,
        patrol_zones: list[tuple[str, list[tuple[float, float]]]],
    ):
        self.history_seconds = float(history_seconds)
        self.stationary_radius_ratio = float(stationary_radius_ratio)
        self.minimum_history_seconds = float(minimum_history_seconds)
        self.patrol_zones = patrol_zones
        self.history = deque()
        self.visited: set[str] = set()
        self.track_id = None

    def reset(self, track_id: int | None = None):
        self.history.clear()
        self.visited.clear()
        self.track_id = track_id

    def update(self, track_id: int, bbox, now: float) -> MovementState:
        if self.track_id != track_id:
            self.reset(track_id)

        point = bbox_footpoint(bbox)
        diag = max(bbox_diagonal(bbox), 1.0)
        self.history.append((now, point, diag))
        while self.history and now - self.history[0][0] > self.history_seconds:
            self.history.popleft()

        for name, polygon in self.patrol_zones:
            if point_in_polygon(point, polygon):
                self.visited.add(name)

        duration = self.history[-1][0] - self.history[0][0] if len(self.history) >= 2 else 0.0
        radius_ratio = None
        stationary = False
        if duration >= self.minimum_history_seconds and len(self.history) >= 2:
            pts = np.asarray([x[1] for x in self.history], dtype=float)
            center = np.median(pts, axis=0)
            radius = float(np.max(np.linalg.norm(pts - center, axis=1)))
            scale = float(np.median([x[2] for x in self.history]))
            radius_ratio = radius / max(scale, 1.0)
            stationary = radius_ratio <= self.stationary_radius_ratio

        total_zones = len(self.patrol_zones)
        coverage = None if total_zones == 0 else len(self.visited) / total_zones
        return MovementState(
            stationary=stationary,
            history_seconds=duration,
            radius_ratio=radius_ratio,
            visited_patrol_zones=tuple(sorted(self.visited)),
            patrol_coverage_ratio=coverage,
        )
