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
        smoothing_seconds: float = 0.25,
        radius_quantile: float = 0.90,
        max_gap_seconds: float = 2.0,
        border_margin_ratio: float = 0.015,
        max_box_scale_change: float = 0.25,
    ):
        self.history_seconds = float(history_seconds)
        self.stationary_radius_ratio = float(stationary_radius_ratio)
        self.minimum_history_seconds = float(minimum_history_seconds)
        self.patrol_zones = patrol_zones
        self.history = deque()
        self.visited: set[str] = set()
        self.track_id = None
        self.smoothing_seconds = float(smoothing_seconds)
        self.radius_quantile = float(radius_quantile)
        self.max_gap_seconds = float(max_gap_seconds)
        self.raw_history = deque()
        self.border_margin_ratio = float(border_margin_ratio)
        self.max_box_scale_change = float(max_box_scale_change)
        self.last_diagonal = None

    def reset(self, track_id: int | None = None):
        self.history.clear()
        self.raw_history.clear()
        self.visited.clear()
        self.track_id = track_id
        self.last_diagonal = None

    def invalidate(self):
        """Discard unaligned motion history without erasing observed patrol visits."""
        self.history.clear()
        self.raw_history.clear()
        self.last_diagonal = None

    def compensate(self, matrix):
        """Align stored positions to this frame before measuring body movement."""
        matrix = np.asarray(matrix, dtype=float)
        if matrix.shape != (2, 3) or not np.isfinite(matrix).all():
            return
        scale = float(np.hypot(matrix[0, 0], matrix[1, 0]))
        def point(p):
            return tuple(matrix[:, :2] @ np.asarray(p) + matrix[:, 2])
        self.history = deque((t, point(p), d * scale) for t, p, d in self.history)
        self.raw_history = deque((t, point(p)) for t, p in self.raw_history)
        if self.last_diagonal is not None:
            self.last_diagonal *= scale

    def update(self, track_id: int, bbox, now: float, *, frame_shape=None,
               camera_state=None, assume_static_camera=False) -> MovementState:
        if self.track_id != track_id:
            self.reset(track_id)
        if self.history and now - self.history[-1][0] > self.max_gap_seconds:
            self.invalidate()

        def uncertain(reason, anchor=False):
            self.invalidate()
            return MovementState(reason=reason, anchor_visible=anchor,
                                 visited_patrol_zones=tuple(sorted(self.visited)),
                                 patrol_coverage_ratio=(len(self.visited) / len(self.patrol_zones)
                                                        if self.patrol_zones else None))

        source = "bbox_history"  # Compatibility for callers without frame context.
        if frame_shape is not None:
            h, w = frame_shape[:2]
            x1, y1, x2, y2 = bbox
            margin = min(h, w) * self.border_margin_ratio
            if x1 <= margin or y1 <= margin or x2 >= w - margin or y2 >= h - margin:
                return uncertain("guard_box_truncated")
        if camera_state is not None:
            if camera_state.get("scene_change"):
                return uncertain("scene_change")
            if camera_state.get("valid"):
                source = "camera_compensated"
            elif assume_static_camera:
                source = "configured_static_camera"
            else:
                return uncertain("camera_motion_unresolved", True)

        point = bbox_footpoint(bbox)
        diag = max(bbox_diagonal(bbox), 1.0)
        if (self.last_diagonal is not None
                and abs(diag / self.last_diagonal - 1.0) > self.max_box_scale_change):
            return uncertain("box_scale_discontinuity")
        self.last_diagonal = diag
        self.raw_history.append((now, point))
        while self.raw_history and now - self.raw_history[0][0] > self.smoothing_seconds:
            self.raw_history.popleft()
        point = tuple(np.median([entry[1] for entry in self.raw_history], axis=0))
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
            radius = float(np.quantile(np.linalg.norm(pts - center, axis=1), self.radius_quantile))
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
            reliable=duration >= self.minimum_history_seconds and len(self.history) >= 2,
            reason="measured" if radius_ratio is not None else "insufficient_history",
            motion_source=source,
            anchor_visible=True,
        )
