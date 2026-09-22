"""
Restricted-zone geometry.

Zones are normalised (x, y in 0..1) so the same polygon stays correct
regardless of the resolution the source is processed at. There is no
persistence here on purpose: in this module zones are drawn by the
frontend fresh for every session and only ever live in memory (see
monitor.set_zones) - they are never written to disk.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple

import cv2
import numpy as np

Point = Tuple[float, float]

PALETTE = [
    (0, 0, 255), (0, 165, 255), (0, 255, 255), (255, 0, 255),
    (255, 0, 0), (0, 255, 0), (128, 0, 255), (255, 128, 0),
]


@dataclass
class Zone:
    name: str
    points: List[Point]  # normalised (0..1)
    color: Tuple[int, int, int] = (0, 0, 255)  # BGR

    def to_pixels(self, w: int, h: int) -> np.ndarray:
        pts = np.array([[int(x * w), int(y * h)] for x, y in self.points], dtype=np.int32)
        return pts.reshape((-1, 1, 2))

    def contains(self, px: float, py: float, w: int, h: int) -> bool:
        """True if pixel point (px, py) is inside (or on the edge of) the zone."""
        if len(self.points) < 3:
            return False
        contour = self.to_pixels(w, h).astype(np.float32)
        return cv2.pointPolygonTest(contour, (float(px), float(py)), False) >= 0


@dataclass
class ZoneSet:
    zones: List[Zone] = field(default_factory=list)

    def zones_containing(self, px: float, py: float, w: int, h: int) -> List[Zone]:
        return [z for z in self.zones if z.contains(px, py, w, h)]

    def __len__(self) -> int:
        return len(self.zones)
