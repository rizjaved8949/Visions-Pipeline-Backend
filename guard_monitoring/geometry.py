from __future__ import annotations

import math
from typing import Iterable

import numpy as np

from .types import BBox, Point


def clamp_bbox(box: BBox, width: int, height: int) -> BBox:
    x1, y1, x2, y2 = box
    x1 = max(0.0, min(float(width - 1), x1))
    y1 = max(0.0, min(float(height - 1), y1))
    x2 = max(x1 + 1.0, min(float(width), x2))
    y2 = max(y1 + 1.0, min(float(height), y2))
    return x1, y1, x2, y2


def expand_bbox(box: BBox, ratio: float, width: int | None = None, height: int | None = None) -> BBox:
    x1, y1, x2, y2 = box
    w = x2 - x1
    h = y2 - y1
    expanded = (x1 - w * ratio, y1 - h * ratio, x2 + w * ratio, y2 + h * ratio)
    if width is not None and height is not None:
        return clamp_bbox(expanded, width, height)
    return expanded


def bbox_center(box: BBox) -> Point:
    x1, y1, x2, y2 = box
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0


def bbox_footpoint(box: BBox) -> Point:
    x1, _, x2, y2 = box
    return (x1 + x2) / 2.0, y2


def bbox_diagonal(box: BBox) -> float:
    x1, y1, x2, y2 = box
    return math.hypot(x2 - x1, y2 - y1)


def point_in_polygon(point: Point, polygon: Iterable[Point]) -> bool:
    x, y = point
    pts = list(polygon)
    inside = False
    j = len(pts) - 1
    for i in range(len(pts)):
        xi, yi = pts[i]
        xj, yj = pts[j]
        # Boundary points belong to the duty zone. A full-frame box often has
        # its footpoint exactly on the bottom edge; it must remain selectable.
        cross = (x - xi) * (yj - yi) - (y - yi) * (xj - xi)
        tolerance = 1e-7 * max(1.0, abs(xj - xi), abs(yj - yi))
        if (abs(cross) <= tolerance and min(xi, xj) - tolerance <= x <= max(xi, xj) + tolerance
                and min(yi, yj) - tolerance <= y <= max(yi, yj) + tolerance):
            return True
        intersects = ((yi > y) != (yj > y)) and (
            x < (xj - xi) * (y - yi) / ((yj - yi) + 1e-12) + xi
        )
        if intersects:
            inside = not inside
        j = i
    return inside


def normalized_polygon_to_pixels(points: list[list[float]], width: int, height: int) -> list[Point]:
    return [(float(x) * width, float(y) * height) for x, y in points]


def euclidean(a: Point, b: Point) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def angle_abc(a: Point, b: Point, c: Point) -> float:
    """Angle ABC in degrees."""
    ba = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    bc = np.asarray(c, dtype=float) - np.asarray(b, dtype=float)
    denom = float(np.linalg.norm(ba) * np.linalg.norm(bc))
    if denom <= 1e-9:
        return float("nan")
    cosine = float(np.clip(np.dot(ba, bc) / denom, -1.0, 1.0))
    return math.degrees(math.acos(cosine))


def point_to_box_contains(point: Point, box: BBox) -> bool:
    x, y = point
    x1, y1, x2, y2 = box
    return x1 <= x <= x2 and y1 <= y <= y2
