"""Overlay rendering of zones, tracked objects and the breach banner onto frames."""

from __future__ import annotations

import time
from typing import Dict, List

import cv2
import numpy as np

from .detector import Detection
from .zones import ZoneSet


def draw_zones(img: np.ndarray, zones: ZoneSet, active: set | None = None) -> None:
    h, w = img.shape[:2]
    overlay = img.copy()
    for z in zones.zones:
        pts = z.to_pixels(w, h)
        hot = active and z.name in active
        color = (0, 0, 255) if hot else z.color
        cv2.fillPoly(overlay, [pts], color)
        cv2.polylines(img, [pts], True, color, 3 if hot else 2)
        cx, cy = pts.reshape(-1, 2).mean(axis=0).astype(int)
        cv2.putText(img, z.name, (cx - 30, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.addWeighted(overlay, 0.35 if active else 0.2, img, 0.8 if not active else 0.65, 0, dst=img)


def draw_detections(img: np.ndarray, dets: List[Detection], inside_map: Dict[int, List[str]],
                    anchor: str = "center") -> None:
    for d in dets:
        x1, y1, x2, y2 = map(int, (d.x1, d.y1, d.x2, d.y2))
        breaching = d.track_id in inside_map
        color = (0, 0, 255) if breaching else (0, 200, 0)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        px, py = (d.bottom_center if anchor == "bottom" else d.center)
        cv2.circle(img, (int(px), int(py)), 5, color, -1)
        cv2.circle(img, (int(px), int(py)), 8, (255, 255, 255), 1)
        tag = f"{d.cls_name} #{d.track_id if d.track_id >= 0 else '?'} {d.conf:.2f}"
        if breaching:
            tag = "BREACH " + tag
        (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(img, (x1, y1 - th - 8), (x1 + tw + 6, y1), color, -1)
        cv2.putText(img, tag, (x1 + 3, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)


class Banner:
    """Flashing red banner shown for a few seconds after every alert."""

    def __init__(self, duration: float = 2.5):
        self.duration = duration
        self.until = 0.0
        self.text = ""

    def trigger(self, text: str) -> None:
        self.text = text
        self.until = time.time() + self.duration

    def draw(self, img: np.ndarray) -> None:
        if time.time() > self.until:
            return
        h, w = img.shape[:2]
        if int(time.time() * 4) % 2 == 0:
            cv2.rectangle(img, (0, 0), (w, 50), (0, 0, 255), -1)
            cv2.putText(img, self.text, (12, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
        cv2.rectangle(img, (2, 2), (w - 3, h - 3), (0, 0, 255), 6)


def draw_hud(img: np.ndarray, fps: float, n_tracks: int, n_inside: int, total: int) -> None:
    h, w = img.shape[:2]
    text = f"FPS {fps:5.1f}   tracked {n_tracks}   inside {n_inside}   breaches {total}"
    y = h - 14
    cv2.putText(img, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3)
    cv2.putText(img, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
