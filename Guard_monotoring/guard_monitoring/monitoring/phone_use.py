from __future__ import annotations

import math

import numpy as np

from ..geometry import bbox_center, bbox_diagonal, expand_bbox, point_to_box_contains
from ..types import PhoneState


HEAD_IDS = [0, 3, 4]  # nose, left ear, right ear
WRIST_IDS = [9, 10]


class PhoneUseAnalyzer:
    def __init__(self, cfg: dict, keypoint_confidence: float = 0.25):
        self.expand_ratio = float(cfg.get("guard_box_expand_ratio", 0.10))
        self.hand_ratio = float(cfg.get("hand_distance_ratio", 0.22))
        self.head_ratio = float(cfg.get("head_distance_ratio", 0.20))
        self.kp_conf = float(keypoint_confidence)

    def analyze(self, phone_detections: list[dict], guard_box, keypoints: np.ndarray | None) -> PhoneState:
        if not phone_detections:
            return PhoneState()

        expanded = expand_bbox(guard_box, self.expand_ratio)
        candidates = [d for d in phone_detections if point_to_box_contains(bbox_center(d["bbox"]), expanded)]
        if not candidates:
            return PhoneState()

        scale = max(bbox_diagonal(guard_box), 1.0)
        head_points = self._valid_points(keypoints, HEAD_IDS)
        wrist_points = self._valid_points(keypoints, WRIST_IDS)

        ranked = []
        for d in candidates:
            c = bbox_center(d["bbox"])
            hand = self._nearest_ratio(c, wrist_points, scale)
            head = self._nearest_ratio(c, head_points, scale)
            association = min(x for x in [hand, head, 1.0] if x is not None)
            ranked.append((association, -float(d["confidence"]), d, hand, head))
        _, _, best, hand_ratio, head_ratio = min(ranked, key=lambda x: (x[0], x[1]))

        usage = "visible"
        if head_ratio is not None and head_ratio <= self.head_ratio:
            usage = "call"
        elif hand_ratio is not None and hand_ratio <= self.hand_ratio:
            usage = "screen_use"

        return PhoneState(
            detected=True,
            usage=usage,
            bbox=best["bbox"],
            confidence=float(best["confidence"]),
            nearest_hand_ratio=hand_ratio,
            nearest_head_ratio=head_ratio,
        )

    def _valid_points(self, keypoints, ids):
        if keypoints is None:
            return []
        pts = []
        for i in ids:
            if i < len(keypoints) and keypoints[i, 2] >= self.kp_conf:
                pts.append((float(keypoints[i, 0]), float(keypoints[i, 1])))
        return pts

    @staticmethod
    def _nearest_ratio(center, points, scale):
        if not points:
            return None
        return min(math.hypot(center[0] - p[0], center[1] - p[1]) for p in points) / scale
