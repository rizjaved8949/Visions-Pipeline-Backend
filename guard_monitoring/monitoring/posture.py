from __future__ import annotations

import math

import numpy as np

from ..geometry import angle_abc
from ..types import PostureState


# COCO-17 indices used by YOLO26 pose.
NOSE = 0
LEFT_SHOULDER, RIGHT_SHOULDER = 5, 6
LEFT_HIP, RIGHT_HIP = 11, 12
LEFT_KNEE, RIGHT_KNEE = 13, 14
LEFT_ANKLE, RIGHT_ANKLE = 15, 16


class PostureAnalyzer:
    def __init__(self, cfg: dict):
        self.kp_conf = float(cfg.get("keypoint_confidence", 0.25))
        self.sitting_knee_max = float(cfg.get("sitting_knee_angle_max_deg", 140.0))
        self.standing_knee_min = float(cfg.get("standing_knee_angle_min_deg", 155.0))
        self.torso_lean_threshold = float(cfg.get("torso_lean_deg", 28.0))
        self.head_down_ratio = float(cfg.get("head_down_ratio", 0.32))

    def analyze(self, keypoints: np.ndarray | None) -> PostureState:
        if keypoints is None or len(keypoints) < 17:
            return PostureState()

        def valid(i):
            return keypoints[i, 2] >= self.kp_conf

        def point(i):
            return float(keypoints[i, 0]), float(keypoints[i, 1])

        knee_angles = []
        left_angle = None
        right_angle = None
        if all(valid(i) for i in [LEFT_HIP, LEFT_KNEE, LEFT_ANKLE]):
            left_angle = angle_abc(point(LEFT_HIP), point(LEFT_KNEE), point(LEFT_ANKLE))
            if not math.isnan(left_angle):
                knee_angles.append(left_angle)
        if all(valid(i) for i in [RIGHT_HIP, RIGHT_KNEE, RIGHT_ANKLE]):
            right_angle = angle_abc(point(RIGHT_HIP), point(RIGHT_KNEE), point(RIGHT_ANKLE))
            if not math.isnan(right_angle):
                knee_angles.append(right_angle)

        posture = "unknown"
        if knee_angles:
            if min(knee_angles) <= self.sitting_knee_max:
                posture = "sitting"
            elif min(knee_angles) >= self.standing_knee_min:
                posture = "standing"

        torso_lean = None
        shoulder_center = None
        hip_center = None
        if all(valid(i) for i in [LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_HIP, RIGHT_HIP]):
            shoulder_center = np.mean([point(LEFT_SHOULDER), point(RIGHT_SHOULDER)], axis=0)
            hip_center = np.mean([point(LEFT_HIP), point(RIGHT_HIP)], axis=0)
            dx = float(hip_center[0] - shoulder_center[0])
            dy = float(hip_center[1] - shoulder_center[1])
            torso_lean = math.degrees(math.atan2(abs(dx), max(abs(dy), 1e-6)))

        head_down = False
        head_down_known = False
        if shoulder_center is not None and hip_center is not None and valid(NOSE):
            torso_len = float(np.linalg.norm(np.asarray(hip_center) - np.asarray(shoulder_center)))
            if torso_len > 1.0:
                head_down_known = True
                nose_y = point(NOSE)[1]
                head_clearance = float(shoulder_center[1] - nose_y)
                ratio = head_clearance / torso_len
                head_down = ratio < self.head_down_ratio

        return PostureState(
            posture=posture,
            head_down=head_down,
            torso_lean_deg=torso_lean,
            left_knee_angle_deg=left_angle,
            right_knee_angle_deg=right_angle,
            head_down_known=head_down_known,
        )
