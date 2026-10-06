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
        self.head_ear_drop_ratio = float(cfg.get("head_ear_drop_ratio", 0.35))
        # Additive: sitting inferred from bbox shape when knees/ankles are
        # occluded (typical desk/counter scene). See config.py for the
        # sitting/standing aspect-ratio thresholds.
        self.aspect_fallback_enabled = bool(cfg.get("aspect_fallback_enabled", False))
        self.sitting_aspect_max = float(cfg.get("sitting_aspect_ratio_max", 1.55))
        self.standing_aspect_min = float(cfg.get("standing_aspect_ratio_min", 2.10))

    def analyze(self, keypoints: np.ndarray | None,
                bbox: tuple[float, float, float, float] | None = None) -> PostureState:
        """Analyze posture. Optional ``bbox`` unlocks the occlusion-aware
        aspect-ratio fallback (see class docstring). Callers that pass only
        keypoints (existing tests, existing pipeline call sites) continue
        to work unchanged; ``bbox`` is a keyword-friendly positional add.
        """
        aspect_ratio = None
        if bbox is not None:
            try:
                x1, y1, x2, y2 = bbox
                w = max(1.0, float(x2 - x1))
                h = max(1.0, float(y2 - y1))
                aspect_ratio = h / w
            except Exception:
                aspect_ratio = None
        if keypoints is None or len(keypoints) < 17:
            return self._aspect_only_posture(aspect_ratio)

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
        thighs = []
        for hip, knee, ankle, knee_angle in (
            (LEFT_HIP, LEFT_KNEE, LEFT_ANKLE, left_angle),
            (RIGHT_HIP, RIGHT_KNEE, RIGHT_ANKLE, right_angle),
        ):
            if all(valid(i) for i in (hip, knee)):
                thigh = keypoints[knee, :2] - keypoints[hip, :2]
                length = float(np.linalg.norm(thigh))
                if length > 5:
                    thighs.append((float(thigh[1] / length), knee_angle))
        if thighs:
            # Walking bends a knee too. Seated posture additionally needs a
            # near-horizontal thigh; a bent walking leg must not become Sitting.
            if len(thighs) == 2 and all(
                    abs(vertical) < .6 and (angle is None or angle <= self.sitting_knee_max)
                    for vertical, angle in thighs):
                posture = "sitting"
            elif all(vertical >= .75 for vertical, _ in thighs):
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

        # A close-up often hides both hips. Independently estimate head drop
        # from high-confidence nose/eye/ear landmarks; leave body labels alone.
        if not head_down_known and keypoints[NOSE, 2] >= max(self.kp_conf, 0.7):
            drops = []
            for eye_idx, ear_idx in ((1, 3), (2, 4)):
                if min(keypoints[eye_idx, 2], keypoints[ear_idx, 2]) >= max(self.kp_conf, 0.7):
                    distance = float(np.linalg.norm(keypoints[NOSE, :2] - keypoints[ear_idx, :2]))
                    if distance > 5:
                        drops.append((keypoints[NOSE, 1] - keypoints[ear_idx, 1]) / distance)
            if drops:
                head_down_known = True
                # If both ears are visible, require agreement to avoid treating
                # a sideways roll alone as downward head pitch.
                head_down = bool(min(drops) >= self.head_ear_drop_ratio)

        posture_source = "keypoints" if posture != "unknown" else "unknown"
        # Occlusion fallback: use bbox aspect only when the knee-angle
        # signal is inconclusive AND the caller has supplied a bbox.
        if posture == "unknown" and self.aspect_fallback_enabled and aspect_ratio is not None:
            inferred = self._posture_from_aspect(aspect_ratio)
            if inferred is not None:
                posture = inferred
                posture_source = "occlusion_inferred"

        return PostureState(
            posture=posture,
            head_down=head_down,
            torso_lean_deg=torso_lean,
            left_knee_angle_deg=left_angle,
            right_knee_angle_deg=right_angle,
            head_down_known=head_down_known,
            posture_source=posture_source,
            bbox_aspect_ratio=aspect_ratio,
        )

    def _aspect_only_posture(self, aspect_ratio):
        """Fallback path when we have no keypoints at all."""
        if not self.aspect_fallback_enabled or aspect_ratio is None:
            return PostureState(bbox_aspect_ratio=aspect_ratio)
        inferred = self._posture_from_aspect(aspect_ratio)
        if inferred is None:
            return PostureState(bbox_aspect_ratio=aspect_ratio, posture_source="unknown")
        return PostureState(
            posture=inferred,
            bbox_aspect_ratio=aspect_ratio,
            posture_source="bbox_shape",
        )

    def _posture_from_aspect(self, aspect_ratio):
        if aspect_ratio is None:
            return None
        if aspect_ratio <= self.sitting_aspect_max:
            return "sitting"
        if aspect_ratio >= self.standing_aspect_min:
            return "standing"
        return None
