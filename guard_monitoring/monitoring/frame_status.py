"""Current-frame status composition, independent from duration-based alerts."""
from __future__ import annotations

import numpy as np

from ..types import MovementState


PRIMARY_STATUSES = ("Sleeping", "Using Mobile", "Moving", "Stationary", "Sitting", "Standing")


def pose_matches_guard(pose, guard_box):
    if pose is None:
        return False
    box, guard = np.asarray(pose.bbox), np.asarray(guard_box)
    intersection = np.maximum(0, np.minimum(box[2:], guard[2:]) - np.maximum(box[:2], guard[:2])).prod()
    union = np.maximum(0, box[2:] - box[:2]).prod() + np.maximum(0, guard[2:] - guard[:2]).prod() - intersection
    return bool(intersection / max(union, 1) >= .35)


class FrameMovementAnalyzer:
    """Measure consecutive observed frames, including movement while seated.

    Background-compensated optical flow measures body displacement. Pose changes
    relative to the shoulders can establish articulation even when camera motion
    is unavailable; they cannot establish that the whole person is stationary.
    """

    def __init__(self, cfg):
        self.min_pixels = float(cfg.get("frame_min_motion_pixels", 2.0))
        self.speed_threshold = float(cfg.get("frame_speed_ratio", 0.06))
        self.joint_ratio = float(cfg.get("frame_joint_change_ratio", 0.012))
        self.max_gap = float(cfg.get("max_gap_seconds", 2.0))
        self.reset()

    def reset(self):
        self.previous = self.previous_box = self.previous_pose = None
        self.previous_time = self.track_id = None

    @staticmethod
    def _relative_pose(keypoints):
        if keypoints is None or len(keypoints) < 17:
            return None
        if min(keypoints[5, 2], keypoints[6, 2]) < .55:
            return None
        left, right = keypoints[5, :2], keypoints[6, :2]
        width = float(np.linalg.norm(right - left))
        if width < 12:
            return None
        x_axis = (right - left) / width
        y_axis = np.array([-x_axis[1], x_axis[0]])
        positions = (keypoints[:, :2] - (left + right) / 2) @ np.column_stack((x_axis, y_axis)) / width
        return positions, keypoints[:, 2] >= .55, width, keypoints[:, :2].copy()

    def update(self, frame, track_id, bbox, keypoints, now, camera_state):
        import cv2

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Bound optical-flow cost independently of the detector resolution.
        resize = min(1.0, 640 / gray.shape[1])
        gray = cv2.resize(gray, None, fx=resize, fy=resize)
        pose = self._relative_pose(keypoints)
        old, old_box, old_pose, old_time = self.previous, self.previous_box, self.previous_pose, self.previous_time
        continuous = (self.track_id == track_id and old_time is not None
                      and 0 < now - old_time <= self.max_gap and old.shape == gray.shape
                      and not camera_state.get("scene_change"))
        self.previous, self.previous_box, self.previous_pose = gray, tuple(bbox), pose
        self.previous_time, self.track_id = now, track_id
        if not continuous:
            return MovementState(reason="first_frame_observation")
        dt = now - old_time
        articulated = False
        if old_pose is not None and pose is not None:
            # Exclude shoulder anchors, and require two independently measured
            # points so one bad wrist/knee cannot declare the whole guard moving.
            indices = [0, 1, 2, 3, 4, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
            keep = old_pose[1][indices] & pose[1][indices]
            changes = np.linalg.norm(pose[0][indices] - old_pose[0][indices], axis=1)
            threshold = max(self.joint_ratio, self.min_pixels / min(old_pose[2], pose[2]))
            articulated = bool(np.sum(keep & (changes >= threshold)) >= 2)

        camera_known = bool(camera_state.get("valid"))
        measured = False
        translating = False
        pose_measured = pose_translating = False
        diagonal = max(float(np.hypot(bbox[2] - bbox[0], bbox[3] - bbox[1])), 1)
        if camera_known and old_pose is not None and pose is not None:
            anchors = [0, 3, 4, 5, 6, 11, 12]
            keep = old_pose[1][anchors] & pose[1][anchors]
            if keep.sum() >= 3:
                affine = np.asarray(camera_state["affine"], dtype=float)
                aligned = old_pose[3][anchors] @ affine[:, :2].T + affine[:, 2]
                displacement = float(np.median(np.linalg.norm(pose[3][anchors][keep] - aligned[keep], axis=1)))
                pose_measured = True
                pose_translating = (displacement >= self.min_pixels
                                    and displacement / diagonal / dt >= self.speed_threshold)
        if camera_known:
            mask = np.zeros(old.shape, dtype=np.uint8)
            x1, y1, x2, y2 = np.asarray(old_box) * resize
            x1, y1 = max(0, int(x1)), max(0, int(y1))
            x2, y2 = min(old.shape[1], int(x2)), min(old.shape[0], int(y2))
            # Avoid the outer detector edge and annotation area.
            pad = max(3, int((x2 - x1) * .08))
            mask[min(y2, y1 + 35):max(y1, y2 - pad), min(x2, x1 + pad):max(x1, x2 - pad)] = 255
            points = cv2.goodFeaturesToTrack(old, maxCorners=100, qualityLevel=.03, minDistance=7, mask=mask)
            if points is not None and len(points) >= 6:
                target, status, _ = cv2.calcOpticalFlowPyrLK(old, gray, points, None)
                if target is not None:
                    back, reverse, _ = cv2.calcOpticalFlowPyrLK(gray, old, target, None)
                    if back is not None and reverse is not None:
                        source, target = points.reshape(-1, 2), target.reshape(-1, 2)
                        good = status.reshape(-1).astype(bool) & reverse.reshape(-1).astype(bool)
                        good &= np.isfinite(target).all(axis=1)
                        good &= np.linalg.norm(back.reshape(-1, 2) - source, axis=1) <= 1.5
                        good &= ((target[:, 0] >= bbox[0] * resize) & (target[:, 0] <= bbox[2] * resize)
                                 & (target[:, 1] >= bbox[1] * resize) & (target[:, 1] <= bbox[3] * resize))
                        if good.sum() >= 6:
                            affine = np.asarray(camera_state["affine"], dtype=float).copy()
                            affine[:, 2] *= resize
                            aligned = source @ affine[:, :2].T + affine[:, 2]
                            residual = float(np.median(np.linalg.norm(target[good] - aligned[good], axis=1))) / resize
                            translating = residual >= self.min_pixels and residual / diagonal / dt >= self.speed_threshold
                            measured = True
        if not measured:
            measured, translating = pose_measured, pose_translating
        reliable = measured or articulated
        return MovementState(stationary=not (translating or articulated), reliable=reliable,
                             reason="frame_motion" if reliable else "frame_motion_unknown",
                             motion_source="body_and_background" if measured else "articulated_pose",
                             history_seconds=dt, anchor_visible=reliable)


def compose_frame_statuses(phone, sleep, eye, posture, movement, *, seen_now=True):
    """All compatible observed labels; timers never delay or hold these labels."""
    if not seen_now:
        return ["Status Unknown"]
    statuses = []
    eyes_known = eye.quality_ok and eye.eyes_closed is not None
    head_support = ((posture.head_down_known and posture.head_down)
                    or (eye.head_roll_degrees is not None and abs(eye.head_roll_degrees) >= 15)
                    or (eye.head_pitch_degrees is not None
                        and (eye.head_pitch_degrees <= -8 or eye.head_pitch_degrees >= 20))
                    or (posture.torso_lean_deg is not None and posture.torso_lean_deg >= 28))
    # Sleep/phone/body activity are independent dimensions. A held phone does
    # not veto fresh closed eyes and supporting sleep posture.
    sleeping = bool((eyes_known and eye.eyes_closed and head_support)
                    or (sleep.candidate and sleep.evidence_quality == "high"
                        and (not eyes_known or eye.eyes_closed)))
    if sleeping:
        statuses.append("Sleeping")
    elif ((eyes_known and eye.eyes_closed)
          or (not eyes_known and head_support and phone.usage not in {"call", "screen_use"})):
        statuses.append("Possible Sleep")
    if phone.detected:
        statuses.append("Using Mobile" if phone.usage in {"call", "screen_use"} and not sleeping else "Phone Visible")
    if movement.reliable:
        statuses.append("Stationary" if movement.stationary else "Moving")
    # A short/wide detector rectangle is not evidence of a seated person.
    # Sitting and Standing require body keypoints; motion is never replaced.
    if posture.posture_source == "keypoints":
        if posture.posture == "sitting":
            statuses.append("Sitting")
        elif posture.posture == "standing":
            statuses.append("Standing")
    if not statuses:
        statuses.append("Status Unknown")
    return statuses
