from __future__ import annotations

import numpy as np


class CameraMotionEstimator:
    """Background-only affine motion with explicit uncertainty and cut detection.

    An absent fit is never proof of a fixed camera. Feature consensus, spatial
    spread and forward/backward consistency gate compensation. Scene changes
    invalidate person state; ambiguous frames only invalidate motion evidence.
    """

    def __init__(self, cfg):
        self.enabled = bool(cfg.get('enabled', True))
        self.width = int(cfg.get('width', 320))
        self.min_points = int(cfg.get('min_points', 12))
        self.min_inlier_ratio = float(cfg.get('min_inlier_ratio', .70))
        self.max_scale_change = float(cfg.get('max_scale_change', .15))
        self.max_fb_error = float(cfg.get('max_fb_error', 1.5))
        self.min_spatial_spread = float(cfg.get('min_spatial_spread', .12))
        self.scene_difference = float(cfg.get('scene_difference', .18))
        self.max_gap_seconds = float(cfg.get('max_gap_seconds', 2.0))
        self.reset()

    def reset(self):
        self.previous = self.previous_mask = self.previous_shape = self.previous_time = None

    def state(self, reason, **kwargs):
        result = dict(enabled=self.enabled, valid=False, reason=reason,
                      scene_change=False, affine=None, tracked_points=0,
                      inlier_ratio=None, background_difference=None,
                      scale=None, rotation_degrees=None)
        result.update(kwargs)
        return result

    def assess_fit(self, matrix, inliers, source, shape, difference):
        """Pure numerical gates, also used by deterministic regression tests."""
        result = self.state('fit_unavailable', tracked_points=len(source),
                            background_difference=float(difference))
        if matrix is None or inliers is None or not np.isfinite(matrix).all():
            result['scene_change'] = difference >= self.scene_difference
            return result
        keep = np.asarray(inliers).reshape(-1).astype(bool)
        result['inlier_ratio'] = float(keep.mean()) if len(keep) else 0.0
        if keep.sum() < self.min_points or result['inlier_ratio'] < self.min_inlier_ratio:
            result.update(reason='low_consensus', scene_change=difference >= self.scene_difference)
            return result
        extent = np.ptp(np.asarray(source)[keep], axis=0)
        if extent[0] < shape[1] * self.min_spatial_spread or extent[1] < shape[0] * self.min_spatial_spread:
            result['reason'] = 'localized_background'
            return result
        scale = float(np.hypot(matrix[0, 0], matrix[1, 0]))
        result.update(scale=scale, rotation_degrees=float(np.degrees(np.arctan2(matrix[1, 0], matrix[0, 0]))))
        if abs(scale - 1.0) > self.max_scale_change:
            result.update(reason='scale_discontinuity', scene_change=True)
            return result
        result.update(valid=True, reason='background_fit', affine=np.asarray(matrix).tolist())
        return result

    def update(self, frame, boxes, now=None):
        if not self.enabled:
            return self.state('disabled')
        import cv2
        h, w = frame.shape[:2]
        resize = min(1.0, self.width / max(w, 1))
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (max(1, round(w * resize)), max(1, round(h * resize))))
        mask = np.full(gray.shape, 255, dtype=np.uint8)
        for box in boxes:
            if not np.isfinite(box).all():
                continue
            x1, y1, x2, y2 = [int(round(v * resize)) for v in box]
            pad = max(2, int((x2 - x1) * .08))
            x1, x2 = np.clip([x1-pad, x2+pad], 0, mask.shape[1])
            y1, y2 = np.clip([y1-pad, y2+pad], 0, mask.shape[0])
            mask[y1:y2, x1:x2] = 0
        previous, old_mask, old_time = self.previous, self.previous_mask, self.previous_time
        same_shape = self.previous_shape == (h, w)
        self.previous, self.previous_mask, self.previous_shape, self.previous_time = gray, mask, (h, w), now
        if previous is None or not same_shape:
            return self.state('initial_frame', scene_change=previous is not None)
        if now is not None and old_time is not None and (now <= old_time or now-old_time > self.max_gap_seconds):
            return self.state('frame_gap')
        common = (mask > 0) & (old_mask > 0)
        if common.mean() < .02:
            return self.state('background_occluded')
        difference = float(np.abs(gray.astype(float)-previous.astype(float))[common].mean() / 255.0)
        points = cv2.goodFeaturesToTrack(previous, maxCorners=200, qualityLevel=.02,
                                        minDistance=7, mask=old_mask)
        if points is None or len(points) < self.min_points:
            return self.state('insufficient_features', background_difference=difference)
        moved, status, _ = cv2.calcOpticalFlowPyrLK(previous, gray, points, None,
                                                  winSize=(21, 21), maxLevel=3)
        if moved is None or status is None:
            return self.state('flow_unavailable', background_difference=difference)
        back, back_status, _ = cv2.calcOpticalFlowPyrLK(gray, previous, moved, None,
                                                      winSize=(21, 21), maxLevel=3)
        if back is None or back_status is None:
            return self.state('backward_flow_unavailable', background_difference=difference)
        source, target = points.reshape(-1, 2), moved.reshape(-1, 2)
        good = status.reshape(-1).astype(bool) & back_status.reshape(-1).astype(bool)
        good &= np.isfinite(target).all(axis=1)
        good &= np.linalg.norm(back.reshape(-1, 2)-source, axis=1) <= self.max_fb_error
        good &= (target[:, 0] >= 0) & (target[:, 0] < mask.shape[1])
        good &= (target[:, 1] >= 0) & (target[:, 1] < mask.shape[0])
        indexes = [i for i in np.flatnonzero(good) if mask[int(target[i, 1]), int(target[i, 0])] > 0]
        if len(indexes) < self.min_points:
            return self.state('insufficient_matches', tracked_points=len(indexes),
                              background_difference=difference,
                              scene_change=difference >= self.scene_difference)
        source, target = source[indexes], target[indexes]
        matrix, inliers = cv2.estimateAffinePartial2D(source, target, method=cv2.RANSAC,
                                                     ransacReprojThreshold=1.5)
        result = self.assess_fit(matrix, inliers, source, gray.shape, difference)
        if result['valid']:
            matrix[:, 2] /= resize
            result['affine'] = matrix.tolist()
        return result
