from __future__ import annotations

import numpy as np

from ..geometry import clamp_bbox, expand_bbox
from ..types import PoseObservation


class YOLOPoseEstimator:
    def __init__(self, weights: str, confidence: float = 0.30, imgsz: int = 640):
        from ultralytics import YOLO

        self.model = YOLO(weights)
        self.confidence = float(confidence)
        self.imgsz = int(imgsz)

    @staticmethod
    def select_guard_pose(boxes, confidences, guard_box):
        """Associate pose with the selected guard, not a nearby confident person."""
        boxes = np.asarray(boxes, dtype=float)
        guard = np.asarray(guard_box, dtype=float)
        overlap = np.maximum(0, np.minimum(boxes[:, 2:], guard[2:]) - np.maximum(boxes[:, :2], guard[:2]))
        intersection = overlap.prod(axis=1)
        areas = np.maximum(0, boxes[:, 2:] - boxes[:, :2]).prod(axis=1)
        guard_area = np.maximum(0, guard[2:] - guard[:2]).prod()
        iou = intersection / np.maximum(areas + guard_area - intersection, 1)
        valid = np.flatnonzero(iou >= .1)
        if len(valid) == 0:
            return None
        return int(max(valid, key=lambda i: (iou[i], confidences[i])))

    def predict(self, frame_bgr, guard_box, expand_ratio: float = 0.05) -> PoseObservation | None:
        h, w = frame_bgr.shape[:2]
        crop_box = expand_bbox(guard_box, expand_ratio, w, h)
        x1, y1, x2, y2 = [int(v) for v in clamp_bbox(crop_box, w, h)]
        crop = frame_bgr[y1:y2, x1:x2]
        if crop.size == 0:
            return None

        result = self.model.predict(crop, conf=self.confidence, imgsz=self.imgsz, verbose=False)[0]
        if result.keypoints is None or result.boxes is None or len(result.boxes) == 0:
            return None

        confs = result.boxes.conf.detach().cpu().numpy()
        full_boxes = result.boxes.xyxy.detach().cpu().numpy().astype(float)
        full_boxes += np.array([x1, y1, x1, y1])
        idx = self.select_guard_pose(full_boxes, confs, guard_box)
        if idx is None:
            return None
        data = result.keypoints.data[idx].detach().cpu().numpy()
        if data.shape[1] == 2:
            data = np.concatenate([data, np.ones((data.shape[0], 1), dtype=data.dtype)], axis=1)
        data = data.astype(float)
        data[:, 0] += x1
        data[:, 1] += y1

        pose_box = tuple(float(v) for v in full_boxes[idx])
        return PoseObservation(keypoints=data, bbox=pose_box, confidence=float(confs[idx]))
