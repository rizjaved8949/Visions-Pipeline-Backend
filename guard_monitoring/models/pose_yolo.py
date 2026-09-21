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
        idx = int(np.argmax(confs))
        data = result.keypoints.data[idx].detach().cpu().numpy()
        if data.shape[1] == 2:
            data = np.concatenate([data, np.ones((data.shape[0], 1), dtype=data.dtype)], axis=1)
        data = data.astype(float)
        data[:, 0] += x1
        data[:, 1] += y1

        b = result.boxes.xyxy[idx].detach().cpu().numpy().astype(float)
        pose_box = (float(b[0] + x1), float(b[1] + y1), float(b[2] + x1), float(b[3] + y1))
        return PoseObservation(keypoints=data, bbox=pose_box, confidence=float(confs[idx]))
