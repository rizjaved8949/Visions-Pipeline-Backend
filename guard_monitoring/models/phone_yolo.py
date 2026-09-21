from __future__ import annotations

import numpy as np

from ..geometry import clamp_bbox, expand_bbox


class YOLOPhoneDetector:
    def __init__(self, weights: str, confidence: float, imgsz: int, target_labels: list[str]):
        from ultralytics import YOLO

        self.model = YOLO(weights)
        self.confidence = float(confidence)
        self.imgsz = int(imgsz)
        self.target_labels = {x.lower() for x in target_labels}

    def predict(self, frame_bgr, guard_box, expand_ratio: float = 0.10) -> list[dict]:
        h, w = frame_bgr.shape[:2]
        crop_box = expand_bbox(guard_box, expand_ratio, w, h)
        x1, y1, x2, y2 = [int(v) for v in clamp_bbox(crop_box, w, h)]
        crop = frame_bgr[y1:y2, x1:x2]
        if crop.size == 0:
            return []

        result = self.model.predict(crop, conf=self.confidence, imgsz=self.imgsz, verbose=False)[0]
        if result.boxes is None or len(result.boxes) == 0:
            return []

        boxes = result.boxes.xyxy.detach().cpu().numpy().astype(float)
        confs = result.boxes.conf.detach().cpu().numpy().astype(float)
        classes = result.boxes.cls.detach().cpu().numpy().astype(int)
        names = result.names
        found = []
        for b, conf, cls_id in zip(boxes, confs, classes):
            name = str(names[int(cls_id)]).lower()
            if name not in self.target_labels:
                continue
            found.append(
                {
                    "bbox": (float(b[0] + x1), float(b[1] + y1), float(b[2] + x1), float(b[3] + y1)),
                    "confidence": float(conf),
                    "label": name,
                }
            )
        return found
