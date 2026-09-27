from __future__ import annotations

import numpy as np

from ..geometry import bbox_diagonal, clamp_bbox, expand_bbox


class YOLOPhoneDetector:
    def __init__(self, weights: str, confidence: float, imgsz: int, target_labels: list[str]):
        from ultralytics import YOLO

        self.model = YOLO(weights)
        self.confidence = float(confidence)
        self.imgsz = int(imgsz)
        self.target_labels = {x.lower() for x in target_labels}

    def predict(self, frame_bgr, guard_box, expand_ratio: float = 0.10,
                extra_crops: list[dict] | None = None) -> list[dict]:
        """Run phone detection on the guard bbox plus any ``extra_crops``.

        ``extra_crops`` is a list of ``{"bbox": (x1,y1,x2,y2), "source": str,
        "confidence": float}`` entries. Each is run through the same model
        with the provided confidence override (typically lower, because
        wrist crops are visually tighter and harder to detect at). Results
        are merged and non-max-suppressed against the guard-box results so
        the same phone is never returned twice.
        """
        h, w = frame_bgr.shape[:2]

        combined: list[dict] = []
        # Primary crop
        combined.extend(self._detect_in_crop(
            frame_bgr, guard_box, expand_ratio,
            confidence=self.confidence, source="guard_box", w=w, h=h,
        ))

        # Extra crops (optional). We keep the primary results as-is and
        # only add extras that don't overlap heavily with them.
        for extra in (extra_crops or []):
            try:
                extra_bbox = extra["bbox"]
            except Exception:
                continue
            extra_conf = float(extra.get("confidence", self.confidence))
            source = str(extra.get("source", "extra_crop"))
            fresh = self._detect_in_crop(
                frame_bgr, extra_bbox, expand_ratio=0.0,
                confidence=extra_conf, source=source, w=w, h=h,
            )
            combined = self._merge_with_nms(combined, fresh, iou_threshold=0.5)

        return combined

    def _detect_in_crop(self, frame_bgr, crop_box, expand_ratio, *,
                        confidence, source, w, h) -> list[dict]:
        crop_box = expand_bbox(crop_box, expand_ratio, w, h) if expand_ratio else crop_box
        x1, y1, x2, y2 = [int(v) for v in clamp_bbox(crop_box, w, h)]
        crop = frame_bgr[y1:y2, x1:x2]
        if crop.size == 0:
            return []
        result = self.model.predict(crop, conf=confidence, imgsz=self.imgsz, verbose=False)[0]
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
                    "source": source,
                }
            )
        return found

    @staticmethod
    def _merge_with_nms(primary, new, iou_threshold=0.5):
        merged = list(primary)
        for candidate in new:
            keep = True
            for existing in merged:
                if YOLOPhoneDetector._iou(candidate["bbox"], existing["bbox"]) >= iou_threshold:
                    keep = False
                    break
            if keep:
                merged.append(candidate)
        return merged

    @staticmethod
    def _iou(box_a, box_b):
        ax1, ay1, ax2, ay2 = box_a
        bx1, by1, bx2, by2 = box_b
        ix1 = max(ax1, bx1); iy1 = max(ay1, by1)
        ix2 = min(ax2, bx2); iy2 = min(ay2, by2)
        iw = max(0.0, ix2 - ix1); ih = max(0.0, iy2 - iy1)
        inter = iw * ih
        area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
        area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
        union = area_a + area_b - inter
        return 0.0 if union <= 0 else inter / union
