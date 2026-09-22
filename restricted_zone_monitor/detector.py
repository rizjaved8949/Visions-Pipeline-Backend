"""
Object detection + multi-object tracking.

Uses Ultralytics YOLO (v8/11) with the built-in ByteTrack / BoT-SORT tracker.
Every detection returned carries a persistent `track_id`, which is what lets
us raise exactly ONE alert per real-world object per entry.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np
from ultralytics import YOLO

from . import config


@dataclass
class Detection:
    track_id: int          # -1 if the tracker has not assigned an id yet
    cls_id: int
    cls_name: str
    conf: float
    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def center(self):
        return (self.x1 + self.x2) / 2.0, (self.y1 + self.y2) / 2.0

    @property
    def bottom_center(self):
        """Feet position - better than the box centre for people on the ground
        when the camera is mounted high and looks down."""
        return (self.x1 + self.x2) / 2.0, self.y2


class Detector:
    def __init__(
        self,
        weights: Optional[str] = None,
        conf: float = config.DETECTION_CONF,
        iou: float = config.DETECTION_IOU,
        imgsz: int = config.DETECTION_IMGSZ,
        classes: Optional[List[int]] = config.DETECTION_CLASSES,
        tracker: str = config.TRACKER,
        device: Optional[str] = None,
        half: Optional[bool] = None,
    ):
        weights = weights or config.PERSON_MODEL_PATH
        device = device or config.DEVICE
        self.device = device
        self.half = (config.HALF if half is None else half) and device.startswith("cuda")
        self.conf, self.iou, self.imgsz = conf, iou, imgsz
        self.classes = classes if classes else None
        self.tracker = tracker

        print(f"[Detector] Loading {weights} on {device} (half={self.half})")
        self.model = YOLO(weights)
        self.names = self.model.names
        # warm-up so the first real frame is not slow
        self.model.predict(np.zeros((imgsz, imgsz, 3), np.uint8), device=device,
                           half=self.half, verbose=False)

    def __call__(self, frame: np.ndarray) -> List[Detection]:
        results = self.model.track(
            frame,
            persist=True,
            conf=self.conf,
            iou=self.iou,
            imgsz=self.imgsz,
            classes=self.classes,
            tracker=self.tracker,
            device=self.device,
            half=self.half,
            verbose=False,
        )
        dets: List[Detection] = []
        if not results:
            return dets
        r = results[0]
        if r.boxes is None or len(r.boxes) == 0:
            return dets

        xyxy = r.boxes.xyxy.cpu().numpy()
        confs = r.boxes.conf.cpu().numpy()
        clss = r.boxes.cls.cpu().numpy().astype(int)
        ids = r.boxes.id.cpu().numpy().astype(int) if r.boxes.id is not None else np.full(len(xyxy), -1)

        for (x1, y1, x2, y2), c, k, tid in zip(xyxy, confs, clss, ids):
            dets.append(Detection(int(tid), int(k), str(self.names[int(k)]), float(c),
                                  float(x1), float(y1), float(x2), float(y2)))
        return dets

    def reset_tracker(self) -> None:
        """Call when switching to a new video so ids start again."""
        if hasattr(self.model, "predictor") and self.model.predictor is not None:
            self.model.predictor.trackers = []
