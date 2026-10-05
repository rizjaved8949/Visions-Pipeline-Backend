"""
Stage 1 - Vehicle detection + multi-object tracking.

Uses an Ultralytics YOLO model pretrained on COCO. COCO already contains
bicycle / car / motorcycle / bus / truck, so nothing has to be trained.
`model.track()` runs detection and ByteTrack/BoT-SORT in one call and gives
every vehicle a persistent integer ID across frames - this ID is what lets the
pipeline save each plate exactly once.
"""
from dataclasses import dataclass

import numpy as np
from ultralytics import YOLO

COCO_NAMES = {1: "bicycle", 2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}


@dataclass
class Vehicle:
    track_id: int
    cls_id: int
    cls_name: str
    conf: float
    xyxy: np.ndarray  # (4,) int, full-frame coordinates


class VehicleDetector:
    def __init__(self, cfg, device, half):
        self.cfg = cfg
        self.model = YOLO(cfg.weights)
        self.device = device
        self.half = half and device != "cpu"
        self.names = self.model.names
        # warm-up
        self.model.predict(np.zeros((cfg.imgsz, cfg.imgsz, 3), np.uint8), imgsz=cfg.imgsz,
                           device=device, half=self.half, verbose=False)

    def track(self, frame) -> list[Vehicle]:
        res = self.model.track(
            frame,
            persist=True,
            tracker=self.cfg.tracker,
            conf=self.cfg.conf,
            iou=self.cfg.iou,
            imgsz=self.cfg.imgsz,
            classes=list(self.cfg.classes),
            device=self.device,
            half=self.half,
            verbose=False,
        )[0]

        out: list[Vehicle] = []
        if res.boxes is None or res.boxes.id is None:
            return out

        xyxy = res.boxes.xyxy.cpu().numpy().astype(int)
        ids = res.boxes.id.cpu().numpy().astype(int)
        confs = res.boxes.conf.cpu().numpy()
        clss = res.boxes.cls.cpu().numpy().astype(int)

        h, w = frame.shape[:2]
        for box, tid, c, k in zip(xyxy, ids, confs, clss):
            box = np.clip(box, [0, 0, 0, 0], [w - 1, h - 1, w - 1, h - 1])
            if max(box[2] - box[0], box[3] - box[1]) < self.cfg.min_box_px:
                continue
            out.append(Vehicle(int(tid), int(k), COCO_NAMES.get(int(k), self.names.get(int(k), str(k))),
                               float(c), box))
        return out
