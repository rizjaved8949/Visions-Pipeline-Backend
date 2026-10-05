"""
Stage 2 - License plate detection.

Runs a second, small YOLO model - NOT on the whole frame but on each vehicle
crop. Reasons:
  * a plate is tiny in a 1080p traffic frame (often < 1% of the image); on a
    vehicle crop it is 10-30% of the image, so the detector sees it far better;
  * we already know which vehicle the plate belongs to (the track ID), so there
    is no plate<->vehicle association problem;
  * all crops of a frame are batched into a single GPU call.

Works with any Ultralytics-format plate model: the pretrained one downloaded
by setup_weights.py, or your own fine-tuned one (train/finetune_plates.py).
"""
from dataclasses import dataclass

import cv2
import numpy as np
from ultralytics import YOLO


@dataclass
class PlateDet:
    conf: float
    xyxy_frame: np.ndarray   # plate box in full-frame coords
    crop: np.ndarray         # plate pixels (BGR), un-enhanced
    score: float             # quality score used to pick the best frame for a track
    low_conf: bool = False   # below cfg.conf: kept only so a weak-but-real plate is visible
                             # in output/plates_lowconf instead of silently vanishing


def sharpness(gray: np.ndarray) -> float:
    """Variance of Laplacian - higher = sharper / less motion blur."""
    if gray.size == 0:
        return 0.0
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


class PlateDetector:
    def __init__(self, cfg, device, half):
        self.cfg = cfg
        self.model = YOLO(cfg.weights)
        self.device = device
        self.half = half and device != "cpu"
        self.model.predict(np.zeros((cfg.imgsz, cfg.imgsz, 3), np.uint8), imgsz=cfg.imgsz,
                           device=device, half=self.half, verbose=False)

    def _expanded_crop(self, frame, box):
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = box
        mx = int((x2 - x1) * self.cfg.crop_margin)
        my = int((y2 - y1) * self.cfg.crop_margin)
        x1, y1 = max(0, x1 - mx), max(0, y1 - my)
        x2, y2 = min(w, x2 + mx), min(h, y2 + my)
        return frame[y1:y2, x1:x2], (x1, y1)

    def detect(self, frame, vehicle_boxes: list[np.ndarray]) -> list[PlateDet | None]:
        """One vehicle box in -> one best plate (or None) out, same order."""
        if not vehicle_boxes:
            return []
        crops, offsets = [], []
        for b in vehicle_boxes:
            c, off = self._expanded_crop(frame, b)
            if c.size == 0:
                c = np.zeros((32, 32, 3), np.uint8)
            crops.append(c)
            offsets.append(off)

        # Detect down to conf_floor, but treat anything below cfg.conf as "low confidence":
        # it is only used when a track has NO confident detection at all, so the main output
        # keeps its purity while a weak-but-real plate still shows up for review.
        floor = min(self.cfg.conf, getattr(self.cfg, "conf_floor", self.cfg.conf))
        results = self.model.predict(crops, imgsz=self.cfg.imgsz, conf=floor,
                                     device=self.device, half=self.half, verbose=False)

        out: list[PlateDet | None] = []
        for res, crop, (ox, oy) in zip(results, crops, offsets):
            best = None
            best_low = None
            if res.boxes is not None and len(res.boxes):
                xyxy = res.boxes.xyxy.cpu().numpy().astype(int)
                confs = res.boxes.conf.cpu().numpy()
                ch = crop.shape[0]
                for box, conf in zip(xyxy, confs):
                    x1, y1, x2, y2 = box
                    pw, ph = x2 - x1, y2 - y1
                    if pw < self.cfg.min_width_px or ph < 8:
                        continue
                    # real plates sit low on the vehicle; roof-mounted signs (taxi tags etc.)
                    # don't - reject candidates whose vertical center is too close to the top
                    if (y1 + y2) / 2 < ch * self.cfg.roof_exclude_frac:
                        continue
                    # real plates are always noticeably wider than tall; square/blob-shaped
                    # boxes (taillights, bumper clutter) are not
                    if pw / ph < self.cfg.min_aspect_ratio:
                        continue
                    plate = crop[y1:y2, x1:x2]
                    if plate.size == 0:
                        continue
                    gray = cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY)
                    sh = sharpness(gray)
                    # Quality score: confidence x size (saturating at quality_width_saturation_px)
                    # x sharpness (saturating ~300)
                    sat = self.cfg.quality_width_saturation_px
                    score = float(conf) * min(pw, sat) / sat * min(sh, 300.0) / 300.0
                    is_low = float(conf) < self.cfg.conf
                    target = best_low if is_low else best
                    if target is None or score > target.score:
                        det = PlateDet(
                            conf=float(conf),
                            xyxy_frame=np.array([x1 + ox, y1 + oy, x2 + ox, y2 + oy]),
                            crop=plate.copy(),
                            score=score,
                            low_conf=is_low,
                        )
                        if is_low:
                            best_low = det
                        else:
                            best = det
            # a confident detection always wins; the weak one is a fallback, never a competitor
            out.append(best if best is not None else best_low)
        return out
