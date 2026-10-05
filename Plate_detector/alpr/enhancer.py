"""
Stage 4a - Plate image enhancement.

Pipeline:  deskew -> super-resolution (FSRCNN / ESPCN via OpenCV dnn_superres,
           or classic bicubic) -> CLAHE contrast -> unsharp mask -> resize to target width

Honest note: super-resolution cannot invent detail that was never captured.
It makes a 60-px-wide plate noticeably more readable; it will not rescue a
15-px smudge. Camera placement/zoom always beats post-processing.
"""
from pathlib import Path

import cv2
import numpy as np


class PlateEnhancer:
    def __init__(self, cfg):
        self.cfg = cfg
        self.sr = None
        method = cfg.method.lower()
        if method in ("fsrcnn", "espcn"):
            model_path = Path(cfg.model_dir) / f"{method.upper()}_x{cfg.scale}.pb"
            if model_path.exists() and hasattr(cv2, "dnn_superres"):
                self.sr = cv2.dnn_superres.DnnSuperResImpl_create()
                self.sr.readModel(str(model_path))
                self.sr.setModel(method, int(cfg.scale))
                try:  # use CUDA if OpenCV was built with it (pip wheels usually are not - CPU is still fast for tiny crops)
                    self.sr.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
                    self.sr.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
                except Exception:  # noqa: BLE001
                    pass
                print(f"[enhance] super-resolution: {method} x{cfg.scale}")
            else:
                print(f"[enhance] {model_path.name} not found or dnn_superres missing -> classic upscale")
        else:
            print("[enhance] classic upscale (bicubic + sharpen)")

    # ------------------------------------------------------------------ #
    @staticmethod
    def deskew(img: np.ndarray) -> np.ndarray:
        """Straighten small rotations using the dominant edge orientation."""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=30,
                                minLineLength=max(10, img.shape[1] // 3), maxLineGap=5)
        if lines is None:
            return img
        angles = []
        # OpenCV <5 returns shape (N,1,4); OpenCV >=5 returns (N,4) directly - reshape covers both
        for x1, y1, x2, y2 in lines.reshape(-1, 4):
            a = np.degrees(np.arctan2(y2 - y1, x2 - x1))
            if abs(a) < 20:          # only near-horizontal lines (plate top/bottom edges)
                angles.append(a)
        if not angles:
            return img
        angle = float(np.median(angles))
        if abs(angle) < 0.5:
            return img
        h, w = img.shape[:2]
        M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
        return cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

    def upscale(self, img: np.ndarray) -> np.ndarray:
        if self.sr is not None:
            try:
                return self.sr.upsample(img)
            except Exception:  # noqa: BLE001
                pass
        s = int(self.cfg.scale)
        return cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_CUBIC)

    def enhance(self, plate: np.ndarray) -> np.ndarray:
        if plate is None or plate.size == 0:
            return plate
        img = plate
        if self.cfg.deskew:
            img = self.deskew(img)
        img = self.upscale(img)

        if self.cfg.clahe:
            lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            l = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4)).apply(l)
            img = cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)

        if self.cfg.sharpen and self.cfg.sharpen > 0:
            blur = cv2.GaussianBlur(img, (0, 0), 2.0)
            img = cv2.addWeighted(img, 1 + self.cfg.sharpen, blur, -self.cfg.sharpen, 0)

        if self.cfg.target_width and img.shape[1] != self.cfg.target_width:
            r = self.cfg.target_width / img.shape[1]
            img = cv2.resize(img, (self.cfg.target_width, max(1, int(img.shape[0] * r))),
                             interpolation=cv2.INTER_AREA if r < 1 else cv2.INTER_CUBIC)
        return img
