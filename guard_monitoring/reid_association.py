import cv2
import numpy as np
from typing import Optional, Tuple
from .types import BBox


class GuardReIDSignature:
    """
    Lightweight appearance feature extractor (HSV color histogram + grayscale patch)
    for re-adopting guard track IDs across occlusions and view rotations.
    """

    def __init__(self):
        self.hist: Optional[np.ndarray] = None
        self.patch: Optional[np.ndarray] = None
        self.last_bbox: Optional[BBox] = None

    def update(self, frame: np.ndarray, bbox: BBox) -> None:
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = bbox.to_int_tuple()
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)

        if x2 - x1 < 10 or y2 - y1 < 10:
            return

        crop = frame[y1:y2, x1:x2]

        # Extract torso crop (upper 50% of bounding box)
        torso_h = max(5, crop.shape[0] // 2)
        torso = crop[0:torso_h, :]

        # 1. Color Histogram in HSV
        hsv_torso = cv2.cvtColor(torso, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv_torso], [0, 1], None, [16, 16], [0, 180, 0, 256])
        cv2.normalize(hist, hist, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)

        # 2. Normalized grayscale patch
        gray_torso = cv2.cvtColor(torso, cv2.COLOR_BGR2GRAY)
        patch = cv2.resize(gray_torso, (16, 32))

        self.hist = hist
        self.patch = patch
        self.last_bbox = bbox

    def compute_similarity(self, frame: np.ndarray, candidate_bbox: BBox) -> float:
        if self.hist is None or self.patch is None or self.last_bbox is None:
            return 0.0

        h, w = frame.shape[:2]
        x1, y1, x2, y2 = candidate_bbox.to_int_tuple()
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)

        if x2 - x1 < 10 or y2 - y1 < 10:
            return 0.0

        crop = frame[y1:y2, x1:x2]
        torso_h = max(5, crop.shape[0] // 2)
        torso = crop[0:torso_h, :]

        # 1. Histogram similarity (Bhattacharyya)
        hsv_torso = cv2.cvtColor(torso, cv2.COLOR_BGR2HSV)
        cand_hist = cv2.calcHist([hsv_torso], [0, 1], None, [16, 16], [0, 180, 0, 256])
        cv2.normalize(cand_hist, cand_hist, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)

        bhatt_dist = cv2.compareHist(self.hist, cand_hist, cv2.HISTCMP_BHATTACHARYYA)
        hist_sim = max(0.0, 1.0 - bhatt_dist)

        # 2. Patch normalized cross-correlation
        gray_torso = cv2.cvtColor(torso, cv2.COLOR_BGR2GRAY)
        cand_patch = cv2.resize(gray_torso, (16, 32))
        res = cv2.matchTemplate(cand_patch, self.patch, cv2.TM_CCOEFF_NORMED)
        patch_sim = max(0.0, float(res[0][0]))

        # 3. Spatial proximity (distance between feet / centers)
        c1 = np.array(self.last_bbox.center)
        c2 = np.array(candidate_bbox.center)
        dist = np.linalg.norm(c1 - c2)
        diag = np.sqrt(w**2 + h**2)
        spatial_sim = max(0.0, 1.0 - (dist / (diag * 0.3)))

        score = (0.45 * hist_sim) + (0.35 * patch_sim) + (0.20 * spatial_sim)
        return float(score)