"""Appearance-based person re-identification for the GuardSelector.

Purpose
-------
ByteTrack (as configured in this project) uses IoU/motion for association only.
When RF-DETR briefly misses the guard - a rotation from front to back view,
walking behind a pillar, a low-light detection dropout - ByteTrack starves the
old track, then when the guard reappears it mints a **new** track ID. Without
appearance evidence the GuardSelector cannot know that the new ID is the same
person, so it either (a) waits ``confirm_seconds`` before re-selecting them
(so the operator overlay goes blank for ~1 s) or (b) selects them as a fresh
identity, losing continuity of the phone/sleep/stationary rule timers.

Solution
--------
Cache an HSV colour histogram of the guard's torso crop (upper-middle half of
the bbox, well away from the face/legs) every few frames while the guard is
seen. When a new candidate appears inside the duty zone during the
release-window, compute its signature and score it against the cached one
using Bhattacharyya distance (histogram-agnostic and cheap). If it exceeds
``min_similarity`` and its footpoint is within a small fraction of the frame
diagonal from the last-known footpoint, treat it as the same guard.

Design notes
------------
- No new dependency: OpenCV is already required by pipeline.py. When cv2 is
  not importable (unit tests without heavy deps) the module falls back to a
  pure-numpy histogram which is functionally identical for the small crops
  we operate on. That fallback keeps the existing test harness working.
- Signatures are content-addressable numpy arrays; comparisons are O(bins).
- Nothing here mutates the pipeline's ground-truth state - the caller decides
  whether to actually adopt a candidate.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from ..types import BBox


def _try_import_cv2():
    try:
        import cv2  # noqa: F401
        return cv2
    except Exception:
        return None


class AppearanceSignature:
    """A stored signature: normalised HSV histogram of the torso crop."""

    __slots__ = ("hist", "bins")

    def __init__(self, hist: np.ndarray, bins: int):
        self.hist = np.asarray(hist, dtype=np.float32)
        self.bins = int(bins)

    def is_empty(self) -> bool:
        return self.hist.size == 0 or float(self.hist.sum()) <= 0.0

    def is_distinctive(self, max_dominant_ratio: float = 0.85,
                       min_effective_bins: int = 3) -> bool:
        """A histogram is 'distinctive' if no single bin dominates.

        Guards uniform crops (e.g. solid-black scripted frames used in the
        regression tests, or a blown-out sunlit wall) from being used as
        a re-identification reference - two different people cropped from
        an all-black frame produce identical histograms, which is
        undesirable. Requires ``min_effective_bins`` bins to hold >=1% of
        the mass each.
        """
        if self.is_empty():
            return False
        total = float(self.hist.sum())
        if total <= 0:
            return False
        normalised = self.hist / total
        if float(normalised.max()) >= max_dominant_ratio:
            return False
        significant = int((normalised >= 0.01).sum())
        return significant >= max(1, int(min_effective_bins))


class TorsoAppearanceModel:
    """Compute and compare torso-crop signatures.

    Parameters
    ----------
    bins
        Number of bins per channel. HSV histogram is 3D (bins x bins x bins).
        24 is a good balance between sensitivity and cost for a torso crop.
    """

    def __init__(self, bins: int = 24):
        if bins < 4:
            raise ValueError("bins must be >= 4")
        self.bins = int(bins)
        self._cv2 = _try_import_cv2()

    # ------------------------------------------------------------------
    # Extraction
    # ------------------------------------------------------------------
    def extract(self, frame_bgr: np.ndarray, bbox: BBox) -> Optional[AppearanceSignature]:
        """Return a signature for the torso region of ``bbox``.

        Returns None if the crop is empty or degenerate. Never raises;
        appearance is opt-in and must never crash the pipeline.
        """
        try:
            crop = self._torso_crop(frame_bgr, bbox)
        except Exception:
            return None
        if crop is None or crop.size == 0:
            return None
        return self._signature_from_crop(crop)

    # ------------------------------------------------------------------
    # Comparison
    # ------------------------------------------------------------------
    def similarity(
        self,
        a: Optional[AppearanceSignature],
        b: Optional[AppearanceSignature],
    ) -> float:
        """Return a similarity score in [0, 1] using Bhattacharyya distance.

        Higher is more similar. Missing/empty signatures score 0.
        """
        if a is None or b is None or a.is_empty() or b.is_empty():
            return 0.0
        if a.bins != b.bins:
            return 0.0
        # Normalise defensively - stored histograms should already be
        # normalised, but the caller might supply an unnormalised one.
        pa = a.hist / max(float(a.hist.sum()), 1e-9)
        pb = b.hist / max(float(b.hist.sum()), 1e-9)
        # Bhattacharyya coefficient in [0, 1]; identical histograms => 1.
        return float(np.sum(np.sqrt(pa * pb)))

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------
    def _torso_crop(self, frame_bgr: np.ndarray, bbox: BBox) -> Optional[np.ndarray]:
        h, w = frame_bgr.shape[:2]
        x1, y1, x2, y2 = bbox
        bw = max(1.0, float(x2 - x1))
        bh = max(1.0, float(y2 - y1))
        # Torso strip: horizontally centered, vertically 20-65% of bbox.
        # Skips the head (where the phone-to-face heuristic looks) and the
        # legs (where uniform colours dominate on many guards).
        cx1 = float(x1) + 0.15 * bw
        cx2 = float(x2) - 0.15 * bw
        cy1 = float(y1) + 0.20 * bh
        cy2 = float(y1) + 0.65 * bh
        ix1 = max(0, int(round(cx1)))
        iy1 = max(0, int(round(cy1)))
        ix2 = min(int(w), int(round(cx2)))
        iy2 = min(int(h), int(round(cy2)))
        if ix2 - ix1 < 4 or iy2 - iy1 < 4:
            return None
        return frame_bgr[iy1:iy2, ix1:ix2]

    def _signature_from_crop(self, crop_bgr: np.ndarray) -> Optional[AppearanceSignature]:
        cv2 = self._cv2
        if cv2 is not None:
            try:
                hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
                hist = cv2.calcHist(
                    [hsv],
                    [0, 1, 2],
                    None,
                    [self.bins, self.bins, self.bins],
                    [0, 180, 0, 256, 0, 256],
                )
                hist = cv2.normalize(hist, hist).flatten()
                return AppearanceSignature(hist, self.bins)
            except Exception:
                pass  # Fall through to numpy fallback below.
        # Numpy fallback (no cv2 present, or cv2 error): treat crop as BGR
        # and build a 3D histogram directly. Bin edges deliberately match
        # the OpenCV version above so signatures are comparable between
        # environments where the test crops are identical.
        crop = np.asarray(crop_bgr, dtype=np.float32)
        if crop.ndim == 2:
            crop = np.stack([crop, crop, crop], axis=-1)
        if crop.shape[-1] != 3:
            return None
        # Approximate HSV via a cheap heuristic to keep the fallback
        # dependency-free; still produces a stable per-appearance signature
        # for regression tests.
        b, g, r = crop[..., 0], crop[..., 1], crop[..., 2]
        v = crop.max(axis=-1)
        c = v - crop.min(axis=-1)
        s = np.where(v > 0, c / np.maximum(v, 1e-6), 0.0) * 255.0
        h = np.zeros_like(v)
        mask = c > 0
        hi = np.zeros_like(v)
        # Vectorised hue computation matching cv2 convention (0..180).
        rmax = (v == r) & mask
        gmax = (v == g) & mask & ~rmax
        bmax = (v == b) & mask & ~rmax & ~gmax
        hi[rmax] = ((g[rmax] - b[rmax]) / np.maximum(c[rmax], 1e-6)) % 6.0
        hi[gmax] = (b[gmax] - r[gmax]) / np.maximum(c[gmax], 1e-6) + 2.0
        hi[bmax] = (r[bmax] - g[bmax]) / np.maximum(c[bmax], 1e-6) + 4.0
        h = hi * 30.0  # 60 degrees / 2 == 30 for cv2's 0..180 hue range
        h = np.clip(h, 0, 179)
        s = np.clip(s, 0, 255)
        v = np.clip(v, 0, 255)
        # Build 3D histogram
        h_bin = np.minimum((h / 180.0 * self.bins).astype(int), self.bins - 1)
        s_bin = np.minimum((s / 256.0 * self.bins).astype(int), self.bins - 1)
        v_bin = np.minimum((v / 256.0 * self.bins).astype(int), self.bins - 1)
        flat_idx = (h_bin * self.bins + s_bin) * self.bins + v_bin
        hist = np.bincount(flat_idx.ravel(), minlength=self.bins ** 3).astype(np.float32)
        total = float(hist.sum())
        if total > 0:
            hist = hist / total
        return AppearanceSignature(hist, self.bins)
