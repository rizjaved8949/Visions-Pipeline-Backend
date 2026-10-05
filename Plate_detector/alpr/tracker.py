"""
Stage 3 - Track bookkeeping (the "no duplicates" logic).

ByteTrack (inside VehicleDetector) gives each vehicle a stable ID. This module
keeps, per ID, the best plate crops seen so far and decides *once* when a track
is "finalized" - i.e. sent to enhancement / OCR and saved. A track is finalized
when:
    * its best plate score hasn't improved for `stable_frames` frames  (vehicle is
      as close/sharp as it's going to get), or
    * the vehicle has left the scene for `lost_frames` frames, or
    * we've tried `max_attempts` frames without ever seeing a plate (skip it).

Each track ID produces at most ONE output. If the tracker loses a vehicle and
re-detects it with a new ID, you may get a second result - tune lost_frames /
use botsort.yaml (re-ID) if that happens a lot in your camera setup.
"""
import time
from dataclasses import dataclass, field

import numpy as np

from .plate_detector import PlateDet
from .vehicle_detector import Vehicle


@dataclass
class TrackState:
    track_id: int
    cls_name: str
    first_seen: int
    last_seen: int
    frames_seen: int = 0
    best: list[PlateDet] = field(default_factory=list)   # sorted desc by score, len <= keep_top_k
    best_score: float = 0.0
    best_score_frame: int = 0
    finalized: bool = False
    result_text: str | None = None      # filled once OCR/enhance worker returns
    result_status: str = ""             # "pending" / "done" / "skipped"
    last_plate_box: np.ndarray | None = None
    last_plate_conf: float = 0.0        # shown on the annotated video so a reviewer can see
                                        # why a detection was kept or dropped
    last_vehicle_box: np.ndarray | None = None
    last_vehicle_crop: np.ndarray | None = None
    created_at: float = field(default_factory=time.time)


class TrackManager:
    def __init__(self, cfg):
        self.cfg = cfg
        self.tracks: dict[int, TrackState] = {}

    # ------------------------------------------------------------------ #
    def update(self, frame_id: int, vehicles: list[Vehicle], plates: list[PlateDet | None], frame):
        """Feed this frame's detections. Returns list of TrackStates ready to be finalized."""
        seen = set()
        for v, p in zip(vehicles, plates):
            seen.add(v.track_id)
            t = self.tracks.get(v.track_id)
            if t is None:
                t = TrackState(v.track_id, v.cls_name, frame_id, frame_id)
                self.tracks[v.track_id] = t
            t.last_seen = frame_id
            t.frames_seen += 1
            t.last_vehicle_box = v.xyxy
            t.last_plate_box = p.xyxy_frame if p is not None else None
            t.last_plate_conf = p.conf if p is not None else 0.0

            if t.finalized or p is None:
                continue

            # insert into top-k. A confident detection outranks a low-confidence one
            # regardless of quality score, so t.best[0] is only low_conf when the track
            # never produced a confident plate at all.
            t.best.append(p)
            t.best.sort(key=lambda d: (not d.low_conf, d.score), reverse=True)
            t.best = t.best[: self.cfg.keep_top_k]
            if p.score > t.best_score:
                t.best_score = p.score
                t.best_score_frame = frame_id
                x1, y1, x2, y2 = v.xyxy
                t.last_vehicle_crop = frame[y1:y2, x1:x2].copy()

        ready = []
        for tid, t in self.tracks.items():
            if t.finalized:
                continue
            has_plate = len(t.best) > 0
            lost = frame_id - t.last_seen
            stable = has_plate and (frame_id - t.best_score_frame) >= self.cfg.stable_frames
            gone = lost >= self.cfg.lost_frames
            hopeless = (not has_plate) and t.frames_seen >= self.cfg.max_attempts

            if stable or (gone and has_plate):
                if t.best_score >= self.cfg.min_score_to_save:
                    t.finalized = True
                    t.result_status = "pending"
                    ready.append(t)
                elif gone:
                    t.finalized = True
                    t.result_status = "skipped"
            elif gone or hopeless:
                t.finalized = True
                t.result_status = "skipped"
        return ready

    def active(self, frame_id: int, max_age: int = 2) -> dict[int, TrackState]:
        """Tracks seen within the last `max_age` frames (for drawing)."""
        return {tid: t for tid, t in self.tracks.items() if frame_id - t.last_seen <= max_age}

    def gc(self, frame_id: int, keep_frames: int = 600):
        """Drop very old finalized tracks to keep memory flat on long streams."""
        for tid in [tid for tid, t in self.tracks.items() if t.finalized and frame_id - t.last_seen > keep_frames]:
            del self.tracks[tid]
