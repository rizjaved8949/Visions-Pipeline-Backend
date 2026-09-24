from __future__ import annotations

from dataclasses import dataclass
import math

from ..geometry import bbox_footpoint, point_in_polygon
from ..types import GuardTrack


@dataclass
class _Candidate:
    first_seen: float
    last_seen: float


class GuardSelector:
    """Persist a confirmed identity through a bounded gap.

    A retained box has seen_now=False. It is not a detection and cannot be used
    for new inference or positive evidence. Different IDs are never merged
    just because their boxes are close.
    """

    def __init__(self, duty_zone_pixels, confirm_seconds=1.0, release_seconds=5.0,
                 presence_grace_seconds=2.0, manual_track_id=None,
                 candidate_gap_seconds=0.5):
        self.zone = list(duty_zone_pixels)
        self.confirm_seconds = float(confirm_seconds)
        self.release_seconds = float(release_seconds)
        self.presence_grace_seconds = min(float(presence_grace_seconds), self.release_seconds)
        self.candidate_gap_seconds = float(candidate_gap_seconds)
        self.manual_track_id = manual_track_id
        self.active_id = None
        self.last_seen_time = None
        self.last_box = None
        self.last_confidence = 0.0
        self.candidates: dict[int, _Candidate] = {}
        self.generation = 0

    def reset(self):
        self._release()
        self.candidates.clear()

    def _release(self):
        self.active_id = None
        self.last_box = None
        self.last_seen_time = None
        self.last_confidence = 0.0

    def retained(self, now):
        if self.last_seen_time is None or now - self.last_seen_time > self.release_seconds:
            self._release()
            return None
        if self.active_id is None or self.last_box is None:
            return None
        return GuardTrack(self.active_id, self.last_box, self.last_confidence, False)

    def update(self, tracked_detections, now):
        visible = {}
        ids = getattr(tracked_detections, "tracker_id", None)
        if ids is not None:
            for idx, tid in enumerate(ids):
                if tid is None or int(tid) < 0:
                    continue
                box = tuple(float(v) for v in tracked_detections.xyxy[idx])
                if not all(math.isfinite(v) for v in box) or box[2] <= box[0] or box[3] <= box[1]:
                    continue
                confidence = getattr(tracked_detections, "confidence", None)
                conf = float(confidence[idx]) if confidence is not None else 0.0
                visible[int(tid)] = GuardTrack(int(tid), box, conf, True)
        if self.active_id is not None:
            self.retained(now)
        # Track replacement candidates during the gap, not only after release.
        inside = {tid for tid, t in visible.items()
                  if point_in_polygon(bbox_footpoint(t.bbox), self.zone)}
        for tid in inside:
            previous = self.candidates.get(tid)
            if previous is None or now - previous.last_seen > self.candidate_gap_seconds:
                self.candidates[tid] = _Candidate(now, now)
            else:
                previous.last_seen = now
        for tid in list(self.candidates):
            if tid not in inside and now - self.candidates[tid].last_seen > self.candidate_gap_seconds:
                del self.candidates[tid]
        if self.manual_track_id is not None:
            track = visible.get(int(self.manual_track_id))
            if track is not None:
                if self.active_id is None:
                    self.generation += 1
                self._remember(track, now)
                return track
            return self.retained(now)
        if self.active_id in visible:
            track = visible[self.active_id]
            self._remember(track, now)
            return track
        if self.active_id is not None:
            return self.retained(now)
        eligible = [
            (c.last_seen - c.first_seen, visible[tid].confidence, tid)
            for tid, c in self.candidates.items()
            if tid in inside and c.last_seen - c.first_seen + 1e-9 >= self.confirm_seconds
        ]
        if not eligible:
            return None
        track = visible[max(eligible)[2]]
        self.generation += 1
        self._remember(track, now)
        return track

    def is_present(self, now):
        return (self.active_id is not None and self.last_seen_time is not None
                and now - self.last_seen_time <= self.presence_grace_seconds)

    def _remember(self, track, now):
        self.active_id = track.track_id
        self.last_seen_time = now
        self.last_box = track.bbox
        self.last_confidence = track.confidence

    def _stale_or_none(self, now):
        return self.retained(now)
