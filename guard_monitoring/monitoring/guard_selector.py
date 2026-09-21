from __future__ import annotations

from dataclasses import dataclass

from ..geometry import bbox_footpoint, point_in_polygon
from ..types import GuardTrack


@dataclass
class _Candidate:
    first_seen: float
    last_seen: float


class GuardSelector:
    """Select one persistent guard track from tracked people inside a configured duty zone."""

    def __init__(
        self,
        duty_zone_pixels,
        confirm_seconds: float = 1.0,
        release_seconds: float = 5.0,
        presence_grace_seconds: float = 2.0,
        manual_track_id: int | None = None,
    ):
        self.zone = list(duty_zone_pixels)
        self.confirm_seconds = float(confirm_seconds)
        self.release_seconds = float(release_seconds)
        self.presence_grace_seconds = float(presence_grace_seconds)
        self.manual_track_id = manual_track_id
        self.active_id: int | None = None
        self.last_seen_time: float | None = None
        self.last_box = None
        self.last_confidence = 0.0
        self.candidates: dict[int, _Candidate] = {}

    def update(self, tracked_detections, now: float) -> GuardTrack | None:
        visible: dict[int, GuardTrack] = {}
        tracker_ids = getattr(tracked_detections, "tracker_id", None)
        if tracker_ids is not None:
            for idx, track_id in enumerate(tracker_ids):
                if track_id is None or int(track_id) < 0:
                    continue
                box = tuple(float(x) for x in tracked_detections.xyxy[idx])
                confidence = 0.0
                if tracked_detections.confidence is not None:
                    confidence = float(tracked_detections.confidence[idx])
                visible[int(track_id)] = GuardTrack(int(track_id), box, confidence, True)

        # Manual mode is useful for debugging a known ByteTrack ID.
        if self.manual_track_id is not None:
            track = visible.get(int(self.manual_track_id))
            if track is not None:
                self._remember(track, now)
                self.active_id = track.track_id
                return track
            return self._stale_or_none(now)

        # Keep the current guard while the same track is visible.
        if self.active_id is not None and self.active_id in visible:
            track = visible[self.active_id]
            self._remember(track, now)
            return track

        # Release a guard only after a configured missing interval.
        if self.active_id is not None and self.last_seen_time is not None:
            if now - self.last_seen_time <= self.release_seconds:
                return None
            self.active_id = None
            self.last_box = None
            self.candidates.clear()

        # Build dwell time for people whose foot-point is inside the duty zone.
        inside_ids = set()
        for track_id, track in visible.items():
            if not point_in_polygon(bbox_footpoint(track.bbox), self.zone):
                continue
            inside_ids.add(track_id)
            if track_id not in self.candidates:
                self.candidates[track_id] = _Candidate(now, now)
            else:
                self.candidates[track_id].last_seen = now

        for track_id in list(self.candidates):
            if track_id not in inside_ids and now - self.candidates[track_id].last_seen > self.release_seconds:
                del self.candidates[track_id]

        eligible = []
        for track_id, c in self.candidates.items():
            dwell = c.last_seen - c.first_seen
            if dwell >= self.confirm_seconds and track_id in visible:
                eligible.append((dwell, visible[track_id].confidence, track_id))

        if eligible:
            _, _, chosen = max(eligible)
            track = visible[chosen]
            self.active_id = chosen
            self._remember(track, now)
            return track
        return None

    def is_present(self, now: float) -> bool:
        return self.last_seen_time is not None and (now - self.last_seen_time <= self.presence_grace_seconds)

    def _remember(self, track: GuardTrack, now: float) -> None:
        self.last_seen_time = now
        self.last_box = track.bbox
        self.last_confidence = track.confidence

    def _stale_or_none(self, now: float) -> GuardTrack | None:
        if self.active_id is None or self.last_seen_time is None or self.last_box is None:
            return None
        if now - self.last_seen_time <= self.presence_grace_seconds:
            return GuardTrack(self.active_id, self.last_box, self.last_confidence, False)
        return None
