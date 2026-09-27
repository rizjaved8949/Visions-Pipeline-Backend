from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Optional

import numpy as np

from ..geometry import bbox_diagonal, bbox_footpoint, point_in_polygon
from ..types import GuardTrack
from .kalman import BBoxKalman
from .reid import AppearanceSignature, TorsoAppearanceModel


@dataclass
class _Candidate:
    first_seen: float
    last_seen: float


class GuardSelector:
    """Persist a confirmed identity through a bounded gap.

    A retained box has seen_now=False. It is not a detection and cannot be used
    for new inference or positive evidence. Different IDs are never merged
    just because their boxes are close.

    Robust tracking add-ons (opt-in, all safe defaults):
      * ``reid_enabled`` - keep a torso HSV histogram of the selected guard,
        and when the underlying ByteTrack ID disappears but a new candidate
        with a similar appearance appears inside the duty zone during the
        release window, adopt the new ID as the same guard. Never activates
        unless the caller passes a ``frame`` to ``update()`` and the cached
        signature is *distinctive* (see AppearanceSignature.is_distinctive),
        so the existing unit tests (scripted, uniform-black frames) are not
        affected.
      * ``predict_during_grace`` - roll a small constant-velocity Kalman
        filter forward while the guard is retained, so ``retained(now)``
        returns a bbox that tracks the last known motion instead of a
        stale frozen rectangle. The retained track is still marked
        ``seen_now=False`` and its rule conditions are still paused - the
        change is presentational only.
    """

    def __init__(self, duty_zone_pixels, confirm_seconds=1.0, release_seconds=5.0,
                 presence_grace_seconds=2.0, manual_track_id=None,
                 candidate_gap_seconds=0.5,
                 *,
                 reid_enabled: bool = False,
                 reid_hist_bins: int = 24,
                 reid_min_similarity: float = 0.70,
                 reid_max_footpoint_ratio: float = 0.35,
                 reid_update_every_frames: int = 5,
                 predict_during_grace: bool = False,
                 kalman_max_seconds: Optional[float] = None):
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

        self.reid_enabled = bool(reid_enabled)
        self.reid_min_similarity = float(reid_min_similarity)
        self.reid_max_footpoint_ratio = float(reid_max_footpoint_ratio)
        self.reid_update_every_frames = max(1, int(reid_update_every_frames))
        self._reid_model = TorsoAppearanceModel(bins=int(reid_hist_bins)) if self.reid_enabled else None
        self._reid_reference: Optional[AppearanceSignature] = None
        self._reid_updates: int = 0
        self._frames_since_reid_update: int = 0
        # Aliases: ByteTrack IDs the selector has decided are the same guard
        # as ``active_id`` via appearance re-identification. Everyone
        # downstream continues to see ``active_id``, so rule timers and
        # movement history are preserved. Alias entries expire on the same
        # release schedule as the primary identity.
        self._alias_ids: dict[int, float] = {}  # tid -> last_seen

        self.predict_during_grace = bool(predict_during_grace)
        self._kalman = BBoxKalman(
            max_predict_seconds=(self.release_seconds if kalman_max_seconds is None
                                 else float(kalman_max_seconds))
        ) if self.predict_during_grace else None

        # Track why the last returned observation was categorised the way it
        # was. Exposed for the pipeline's frame log (diagnostics only).
        self.last_association_reason: str = "idle"

    # ------------------------------------------------------------------
    # State management
    # ------------------------------------------------------------------
    def reset(self):
        self._release()
        self.candidates.clear()

    def _release(self):
        self.active_id = None
        self.last_box = None
        self.last_seen_time = None
        self.last_confidence = 0.0
        self._reid_reference = None
        self._reid_updates = 0
        self._frames_since_reid_update = 0
        self._alias_ids.clear()
        if self._kalman is not None:
            self._kalman.reset()

    def retained(self, now):
        if self.last_seen_time is None or now - self.last_seen_time > self.release_seconds:
            self._release()
            return None
        if self.active_id is None or self.last_box is None:
            return None
        # Kalman prediction is a *display* refinement. The event pipeline
        # still treats the return value as seen_now=False, so no rule can
        # advance from it.
        predicted = None
        if self.predict_during_grace and self._kalman is not None:
            predicted = self._kalman.predict(now)
        display_box = predicted if predicted is not None else self.last_box
        source = "predicted" if predicted is not None else "retained"
        return GuardTrack(
            self.active_id, tuple(display_box), self.last_confidence, False,
            source=source,
            predicted_bbox=tuple(predicted) if predicted is not None else None,
        )

    # ------------------------------------------------------------------
    # Update - main entry
    # ------------------------------------------------------------------
    def update(self, tracked_detections, now, *, frame=None):
        """Process one frame of tracker output.

        Parameters
        ----------
        tracked_detections
            The supervision.Detections-like object returned by ByteTrack.
        now
            Timestamp of this frame in seconds.
        frame
            Optional BGR frame. When supplied *and* ReID is enabled, the
            selector maintains an appearance model of the current guard
            and may adopt a new track ID as the same guard on re-appearance.
            When ``None`` (the default in every existing test), ReID never
            activates and behavior is identical to the historical selector.
        """
        visible: dict[int, GuardTrack] = {}
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
                visible[int(tid)] = GuardTrack(int(tid), box, conf, True, source="detected")
        # Legacy call: expire retention lazily (may release active_id).
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

        # Manual override path - retained for the operator-selected-guard
        # workflow used by the live service. Manual selection bypasses ReID
        # because the operator has already committed to a specific ID.
        if self.manual_track_id is not None:
            track = visible.get(int(self.manual_track_id))
            if track is not None:
                if self.active_id is None:
                    self.generation += 1
                self._remember(track, now, frame=frame)
                self.last_association_reason = "manual_match"
                return track
            self.last_association_reason = "manual_missing"
            return self.retained(now)

        # Expire stale aliases first so we don't re-match a track ID that
        # ByteTrack has since re-assigned to a different person.
        if self._alias_ids and self.active_id is not None:
            self._alias_ids = {
                tid: last_seen
                for tid, last_seen in self._alias_ids.items()
                if now - last_seen <= self.release_seconds
            }

        # Happy path - the current guard's underlying track ID (or one of
        # its established aliases) is still in this frame's tracker output.
        matching_id = self._match_visible(visible)
        if matching_id is not None:
            candidate_track = visible[matching_id]
            # Downstream must always see the ORIGINAL active_id so rule
            # timers and movement history carry across ReID adoptions.
            display_track = GuardTrack(
                self.active_id, candidate_track.bbox, candidate_track.confidence,
                True,
                source=("id_match" if matching_id == self.active_id else "alias_match"),
            )
            if matching_id != self.active_id:
                self._alias_ids[matching_id] = now
            self._remember(display_track, now, frame=frame)
            self.last_association_reason = (
                "id_match" if matching_id == self.active_id else "alias_match"
            )
            return display_track

        # The current guard's IDs are all missing this frame. Try
        # appearance-based re-identification with any *other* visible ID
        # whose footprint is near the last known one. This is the
        # ByteTrack-lost-me-because-they-turned-around case. If nothing
        # scores high enough, fall back to the retention/candidate path
        # below.
        adopted = self._maybe_adopt_via_reid(visible, inside, now, frame)
        if adopted is not None:
            self.last_association_reason = "reid_adopted"
            return adopted

        # Advance the Kalman filter one step even without a measurement, so
        # a subsequent retained() call returns a fresher prediction.
        if self.active_id is not None and self._kalman is not None and self.predict_during_grace:
            self._kalman.advance_without_measurement(now)

        if self.active_id is not None:
            self.last_association_reason = "retained"
            return self.retained(now)

        # Fresh selection: pick the candidate that has been inside long
        # enough. Ties broken by detection confidence.
        eligible = [
            (c.last_seen - c.first_seen, visible[tid].confidence, tid)
            for tid, c in self.candidates.items()
            if tid in inside and c.last_seen - c.first_seen + 1e-9 >= self.confirm_seconds
        ]
        if not eligible:
            self.last_association_reason = "no_candidate"
            return None
        track = visible[max(eligible)[2]]
        self.generation += 1
        self._remember(track, now, frame=frame)
        self.last_association_reason = "confirmed"
        return track

    def is_present(self, now):
        return (self.active_id is not None and self.last_seen_time is not None
                and now - self.last_seen_time <= self.presence_grace_seconds)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------
    def _remember(self, track, now, *, frame=None):
        self.active_id = track.track_id
        self.last_seen_time = now
        self.last_box = track.bbox
        self.last_confidence = track.confidence
        # Kalman: measurement update (initialises on the first call).
        if self._kalman is not None:
            self._kalman.update(track.bbox, now)
        # ReID: cache/refresh the torso signature at a bounded frequency so
        # brief phone-out or one-side-facing frames don't spoil the model.
        if self._reid_model is not None and frame is not None:
            self._frames_since_reid_update += 1
            if (self._reid_reference is None
                    or self._frames_since_reid_update >= self.reid_update_every_frames):
                signature = self._reid_model.extract(frame, track.bbox)
                if signature is not None and signature.is_distinctive():
                    # Blend with the previous reference so brief pose
                    # changes don't cause reference drift (e.g. body turned
                    # sideways). 30% fresh + 70% old keeps ID stable.
                    if self._reid_reference is None or self._reid_reference.is_empty():
                        self._reid_reference = signature
                    else:
                        blended = 0.30 * signature.hist + 0.70 * self._reid_reference.hist
                        total = float(blended.sum())
                        if total > 0:
                            blended = blended / total
                        self._reid_reference = AppearanceSignature(blended, signature.bins)
                    self._reid_updates += 1
                    self._frames_since_reid_update = 0

    def _match_visible(self, visible: dict[int, GuardTrack]) -> Optional[int]:
        """Return the id in ``visible`` that is our current guard or an alias.

        When multiple aliases are present, prefer the primary active_id, then
        the alias closest to the last known box (min L1 corner distance).
        """
        if self.active_id in visible:
            return int(self.active_id)
        if not self._alias_ids or self.last_box is None:
            return None
        candidates = [tid for tid in self._alias_ids if tid in visible]
        if not candidates:
            return None
        if len(candidates) == 1:
            return int(candidates[0])
        # Break ties by proximity to the last known box.
        def _distance(tid):
            box = visible[tid].bbox
            return sum(abs(a - b) for a, b in zip(box, self.last_box))
        return int(min(candidates, key=_distance))

    def _maybe_adopt_via_reid(self, visible, inside, now, frame) -> Optional[GuardTrack]:
        if (self._reid_model is None or frame is None
                or self._reid_reference is None
                or not self._reid_reference.is_distinctive()
                or self.active_id is None or self.last_box is None):
            return None
        if now - (self.last_seen_time or now) > self.release_seconds:
            return None  # Too late to adopt - the active guard has been released.
        if not visible:
            return None
        # Only consider candidates whose footprint is close to the last
        # known one, so a completely different person on the opposite side
        # of the frame can never be adopted as the guard.
        frame_diag = math.hypot(*frame.shape[:2][::-1]) if hasattr(frame, "shape") else 0.0
        max_dist = self.reid_max_footpoint_ratio * frame_diag
        last_foot = bbox_footpoint(self.last_box)
        best_score = -1.0
        best_track: Optional[GuardTrack] = None
        for tid, track in visible.items():
            if tid == self.active_id or tid in self._alias_ids:
                continue
            foot = bbox_footpoint(track.bbox)
            if frame_diag > 0 and math.hypot(foot[0] - last_foot[0], foot[1] - last_foot[1]) > max_dist:
                continue
            candidate_sig = self._reid_model.extract(frame, track.bbox)
            if candidate_sig is None or not candidate_sig.is_distinctive():
                continue
            score = self._reid_model.similarity(self._reid_reference, candidate_sig)
            if score < self.reid_min_similarity:
                continue
            if tid in inside:
                # Small bonus for being inside the duty polygon.
                score += 0.01
            if score > best_score:
                best_score = score
                best_track = track
        if best_track is None:
            return None
        # Adopt: register the new track id as an alias of the current
        # identity. active_id and generation do NOT change - so downstream
        # (rule engine, movement monitor, sleep analyzer) sees the same
        # track_id and does not reset any timer. The returned observation
        # carries the original active_id as track_id so the pipeline's
        # identity check does not fire either.
        self._alias_ids[int(best_track.track_id)] = now
        display_track = GuardTrack(
            self.active_id, best_track.bbox, best_track.confidence, True,
            source="reid_adopted",
        )
        self._remember(display_track, now, frame=frame)
        return display_track

    def _stale_or_none(self, now):
        return self.retained(now)
