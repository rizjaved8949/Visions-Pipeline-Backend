from __future__ import annotations

import math
from collections import deque

from ..types import SleepState


class SleepAnalyzer:
    """Sleep suspicion from sustained eye evidence or a strict posture fallback.

    Stillness or sitting alone never implies sleeping. Eye statistics are
    time-weighted, freshness-bounded, and count each inference sample once.
    """

    def __init__(self, cfg):
        self.window = float(cfg.get("perclos_window_seconds", 30.0))
        self.perclos_threshold = float(cfg.get("perclos_threshold", 0.60))
        self.score_threshold = float(cfg.get("sleep_score_threshold", 0.45))
        self.suppress_phone = bool(cfg.get("suppress_when_phone_active", False))
        self.torso_lean_threshold = float(cfg.get("torso_lean_deg", 28.0))
        self.min_eye_seconds = float(cfg.get("min_eye_evidence_seconds", 2.0))
        self.min_closed_seconds = float(cfg.get("min_closed_seconds", 2.0))
        self.eye_max_gap = float(cfg.get("eye_max_gap_seconds", 1.0))
        self.min_coverage = float(cfg.get("min_eye_coverage", 0.60))
        self.fallback_confirm = float(cfg.get("fallback_confirm_seconds", 8.0))
        self.max_gap = float(cfg.get("max_gap_seconds", 2.0))
        self.allow_eye_only = bool(cfg.get("allow_eye_only_sleep", True))
        self.eye_only_seconds = float(cfg.get("eye_only_min_seconds", max(4.0, self.min_eye_seconds, self.min_closed_seconds)))
        self.eye_only_perclos = float(cfg.get("eye_only_perclos_threshold", max(.85, self.perclos_threshold)))
        self.eye_only_coverage = float(cfg.get("eye_only_min_coverage", max(.75, self.min_coverage)))
        self.head_roll_threshold = float(cfg.get("head_roll_threshold", 15.0))
        self.ear_closed = float(cfg.get("ear_closed_threshold", 0.20))
        self.ear_open = float(cfg.get("ear_open_threshold", max(0.23, self.ear_closed)))
        self.ear_smoothing = float(cfg.get("ear_smoothing_seconds", 0.15))
        self.fallback_gap = float(cfg.get("fallback_gap_seconds", 0.5))
        self.wake_seconds = float(cfg.get("wake_confirm_seconds", 0.8))
        self.reset()

    def reset(self, track_id=None):
        self.track_id = track_id
        self.eye_history = deque()
        self.last_eye_sample = None
        self.last_time = None
        self.fallback_seconds = 0.0
        self.last_fallback = False
        self.last_fallback_at = None
        self.filtered_ear = None
        self.filtered_closed = None
        self.filtered_at = None
        self.awake_since = None
        self.awake_confirmed = False

    def _closed_value(self, eye, sample_time):
        """Filter fresh EAR samples only; retain support for boolean adapters."""
        if eye.ear_mean is None or not math.isfinite(eye.ear_mean):
            self.filtered_ear = self.filtered_at = self.filtered_closed = None
            return bool(eye.eyes_closed)
        if self.filtered_at is None or sample_time - self.filtered_at > self.eye_max_gap:
            self.filtered_ear = eye.ear_mean
            self.filtered_closed = eye.ear_mean < self.ear_closed
        elif sample_time <= self.filtered_at:
            return self.filtered_closed
        else:
            dt = sample_time - self.filtered_at
            alpha = 1.0 if self.ear_smoothing <= 0 else -math.expm1(-dt / self.ear_smoothing)
            self.filtered_ear += alpha * (eye.ear_mean - self.filtered_ear)
            if self.filtered_ear < self.ear_closed:
                self.filtered_closed = True
            elif self.filtered_ear >= self.ear_open:
                self.filtered_closed = False
        self.filtered_at = sample_time
        return self.filtered_closed

    def _append_eye(self, timestamp, value):
        if self.last_eye_sample is None or timestamp > self.last_eye_sample:
            self.eye_history.append((timestamp, value))
            self.last_eye_sample = timestamp

    def pause(self, now):
        # Preserve the episode, while explicitly breaking unobserved eye/pose time.
        if self.last_time is not None and now - self.last_time > self.max_gap:
            self.fallback_seconds = 0.0
        self.last_fallback = False
        self.awake_since = None
        self.awake_confirmed = False
        self._append_eye(now, None)

    def _eye_stats(self, now):
        while len(self.eye_history) > 1 and self.eye_history[1][0] <= now - self.window:
            self.eye_history.popleft()
        samples = list(self.eye_history)
        segments = []
        for idx, (timestamp, closed) in enumerate(samples):
            end = samples[idx + 1][0] if idx + 1 < len(samples) else now
            start = max(timestamp, now - self.window)
            end = min(end, now, timestamp + self.eye_max_gap)
            if closed is not None and end > start:
                segments.append((start, end, closed))
        known = sum(b - a for a, b, _ in segments)
        closed = sum(b - a for a, b, value in segments if value)
        span = min(self.window, now - samples[0][0]) if samples else 0.0
        coverage = known / span if span > 0 else 0.0
        continuous = 0.0
        cursor = now
        for start, end, value in reversed(segments):
            if not value or abs(end - cursor) > 1e-6:
                break
            continuous += end - start
            cursor = start
        return (closed / known if known > 0 else None), known, coverage, continuous

    def update(self, track_id, now, eye, posture, movement, phone, availability=None):
        if track_id != self.track_id:
            self.reset(track_id)
        availability = availability or dict(movement=True, posture=True, eyes=True, phone=True)
        dt = 0.0 if self.last_time is None else max(0.0, now - self.last_time)
        if dt > self.max_gap:
            self.fallback_seconds = 0.0
            self.last_fallback = False
        self.last_time = now
        sample_time = eye.observed_at if eye.observed_at is not None else now
        eye_known = bool(availability.get("eyes", False) and eye.quality_ok
                         and eye.eyes_closed is not None
                         and 0 <= now - sample_time <= self.eye_max_gap)
        if eye_known:
            eyes_closed = self._closed_value(eye, sample_time)
            if eyes_closed:
                self.awake_since = None
                self.awake_confirmed = False
            else:
                if self.awake_since is None:
                    self.awake_since = sample_time
                if not self.awake_confirmed and sample_time - self.awake_since + 1e-9 >= self.wake_seconds:
                    # New blinks must not reuse an old sleep episode's PERCLOS.
                    self.eye_history.clear()
                    self.last_eye_sample = None
                    self.awake_confirmed = True
            self._append_eye(sample_time, eyes_closed)
        else:
            eyes_closed = None
            self.awake_since = None
            self.awake_confirmed = False
            self._append_eye(now, None)
        perclos, known_seconds, coverage, continuous_closed = self._eye_stats(now)
        move_known = bool(availability.get("movement", False) and movement.reliable)
        missing = tuple(sorted(
            name for name, good in availability.items()
            if not good or (name == "eyes" and not eye_known) or (name == "movement" and not move_known)
        ))
        base = dict(perclos=perclos, filtered_ear=self.filtered_ear if eye_known else None,
                    filtered_eyes_closed=eyes_closed, eye_samples=sum(v is not None for _, v in self.eye_history),
                    missing_modules=missing, eye_evidence_seconds=known_seconds,
                    eye_coverage=coverage, continuous_closed_seconds=continuous_closed)
        phone_known = bool(availability.get("phone", False))
        phone_active = phone_known and phone.usage in {"call", "screen_use"}
        phone_ambiguous = phone_known and phone.detected and phone.usage == "visible"
        if (move_known and not movement.stationary) or (self.suppress_phone and phone_active):
            self.fallback_seconds = 0.0
            self.last_fallback = False
            return SleepState(reason="phone_active" if phone_active else "moving",
                              evidence_quality="high", **base)
        if self.suppress_phone and phone_ambiguous:
            self.fallback_seconds = 0.0
            self.last_fallback = False
            return SleepState(reason="phone_association_uncertain", **base)
        posture_known = bool(availability.get("posture", False))
        head_down = bool(posture_known and posture.head_down_known and posture.head_down)
        sitting_or_leaning = bool(posture_known and (
            posture.posture == "sitting" or
            (posture.torso_lean_deg is not None and posture.torso_lean_deg >= self.torso_lean_threshold)
        ))
        head_tilt = bool(eye_known and eye.head_roll_degrees is not None
                         and abs(eye.head_roll_degrees) >= self.head_roll_threshold)
        base["posture_support"] = bool(not eye_known and head_down and sitting_or_leaning
                                       and phone_known and not phone.detected
                                       and (not move_known or movement.stationary))
        fallback = bool(move_known and movement.stationary and not eye_known and head_down and sitting_or_leaning
                        and phone_known and not phone.detected)
        if fallback:
            if (self.last_fallback_at is not None and not self.last_fallback
                    and now - self.last_fallback_at > self.fallback_gap):
                self.fallback_seconds = 0.0
            if self.last_fallback and dt <= self.max_gap:
                self.fallback_seconds += dt
            self.last_fallback_at = now
        else:
            # Pause short unavailable intervals; contradictory evidence resets.
            if (eye_known or (posture_known and posture.head_down_known and not head_down)
                    or self.last_fallback_at is None
                    or now - self.last_fallback_at > self.fallback_gap):
                self.fallback_seconds = 0.0
        self.last_fallback = fallback
        high_perclos = bool(perclos is not None and perclos >= self.perclos_threshold
                            and known_seconds >= self.min_eye_seconds and coverage >= self.min_coverage)
        eye_signal = bool(eye_known and eyes_closed and
                          (continuous_closed >= self.min_closed_seconds or high_perclos))
        fallback_signal = fallback and self.fallback_seconds >= self.fallback_confirm
        stronger_eyes = bool(known_seconds >= self.eye_only_seconds
                             and perclos is not None and perclos >= self.eye_only_perclos
                             and coverage >= self.eye_only_coverage)
        independent_eyes = bool(self.allow_eye_only and not move_known and eye_signal
                                and phone_known and (not self.suppress_phone or not phone.detected)
                                and (head_down or head_tilt or sitting_or_leaning or stronger_eyes))
        ordinary_eyes = bool(move_known and movement.stationary and eye_signal)
        score = ((0.25 if move_known and movement.stationary else 0.0)
                 + (0.25 if head_down or head_tilt else 0.0)
                 + (0.30 if eye_signal else 0.0) + (0.15 if high_perclos else 0.0)
                 + (0.05 if sitting_or_leaning else 0.0))
        candidate = bool(score + 1e-9 >= self.score_threshold
                         and (ordinary_eyes or independent_eyes or fallback_signal))
        if candidate:
            quality = "high" if eye_signal else "degraded"
            reason = ("eye_evidence_motion_unknown" if independent_eyes else
                      "sleep_evidence" if ordinary_eyes else "sustained_head_down_posture")
        else:
            quality = "high" if eye_known and eyes_closed is False else "unknown"
            reason = "awake_eye_evidence" if quality == "high" else "insufficient_evidence"
        return SleepState(candidate=candidate, score=score, reason=reason,
                          evidence_quality=quality, fallback_seconds=self.fallback_seconds,
                          decision_basis=("eyes_and_stationary" if candidate and ordinary_eyes else
                                          "eyes_and_head_posture" if candidate and independent_eyes and (head_down or head_tilt or sitting_or_leaning) else
                                          "sustained_eyes" if candidate and independent_eyes else
                                          "posture_and_stationary" if candidate and fallback_signal else "unknown"), **base)


class SleepStatusStabilizer:
    """Independent sleep presentation; held labels never create alert evidence.

    The analyzer already qualifies entry. Display it immediately, then tolerate
    brief eye/track gaps and require sustained awake eyes before clearing it.
    """

    def __init__(self, cfg):
        self.hold_seconds = float(cfg.get("display_hold_seconds", 1.0))
        self.wake_seconds = float(cfg.get("wake_confirm_seconds", 0.8))
        self.posture_hold_seconds = float(cfg.get("posture_display_hold_seconds", 3.0))
        self.max_gap = float(cfg.get("max_gap_seconds", 2.0))
        self.reset()

    def reset(self, track_id=None):
        self.track_id = track_id
        self.label = None
        self.last_positive = None
        self.last_high = None
        self.last_posture_support = None
        self.awake_since = None
        self.last_update = None

    def update(self, sleep, now, *, track_id, present, observed_at=None):
        if (track_id != self.track_id or present is False
                or (self.last_update is not None and now - self.last_update > self.max_gap)):
            self.reset(track_id)
        self.last_update = now
        fresh = False
        if present is False:
            reason = "guard_absent"
        elif sleep.reason in {"moving", "phone_active", "phone_association_uncertain"}:
            self.reset(track_id)
            self.last_update = now
            reason = sleep.reason
        elif sleep.candidate and present is True:
            self.awake_since = None
            self.last_positive = now if observed_at is None else min(now, observed_at)
            if sleep.evidence_quality == "high":
                self.last_high = self.last_positive
                self.label = "Sleeping"
            elif self.last_high is None or now - self.last_high > self.hold_seconds:
                self.label = "Possible Sleep"
            fresh = self.label == (
                "Sleeping" if sleep.evidence_quality == "high" else "Possible Sleep")
            reason = sleep.reason
        else:
            if sleep.posture_support and self.label == "Sleeping":
                self.last_posture_support = now
            if sleep.reason == "awake_eye_evidence":
                if self.awake_since is None:
                    self.awake_since = now
                if now - self.awake_since + 1e-9 >= self.wake_seconds:
                    self.reset(track_id)
                    self.last_update = now
            else:
                self.awake_since = None
            # Bridge sensor reacquisition without extending beyond the bounded
            # interval since the last qualified eye evidence.
            supported_hold = (self.last_posture_support is not None and self.label == "Sleeping"
                              and now - self.last_posture_support <= self.hold_seconds
                              and self.last_high is not None
                              and now - self.last_high <= self.posture_hold_seconds)
            if not supported_hold and (self.last_positive is None or now - self.last_positive > self.hold_seconds):
                self.label = None
            reason = "held_sleep_evidence" if self.label else sleep.reason
        return dict(label=self.label, fresh=fresh, held=self.label is not None and not fresh, reason=reason)
