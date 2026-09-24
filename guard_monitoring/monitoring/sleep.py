from __future__ import annotations

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
        self.suppress_phone = bool(cfg.get("suppress_when_phone_active", True))
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
        self.reset()

    def reset(self, track_id=None):
        self.track_id = track_id
        self.eye_history = deque()
        self.last_eye_sample = None
        self.last_time = None
        self.fallback_seconds = 0.0
        self.last_fallback = False

    def _append_eye(self, timestamp, value):
        if self.last_eye_sample is None or timestamp > self.last_eye_sample:
            self.eye_history.append((timestamp, value))
            self.last_eye_sample = timestamp

    def pause(self, now):
        # Preserve the episode, while explicitly breaking unobserved eye/pose time.
        if self.last_time is not None and now - self.last_time > self.max_gap:
            self.fallback_seconds = 0.0
        self.last_fallback = False
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
            self._append_eye(sample_time, bool(eye.eyes_closed))
        else:
            self._append_eye(now, None)
        perclos, known_seconds, coverage, continuous_closed = self._eye_stats(now)
        move_known = bool(availability.get("movement", False) and movement.reliable)
        missing = tuple(sorted(
            name for name, good in availability.items()
            if not good or (name == "eyes" and not eye_known) or (name == "movement" and not move_known)
        ))
        base = dict(perclos=perclos, eye_samples=sum(v is not None for _, v in self.eye_history),
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
        fallback = bool(move_known and movement.stationary and not eye_known and head_down and sitting_or_leaning
                        and phone_known and not phone.detected)
        if fallback:
            if self.last_fallback and dt <= self.max_gap:
                self.fallback_seconds += dt
        else:
            self.fallback_seconds = 0.0
        self.last_fallback = fallback
        high_perclos = bool(perclos is not None and perclos >= self.perclos_threshold
                            and known_seconds >= self.min_eye_seconds and coverage >= self.min_coverage)
        eye_signal = bool(eye_known and eye.eyes_closed and
                          (continuous_closed >= self.min_closed_seconds or high_perclos))
        fallback_signal = fallback and self.fallback_seconds >= self.fallback_confirm
        stronger_eyes = bool(known_seconds >= self.eye_only_seconds
                             and perclos is not None and perclos >= self.eye_only_perclos
                             and coverage >= self.eye_only_coverage)
        independent_eyes = bool(self.allow_eye_only and not move_known and eye_signal
                                and phone_known and not phone.detected
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
            quality = "high" if eye_known and eye.eyes_closed is False else "unknown"
            reason = "awake_eye_evidence" if quality == "high" else "insufficient_evidence"
        return SleepState(candidate=candidate, score=score, reason=reason,
                          evidence_quality=quality, fallback_seconds=self.fallback_seconds,
                          decision_basis=("eyes_and_stationary" if candidate and ordinary_eyes else
                                          "eyes_and_head_posture" if candidate and independent_eyes and (head_down or head_tilt or sitting_or_leaning) else
                                          "sustained_eyes" if candidate and independent_eyes else
                                          "posture_and_stationary" if candidate and fallback_signal else "unknown"), **base)
