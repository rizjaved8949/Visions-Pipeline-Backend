from __future__ import annotations

from collections import deque

from ..types import EyeState, MovementState, PhoneState, PostureState, SleepState


class SleepAnalyzer:
    """Multi-cue sleep/inactivity suspicion logic.

    This is an operational alert heuristic, not a medical sleep diagnosis. Eye evidence
    is used only when face landmarks are reliable. If the face is unavailable, posture
    and movement can produce degraded evidence without inventing an eyelid state.
    """

    def __init__(self, cfg: dict):
        self.window = float(cfg.get("perclos_window_seconds", 30.0))
        self.perclos_threshold = float(cfg.get("perclos_threshold", 0.70))
        self.score_threshold = float(cfg.get("sleep_score_threshold", 0.55))
        self.suppress_phone = bool(cfg.get("suppress_when_phone_active", True))
        self.torso_lean_threshold = float(cfg.get("torso_lean_deg", 28.0))
        self.eye_history = deque()
        self.track_id = None

    def reset(self, track_id=None):
        self.eye_history.clear()
        self.track_id = track_id

    def update(
        self,
        track_id: int,
        now: float,
        eye: EyeState,
        posture: PostureState,
        movement: MovementState,
        phone: PhoneState,
        availability: dict[str, bool] | None = None,
    ) -> SleepState:
        if self.track_id != track_id:
            self.reset(track_id)

        availability = availability or {
            "movement": True,
            "posture": True,
            "eyes": True,
            "phone": True,
        }
        missing = tuple(sorted(name for name, available in availability.items() if not available))

        # Movement is foundational for the inactivity part of this rule. If it is
        # unavailable, do not create a sleep candidate from pose/eyes alone.
        if not availability.get("movement", True):
            return SleepState(
                candidate=False,
                score=0.0,
                reason="movement_unavailable",
                evidence_quality="unknown",
                missing_modules=missing,
            )

        if availability.get("eyes", True) and eye.quality_ok and eye.eyes_closed is not None:
            self.eye_history.append((now, bool(eye.eyes_closed)))
        while self.eye_history and now - self.eye_history[0][0] > self.window:
            self.eye_history.popleft()

        perclos = None
        if self.eye_history:
            perclos = sum(1 for _, closed in self.eye_history if closed) / len(self.eye_history)

        phone_known = availability.get("phone", True)
        phone_active = phone_known and phone.usage in {"call", "screen_use"}
        if self.suppress_phone and phone_active:
            return SleepState(
                candidate=False,
                score=0.0,
                perclos=perclos,
                eye_samples=len(self.eye_history),
                reason="phone_active",
                evidence_quality="high" if not missing else "degraded",
                missing_modules=missing,
            )

        low_motion = bool(movement.stationary)
        eye_known = availability.get("eyes", True) and eye.quality_ok
        eye_closed = bool(eye_known and eye.eyes_closed is True)
        high_perclos = perclos is not None and perclos >= self.perclos_threshold
        posture_known = availability.get("posture", True)
        head_down = bool(posture_known and posture.head_down)
        sitting_or_leaning = bool(
            posture_known
            and (
                posture.posture == "sitting"
                or (
                    posture.torso_lean_deg is not None
                    and posture.torso_lean_deg >= self.torso_lean_threshold
                )
            )
        )

        score = 0.0
        score += 0.25 if low_motion else 0.0
        score += 0.25 if head_down else 0.0
        score += 0.30 if eye_closed else 0.0
        score += 0.15 if high_perclos else 0.0
        score += 0.05 if sitting_or_leaning else 0.0

        eye_signal = eye_closed or high_perclos
        fallback_signal = (not eye_known) and low_motion and head_down and sitting_or_leaning
        candidate = low_motion and score >= self.score_threshold and (eye_signal or fallback_signal)

        if candidate and eye_signal and phone_known and posture_known:
            quality = "high"
        elif candidate:
            quality = "degraded"
        else:
            quality = "unknown" if missing else "high"

        if candidate and quality == "degraded":
            reason = "sleep_evidence_degraded"
        elif candidate:
            reason = "sleep_evidence"
        else:
            reason = "insufficient_evidence"

        return SleepState(
            candidate=candidate,
            score=score,
            perclos=perclos,
            eye_samples=len(self.eye_history),
            reason=reason,
            evidence_quality=quality,
            missing_modules=missing,
        )
