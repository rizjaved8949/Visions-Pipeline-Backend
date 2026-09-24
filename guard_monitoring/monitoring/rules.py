from __future__ import annotations

import math
from dataclasses import dataclass

from ..types import RuleEvent


@dataclass
class _RuleState:
    active_since: float | None = None
    false_since: float | None = None
    paused_since: float | None = None
    last_time: float | None = None
    last_condition: bool | None = None
    positive_seconds: float = 0.0
    triggered: bool = False


class TimedRuleEngine:
    """Count positive observed time, preserving episodes across bounded gaps.

    False/unknown intervals preserve the one-alert latch within grace, but
    never count toward a threshold. Person-specific time cannot transfer IDs.
    """

    def __init__(self, camera_id: str, cfg: dict):
        self.camera_id = camera_id
        self.warmup = float(cfg.get("warmup_seconds", 0.0))
        self.thresholds = {
            name: float(cfg[f"{name}_seconds"])
            for name in ("sleep", "phone", "stationary", "absence")
        }
        self.grace = {
            name: float(cfg.get("grace_seconds", {}).get(name, 0.0))
            for name in self.thresholds
        }
        self.unknown_reset = float(cfg.get("unknown_reset_seconds", 5.0))
        gap = cfg.get("max_evidence_gap_seconds")
        self.max_gap = None if gap is None else float(gap)
        self.states = {name: _RuleState() for name in self.thresholds}
        self.start_time = None
        self.track_id = None

    def reset_guard(self, track_id=None):
        for name in ("sleep", "phone", "stationary"):
            self.states[name] = _RuleState()
        self.track_id = track_id

    def update(self, name, condition, now, track_id, payload) -> RuleEvent | None:
        now = float(now)
        if not math.isfinite(now):
            raise ValueError("Rule timestamp must be finite")
        if condition is not None and not isinstance(condition, bool):
            raise ValueError("Rule condition must be True, False, or None")
        if name != "absence" and track_id is not None and track_id != self.track_id:
            self.reset_guard(track_id)
        if self.start_time is None:
            self.start_time = now
        if now < self.start_time:
            raise ValueError("Rule timestamps must be monotonic")
        if now - self.start_time < self.warmup:
            return None
        state = self.states[name]
        dt = 0.0 if state.last_time is None else now - state.last_time
        if dt < 0:
            raise ValueError("Rule timestamps must be monotonic")
        expired = (
            (state.false_since is not None and now - state.false_since > self.grace[name])
            or (state.paused_since is not None and now - state.paused_since > self.unknown_reset)
            or (self.max_gap is not None and dt > self.max_gap and dt > self.unknown_reset)
        )
        if expired:
            state = self.states[name] = _RuleState()
            dt = 0.0
        # Unknown -> false -> true must not count the unknown interval.
        if state.active_since is not None and state.last_condition is True:
            if self.max_gap is None or dt <= self.max_gap:
                state.positive_seconds += dt
        state.last_time = now
        state.last_condition = condition
        if condition is None:
            if state.active_since is not None and state.paused_since is None:
                state.paused_since = now
            return None
        state.paused_since = None
        if condition:
            state.false_since = None
            if state.active_since is None:
                state.active_since = now
                state.positive_seconds = 0.0
            if not state.triggered and state.positive_seconds + 1e-9 >= self.thresholds[name]:
                state.triggered = True
                return RuleEvent(
                    rule=name, camera_id=self.camera_id, track_id=track_id,
                    episode_started_at=state.active_since, triggered_at=now,
                    threshold_seconds=self.thresholds[name], payload=payload,
                )
            return None
        if state.active_since is not None:
            if state.false_since is None:
                state.false_since = now
            if self.grace[name] <= 0:
                self.states[name] = _RuleState(last_time=now, last_condition=False)
        return None

    def pause_all(self, now):
        for name in self.states:
            self.update(name, None, now, self.track_id, {})

    def is_triggered(self, name):
        return self.states[name].triggered

    def active_duration(self, name, now):
        # Do not extrapolate an unobserved interval for a UI countdown.
        return self.states[name].positive_seconds
