from __future__ import annotations

from dataclasses import dataclass

from ..types import RuleEvent


@dataclass
class _RuleState:
    active_since: float | None = None
    false_since: float | None = None
    paused_since: float | None = None
    triggered: bool = False


class TimedRuleEngine:
    """Duration rules with tri-state input.

    condition=True   -> accumulate duration.
    condition=False  -> apply the configured grace/reset behavior.
    condition=None   -> evidence is unknown; pause the timer. Unknown time is not
                        counted as positive evidence and is not treated as false.
    """

    def __init__(self, camera_id: str, cfg: dict):
        self.camera_id = camera_id
        self.warmup = float(cfg.get("warmup_seconds", 0.0))
        self.thresholds = {
            "sleep": float(cfg["sleep_seconds"]),
            "phone": float(cfg["phone_seconds"]),
            "stationary": float(cfg["stationary_seconds"]),
            "absence": float(cfg["absence_seconds"]),
        }
        grace = cfg.get("grace_seconds", {})
        self.grace = {name: float(grace.get(name, 0.0)) for name in self.thresholds}
        self.states = {name: _RuleState() for name in self.thresholds}
        self.start_time = None

    def update(
        self,
        name: str,
        condition: bool | None,
        now: float,
        track_id: int | None,
        payload: dict,
    ) -> RuleEvent | None:
        if self.start_time is None:
            self.start_time = now
        if now - self.start_time < self.warmup:
            return None

        state = self.states[name]

        # Unknown evidence pauses an active timer rather than counting it as true
        # or resetting it as false.
        if condition is None:
            if state.active_since is not None and state.paused_since is None:
                state.paused_since = now
            return None

        # Resume from an unknown interval. Shift active_since forward so unknown
        # wall-clock time is excluded from the accumulated positive duration.
        if state.paused_since is not None:
            if condition and state.active_since is not None:
                state.active_since += max(0.0, now - state.paused_since)
            state.paused_since = None

        if condition:
            state.false_since = None
            if state.active_since is None:
                state.active_since = now
            duration = now - state.active_since
            if not state.triggered and duration >= self.thresholds[name]:
                state.triggered = True
                return RuleEvent(
                    rule=name,
                    camera_id=self.camera_id,
                    track_id=track_id,
                    episode_started_at=state.active_since,
                    triggered_at=now,
                    threshold_seconds=self.thresholds[name],
                    payload=payload,
                )
            return None

        if state.active_since is not None:
            if state.false_since is None:
                state.false_since = now
            if now - state.false_since > self.grace[name]:
                state.active_since = None
                state.false_since = None
                state.paused_since = None
                state.triggered = False
        return None

    def is_triggered(self, name: str) -> bool:
        return self.states[name].triggered

    def active_duration(self, name: str, now: float) -> float:
        state = self.states[name]
        if state.active_since is None:
            return 0.0
        effective_now = state.paused_since if state.paused_since is not None else now
        return max(0.0, effective_now - state.active_since)
