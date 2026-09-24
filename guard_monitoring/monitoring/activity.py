from __future__ import annotations

ACTIVITY_LABELS = frozenset({"Sleeping", "Moving", "Using Mobile", "Stationary"})


class ActivityStabilizer:
    """Presentation state only; independent event timers consume raw evidence."""

    def __init__(self, cfg):
        self.confirm_seconds = float(cfg.get("confirm_seconds", 0.4))
        self.sleep_confirm_seconds = float(cfg.get("sleep_confirm_seconds", 0.4))
        self.hold_seconds = float(cfg.get("hold_seconds", 1.0))
        self.max_gap = float(cfg.get("max_evidence_gap_seconds", 2.0))
        self.reset()

    def reset(self):
        self.label = None
        self.pending = None
        self.pending_since = None
        self.last_valid = None
        self.last_update = None

    def update(self, candidate, now, *, present):
        if candidate is not None and candidate not in ACTIVITY_LABELS:
            raise ValueError("Unsupported guard activity")
        if present is False:
            self.reset()
            return {"label": None, "fresh": False, "held": False}
        if self.last_update is not None and now - self.last_update > self.max_gap:
            self.reset()
        self.last_update = now
        # Conflicting candidate labels must not keep an unsupported old label
        # alive indefinitely. Only evidence for the displayed label renews it.
        if (self.label is not None and candidate != self.label
                and self.last_valid is not None
                and now - self.last_valid > self.hold_seconds):
            self.label = None
        if candidate is None:
            self.pending = None
            self.pending_since = None
            if self.last_valid is None or now - self.last_valid > self.hold_seconds:
                self.label = None
            return {"label": self.label, "fresh": False, "held": self.label is not None}
        if candidate == self.label:
            self.last_valid = now
            self.pending = None
            self.pending_since = None
        else:
            if self.pending != candidate:
                self.pending = candidate
                self.pending_since = now
            delay = self.sleep_confirm_seconds if candidate == "Sleeping" else self.confirm_seconds
            if now - self.pending_since + 1e-9 >= delay:
                self.label = candidate
                self.last_valid = now
                self.pending = None
                self.pending_since = None
        return {"label": self.label, "fresh": self.label == candidate,
                "held": self.label is not None and self.label != candidate}
