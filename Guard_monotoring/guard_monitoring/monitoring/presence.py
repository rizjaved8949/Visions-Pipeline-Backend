from __future__ import annotations

from dataclasses import dataclass


@dataclass
class PresenceState:
    """Tri-state presence result.

    present=True/False is only emitted when tracking was available. If tracking is
    unavailable for a frame, present=None prevents the absence timer from fabricating
    an absence event.
    """

    present: bool | None
    reason: str


def presence_from_tracker(selector, tracking_available: bool, now: float) -> PresenceState:
    if not tracking_available:
        return PresenceState(present=None, reason="tracking_unavailable")
    return PresenceState(present=bool(selector.is_present(now)), reason="tracking_available")
