"""
Breach logic - the part that decides WHEN an alert is raised.

Design goals
------------
* Exactly one alert per (object, zone) ENTRY. A person standing inside the zone
  for 5 minutes produces one alert, not 9000.
* No flicker: the anchor point must be inside for `enter_frames` consecutive
  frames before it counts as an entry, and outside for `exit_frames` consecutive
  frames before it counts as an exit. This suppresses jitter on the boundary.
* Tracker id swaps are tolerated: if the same physical object gets a new id
  (occlusion, id switch) while still inside the zone, we suppress the duplicate
  if a very recent alert happened at nearly the same position (`dup_radius` /
  `dup_window_s`).
* A track that disappears is forgotten after `track_ttl_s` seconds so memory
  never grows unbounded on a 24/7 stream.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from .detector import Detection
from .zones import Zone, ZoneSet


@dataclass
class BreachEvent:
    track_id: int
    cls_name: str
    zone: Zone
    point: Tuple[float, float]
    bbox: Tuple[float, float, float, float]
    timestamp: float
    frame_index: int


@dataclass
class _TrackState:
    inside: Dict[str, bool] = field(default_factory=dict)         # zone name -> currently inside?
    inside_count: Dict[str, int] = field(default_factory=dict)    # consecutive frames inside
    outside_count: Dict[str, int] = field(default_factory=dict)   # consecutive frames outside
    last_seen: float = 0.0
    cls_name: str = ""


class BreachManager:
    def __init__(
        self,
        zones: ZoneSet,
        frame_w: int,
        frame_h: int,
        anchor: str = "center",        # "center" | "bottom"
        enter_frames: int = 3,
        exit_frames: int = 10,
        track_ttl_s: float = 5.0,
        dup_radius: float = 60.0,      # px
        dup_window_s: float = 2.0,
        re_alert_cooldown_s: float = 0.0,  # 0 = alert again on every fresh entry
    ):
        self.zones = zones
        self.w, self.h = frame_w, frame_h
        self.anchor = anchor
        self.enter_frames = max(1, enter_frames)
        self.exit_frames = max(1, exit_frames)
        self.track_ttl_s = track_ttl_s
        self.dup_radius = dup_radius
        self.dup_window_s = dup_window_s
        self.re_alert_cooldown_s = re_alert_cooldown_s

        self._tracks: Dict[int, _TrackState] = {}
        self._recent_alerts: List[Tuple[float, str, float, float]] = []  # (t, zone, x, y)
        self._last_alert_per_track_zone: Dict[Tuple[int, str], float] = {}
        self.total_breaches = 0

    # ------------------------------------------------------------------
    def _point(self, d: Detection) -> Tuple[float, float]:
        return d.bottom_center if self.anchor == "bottom" else d.center

    def _is_duplicate(self, now: float, zone_name: str, x: float, y: float) -> bool:
        for t, zn, ax, ay in self._recent_alerts:
            if zn == zone_name and now - t <= self.dup_window_s:
                if (ax - x) ** 2 + (ay - y) ** 2 <= self.dup_radius ** 2:
                    return True
        return False

    def _gc(self, now: float) -> None:
        dead = [tid for tid, s in self._tracks.items() if now - s.last_seen > self.track_ttl_s]
        for tid in dead:
            del self._tracks[tid]
            for key in [k for k in self._last_alert_per_track_zone if k[0] == tid]:
                del self._last_alert_per_track_zone[key]
        self._recent_alerts = [a for a in self._recent_alerts if now - a[0] <= self.dup_window_s]

    # ------------------------------------------------------------------
    def update(self, detections: List[Detection], frame_index: int,
               now: Optional[float] = None) -> Tuple[List[BreachEvent], Dict[int, List[str]]]:
        """
        Returns
        -------
        events   : new breach events raised on THIS frame
        inside   : {track_id: [zone names the object is currently inside]}
                   (used for drawing - objects inside are highlighted)
        """
        now = time.time() if now is None else now
        events: List[BreachEvent] = []
        inside_map: Dict[int, List[str]] = {}

        for d in detections:
            if d.track_id < 0:
                # untracked detection (tracker still initialising) - ignore for
                # alert purposes; it will get an id within a frame or two.
                continue
            st = self._tracks.setdefault(d.track_id, _TrackState())
            st.last_seen = now
            st.cls_name = d.cls_name
            px, py = self._point(d)

            for z in self.zones.zones:
                zn = z.name
                is_in = z.contains(px, py, self.w, self.h)
                was_in = st.inside.get(zn, False)

                if is_in:
                    st.inside_count[zn] = st.inside_count.get(zn, 0) + 1
                    st.outside_count[zn] = 0
                else:
                    st.outside_count[zn] = st.outside_count.get(zn, 0) + 1
                    st.inside_count[zn] = 0

                # ---- ENTER transition ----
                if not was_in and st.inside_count[zn] >= self.enter_frames:
                    st.inside[zn] = True
                    last = self._last_alert_per_track_zone.get((d.track_id, zn), -1e9)
                    cooldown_ok = (now - last) >= self.re_alert_cooldown_s
                    if cooldown_ok and not self._is_duplicate(now, zn, px, py):
                        ev = BreachEvent(d.track_id, d.cls_name, z, (px, py),
                                         (d.x1, d.y1, d.x2, d.y2), now, frame_index)
                        events.append(ev)
                        self.total_breaches += 1
                        self._recent_alerts.append((now, zn, px, py))
                        self._last_alert_per_track_zone[(d.track_id, zn)] = now

                # ---- EXIT transition ----
                elif was_in and st.outside_count[zn] >= self.exit_frames:
                    st.inside[zn] = False

                if st.inside.get(zn, False):
                    inside_map.setdefault(d.track_id, []).append(zn)

        self._gc(now)
        return events, inside_map

    def reset(self) -> None:
        self._tracks.clear()
        self._recent_alerts.clear()
        self._last_alert_per_track_zone.clear()
