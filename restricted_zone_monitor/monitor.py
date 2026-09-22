"""
Session engine for restricted-zone breach monitoring.

Mirrors Attendance/attendance.py's shape: module-level state, one active
session at a time, a background capture thread, and an explicit
start/stop/report/wipe lifecycle driven by the API layer.

Zones are the one thing that works differently from every other module here:
there is no persistence at all. The frontend asks the user to draw the
zone(s) fresh before every session (see set_zones / get_zone_frame_jpeg) and
whatever was drawn is discarded - along with the rest of the session - the
moment a report is downloaded or /wipe_data is called.
"""

from __future__ import annotations

import os
import re
import shutil
import threading
import time
import uuid
from datetime import datetime
from typing import List, Optional

import cv2

from guard_monitoring.io import make_writer

from . import config
from .breach import BreachManager
from .detector import Detector
from .video_source import VideoSource
from .visualize import Banner, draw_detections, draw_hud, draw_zones
from .zones import PALETTE, Zone, ZoneSet

# === Locks ===
video_source_lock = threading.Lock()
zones_lock = threading.Lock()
session_lock = threading.Lock()
frame_lock = threading.Lock()
model_lock = threading.Lock()

# === State ===
VIDEO_SOURCE: object = 0
current_zones = ZoneSet()
detector: Optional[Detector] = None
current_session: Optional[dict] = None
session_counter = 0
is_capturing = False
capture_thread_running = False
latest_frame = None
latest_person_count = 0
latest_inside_count = 0

_capture_thread: Optional[threading.Thread] = None


# ------------------------------------------------------------------
# Video source
# ------------------------------------------------------------------
def get_video_source():
    with video_source_lock:
        return VIDEO_SOURCE


def set_video_source(source) -> object:
    """Local device index ("0", "1", ...), a network stream URL
    (rtsp://... / http://...), or a path to an uploaded video file."""
    global VIDEO_SOURCE
    if is_capturing:
        raise ValueError("Stop the current session before changing the source.")
    source = str(source).strip()
    if not source:
        raise ValueError("Video source is required.")
    parsed = int(source) if source.isdigit() else source
    with video_source_lock:
        VIDEO_SOURCE = parsed
    return parsed


def list_available_cameras(max_index: int = 5) -> List[int]:
    available = []
    for i in range(max_index):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            available.append(i)
        cap.release()
    return available


def save_uploaded_video(filename: str, content: bytes) -> str:
    """Save an uploaded video under config.UPLOAD_DIR and return its path,
    ready to be passed to set_video_source()."""
    safe = re.sub(r"[^A-Za-z0-9_.-]", "_", filename or "upload.mp4")
    path = os.path.join(str(config.UPLOAD_DIR), f"{uuid.uuid4().hex}_{safe}")
    with open(path, "wb") as f:
        f.write(content)
    return path


# ------------------------------------------------------------------
# Zones - drawn fresh every time, kept in memory only
# ------------------------------------------------------------------
def get_zones() -> ZoneSet:
    with zones_lock:
        return ZoneSet(list(current_zones.zones))


def set_zones(zone_payload: List[dict]) -> ZoneSet:
    """zone_payload: [{"name": str, "points": [[x, y], ...]}, ...] with
    points normalised to 0..1. Replaces the zone set used by the next
    session started. Never written to disk."""
    if is_capturing:
        raise ValueError("Stop the current session before changing zones.")
    zones = []
    for i, z in enumerate(zone_payload):
        pts = [(float(p[0]), float(p[1])) for p in z["points"]]
        if len(pts) < 3:
            raise ValueError(f"Zone '{z.get('name') or i + 1}' needs at least 3 points.")
        name = str(z.get("name") or f"Zone {i + 1}")
        color = tuple(z["color"]) if z.get("color") else PALETTE[i % len(PALETTE)]
        zones.append(Zone(name=name, points=pts, color=color))
    if not zones:
        raise ValueError("At least one zone is required.")

    global current_zones
    with zones_lock:
        current_zones = ZoneSet(zones)
    return current_zones


def get_zone_frame_jpeg():
    """Grab one fresh frame from the configured source (without starting a
    session) so the frontend can let the user draw zones on it. Returns
    (jpeg_bytes, width, height)."""
    source = get_video_source()
    vs = VideoSource(source)
    try:
        frame = None
        for _ in range(50):
            ok, f = vs.read()
            if ok:
                frame = f
                break
            time.sleep(0.05)
        if frame is None:
            raise RuntimeError("Could not read a frame from the source.")
        ok, buf = cv2.imencode(".jpg", frame)
        if not ok:
            raise RuntimeError("Could not encode the frame.")
        return buf.tobytes(), frame.shape[1], frame.shape[0]
    finally:
        vs.release()


# ------------------------------------------------------------------
# Detector (heavy - loaded once, lazily)
# ------------------------------------------------------------------
def _ensure_detector() -> Detector:
    global detector
    if detector is None:
        with model_lock:
            if detector is None:
                detector = Detector()
    return detector


# ------------------------------------------------------------------
# Session lifecycle
# ------------------------------------------------------------------
def start_session() -> dict:
    global current_session, session_counter, is_capturing

    with zones_lock:
        zones_snapshot = ZoneSet(list(current_zones.zones))
    if len(zones_snapshot) == 0:
        raise ValueError("Draw at least one restricted zone before starting a session.")

    with session_lock:
        if is_capturing:
            raise ValueError("A session is already running. Stop it first.")
        session_counter += 1
        current_session = {
            "session_id": session_counter,
            "start_time": datetime.now(),
            "end_time": None,
            "zones": zones_snapshot,
            "events": [],
            "frame_count": 0,
            "total_breaches": 0,
            "video_path": None,
        }
        is_capturing = True

    _ensure_capture_thread()
    return get_status()


def stop_session() -> dict:
    global capture_thread_running
    with session_lock:
        if not is_capturing or current_session is None:
            raise ValueError("No active session to stop.")
    capture_thread_running = False
    if _capture_thread is not None:
        _capture_thread.join(timeout=5)
    # _capture_loop's `finally` block calls _finalize_session() once the
    # thread actually exits, so the session record is ready by the time we
    # get here.
    return get_status()


def _ensure_capture_thread() -> None:
    global _capture_thread
    if _capture_thread is None or not _capture_thread.is_alive():
        _capture_thread = threading.Thread(target=_capture_loop, daemon=True)
        _capture_thread.start()


def _finalize_session() -> None:
    global is_capturing
    with session_lock:
        if current_session is None:
            return
        current_session["end_time"] = datetime.now()
        is_capturing = False
        print(f"[restricted_zone_monitor] Session {current_session['session_id']} ended - "
              f"{current_session['total_breaches']} breach(es).")


def _capture_loop() -> None:
    global latest_frame, capture_thread_running, latest_person_count, latest_inside_count

    with session_lock:
        session = current_session
        zones = session["zones"]

    source = get_video_source()
    try:
        video = VideoSource(source)
    except Exception as e:  # noqa: BLE001
        print(f"[restricted_zone_monitor] Could not open source {source!r}: {e}")
        _finalize_session()
        return

    det = _ensure_detector()
    breach_mgr = BreachManager(
        zones, video.width, video.height, anchor=config.ANCHOR,
        enter_frames=config.ENTER_FRAMES, exit_frames=config.EXIT_FRAMES,
        track_ttl_s=config.TRACK_TTL_S, dup_radius=config.DUP_RADIUS_PX,
        dup_window_s=config.DUP_WINDOW_S, re_alert_cooldown_s=config.RE_ALERT_COOLDOWN_S,
    )
    banner = Banner()

    capture_thread_running = True
    frame_idx = 0
    session_id = session["session_id"]
    snap_dir = os.path.join(str(config.SNAPSHOT_DIR), str(session_id))
    os.makedirs(snap_dir, exist_ok=True)

    video_path = os.path.join(str(config.OUTPUT_DIR), f"{session_id}.mp4")
    writer = None
    try:
        writer = make_writer(video_path, video.width, video.height, video.fps)
        with session_lock:
            session["video_path"] = video_path
    except Exception as e:  # noqa: BLE001
        # A replayable video is a nice-to-have, not essential - the live
        # stream, breach log and report still work without it.
        print(f"[restricted_zone_monitor] Could not open output video writer: {e}")
        writer = None

    try:
        while capture_thread_running:
            ok, frame = video.read()
            if not ok:
                print("[restricted_zone_monitor] End of source / read failure - ending session.")
                break
            frame_idx += 1
            now = (frame_idx / video.fps) if video.is_file else time.time()

            dets = det(frame)
            events, inside_map = breach_mgr.update(dets, frame_idx, now=now)

            latest_person_count = sum(1 for d in dets if d.cls_name == "person")
            latest_inside_count = len(inside_map)

            active = {zn for zl in inside_map.values() for zn in zl}
            draw_zones(frame, zones, active)
            draw_detections(frame, dets, inside_map, anchor=config.ANCHOR)

            for ev in events:
                banner.trigger(f"BREACH: {ev.cls_name} #{ev.track_id} entered {ev.zone.name}")
                with session_lock:
                    event_no = len(session["events"]) + 1
                    snap_path = os.path.join(snap_dir, f"{event_no}_{ev.track_id}_{ev.cls_name}.jpg")
                    cv2.imwrite(snap_path, frame)
                    session["events"].append({
                        "track_id": ev.track_id,
                        "cls_name": ev.cls_name,
                        "zone": ev.zone.name,
                        "time": datetime.now().strftime("%H:%M:%S"),
                        "snapshot_path": snap_path,
                    })
                    session["total_breaches"] += 1

            banner.draw(frame)
            draw_hud(frame, 0.0, len(dets), len(inside_map), breach_mgr.total_breaches)

            if writer is not None:
                writer.write(frame)

            with session_lock:
                session["frame_count"] = frame_idx

            with frame_lock:
                latest_frame = frame

            if not video.is_file:
                time.sleep(0.01)
    finally:
        video.release()
        if writer is not None:
            writer.release()
        with frame_lock:
            latest_frame = None
        latest_person_count = 0
        latest_inside_count = 0
        capture_thread_running = False
        with session_lock:
            still_active = is_capturing and current_session is session
        if still_active:
            # The video simply ended (EOF) rather than an explicit /session/stop -
            # finalize automatically so the report/download flow can proceed.
            _finalize_session()


# ------------------------------------------------------------------
# Status / live feed
# ------------------------------------------------------------------
def get_status() -> dict:
    with session_lock:
        if current_session is None:
            return {"is_capturing": False, "has_report": False, "has_video": False}
        s = current_session
        return {
            "is_capturing": is_capturing,
            "session_id": s["session_id"],
            "start_time": s["start_time"].strftime("%H:%M:%S"),
            "frame_count": s["frame_count"],
            "total_breaches": s["total_breaches"],
            "persons_detected": latest_person_count,
            "currently_inside": latest_inside_count,
            "zones": [z.name for z in s["zones"].zones],
            "recent_events": [
                {"track_id": e["track_id"], "cls_name": e["cls_name"],
                 "zone": e["zone"], "time": e["time"]}
                for e in list(reversed(s["events"]))[:20]
            ],
            "has_report": not is_capturing,
            "has_video": not is_capturing and bool(s["video_path"]) and os.path.exists(s["video_path"]),
        }


def get_session_video_path() -> Optional[str]:
    """The just-finished session's annotated video, kept around after stop
    for replay - same lifetime as the report (see wipe_all_data). None while
    a session is still running or if no session has been recorded yet."""
    with session_lock:
        if current_session is None or is_capturing:
            return None
        path = current_session["video_path"]
        return path if path and os.path.exists(path) else None


def get_latest_jpeg():
    with frame_lock:
        frame = None if latest_frame is None else latest_frame.copy()
    if frame is None:
        return None
    ok, buf = cv2.imencode(".jpg", frame)
    return buf.tobytes() if ok else None


# ------------------------------------------------------------------
# Wipe - called after a report is downloaded, or on demand
# ------------------------------------------------------------------
def wipe_all_data() -> None:
    """Delete the finished session's breach snapshots and clear all session
    + zone state. Called once a report has been built (see reporting.py),
    from /wipe_data (user closed the download dialog / left the page), and
    on server startup/shutdown so nothing ever survives across restarts."""
    global current_session, is_capturing, current_zones

    with session_lock:
        if current_session is not None:
            snap_dir = os.path.join(str(config.SNAPSHOT_DIR), str(current_session["session_id"]))
            shutil.rmtree(snap_dir, ignore_errors=True)
            video_path = current_session["video_path"]
            if video_path:
                try:
                    os.remove(video_path)
                except OSError:
                    pass
        current_session = None
        is_capturing = False

    with zones_lock:
        current_zones = ZoneSet()

    print("[restricted_zone_monitor] Wiped session and zone data.")
