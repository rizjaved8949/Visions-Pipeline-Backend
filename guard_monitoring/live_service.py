from __future__ import annotations

import atexit
import os
import threading
import time
import uuid
from collections import deque
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Iterator
from urllib.parse import urlsplit, urlunsplit

from .config import load_config

_RULE_NAMES = ("sleep", "phone", "stationary", "absence")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    return default if raw is None or raw.strip() == "" else float(raw)


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    return default if raw is None or raw.strip() == "" else int(raw)


def _redact_source(source: str) -> str:
    """Hide embedded credentials when a live-session record is returned by the API."""
    try:
        parts = urlsplit(source)
    except Exception:
        return source
    if not parts.scheme or not parts.netloc or "@" not in parts.netloc:
        return source
    host = parts.netloc.rsplit("@", 1)[-1]
    return urlunsplit((parts.scheme, f"***:***@{host}", parts.path, parts.query, parts.fragment))


def _validate_rule_settings(settings: dict[str, float] | None) -> dict[str, float]:
    if not settings:
        return {}
    result: dict[str, float] = {}
    for name, value in settings.items():
        if name not in _RULE_NAMES:
            raise ValueError(f"Unsupported live rule setting: {name}")
        number = float(value)
        if number < 0:
            raise ValueError(f"{name}_seconds must be >= 0")
        result[name] = number
    return result


@dataclass
class _LiveSession:
    session_id: str
    camera_id: str
    source: str
    analysis_fps: float
    reconnect_seconds: float
    jpeg_quality: int
    created_at: str = field(default_factory=_utc_now)
    status: str = "starting"
    error: str | None = None
    frames_processed: int = 0
    reconnect_count: int = 0
    last_frame_at: str | None = None
    latest_state: dict[str, Any] | None = None
    latest_jpeg: bytes | None = None
    frame_version: int = 0
    stats: dict[str, int] = field(
        default_factory=lambda: {
            "visible_frames": 0,
            "stationary_frames": 0,
            "sleep_candidate_frames": 0,
            "phone_use_frames": 0,
            "absence_frames": 0,
        }
    )
    events: deque = field(default_factory=lambda: deque(maxlen=1000))
    module_health: dict[str, Any] = field(default_factory=dict)
    settings: dict[str, float] = field(default_factory=dict)
    stop_event: threading.Event = field(default_factory=threading.Event, repr=False)
    lock: threading.RLock = field(default_factory=threading.RLock, repr=False)
    thread: threading.Thread | None = field(default=None, repr=False)
    started_monotonic: float | None = field(default=None, repr=False)
    rules: Any = field(default=None, repr=False)

    def public(self) -> dict[str, Any]:
        with self.lock:
            uptime = 0.0
            if self.started_monotonic is not None:
                uptime = max(0.0, time.monotonic() - self.started_monotonic)
            return {
                "session_id": self.session_id,
                "camera_id": self.camera_id,
                "source": _redact_source(self.source),
                "status": self.status,
                "analysis_fps": self.analysis_fps,
                "frames_processed": self.frames_processed,
                "reconnect_count": self.reconnect_count,
                "created_at": self.created_at,
                "last_frame_at": self.last_frame_at,
                "uptime_seconds": round(uptime, 3),
                "settings": {f"{name}_seconds": float(value) for name, value in self.settings.items()},
                "error": self.error,
                "stats": dict(self.stats),
            }


class GuardLiveService:
    """In-memory real-time guard-monitoring sessions.

    This service is additive: the existing uploaded-video GuardJobService is unchanged.
    Heavy computer-vision modules are imported only inside the live worker thread so
    importing the FastAPI router still does not initialize RF-DETR/YOLO/MediaPipe.

    Sessions use wall-clock monotonic time for alert durations. This is important for
    real cameras because inference FPS can be lower than the camera's native FPS.
    """

    def __init__(self):
        self._lock = threading.RLock()
        self._sessions: dict[str, _LiveSession] = {}
        self.max_sessions = max(1, _env_int("GUARD_LIVE_MAX_SESSIONS", 1))
        self._latest_session_id: str | None = None
        atexit.register(self.stop_all)

    def current_session_id(self) -> str | None:
        """The most recently started session, regardless of whether it is still
        running. Lets a caller reach "whatever is live right now" through a
        stable URL instead of needing the session_id up front (mirrors the
        Attendance module's single fixed /api/video_feed endpoint)."""
        with self._lock:
            return self._latest_session_id

    def start(
        self,
        *,
        source: str,
        camera_id: str | None = None,
        analysis_fps: float | None = None,
        reconnect_seconds: float | None = None,
        rule_settings: dict[str, float] | None = None,
    ) -> dict[str, Any]:
        source = str(source).strip()
        if not source:
            raise ValueError("source is required")

        target_fps = float(
            analysis_fps
            if analysis_fps is not None
            else _env_float("GUARD_LIVE_ANALYSIS_FPS", 3.0)
        )
        if not (0.2 <= target_fps <= 30.0):
            raise ValueError("analysis_fps must be between 0.2 and 30")

        reconnect = float(
            reconnect_seconds
            if reconnect_seconds is not None
            else _env_float("GUARD_LIVE_RECONNECT_SECONDS", 2.0)
        )
        if reconnect < 0.1:
            raise ValueError("reconnect_seconds must be >= 0.1")

        requested_settings = _validate_rule_settings(rule_settings)
        # Resolve the same rule defaults used by uploaded-video jobs so the live API
        # reports complete, effective settings even when the request omits them.
        base_cfg = load_config(
            source=source,
            output_dir="local_data/guard/live",
            display=False,
        )
        settings = {
            name: float(base_cfg["rules"][f"{name}_seconds"])
            for name in _RULE_NAMES
        }
        settings.update(requested_settings)

        with self._lock:
            active = [
                s for s in self._sessions.values()
                if s.status in {"starting", "running", "reconnecting"}
            ]
            if len(active) >= self.max_sessions:
                raise RuntimeError(
                    f"Maximum live sessions reached ({self.max_sessions}). "
                    "Stop an existing live session first."
                )

            session_id = uuid.uuid4().hex
            session = _LiveSession(
                session_id=session_id,
                camera_id=(camera_id or str(base_cfg["project"].get("camera_id", "camera-01"))).strip()
                or "camera-01",
                source=source,
                analysis_fps=target_fps,
                reconnect_seconds=reconnect,
                jpeg_quality=max(40, min(95, _env_int("GUARD_LIVE_JPEG_QUALITY", 80))),
                settings=settings,
            )
            self._sessions[session_id] = session
            self._latest_session_id = session_id
            thread = threading.Thread(
                target=self._run,
                args=(session,),
                name=f"guard-live-{session_id[:8]}",
                daemon=True,
            )
            session.thread = thread
            thread.start()
            return session.public()

    def list(self) -> list[dict[str, Any]]:
        with self._lock:
            sessions = list(self._sessions.values())
        return [session.public() for session in sessions]

    def get(self, session_id: str) -> _LiveSession | None:
        with self._lock:
            return self._sessions.get(session_id)

    def state(self, session_id: str) -> dict[str, Any] | None:
        session = self.get(session_id)
        if session is None:
            return None
        public = session.public()
        with session.lock:
            current = deepcopy(session.latest_state)
        public["state"] = current
        return public

    def events(self, session_id: str) -> list[dict[str, Any]] | None:
        session = self.get(session_id)
        if session is None:
            return None
        with session.lock:
            return [deepcopy(item) for item in session.events]

    def module_health(self, session_id: str) -> dict[str, Any] | None:
        session = self.get(session_id)
        if session is None:
            return None
        with session.lock:
            return deepcopy(session.module_health)

    def update_settings(self, session_id: str, settings: dict[str, float]) -> dict[str, Any] | None:
        session = self.get(session_id)
        if session is None:
            return None
        updates = _validate_rule_settings(settings)
        with session.lock:
            session.settings.update(updates)
            if session.rules is not None:
                for name, value in updates.items():
                    session.rules.thresholds[name] = float(value)
        return session.public()

    def stop(self, session_id: str, *, join_timeout: float = 5.0) -> dict[str, Any] | None:
        session = self.get(session_id)
        if session is None:
            return None
        session.stop_event.set()
        thread = session.thread
        if thread is not None and thread.is_alive() and thread is not threading.current_thread():
            thread.join(timeout=max(0.0, join_timeout))
        with session.lock:
            if session.status not in {"failed", "stopped"}:
                session.status = "stopping" if thread is not None and thread.is_alive() else "stopped"
        return session.public()

    def stop_all(self) -> None:
        with self._lock:
            ids = list(self._sessions)
        for session_id in ids:
            try:
                self.stop(session_id, join_timeout=1.0)
            except Exception:
                pass

    def mjpeg(self, session_id: str) -> Iterator[bytes]:
        session = self.get(session_id)
        if session is None:
            return
        last_version = -1
        while True:
            with session.lock:
                jpeg = session.latest_jpeg
                version = session.frame_version
                status = session.status
            if jpeg is not None and version != last_version:
                last_version = version
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n"
                    b"Cache-Control: no-cache\r\n\r\n"
                    + jpeg
                    + b"\r\n"
                )
            if status in {"stopped", "failed"}:
                break
            time.sleep(0.05)

    def _run(self, session: _LiveSession) -> None:
        # Heavy imports intentionally live here to keep FastAPI import lightweight.
        import cv2

        from .geometry import normalized_polygon_to_pixels
        from .io import open_capture, video_metadata
        from .monitoring.guard_selector import GuardSelector
        from .monitoring.movement import MovementMonitor
        from .monitoring.rules import TimedRuleEngine
        from .pipeline import GuardMonitoringPipeline

        cap = None
        session.started_monotonic = time.monotonic()
        cfg = load_config(source=session.source, output_dir="local_data/guard/live")
        cfg["project"]["camera_id"] = session.camera_id
        for name, value in session.settings.items():
            cfg["rules"][f"{name}_seconds"] = float(value)

        pipeline = GuardMonitoringPipeline(cfg)
        frame_idx = 0
        next_due = 0.0
        runtime = None

        try:
            while not session.stop_event.is_set():
                if cap is None or not cap.isOpened():
                    try:
                        cap = open_capture(session.source)
                        # Best-effort low-latency request; unsupported backends simply ignore it.
                        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                        width, height, native_fps = video_metadata(cap)
                        pipeline.video_fps = session.analysis_fps
                        runtime = self._build_runtime(
                            pipeline,
                            cfg,
                            width,
                            height,
                            session.analysis_fps,
                            GuardSelector,
                            MovementMonitor,
                            TimedRuleEngine,
                            normalized_polygon_to_pixels,
                        )
                        with session.lock:
                            # Re-apply settings that may have been PATCHed while the
                            # camera was disconnected or before the first frame arrived.
                            for name, value in session.settings.items():
                                runtime["rules"].thresholds[name] = float(value)
                            session.rules = runtime["rules"]
                            session.status = "running"
                            session.error = None
                    except Exception as exc:
                        if cap is not None:
                            cap.release()
                            cap = None
                        with session.lock:
                            session.status = "reconnecting"
                            session.error = f"{type(exc).__name__}: {exc}"
                            session.reconnect_count += 1
                        session.stop_event.wait(session.reconnect_seconds)
                        continue

                ok_read, frame = cap.read()
                if not ok_read:
                    cap.release()
                    cap = None
                    runtime = None
                    with session.lock:
                        session.status = "reconnecting"
                        session.error = "Camera read failed; reconnecting"
                        session.reconnect_count += 1
                    session.stop_event.wait(session.reconnect_seconds)
                    continue

                elapsed = time.monotonic() - session.started_monotonic
                if elapsed < next_due:
                    continue
                next_due = elapsed + (1.0 / session.analysis_fps)

                if runtime is None:
                    continue

                try:
                    result = pipeline._process_frame(
                        frame=frame,
                        frame_idx=frame_idx,
                        now=elapsed,
                        selector=runtime["selector"],
                        movement_monitor=runtime["movement_monitor"],
                        rules=runtime["rules"],
                        duty_zone=runtime["duty_zone"],
                        patrol_zones=runtime["patrol_zones"],
                    )
                    state = deepcopy(result["frame_log"])
                    state.update(
                        {
                            "session_id": session.session_id,
                            "camera_id": session.camera_id,
                            "wall_time": _utc_now(),
                            "uptime_seconds": round(elapsed, 3),
                            "rule_durations_seconds": {
                                name: round(runtime["rules"].active_duration(name, elapsed), 3)
                                for name in _RULE_NAMES
                            },
                        }
                    )
                    ok_encode, encoded = cv2.imencode(
                        ".jpg",
                        result["annotated"],
                        [int(cv2.IMWRITE_JPEG_QUALITY), session.jpeg_quality],
                    )
                    jpeg = encoded.tobytes() if ok_encode else None

                    fl = result["frame_log"]
                    mv = fl.get("movement") or {}
                    sl = fl.get("sleep") or {}
                    ph = fl.get("phone") or {}

                    with session.lock:
                        session.status = "running"
                        session.error = None
                        session.frames_processed += 1
                        session.last_frame_at = state["wall_time"]
                        session.latest_state = state
                        session.module_health = pipeline.health.snapshot()
                        session.rules = runtime["rules"]
                        if result["guard_seen_now"]:
                            session.stats["visible_frames"] += 1
                        if mv.get("stationary"):
                            session.stats["stationary_frames"] += 1
                        if sl.get("candidate"):
                            session.stats["sleep_candidate_frames"] += 1
                        if ph.get("usage") in ("call", "screen_use"):
                            session.stats["phone_use_frames"] += 1
                        if fl.get("present") is False:
                            session.stats["absence_frames"] += 1
                        if jpeg is not None:
                            session.latest_jpeg = jpeg
                            session.frame_version += 1
                        for event in result["events"]:
                            enriched = deepcopy(event)
                            enriched["session_id"] = session.session_id
                            enriched["wall_time"] = state["wall_time"]
                            session.events.append(enriched)
                    frame_idx += 1
                except Exception as exc:
                    # A single orchestration failure must not permanently kill the camera loop.
                    with session.lock:
                        session.error = f"FramePipelineError: {type(exc).__name__}: {exc}"
                    frame_idx += 1

        except Exception as exc:
            with session.lock:
                session.status = "failed"
                session.error = f"{type(exc).__name__}: {exc}"
        finally:
            if cap is not None:
                cap.release()
            with session.lock:
                if session.status != "failed":
                    session.status = "stopped"

    @staticmethod
    def _build_runtime(
        pipeline,
        cfg,
        width,
        height,
        analysis_fps,
        GuardSelector,
        MovementMonitor,
        TimedRuleEngine,
        normalized_polygon_to_pixels,
    ) -> dict[str, Any]:
        duty_zone = normalized_polygon_to_pixels(
            cfg["guard_selection"]["duty_zone"], width, height
        )
        patrol_zones = [
            (z["name"], normalized_polygon_to_pixels(z["polygon"], width, height))
            for z in cfg["movement"].get("patrol_zones", [])
        ]
        sel_cfg = cfg["guard_selection"]
        selector = GuardSelector(
            duty_zone,
            confirm_seconds=sel_cfg.get("confirm_seconds", 1.0),
            release_seconds=sel_cfg.get("release_seconds", 5.0),
            presence_grace_seconds=sel_cfg.get("presence_grace_seconds", 2.0),
            manual_track_id=sel_cfg.get("manual_track_id"),
        )
        move_cfg = cfg["movement"]
        movement_monitor = MovementMonitor(
            history_seconds=move_cfg.get("history_seconds", 5.0),
            stationary_radius_ratio=move_cfg.get("stationary_radius_ratio", 0.035),
            minimum_history_seconds=move_cfg.get("minimum_history_seconds", 2.0),
            patrol_zones=patrol_zones,
        )
        rules = TimedRuleEngine(pipeline.camera_id, cfg["rules"])
        pipeline.video_fps = analysis_fps
        return {
            "duty_zone": duty_zone,
            "patrol_zones": patrol_zones,
            "selector": selector,
            "movement_monitor": movement_monitor,
            "rules": rules,
        }


_live_service: GuardLiveService | None = None
_live_service_lock = threading.Lock()


def get_live_service() -> GuardLiveService:
    global _live_service
    if _live_service is None:
        with _live_service_lock:
            if _live_service is None:
                _live_service = GuardLiveService()
    return _live_service
