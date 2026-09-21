from __future__ import annotations

import threading
from collections.abc import Callable
from typing import Any

from .contracts import ModuleResult, disabled, error_result, ok
from .health import ModuleHealthRegistry

# Heavy, stateless models (detector/pose/phone/eyes) are expensive to load
# (multi-second GPU/model-graph setup) but safe to share across every job and
# live session, since inference calls don't carry any per-session memory.
# Without this, every single uploaded video or "Connect camera" click paid the
# full load cost again - the dominant part of "why does this take so long".
# The tracker is deliberately excluded: it holds per-session track history and
# must never be shared between unrelated videos/sessions.
_GLOBAL_MODEL_CACHE: dict[tuple[str, tuple[Any, ...]], object] = {}
_GLOBAL_CACHE_LOCK = threading.Lock()


class LazyModelRegistry:
    """Lazy-load heavy CV models only when a guard analysis job actually runs.

    A model-load failure is cached (per registry instance, i.e. per job/session)
    and reported to module health instead of crashing the existing FastAPI
    application process. Successful loads are also cached process-wide (see
    _GLOBAL_MODEL_CACHE) so fixing a config issue (e.g. adding a missing
    weights file) takes effect on the very next job without a server restart,
    while still avoiding a redundant reload every time.
    """

    def __init__(self, cfg: dict, health: ModuleHealthRegistry | None = None):
        self.cfg = cfg
        self.health = health
        self._instances: dict[str, object] = {}
        self._load_errors: dict[str, Exception] = {}

    def _module_enabled(self, name: str, default: bool = True) -> bool:
        return bool(self.cfg.get("modules", {}).get(name, {}).get("enabled", default))

    def _get(
        self,
        name: str,
        builder: Callable[[], object],
        enabled: bool = True,
        *,
        cache_key: tuple[Any, ...] | None = None,
    ) -> ModuleResult:
        if not enabled:
            result = disabled()
            self._record(f"{name}_load", result)
            return result
        if name in self._instances:
            result = ok(self._instances[name], detail="cached")
            self._record(f"{name}_load", result)
            return result
        if name in self._load_errors:
            exc = self._load_errors[name]
            result = error_result(exc, detail=f"cached_load_error: {exc}")
            self._record(f"{name}_load", result)
            return result

        global_key = (name, cache_key) if cache_key is not None else None
        if global_key is not None:
            with _GLOBAL_CACHE_LOCK:
                cached = _GLOBAL_MODEL_CACHE.get(global_key)
            if cached is not None:
                self._instances[name] = cached
                result = ok(cached, detail="cached")
                self._record(f"{name}_load", result)
                return result

        try:
            instance = builder()
            self._instances[name] = instance
            if global_key is not None:
                with _GLOBAL_CACHE_LOCK:
                    _GLOBAL_MODEL_CACHE[global_key] = instance
            result = ok(instance, detail="loaded")
        except Exception as exc:
            self._load_errors[name] = exc
            result = error_result(exc, detail=f"model_load_failed: {exc}")
        self._record(f"{name}_load", result)
        return result

    def _record(self, name: str, result: ModuleResult) -> None:
        if self.health is not None:
            self.health.record(name, result)

    def guard_detector(self) -> ModuleResult:
        cfg = self.cfg["models"]["guard_detector"]
        size = cfg.get("size", "medium")
        weights = cfg.get("weights", "")
        confidence = cfg.get("confidence", 0.35)
        target_label = cfg.get("target_label", "")
        require_finetuned = cfg.get("require_finetuned", True)

        def build():
            from .models.detector_rfdetr import RFDETRPersonDetector

            return RFDETRPersonDetector(
                size=size,
                weights=weights,
                confidence=confidence,
                target_label=target_label,
                require_finetuned=require_finetuned,
            )

        return self._get(
            "guard_detector",
            build,
            self._module_enabled("guard_detection", True),
            cache_key=(size, weights, confidence, target_label, require_finetuned),
        )

    def tracker(self, frame_rate: float = 30.0) -> ModuleResult:
        # Never cached globally - a tracker holds per-job/session track history
        # and must start fresh for every video/live session.
        cfg = self.cfg["tracker"]

        def build():
            from .models.tracker_bytetrack import GuardByteTracker

            return GuardByteTracker(cfg, frame_rate=frame_rate)

        return self._get("tracker", build, self._module_enabled("tracking", True))

    def pose(self) -> ModuleResult:
        cfg = self.cfg["models"]["pose"]
        weights = cfg["weights"]
        confidence = cfg.get("confidence", 0.30)
        imgsz = cfg.get("imgsz", 640)

        def build():
            from .models.pose_yolo import YOLOPoseEstimator

            return YOLOPoseEstimator(weights=weights, confidence=confidence, imgsz=imgsz)

        return self._get(
            "pose",
            build,
            self._module_enabled("pose", True),
            cache_key=(weights, confidence, imgsz),
        )

    def phone(self) -> ModuleResult:
        cfg = self.cfg["models"]["phone"]
        weights = cfg["weights"]
        confidence = cfg.get("confidence", 0.20)
        imgsz = cfg.get("imgsz", 640)
        target_labels = tuple(cfg.get("target_labels", ["cell phone"]))

        def build():
            from .models.phone_yolo import YOLOPhoneDetector

            return YOLOPhoneDetector(
                weights=weights,
                confidence=confidence,
                imgsz=imgsz,
                target_labels=list(target_labels),
            )

        return self._get(
            "phone",
            build,
            self._module_enabled("phone", True),
            cache_key=(weights, confidence, imgsz, target_labels),
        )

    def eyes(self) -> ModuleResult:
        # Not cached globally, unlike the other models: MediaPipe's FaceLandmarker
        # runs in VIDEO mode here, which requires strictly increasing timestamps
        # for the lifetime of the instance. Every job/session restarts its own
        # timestamp counter at 0, so sharing this instance across two videos
        # raises "Input timestamp must be monotonically increasing" and silently
        # fails eye analysis on every job after the first. It's cheap to build
        # (no GPU weights) so reloading it per job/session isn't a real cost.
        cfg = self.cfg["models"]["eyes"]
        enabled = bool(cfg.get("enabled", True)) and self._module_enabled("eyes", True)

        def build():
            from .models.face_eyes_mediapipe import MediaPipeEyeAnalyzer

            return MediaPipeEyeAnalyzer(
                ear_closed_threshold=cfg.get("ear_closed_threshold", 0.20),
                min_face_pixels=cfg.get("min_face_pixels", 48),
                min_eye_width_pixels=cfg.get("min_eye_width_pixels", 4),
                min_eye_symmetry_ratio=cfg.get("min_eye_symmetry_ratio", 0.30),
                face_landmarker_model=cfg.get("face_landmarker_model", ""),
            )

        return self._get("eyes", build, enabled)

    def reset_tracker(self) -> None:
        tracker = self._instances.get("tracker")
        if tracker is not None and hasattr(tracker, "reset"):
            tracker.reset()
