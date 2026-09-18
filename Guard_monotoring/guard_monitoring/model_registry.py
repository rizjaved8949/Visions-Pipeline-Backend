from __future__ import annotations

from collections.abc import Callable

from .contracts import ModuleResult, disabled, error_result, ok
from .health import ModuleHealthRegistry


class LazyModelRegistry:
    """Lazy-load heavy CV models only when a guard analysis job actually runs.

    A model-load failure is cached and reported to module health instead of crashing
    the existing FastAPI application process.
    """

    def __init__(self, cfg: dict, health: ModuleHealthRegistry | None = None):
        self.cfg = cfg
        self.health = health
        self._instances: dict[str, object] = {}
        self._load_errors: dict[str, Exception] = {}

    def _module_enabled(self, name: str, default: bool = True) -> bool:
        return bool(self.cfg.get("modules", {}).get(name, {}).get("enabled", default))

    def _get(self, name: str, builder: Callable[[], object], enabled: bool = True) -> ModuleResult:
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
        try:
            instance = builder()
            self._instances[name] = instance
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

        def build():
            from .models.detector_rfdetr import RFDETRPersonDetector

            return RFDETRPersonDetector(
                size=cfg.get("size", "medium"),
                weights=cfg.get("weights", ""),
                confidence=cfg.get("confidence", 0.35),
                target_label=cfg.get("target_label", ""),
                require_finetuned=cfg.get("require_finetuned", True),
            )

        return self._get("guard_detector", build, self._module_enabled("guard_detection", True))

    def tracker(self, frame_rate: float = 30.0) -> ModuleResult:
        cfg = self.cfg["tracker"]

        def build():
            from .models.tracker_bytetrack import GuardByteTracker

            return GuardByteTracker(cfg, frame_rate=frame_rate)

        return self._get("tracker", build, self._module_enabled("tracking", True))

    def pose(self) -> ModuleResult:
        cfg = self.cfg["models"]["pose"]

        def build():
            from .models.pose_yolo import YOLOPoseEstimator

            return YOLOPoseEstimator(
                weights=cfg["weights"],
                confidence=cfg.get("confidence", 0.30),
                imgsz=cfg.get("imgsz", 640),
            )

        return self._get("pose", build, self._module_enabled("pose", True))

    def phone(self) -> ModuleResult:
        cfg = self.cfg["models"]["phone"]

        def build():
            from .models.phone_yolo import YOLOPhoneDetector

            return YOLOPhoneDetector(
                weights=cfg["weights"],
                confidence=cfg.get("confidence", 0.20),
                imgsz=cfg.get("imgsz", 640),
                target_labels=cfg.get("target_labels", ["cell phone"]),
            )

        return self._get("phone", build, self._module_enabled("phone", True))

    def eyes(self) -> ModuleResult:
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
