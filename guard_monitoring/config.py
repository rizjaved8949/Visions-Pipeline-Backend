from __future__ import annotations

import json
import os
from copy import deepcopy
from pathlib import Path
from typing import Any

try:
    from dotenv import load_dotenv
except ImportError:  # Existing app can still start from process environment.
    def load_dotenv(*args, **kwargs):
        return False

# Load the single repository-level .env file if present. Existing application keys
# remain untouched; guard settings use the GUARD_ prefix.
load_dotenv()


def _env(name: str, default: str) -> str:
    return os.getenv(name, default).strip()


def _bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _int(name: str, default: int) -> int:
    raw = os.getenv(name)
    return default if raw is None or raw.strip() == "" else int(raw)


def _float(name: str, default: float) -> float:
    raw = os.getenv(name)
    return default if raw is None or raw.strip() == "" else float(raw)


def _json(name: str, default: Any) -> Any:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return deepcopy(default)
    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{name} must contain valid JSON") from exc


def guard_enabled() -> bool:
    return _bool("GUARD_MONITORING_ENABLED", False)


def load_config(
    *,
    source: str | None = None,
    output_dir: str | None = None,
    max_frames: int | None = None,
    display: bool | None = None,
) -> dict[str, Any]:
    """Build the complete pipeline configuration from the single .env file.

    No YAML profile is required. Runtime-only values such as source/output can be
    supplied by the API worker or CLI without mutating environment variables.
    """

    duty_zone = _json(
        "GUARD_DUTY_ZONE_JSON",
        [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]],
    )
    patrol_zones = _json("GUARD_PATROL_ZONES_JSON", [])
    target_labels = _json("GUARD_PHONE_LABELS_JSON", ["cell phone", "mobile phone", "phone"])

    cfg: dict[str, Any] = {
        "project": {
            "name": "guard-monitoring",
            "camera_id": _env("GUARD_CAMERA_ID", "camera-01"),
        },
        "input": {
            "source": source or _env("GUARD_DEFAULT_SOURCE", "local_data/guard/uploads/sample.mp4"),
            "max_frames": max_frames,
        },
        "output": {
            "directory": output_dir or _env("GUARD_OUTPUT_DIR", "local_data/guard/outputs/manual-run"),
            "annotated_video": "annotated.mp4",
            "frame_log": "frames.jsonl",
            "event_log": "events.jsonl",
            "summary": "summary.json",
            "module_health": "module_health.json",
            "display": _bool("GUARD_DISPLAY", False) if display is None else bool(display),
        },
        "models": {
            "guard_detector": {
                "size": _env("GUARD_RFDETR_SIZE", "medium"),
                "weights": _env("GUARD_RFDETR_WEIGHTS", ""),
                "confidence": _float("GUARD_DETECTION_CONFIDENCE", 0.35),
                "target_label": _env("GUARD_RFDETR_TARGET_LABEL", ""),
                "require_finetuned": _bool("GUARD_REQUIRE_FINETUNED_GUARD_MODEL", True),
            },
            "pose": {
                "weights": _env("GUARD_POSE_WEIGHTS", "guard_monitoring/weights/yolo26m-pose.pt"),
                "confidence": _float("GUARD_POSE_CONFIDENCE", 0.30),
                "imgsz": _int("GUARD_POSE_IMAGE_SIZE", 640),
                "every_n_frames": _int("GUARD_POSE_EVERY_N_FRAMES", 2),
                "cache_max_frames": _int("GUARD_POSE_CACHE_FRAMES", 3),
            },
            "phone": {
                "weights": _env("GUARD_PHONE_WEIGHTS", "guard_monitoring/weights/yolo26m.pt"),
                "confidence": _float("GUARD_PHONE_CONFIDENCE", 0.20),
                "imgsz": _int("GUARD_PHONE_IMAGE_SIZE", 640),
                "every_n_frames": _int("GUARD_PHONE_EVERY_N_FRAMES", 2),
                "cache_max_frames": _int("GUARD_PHONE_CACHE_FRAMES", 3),
                "target_labels": target_labels,
            },
            "eyes": {
                "enabled": _bool("GUARD_EYES_ENABLED", True),
                "face_landmarker_model": _env("GUARD_FACE_LANDMARKER_MODEL", ""),
                "every_n_frames": _int("GUARD_EYES_EVERY_N_FRAMES", 2),
                "cache_max_frames": _int("GUARD_EYES_CACHE_FRAMES", 3),
                "min_face_pixels": _int("GUARD_MIN_FACE_PIXELS", 48),
                "ear_closed_threshold": _float("GUARD_EAR_CLOSED_THRESHOLD", 0.20),
                "min_eye_width_pixels": _int("GUARD_MIN_EYE_WIDTH_PIXELS", 4),
                "min_eye_symmetry_ratio": _float("GUARD_MIN_EYE_SYMMETRY_RATIO", 0.30),
            },
        },
        "tracker": {
            "lost_track_buffer": _int("GUARD_TRACK_LOST_BUFFER", 60),
            "minimum_consecutive_frames": _int("GUARD_TRACK_MIN_CONSECUTIVE", 2),
            "track_activation_threshold": _float("GUARD_TRACK_ACTIVATION_THRESHOLD", 0.45),
            "high_conf_det_threshold": _float("GUARD_TRACK_HIGH_CONF_THRESHOLD", 0.35),
            "minimum_iou_threshold": _float("GUARD_TRACK_MIN_IOU", 0.10),
        },
        "guard_selection": {
            "duty_zone": duty_zone,
            "confirm_seconds": _float("GUARD_SELECTION_CONFIRM_SECONDS", 1.0),
            "release_seconds": _float("GUARD_SELECTION_RELEASE_SECONDS", 5.0),
            "presence_grace_seconds": _float("GUARD_PRESENCE_GRACE_SECONDS", 2.0),
            "manual_track_id": None,
        },
        "movement": {
            "history_seconds": _float("GUARD_MOVEMENT_HISTORY_SECONDS", 5.0),
            "stationary_radius_ratio": _float("GUARD_STATIONARY_RADIUS_RATIO", 0.035),
            "minimum_history_seconds": _float("GUARD_MOVEMENT_MIN_HISTORY_SECONDS", 2.0),
            "patrol_zones": patrol_zones,
        },
        "pose_logic": {
            "keypoint_confidence": _float("GUARD_KEYPOINT_CONFIDENCE", 0.25),
            "sitting_knee_angle_max_deg": _float("GUARD_SITTING_KNEE_MAX_DEG", 140.0),
            "standing_knee_angle_min_deg": _float("GUARD_STANDING_KNEE_MIN_DEG", 155.0),
            "torso_lean_deg": _float("GUARD_TORSO_LEAN_DEG", 28.0),
            "head_down_ratio": _float("GUARD_HEAD_DOWN_RATIO", 0.32),
        },
        "phone_logic": {
            "guard_box_expand_ratio": _float("GUARD_PHONE_BOX_EXPAND_RATIO", 0.10),
            "hand_distance_ratio": _float("GUARD_PHONE_HAND_DISTANCE_RATIO", 0.22),
            "head_distance_ratio": _float("GUARD_PHONE_HEAD_DISTANCE_RATIO", 0.20),
        },
        "sleep_logic": {
            "perclos_window_seconds": _float("GUARD_PERCLOS_WINDOW_SECONDS", 30.0),
            "perclos_threshold": _float("GUARD_PERCLOS_THRESHOLD", 0.70),
            "sleep_score_threshold": _float("GUARD_SLEEP_SCORE_THRESHOLD", 0.55),
            "suppress_when_phone_active": _bool("GUARD_SLEEP_SUPPRESS_PHONE", True),
            "allow_degraded_rule_trigger": _bool("GUARD_ALLOW_DEGRADED_SLEEP_ALERT", False),
        },
        "rules": {
            "warmup_seconds": _float("GUARD_RULE_WARMUP_SECONDS", 5.0),
            "sleep_seconds": _float("GUARD_SLEEP_SECONDS", 300.0),
            "phone_seconds": _float("GUARD_PHONE_SECONDS", 600.0),
            "stationary_seconds": _float("GUARD_STATIONARY_SECONDS", 1800.0),
            "absence_seconds": _float("GUARD_ABSENCE_SECONDS", 120.0),
            "grace_seconds": {
                "sleep": _float("GUARD_SLEEP_GRACE_SECONDS", 10.0),
                "phone": _float("GUARD_PHONE_GRACE_SECONDS", 3.0),
                "stationary": _float("GUARD_STATIONARY_GRACE_SECONDS", 3.0),
                "absence": _float("GUARD_ABSENCE_GRACE_SECONDS", 1.0),
            },
        },
        "visualization": {
            "draw_zones": _bool("GUARD_DRAW_ZONES", True),
            "draw_pose": _bool("GUARD_DRAW_POSE", True),
            "draw_phone": _bool("GUARD_DRAW_PHONE", True),
            # The debug HUD (posture/phone/eyes/sleep/module-health text) is off by
            # default - that information is served to the frontend via the
            # live-state/summary APIs instead of being burned into the video.
            "draw_panel": _bool("GUARD_DRAW_PANEL", False),
        },
        "modules": {
            "guard_detection": {"enabled": _bool("GUARD_DETECTION_ENABLED", True)},
            "tracking": {"enabled": _bool("GUARD_TRACKING_ENABLED", True)},
            "pose": {"enabled": _bool("GUARD_POSE_ENABLED", True)},
            "phone": {"enabled": _bool("GUARD_PHONE_ENABLED", True)},
            "eyes": {"enabled": _bool("GUARD_EYES_ENABLED", True)},
        },
        "fault_tolerance": {
            "continue_on_frame_error": _bool("GUARD_CONTINUE_ON_FRAME_ERROR", True),
            "unknown_evidence_policy": "pause_rule_timer",
        },
        "api": {
            "progress_every_frames": _int("GUARD_PROGRESS_EVERY_FRAMES", 25),
        },
    }

    if _bool("GUARD_TEST_MODE", False):
        cfg["rules"].update(
            sleep_seconds=_float("GUARD_TEST_SLEEP_SECONDS", 5.0),
            phone_seconds=_float("GUARD_TEST_PHONE_SECONDS", 5.0),
            stationary_seconds=_float("GUARD_TEST_STATIONARY_SECONDS", 8.0),
            absence_seconds=_float("GUARD_TEST_ABSENCE_SECONDS", 5.0),
            warmup_seconds=0.0,
        )

    validate_config(cfg)
    return cfg


def validate_config(cfg: dict[str, Any]) -> None:
    _validate_polygon(cfg["guard_selection"]["duty_zone"], "GUARD_DUTY_ZONE_JSON")
    for idx, zone in enumerate(cfg["movement"].get("patrol_zones", [])):
        if not isinstance(zone, dict) or "name" not in zone or "polygon" not in zone:
            raise ValueError(f"GUARD_PATROL_ZONES_JSON[{idx}] requires name and polygon")
        _validate_polygon(zone["polygon"], f"GUARD_PATROL_ZONES_JSON[{idx}].polygon")
    for key in ("sleep_seconds", "phone_seconds", "stationary_seconds", "absence_seconds"):
        if float(cfg["rules"][key]) < 0:
            raise ValueError(f"rules.{key} must be >= 0")


def _validate_polygon(points: list[list[float]], name: str) -> None:
    if not isinstance(points, list) or len(points) < 3:
        raise ValueError(f"{name} must contain at least 3 [x,y] points")
    for point in points:
        if not isinstance(point, list) or len(point) != 2:
            raise ValueError(f"{name} points must be [x,y]")
        x, y = float(point[0]), float(point[1])
        if not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0):
            raise ValueError(f"{name} uses normalized coordinates in [0,1]")


def runtime_paths() -> dict[str, Path]:
    upload_dir = Path(_env("GUARD_UPLOAD_DIR", "local_data/guard/uploads"))
    output_dir = Path(_env("GUARD_OUTPUT_ROOT", "local_data/guard/outputs"))
    db_path = Path(_env("GUARD_JOB_DB", "local_data/guard/jobs.sqlite3"))
    return {"upload_dir": upload_dir, "output_root": output_dir, "db_path": db_path}
