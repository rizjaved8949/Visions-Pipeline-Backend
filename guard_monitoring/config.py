from __future__ import annotations

import json
import math
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
                "confidence": _float("GUARD_DETECTION_CONFIDENCE", 0.25),
                "target_label": _env("GUARD_RFDETR_TARGET_LABEL", ""),
                "require_finetuned": _bool("GUARD_REQUIRE_FINETUNED_GUARD_MODEL", True),
            },
            "pose": {
                "weights": _env("GUARD_POSE_WEIGHTS", "guard_monitoring/weights/yolo26m-pose.pt"),
                "confidence": _float("GUARD_POSE_CONFIDENCE", 0.20),
                "imgsz": _int("GUARD_POSE_IMAGE_SIZE", 640),
                "every_n_frames": _int("GUARD_POSE_EVERY_N_FRAMES", 2),
                "cache_max_frames": _int("GUARD_POSE_CACHE_FRAMES", 3),
                "cache_max_seconds": _float("GUARD_POSE_CACHE_SECONDS", 0.75),
            },
            "phone": {
                "weights": _env("GUARD_PHONE_WEIGHTS", "guard_monitoring/weights/yolo26m.pt"),
                "confidence": _float("GUARD_PHONE_CONFIDENCE", 0.20),
                "imgsz": _int("GUARD_PHONE_IMAGE_SIZE", 640),
                "every_n_frames": _int("GUARD_PHONE_EVERY_N_FRAMES", 2),
                "cache_max_frames": _int("GUARD_PHONE_CACHE_FRAMES", 3),
                "cache_max_seconds": _float("GUARD_PHONE_CACHE_SECONDS", 0.75),
                "target_labels": target_labels,
            },
            "eyes": {
                "enabled": _bool("GUARD_EYES_ENABLED", True),
                "face_landmarker_model": _env("GUARD_FACE_LANDMARKER_MODEL", ""),
                "every_n_frames": _int("GUARD_EYES_EVERY_N_FRAMES", 2),
                "cache_max_frames": _int("GUARD_EYES_CACHE_FRAMES", 3),
                "cache_max_seconds": _float("GUARD_EYES_CACHE_SECONDS", 0.75),
                "min_face_pixels": _int("GUARD_MIN_FACE_PIXELS", 48),
                "ear_closed_threshold": _float("GUARD_EAR_CLOSED_THRESHOLD", 0.20),
                "min_eye_width_pixels": _int("GUARD_MIN_EYE_WIDTH_PIXELS", 4),
                "min_eye_symmetry_ratio": _float("GUARD_MIN_EYE_SYMMETRY_RATIO", 0.30),
            },
        },
        "tracker": {
            "lost_track_buffer": _int("GUARD_TRACK_LOST_BUFFER", 120),
            "minimum_consecutive_frames": _int("GUARD_TRACK_MIN_CONSECUTIVE", 2),
            "track_activation_threshold": _float("GUARD_TRACK_ACTIVATION_THRESHOLD", 0.45),
            "high_conf_det_threshold": _float("GUARD_TRACK_HIGH_CONF_THRESHOLD", 0.35),
            "minimum_iou_threshold": _float("GUARD_TRACK_MIN_IOU", 0.10),
            # Optional time-based override; legacy buffer remains the default.
            "lost_track_seconds": (
                _float("GUARD_TRACK_LOST_SECONDS", 4.0)
                if os.getenv("GUARD_TRACK_LOST_SECONDS", "").strip() else None
            ),
        },
        "guard_selection": {
            "duty_zone": duty_zone,
            "confirm_seconds": _float("GUARD_SELECTION_CONFIRM_SECONDS", 1.0),
            "release_seconds": _float("GUARD_SELECTION_RELEASE_SECONDS", 5.0),
            "presence_grace_seconds": _float("GUARD_PRESENCE_GRACE_SECONDS", 2.0),
            "manual_track_id": None,
            "candidate_gap_seconds": _float("GUARD_CANDIDATE_GAP_SECONDS", 0.5),
        },
        "camera_motion": {
            # Unverified camera motion cannot be interpreted as guard movement.
            "enabled": _bool("GUARD_CAMERA_MOTION_ENABLED", True),
            "width": _int("GUARD_CAMERA_MOTION_WIDTH", 320),
            "min_points": _int("GUARD_CAMERA_MOTION_MIN_POINTS", 12),
            "min_inlier_ratio": _float("GUARD_CAMERA_MOTION_MIN_INLIERS", 0.70),
            "max_scale_change": _float("GUARD_CAMERA_MOTION_MAX_SCALE_CHANGE", 0.15),
            "max_fb_error": _float("GUARD_CAMERA_MOTION_MAX_FB_ERROR", 1.5),
            "min_spatial_spread": _float("GUARD_CAMERA_MOTION_MIN_SPREAD", 0.12),
            "scene_difference": _float("GUARD_CAMERA_SCENE_DIFFERENCE", 0.18),
            "max_gap_seconds": _float("GUARD_MAX_EVIDENCE_GAP_SECONDS", 2.0),
        },
        "movement": {
            "history_seconds": _float("GUARD_MOVEMENT_HISTORY_SECONDS", 5.0),
            "stationary_radius_ratio": _float("GUARD_STATIONARY_RADIUS_RATIO", 0.035),
            "minimum_history_seconds": _float("GUARD_MOVEMENT_MIN_HISTORY_SECONDS", 2.0),
            "patrol_zones": patrol_zones,
            "border_margin_ratio": _float("GUARD_MOVEMENT_BORDER_MARGIN", 0.015),
            "max_box_scale_change": _float("GUARD_MOVEMENT_MAX_BOX_SCALE_CHANGE", 0.25),
            "assume_static_camera": _bool("GUARD_ASSUME_STATIC_CAMERA", False),
            "smoothing_seconds": _float("GUARD_MOVEMENT_SMOOTHING_SECONDS", 0.25),
            "radius_quantile": _float("GUARD_MOVEMENT_RADIUS_QUANTILE", 0.90),
            "max_gap_seconds": _float("GUARD_MAX_EVIDENCE_GAP_SECONDS", 2.0),
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
            "perclos_threshold": _float("GUARD_PERCLOS_THRESHOLD", 0.60),
            "sleep_score_threshold": _float("GUARD_SLEEP_SCORE_THRESHOLD", 0.45),
            "suppress_when_phone_active": _bool("GUARD_SLEEP_SUPPRESS_PHONE", True),
            "allow_degraded_rule_trigger": _bool("GUARD_ALLOW_DEGRADED_SLEEP_ALERT", False),
            "min_eye_evidence_seconds": _float("GUARD_MIN_EYE_EVIDENCE_SECONDS", 2.0),
            "min_closed_seconds": _float("GUARD_MIN_CLOSED_SECONDS", 2.0),
            "eye_max_gap_seconds": _float("GUARD_EYE_EVIDENCE_MAX_GAP_SECONDS", 1.0),
            "min_eye_coverage": _float("GUARD_MIN_EYE_COVERAGE", 0.60),
            "fallback_confirm_seconds": _float("GUARD_SLEEP_FALLBACK_SECONDS", 8.0),
            "allow_eye_only_sleep": _bool("GUARD_ALLOW_EYE_ONLY_SLEEP", True),
            "eye_only_min_seconds": _float("GUARD_EYE_ONLY_MIN_SECONDS", max(
                4.0, _float("GUARD_MIN_EYE_EVIDENCE_SECONDS", 2.0), _float("GUARD_MIN_CLOSED_SECONDS", 2.0))),
            "eye_only_perclos_threshold": _float("GUARD_EYE_ONLY_PERCLOS", max(.85, _float("GUARD_PERCLOS_THRESHOLD", .60))),
            "eye_only_min_coverage": _float("GUARD_EYE_ONLY_MIN_COVERAGE", max(.75, _float("GUARD_MIN_EYE_COVERAGE", .60))),
            "head_roll_threshold": _float("GUARD_SLEEP_HEAD_ROLL_DEGREES", 15.0),
            "max_gap_seconds": _float("GUARD_MAX_EVIDENCE_GAP_SECONDS", 2.0),
        },
        "rules": {
            "warmup_seconds": _float("GUARD_RULE_WARMUP_SECONDS", 5.0),
            "sleep_seconds": _float("GUARD_SLEEP_SECONDS", 30.0),
            "phone_seconds": _float("GUARD_PHONE_SECONDS", 30.0),
            "stationary_seconds": _float("GUARD_STATIONARY_SECONDS", 60.0),
            "absence_seconds": _float("GUARD_ABSENCE_SECONDS", 30.0),
            "unknown_reset_seconds": _float("GUARD_UNKNOWN_RESET_SECONDS", 5.0),
            "max_evidence_gap_seconds": _float("GUARD_MAX_EVIDENCE_GAP_SECONDS", 2.0),
            "grace_seconds": {
                "sleep": _float("GUARD_SLEEP_GRACE_SECONDS", 10.0),
                "phone": _float("GUARD_PHONE_GRACE_SECONDS", 3.0),
                "stationary": _float("GUARD_STATIONARY_GRACE_SECONDS", 3.0),
                "absence": _float("GUARD_ABSENCE_GRACE_SECONDS", 1.0),
            },
        },
        "activity": {
            "confirm_seconds": _float("GUARD_ACTIVITY_CONFIRM_SECONDS", 0.4),
            "sleep_confirm_seconds": _float("GUARD_ACTIVITY_SLEEP_CONFIRM_SECONDS", 0.4),
            "hold_seconds": _float("GUARD_ACTIVITY_HOLD_SECONDS", 1.0),
            "max_evidence_gap_seconds": _float("GUARD_MAX_EVIDENCE_GAP_SECONDS", 2.0),
        },
        "visualization": {
            "draw_guard_box": _bool("GUARD_DRAW_GUARD_BOX", True),
            "draw_guard_id": _bool("GUARD_DRAW_GUARD_ID", True),
            "draw_zones": _bool("GUARD_DRAW_ZONES", True),
            "draw_pose": _bool("GUARD_DRAW_POSE", True),
            "draw_phone": _bool("GUARD_DRAW_PHONE", True),
            # Legacy options remain parseable for configuration compatibility.
            # The pipeline renders the selected guard box, ID and activity only.
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
    def check_numbers(value, path="config"):
        if isinstance(value, dict):
            for key, child in value.items():
                check_numbers(child, f"{path}.{key}")
        elif isinstance(value, (int, float)) and not isinstance(value, bool):
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{path} must be finite and >= 0")

    check_numbers(cfg)
    for section, key in (
        (cfg["movement"], "radius_quantile"),
        (cfg["sleep_logic"], "perclos_threshold"),
        (cfg["sleep_logic"], "sleep_score_threshold"),
        (cfg["sleep_logic"], "min_eye_coverage"),
        (cfg["camera_motion"], "min_inlier_ratio"),
        (cfg["camera_motion"], "min_spatial_spread"),
        (cfg["camera_motion"], "scene_difference"),
        (cfg["sleep_logic"], "eye_only_perclos_threshold"),
        (cfg["sleep_logic"], "eye_only_min_coverage"),
    ):
        if not 0 < float(section[key]) <= 1:
            raise ValueError(f"{key} must be in (0, 1]")
    sleep = cfg["sleep_logic"]
    if (sleep["eye_only_min_seconds"] < max(sleep["min_eye_evidence_seconds"], sleep["min_closed_seconds"])
            or sleep["eye_only_perclos_threshold"] < sleep["perclos_threshold"]
            or sleep["eye_only_min_coverage"] < sleep["min_eye_coverage"]):
        raise ValueError("Eye-only sleep thresholds must be at least as strict as ordinary eye thresholds")
    if not 0 < sleep["head_roll_threshold"] <= 90:
        raise ValueError("Sleep head roll threshold must be in (0, 90]")
    if not 0 <= cfg["movement"]["border_margin_ratio"] < .25:
        raise ValueError("Movement border margin must be in [0, .25)")
    if cfg["camera_motion"]["max_fb_error"] <= 0:
        raise ValueError("Camera forward/backward error limit must be > 0")
    for model in cfg["models"].values():
        if "confidence" in model and not 0 < model["confidence"] <= 1:
            raise ValueError("Model confidence must be in (0, 1]")
        if "every_n_frames" in model and model["every_n_frames"] < 1:
            raise ValueError("Model every_n_frames must be >= 1")
    if cfg["camera_motion"]["width"] < 32 or cfg["camera_motion"]["min_points"] < 4:
        raise ValueError("Camera motion requires width >= 32 and at least 4 points")
    for key in ("track_activation_threshold", "high_conf_det_threshold", "minimum_iou_threshold"):
        if not 0 < cfg["tracker"][key] <= 1:
            raise ValueError(f"tracker.{key} must be in (0, 1]")
    if cfg["tracker"]["minimum_consecutive_frames"] < 1:
        raise ValueError("Tracker minimum consecutive frames must be >= 1")
    selection = cfg["guard_selection"]
    if selection["presence_grace_seconds"] > selection["release_seconds"]:
        raise ValueError("Presence grace must not exceed selection release time")
    if cfg["activity"]["hold_seconds"] > selection["presence_grace_seconds"]:
        raise ValueError("Activity hold must not exceed presence grace")
    if cfg["movement"]["minimum_history_seconds"] > cfg["movement"]["history_seconds"]:
        raise ValueError("Movement minimum history must not exceed history window")
    for value, name in ((cfg["rules"]["max_evidence_gap_seconds"], "evidence gap"),
                        (cfg["sleep_logic"]["eye_max_gap_seconds"], "eye gap"),
                        (cfg["sleep_logic"]["perclos_window_seconds"], "PERCLOS window")):
        if value <= 0:
            raise ValueError(f"{name} must be > 0")
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
