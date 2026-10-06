"""Real, backend-held workspace settings:

1. Per-pipeline enable/disable, shared across all five modules. Guard and
   ALPR already gate every one of their routes on a live os.environ read
   (guard_monitoring.config.guard_enabled / Plate_detector.api.alpr_enabled)
   - this module reuses those two functions directly rather than keeping a
   second, possibly-drifting copy of the same truth, and extends the same
   pattern to the three modules that didn't have it yet (see their own
   *_enabled()/_require_enabled() additions).

2. A small persisted preferences blob (Notifications minus the email
   digest, and Display) backing /api/system/preferences. Read once per
   session/job by each pipeline's drawing code - not re-read per frame.
"""

from __future__ import annotations

import copy
import json
import os
import threading
from pathlib import Path
from typing import Any

from dotenv import find_dotenv, set_key
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from guard_monitoring.config import guard_enabled
from Plate_detector.api import alpr_enabled

router = APIRouter()

PREFERENCES_PATH = Path(os.getenv("SYSTEM_PREFERENCES_PATH", "local_data/preferences.json"))
_prefs_lock = threading.Lock()

DEFAULT_PREFERENCES: dict[str, Any] = {
    "notifications": {
        "enable_alerts": True,
        "browser_notifications": False,
        "sound_alerts": False,
    },
    "display": {
        "show_detection_boxes": True,
        "show_labels": True,
        "show_confidence": True,
        "compact_view": False,
    },
}


def _bool_env(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


# ------------------------------------------------------------------
# Pipelines - enable/disable, shared registry
# ------------------------------------------------------------------
PIPELINES: dict[str, dict[str, Any]] = {
    "attendance": {
        "name": "Smart Attendance",
        "env_key": "ATTENDANCE_ENABLED",
        "getter": lambda: _bool_env("ATTENDANCE_ENABLED", True),
    },
    "kitchen": {
        "name": "Kitchen Hygiene",
        "env_key": "KITCHEN_HYGIENE_ENABLED",
        "getter": lambda: _bool_env("KITCHEN_HYGIENE_ENABLED", True),
    },
    "plate": {
        "name": "Plate Detection",
        "env_key": "ALPR_ENABLED",
        "getter": alpr_enabled,
    },
    "guard": {
        "name": "Guard Activity",
        "env_key": "GUARD_MONITORING_ENABLED",
        "getter": guard_enabled,
    },
    "restricted_zone": {
        "name": "Restricted Zones",
        "env_key": "RESTRICTED_ZONE_ENABLED",
        "getter": lambda: _bool_env("RESTRICTED_ZONE_ENABLED", True),
    },
}


class ToggleRequest(BaseModel):
    enabled: bool


@router.get("/pipelines")
def list_pipelines():
    return {
        "pipelines": [
            {"key": key, "name": entry["name"], "enabled": entry["getter"]()}
            for key, entry in PIPELINES.items()
        ]
    }


@router.post("/pipelines/{key}/toggle")
def toggle_pipeline(key: str, body: ToggleRequest):
    entry = PIPELINES.get(key)
    if entry is None:
        raise HTTPException(status_code=404, detail=f"Unknown pipeline: {key}")

    value = "true" if body.enabled else "false"
    os.environ[entry["env_key"]] = value
    try:
        env_path = find_dotenv() or str(Path(__file__).resolve().parent / ".env")
        set_key(env_path, entry["env_key"], value)
    except Exception as exc:  # noqa: BLE001 - the in-memory toggle above already took effect
        # Persistence failed but the live toggle didn't - surface it instead
        # of silently leaving .env out of sync with the running process.
        print(f"[system_settings] Could not persist {entry['env_key']}={value} to .env: {exc}")

    return {"key": key, "enabled": entry["getter"]()}


# ------------------------------------------------------------------
# Preferences - Notifications (minus email) + Display
# ------------------------------------------------------------------
def _deep_merge(base: dict, patch: dict) -> dict:
    for k, v in patch.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _deep_merge(base[k], v)
        else:
            base[k] = v
    return base


def _read_preferences() -> dict:
    if not PREFERENCES_PATH.exists():
        return copy.deepcopy(DEFAULT_PREFERENCES)
    try:
        with open(PREFERENCES_PATH, "r", encoding="utf-8") as f:
            stored = json.load(f)
    except (json.JSONDecodeError, OSError):
        return copy.deepcopy(DEFAULT_PREFERENCES)
    merged = copy.deepcopy(DEFAULT_PREFERENCES)
    return _deep_merge(merged, stored)


def _write_preferences(data: dict) -> None:
    PREFERENCES_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = PREFERENCES_PATH.with_suffix(".tmp.json")
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    os.replace(tmp_path, PREFERENCES_PATH)


def get_preferences() -> dict:
    with _prefs_lock:
        return _read_preferences()


def get_display_prefs() -> dict:
    """Called once per session/job start by each pipeline's drawing code -
    not re-read per frame."""
    return get_preferences()["display"]


@router.get("/preferences")
def get_preferences_route():
    return get_preferences()


@router.post("/preferences")
def update_preferences(patch: dict[str, Any]):
    allowed_sections = {"notifications", "display"}
    unknown = set(patch) - allowed_sections
    if unknown:
        raise HTTPException(status_code=400, detail=f"Unknown preference section(s): {sorted(unknown)}")

    with _prefs_lock:
        current = _read_preferences()
        _deep_merge(current, patch)
        _write_preferences(current)
        return current
