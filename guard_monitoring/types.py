from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Optional

import numpy as np

BBox = tuple[float, float, float, float]
Point = tuple[float, float]


@dataclass
class GuardTrack:
    track_id: int
    bbox: BBox
    confidence: float
    seen_now: bool = True


@dataclass
class PoseObservation:
    keypoints: np.ndarray  # [17, 3] -> x, y, confidence in full-frame pixels
    bbox: BBox
    confidence: float


@dataclass
class PostureState:
    posture: str = "unknown"  # sitting | standing | unknown
    head_down: bool = False
    torso_lean_deg: Optional[float] = None
    left_knee_angle_deg: Optional[float] = None
    right_knee_angle_deg: Optional[float] = None


@dataclass
class EyeState:
    available: bool = False
    quality_ok: bool = False
    eyes_closed: Optional[bool] = None
    ear_left: Optional[float] = None
    ear_right: Optional[float] = None
    ear_mean: Optional[float] = None
    reason: str = "not_run"


@dataclass
class PhoneState:
    detected: bool = False
    usage: str = "no_phone"  # no_phone | visible | screen_use | call
    bbox: Optional[BBox] = None
    confidence: Optional[float] = None
    nearest_hand_ratio: Optional[float] = None
    nearest_head_ratio: Optional[float] = None


@dataclass
class MovementState:
    stationary: bool = False
    history_seconds: float = 0.0
    radius_ratio: Optional[float] = None
    visited_patrol_zones: tuple[str, ...] = ()
    patrol_coverage_ratio: Optional[float] = None


@dataclass
class SleepState:
    candidate: bool = False
    score: float = 0.0
    perclos: Optional[float] = None
    eye_samples: int = 0
    reason: str = "insufficient_evidence"
    evidence_quality: str = "unknown"  # high | degraded | unknown
    missing_modules: tuple[str, ...] = ()


@dataclass
class RuleEvent:
    rule: str
    camera_id: str
    track_id: Optional[int]
    episode_started_at: float
    triggered_at: float
    threshold_seconds: float
    payload: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
