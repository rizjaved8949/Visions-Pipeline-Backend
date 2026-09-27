import cv2
import numpy as np
from typing import List, Optional, Dict, Any
from .types import BBox, Detection


def draw_all_persons(
    frame: np.ndarray,
    tracked_detections: List[Detection],
    selected_id: Optional[int],
) -> np.ndarray:
    for det in tracked_detections:
        track_id = getattr(det, "track_id", None)
        if track_id is None or track_id == selected_id:
            continue

        bbox = getattr(det, "bbox", None)
        if bbox is None:
            continue

        x1, y1, x2, y2 = bbox.to_int_tuple()
        cv2.rectangle(frame, (x1, y1), (x2, y2), (120, 120, 120), 1)

        label = f"person #{track_id}"
        cv2.putText(
            frame,
            label,
            (x1, max(y1 - 5, 15)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (180, 180, 180),
            1,
            cv2.LINE_AA,
        )

    return frame


def draw_guard_identity(
    frame: np.ndarray,
    bbox: BBox,
    track_id: Optional[int],
    is_retained: bool = False,
) -> np.ndarray:
    x1, y1, x2, y2 = bbox.to_int_tuple()
    color = (0, 165, 255) if is_retained else (0, 255, 0)

    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    tag = f"GUARD #{track_id}" if track_id else "GUARD"
    if is_retained:
        tag += " (last seen)"

    cv2.putText(
        frame,
        tag,
        (x1, max(y1 - 8, 20)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        color,
        2,
        cv2.LINE_AA,
    )

    return frame


def draw_activity_status(
    frame: np.ndarray, bbox: BBox, activity_label: str
) -> np.ndarray:
    x1, y1, x2, y2 = bbox.to_int_tuple()
    h, w = frame.shape[:2]

    text = f"STATUS: {activity_label.upper()}"

    if bbox.width < 50 or bbox.height < 50:
        pos_x = min(w - 150, x2 + 10)
        pos_y = max(20, y1 + 15)
    else:
        pos_x = max(10, x1)
        pos_y = min(h - 10, y2 + 20)

    cv2.putText(
        frame,
        text,
        (pos_x, pos_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    return frame


def draw_hud(frame: np.ndarray, status_dict: Dict[str, Any]) -> np.ndarray:
    h, w = frame.shape[:2]
    lines = [
        f"Guard Present: {status_dict.get('present', False)}",
        f"Activity: {status_dict.get('activity', 'Unknown')}",
        f"Phone: {status_dict.get('phone', 'No')}",
        f"Sleep: {status_dict.get('sleep', 'No')}",
        f"Movement: {status_dict.get('movement', 'Unknown')}",
    ]

    pad_x, pad_y = 10, h - (len(lines) * 20) - 10
    cv2.rectangle(frame, (pad_x - 5, pad_y - 20), (pad_x + 220, h - 5), (0, 0, 0), -1)

    for idx, line in enumerate(lines):
        y = pad_y + (idx * 20)
        cv2.putText(
            frame,
            line,
            (pad_x, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 255, 255),
            1,
            cv2.LINE_AA,
        )

    return frame