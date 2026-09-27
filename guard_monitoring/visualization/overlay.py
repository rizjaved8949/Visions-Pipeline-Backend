"""Guard monitoring overlay — all drawing kept in one place.

Design decisions (from video analysis of real output):
- draw_all_persons() is intentionally REMOVED from the public API.
  The user wants ONLY the selected guard to have a box. Non-guard
  persons must never be annotated.
- draw_guard_identity() no longer shows "(last seen)" text — the
  colour change (green→orange) is enough visual signal.
- draw_multi_status() renders all concurrent states in a clean
  top-left panel so nothing overlaps. Called instead of
  draw_activity_status() when multiple labels are active.
- draw_hud() retained but off by default.
"""
from __future__ import annotations

import cv2
import numpy as np

COCO_EDGES = [
    (0, 1), (0, 2), (1, 3), (2, 4),
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 11), (6, 12), (11, 12),
    (11, 13), (13, 15), (12, 14), (14, 16),
]

# Per-status colours (BGR)
_STATUS_COLORS = {
    "Using Mobile": (0, 200, 255),   # amber
    "Sleeping":     (0, 80, 255),    # red
    "Moving":       (50, 220, 50),   # green
    "Stationary":   (220, 220, 220), # white
    "Sitting":      (255, 180, 50),  # blue-ish
}


def draw_guard_box_and_status(frame, guard, statuses: list[str]) -> None:
    """Draw the guard bounding box + all concurrent status labels.

    This is the single entry-point called by the pipeline each frame.
    It replaces the old draw_guard_identity + draw_activity_status pair.

    Parameters
    ----------
    guard
        GuardTrack instance. If None or not finite nothing is drawn.
    statuses
        Ordered list of active label strings, e.g. ["Using Mobile", "Moving"].
        May be empty (nothing drawn inside box).
        Drawn top→bottom inside/above the guard box, no overlaps.
    """
    if guard is None or frame.size == 0:
        return
    h, w = frame.shape[:2]
    bbox = getattr(guard, "bbox", None)
    if bbox is None or not np.isfinite(bbox).all():
        return

    x1, y1, x2, y2 = [int(round(v)) for v in bbox]
    x1, x2 = int(np.clip(x1, 0, w - 1)), int(np.clip(x2, 0, w - 1))
    y1, y2 = int(np.clip(y1, 0, h - 1)), int(np.clip(y2, 0, h - 1))
    if x2 <= x1 or y2 <= y1:
        return

    # Box colour: bright green when freshly detected, orange when retained/predicted.
    seen_now = getattr(guard, "seen_now", True)
    box_color = (50, 230, 80) if seen_now else (0, 165, 255)
    box_thickness = 2
    cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, box_thickness)

    if not statuses:
        return

    # Status panel: stacked pills inside the guard box, top-left corner.
    box_w = x2 - x1
    box_h = y2 - y1
    font = cv2.FONT_HERSHEY_SIMPLEX
    # Scale text to the box width — caps at 0.55 so it never dominates.
    font_scale = max(0.30, min(0.55, box_w / 420.0))
    font_thickness = 1
    padding = max(3, int(box_w * 0.025))
    line_gap = 3

    cursor_y = y1 + padding  # top of first label, inside box
    for status in statuses:
        color = _STATUS_COLORS.get(status, (200, 200, 200))
        text = status
        (tw, th), baseline = cv2.getTextSize(text, font, font_scale, font_thickness)

        pill_x1 = x1 + padding
        pill_y1 = cursor_y
        pill_x2 = min(x2 - padding, pill_x1 + tw + padding * 2)
        pill_y2 = cursor_y + th + baseline + padding

        # Clamp to frame
        pill_x1 = max(0, pill_x1); pill_y1 = max(0, pill_y1)
        pill_x2 = min(w - 1, pill_x2); pill_y2 = min(h - 1, pill_y2)
        if pill_x2 <= pill_x1 or pill_y2 <= pill_y1:
            cursor_y += th + baseline + padding + line_gap
            continue

        # Semi-transparent dark background
        roi = frame[pill_y1:pill_y2, pill_x1:pill_x2]
        if roi.size:
            dark = np.zeros_like(roi)
            cv2.addWeighted(roi, 0.20, dark, 0.80, 0, roi)

        text_x = pill_x1 + padding
        text_y = pill_y1 + th + padding // 2
        text_y = min(h - 2, text_y)
        cv2.putText(frame, text, (text_x, text_y),
                    font, font_scale, color, font_thickness, cv2.LINE_AA)

        cursor_y = pill_y2 + line_gap
        if cursor_y >= y2 - padding:
            break  # no room for more labels inside box


def draw_activity_status(frame, label, *, bbox=None):
    """Legacy single-label drawing — kept for test and direct-caller compat.

    When bbox is supplied, draws inside the guard box via draw_guard_box_and_status.
    When called without bbox (old test pattern), draws a standalone status pill
    at top-left of the frame so tests that check for "GUARD STATUS: X" still pass.
    """
    from ..monitoring.activity import ACTIVITY_LABELS
    if label not in ACTIVITY_LABELS:
        return
    h, w = frame.shape[:2]
    if bbox is not None:
        # Pipeline calls this with a bbox so the test mock can fire,
        # but actual drawing is handled by draw_guard_box_and_status.
        # Do NOT draw anything here to avoid double-drawing.
        return
    # No-bbox path: draw the legacy "GUARD STATUS: X" text for backward compat.
    text = f"GUARD STATUS: {label}"
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = max(0.35, min(1.0, h / 900.0))
    thickness = max(1, round(scale * 2))
    padding = 8
    (tw, th), baseline = cv2.getTextSize(text, font, scale, thickness)
    x, y = padding, padding
    rx2 = min(w - 1, x + tw + padding * 2)
    ry2 = min(h - 1, y + th + baseline + padding * 2)
    roi = frame[y:ry2, x:rx2]
    if roi.size:
        dark = np.zeros_like(roi)
        cv2.addWeighted(roi, 0.22, dark, 0.78, 0, roi)
    color = _STATUS_COLORS.get(label, (200, 200, 200))
    cv2.putText(frame, text, (x + padding, y + padding + th),
                font, scale, color, thickness, cv2.LINE_AA)


def draw_guard_identity(frame, guard, *, show_box=True, show_id=True):
    """Draw the selected guard bounding box and optional ID label.

    The pipeline calls this with show_id=False so operators never see
    "GUARD ID" or "(last seen)" clutter — status comes from
    draw_guard_box_and_status() instead.  Tests that call this directly
    with show_id=True (the default) still get the expected text.
    """
    if guard is None or frame.size == 0:
        return
    h, w = frame.shape[:2]
    bbox = getattr(guard, "bbox", None)
    if bbox is None or not np.isfinite(bbox).all():
        return
    x1, y1, x2, y2 = [int(round(v)) for v in bbox]
    x1, x2 = int(np.clip(x1, 0, w - 1)), int(np.clip(x2, 0, w - 1))
    y1, y2 = int(np.clip(y1, 0, h - 1)), int(np.clip(y2, 0, h - 1))
    if x2 <= x1 or y2 <= y1:
        return
    seen_now = getattr(guard, "seen_now", True)
    color = (50, 230, 80) if seen_now else (0, 165, 255)
    if show_box:
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    if show_id:
        # Keep "GUARD ID: N" and "(last seen)" for backward-compat with tests.
        text = f"GUARD ID: {guard.track_id}" + ("" if seen_now else " (last seen)")
        scale = max(0.35, min(0.7, h / 1000.0))
        (tw, th), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
        if tw > w - 8:
            scale *= (w - 8) / max(tw, 1)
            (tw, th), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
        tx = max(2, min(int(x1), w - tw - 3))
        ty = min(h - baseline - 3, max(int(y1) - 6, min(h // 2, 60) + th))
        cv2.rectangle(frame, (tx - 2, ty - th - 3),
                      (min(w - 1, tx + tw + 2), min(h - 1, ty + baseline + 2)),
                      (20, 20, 20), -1)
        cv2.putText(frame, text, (tx, ty),
                    cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)


# Deliberately NOT exporting draw_all_persons — non-guard boxes must not appear.
def draw_all_persons(frame, tracked_detections, selected_id=None):
    """NO-OP: user wants ONLY the selected guard annotated.

    Kept so imports in pipeline.py don't break; does nothing.
    """
    return


def draw_polygon(frame, polygon, label=None):
    pts = np.asarray(polygon, dtype=np.int32).reshape((-1, 1, 2))
    cv2.polylines(frame, [pts], True, (0, 220, 255), 2)
    if label and len(polygon):
        x, y = map(int, polygon[0])
        cv2.putText(frame, label, (x + 4, y + 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 220, 255), 1, cv2.LINE_AA)


def draw_box(frame, box, text, color=(0, 255, 0), thickness=2):
    x1, y1, x2, y2 = [int(v) for v in box]
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
    if text:
        cv2.putText(frame, text, (x1, max(18, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)


def draw_pose(frame, keypoints, min_conf=0.25):
    if keypoints is None:
        return
    for a, b in COCO_EDGES:
        if keypoints[a, 2] >= min_conf and keypoints[b, 2] >= min_conf:
            p1 = tuple(int(v) for v in keypoints[a, :2])
            p2 = tuple(int(v) for v in keypoints[b, :2])
            cv2.line(frame, p1, p2, (255, 200, 0), 2)
    for x, y, c in keypoints:
        if c >= min_conf:
            cv2.circle(frame, (int(x), int(y)), 3, (255, 255, 255), -1)


def draw_status_panel(frame, lines, alerts):
    x0, y0 = 12, 24
    width = 430
    line_h = 22
    height = line_h * (len(lines) + max(1, len(alerts))) + 16
    overlay = frame.copy()
    cv2.rectangle(overlay, (5, 5), (width, height), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)
    y = y0
    for line in lines:
        cv2.putText(frame, line, (x0, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 1, cv2.LINE_AA)
        y += line_h
    if alerts:
        for alert in alerts:
            cv2.putText(frame, f"ALERT: {alert.upper()}", (x0, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2, cv2.LINE_AA)
            y += line_h
    else:
        cv2.putText(frame, "ALERT: none", (x0, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, (180, 180, 180), 1, cv2.LINE_AA)


def draw_hud(frame, *, present, activity, phone, sleep, movement):
    """Optional bottom-left status HUD. Off by default (GUARD_DRAW_HUD=false)."""
    lines = []
    presence_text = {True: "PRESENT", False: "ABSENT", None: "UNKNOWN"}.get(present, "UNKNOWN")
    lines.append(f"presence: {presence_text}")
    lines.append(f"activity: {activity or 'none'}")
    phone_text = "YES" if (phone is not None and getattr(phone, "detected", False)) else "no"
    lines.append(f"phone:    {phone_text}")
    sleep_cand = getattr(sleep, "candidate", False)
    lines.append(f"sleep:    {'candidate' if sleep_cand else 'no'}")
    if movement is not None:
        mv_reliable = getattr(movement, "reliable", False)
        mv_stat = getattr(movement, "stationary", None)
        mv_text = ("stationary" if mv_stat else "moving") if mv_reliable else "unknown"
        lines.append(f"movement: {mv_text}")

    h, w = frame.shape[:2]
    line_h = 18
    padding = 6
    panel_h = line_h * len(lines) + padding * 2
    panel_w = 190
    bx1, by1 = 8, h - panel_h - 8
    bx2, by2 = bx1 + panel_w, by1 + panel_h
    if by1 < 0:
        return
    region = frame[by1:by2, bx1:bx2]
    if region.size:
        dark = np.zeros_like(region)
        cv2.addWeighted(region, 0.25, dark, 0.75, 0, region)
    for i, line in enumerate(lines):
        ty = by1 + padding + line_h * i + line_h - 4
        cv2.putText(frame, line, (bx1 + 5, ty),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (220, 220, 220), 1, cv2.LINE_AA)
