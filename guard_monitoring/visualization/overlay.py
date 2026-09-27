from __future__ import annotations

import cv2
import numpy as np

COCO_EDGES = [
    (0, 1), (0, 2), (1, 3), (2, 4),
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 11), (6, 12), (11, 12),
    (11, 13), (13, 15), (12, 14), (14, 16),
]


def draw_activity_status(frame, label, *, bbox=None):
    """Draw confirmed activity inside the selected box when one is supplied.

    Draw on a sliced view so every text/background pixel stays within the box.
    The no-box form is kept for existing direct callers; the pipeline always
    supplies its selected guard box. This function never infers an activity.
    """
    from ..monitoring.activity import ACTIVITY_LABELS

    if label not in ACTIVITY_LABELS:
        return
    if bbox is not None:
        if len(bbox) != 4 or not np.isfinite(bbox).all():
            return
        frame_height, frame_width = frame.shape[:2]
        x1, y1, x2, y2 = bbox
        # Stay inside both the rectangle outline and the image edges. Check the
        # intersection before slicing so wholly off-screen boxes remain empty.
        left = max(0, int(np.ceil(x1)) + 3)
        top = max(0, int(np.ceil(y1)) + 3)
        right = min(frame_width, int(np.floor(x2)) - 3)
        bottom = min(frame_height, int(np.floor(y2)) - 3)
        if right <= left or bottom <= top:
            return
        frame = frame[top:bottom, left:right]
    height, width = frame.shape[:2]
    if height < 20 or width < 40:
        return
    text = f"GUARD STATUS: {label}"
    padding = max(3, min(14, width // 70))
    scale = max(0.35, min(1.0, height / 900.0))
    thickness = max(1, round(scale * 2))
    (tw, th), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
    if tw > width - padding * 4:
        scale *= (width - padding * 4) / max(tw, 1)
        thickness = 1
        (tw, th), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
    x = padding
    # Keep the activity label at the top-left inside the selected guard box.
    # Identity drawing remains separate; this function only renders activity.
    banner_height = th + baseline + padding * 2
    y = padding if bbox is not None else padding
    right = min(width - 1, x + tw + padding * 2)
    bottom = min(height - 1, y + th + baseline + padding * 2)
    # Blend only the banner region, avoiding a second full-frame copy.
    region = frame[y:bottom, x:right]
    if not region.size:
        return
    dark = np.zeros_like(region)
    cv2.addWeighted(region, 0.22, dark, 0.78, 0, region)
    color = {"Sleeping": (110, 160, 255), "Using Mobile": (70, 215, 255),
             "Moving": (150, 245, 170), "Stationary": (245, 245, 245)}[label]
    cv2.putText(frame, text, (x + padding, y + padding + th),
                cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


def draw_polygon(frame, polygon, label=None):
    pts = np.asarray(polygon, dtype=np.int32).reshape((-1, 1, 2))
    cv2.polylines(frame, [pts], True, (0, 220, 255), 2)
    if label and len(polygon):
        x, y = map(int, polygon[0])
        cv2.putText(frame, label, (x + 4, y + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 220, 255), 1, cv2.LINE_AA)


def draw_box(frame, box, text, color=(0, 255, 0), thickness=2):
    x1, y1, x2, y2 = [int(v) for v in box]
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
    if text:
        cv2.putText(frame, text, (x1, max(18, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)


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
        cv2.putText(frame, line, (x0, y), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 1, cv2.LINE_AA)
        y += line_h
    if alerts:
        for alert in alerts:
            cv2.putText(frame, f"ALERT: {alert.upper()}", (x0, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2, cv2.LINE_AA)
            y += line_h
    else:
        cv2.putText(frame, "ALERT: none", (x0, y), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (180, 180, 180), 1, cv2.LINE_AA)


def draw_all_persons(frame, tracked_detections, selected_id=None):
    """Draw a thin grey box + 'person #N' label for every tracked person
    that is NOT the currently selected guard.

    Parameters
    ----------
    tracked_detections
        The supervision.Detections-like namespace returned by ByteTrack
        (has .tracker_id and .xyxy attributes). May be None; function is
        a no-op in that case.
    selected_id
        The track_id of the currently selected guard. Persons with this
        id are skipped (they get the full selected-guard overlay instead).
    """
    if tracked_detections is None:
        return
    ids = getattr(tracked_detections, "tracker_id", None)
    boxes = getattr(tracked_detections, "xyxy", None)
    if ids is None or boxes is None or len(ids) == 0:
        return
    h, w = frame.shape[:2]
    for idx, tid in enumerate(ids):
        if tid is None:
            continue
        if selected_id is not None and int(tid) == int(selected_id):
            continue
        try:
            x1, y1, x2, y2 = [int(round(float(v))) for v in boxes[idx]]
        except Exception:
            continue
        x1, x2 = max(0, x1), min(w - 1, x2)
        y1, y2 = max(0, y1), min(h - 1, y2)
        if x2 <= x1 or y2 <= y1:
            continue
        # Thin grey box, low alpha to stay unobtrusive.
        color = (140, 140, 140)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 1)
        label = f"person #{tid}"
        scale = max(0.30, min(0.50, h / 1200.0))
        try:
            (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
            tw, th, baseline = int(tw), int(th), int(baseline)
        except (TypeError, ValueError):
            tw, th, baseline = 60, 12, 2
        tx = max(2, min(x1, w - tw - 4))
        ty = max(th + 4, y1 - 4)
        cv2.rectangle(frame, (tx - 1, ty - th - 2), (min(w - 1, tx + tw + 1), min(h - 1, ty + baseline + 1)),
                      (30, 30, 30), -1)
        cv2.putText(frame, label, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)


def draw_hud(frame, *, present, activity, phone, sleep, movement):
    """Bottom-left status HUD: a compact one-line-per-module status block.

    Intended for operators who need to verify the system is working
    even when the guard box is off-frame. Off by default.
    """
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
    # Semi-transparent background
    region = frame[by1:by2, bx1:bx2]
    if region.size:
        dark = np.zeros_like(region)
        cv2.addWeighted(region, 0.25, dark, 0.75, 0, region)
    for i, line in enumerate(lines):
        ty = by1 + padding + line_h * i + line_h - 4
        cv2.putText(frame, line, (bx1 + 5, ty),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (220, 220, 220), 1, cv2.LINE_AA)


def draw_guard_identity(frame, guard, *, show_box=True, show_id=True):
    """Selected identity only; retained observations are explicitly marked."""
    if guard is None or frame.size == 0:
        return
    h, w = frame.shape[:2]
    if not np.isfinite(guard.bbox).all() or h < 20 or w < 40:
        return
    x1, y1, x2, y2 = [int(round(v)) for v in guard.bbox]
    x1, x2 = np.clip([x1, x2], 0, w-1)
    y1, y2 = np.clip([y1, y2], 0, h-1)
    if x2 <= x1 or y2 <= y1:
        return
    color = (80, 230, 110) if guard.seen_now else (0, 190, 255)
    if show_box:
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
    if show_id:
        text = f"GUARD ID: {guard.track_id}" + ("" if guard.seen_now else " (last seen)")
        scale = max(.35, min(.7, h/1000.0))
        (tw, th), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
        if tw > w-8:
            scale *= (w-8)/max(tw, 1)
            (tw, th), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
        tx = max(2, min(int(x1), w-tw-3))
        ty = min(h-baseline-3, max(int(y1)-6, min(h//2, 60)+th))
        cv2.rectangle(frame, (tx-2, ty-th-3), (min(w-1, tx+tw+2), min(h-1, ty+baseline+2)), (20, 20, 20), -1)
        cv2.putText(frame, text, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)
