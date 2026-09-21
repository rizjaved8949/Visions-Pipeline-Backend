from __future__ import annotations

import cv2
import numpy as np

COCO_EDGES = [
    (0, 1), (0, 2), (1, 3), (2, 4),
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 11), (6, 12), (11, 12),
    (11, 13), (13, 15), (12, 14), (14, 16),
]


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
