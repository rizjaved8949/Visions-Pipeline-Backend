from __future__ import annotations

from pathlib import Path

import cv2


def parse_source(source: str):
    if source.isdigit():
        return int(source)
    return source


def open_capture(source: str):
    parsed = parse_source(source)
    cap = cv2.VideoCapture(parsed)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video source: {source}")
    return cap


def video_metadata(cap):
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    if fps <= 0 or fps > 240:
        fps = 25.0
    return width, height, fps


def make_writer(path: str | Path, width: int, height: int, fps: float):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Could not create output video: {path}")
    return writer
