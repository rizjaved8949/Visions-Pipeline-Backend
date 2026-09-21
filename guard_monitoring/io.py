from __future__ import annotations

import bz2
import os
import platform
import urllib.request
from pathlib import Path

import cv2

_OPENH264_VERSION = "2.5.0"
_OPENH264_CACHE_DIR = Path(__file__).resolve().parent.parent / "local_data" / "guard" / "codecs"
_openh264_available: bool | None = None  # None = not yet attempted this process


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


def _ensure_openh264() -> bool:
    """Best-effort: makes real H.264 encoding available so annotated video
    output plays natively in a browser's <video> element. OpenCV's always-
    available MP4 encoder (mp4v / MPEG-4 Part 2) is not decodable by Chrome,
    Edge or Firefox, which otherwise leaves the frontend with a black,
    zero-duration video for every processed job.

    Downloads Cisco's BSD-licensed OpenH264 codec DLL (the standard companion
    OpenCV's FFmpeg backend needs for H.264 on Windows) once, caches it under
    local_data/codecs, and prepends that folder to PATH so the bundled ffmpeg
    plugin's plain LoadLibrary call can find it. Never raises - callers fall
    back to mp4v if this returns False for any reason (offline, non-Windows,
    blocked network, etc.)."""
    global _openh264_available
    if _openh264_available is not None:
        return _openh264_available
    if platform.system() != "Windows":
        _openh264_available = False
        return False

    dll_name = f"openh264-{_OPENH264_VERSION}-win64.dll"
    dll_path = _OPENH264_CACHE_DIR / dll_name
    try:
        if not dll_path.exists():
            _OPENH264_CACHE_DIR.mkdir(parents=True, exist_ok=True)
            url = f"http://ciscobinary.openh264.org/{dll_name}.bz2"
            with urllib.request.urlopen(url, timeout=15) as resp:
                compressed = resp.read()
            dll_path.write_bytes(bz2.decompress(compressed))
        cache_dir_str = str(_OPENH264_CACHE_DIR)
        if cache_dir_str not in os.environ.get("PATH", "").split(os.pathsep):
            os.environ["PATH"] = cache_dir_str + os.pathsep + os.environ.get("PATH", "")
        _openh264_available = True
    except Exception:
        _openh264_available = False
    return _openh264_available


def make_writer(path: str | Path, width: int, height: int, fps: float):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if _ensure_openh264():
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"avc1"), fps, (width, height))
        if writer.isOpened():
            return writer
        writer.release()

    # Fallback: always available, but not browser-playable - only reached
    # when H.264 couldn't be set up (offline, non-Windows, blocked network).
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Could not create output video: {path}")
    return writer
