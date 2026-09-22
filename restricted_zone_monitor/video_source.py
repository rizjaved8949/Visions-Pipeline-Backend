"""
Unified video input.

Accepts:
  * an integer (or numeric string)  -> local webcam / USB camera index
  * a file path                     -> video file (processed frame by frame, nothing dropped)
  * rtsp:// rtmp:// http(s)://      -> network camera / stream (threaded reader,
                                       always serves the NEWEST frame so we never
                                       fall behind real time, auto-reconnects)
"""

from __future__ import annotations

import os
import threading
import time
from typing import Optional, Tuple

import cv2
import numpy as np

STREAM_PREFIXES = ("rtsp://", "rtmp://", "http://", "https://", "udp://", "tcp://")


def _is_stream(src) -> bool:
    return isinstance(src, str) and src.lower().startswith(STREAM_PREFIXES)


def _is_camera(src) -> bool:
    return isinstance(src, int) or (isinstance(src, str) and src.isdigit())


def parse_source(src: str):
    return int(src) if _is_camera(src) else src


class VideoSource:
    def __init__(self, source, reconnect_delay: float = 2.0, max_reconnects: int = 0):
        self.source = parse_source(source)
        self.is_file = not _is_stream(self.source) and not _is_camera(self.source)
        self.is_live = not self.is_file
        self.reconnect_delay = reconnect_delay
        self.max_reconnects = max_reconnects  # 0 = infinite

        if self.is_file and not os.path.exists(self.source):
            raise FileNotFoundError(f"Video file not found: {self.source}")

        self.cap: Optional[cv2.VideoCapture] = None
        self._open()

        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1280
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 720
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 25.0
        if self.fps <= 1 or self.fps > 240:
            self.fps = 25.0
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT)) if self.is_file else -1

        # threaded reader for live inputs
        self._lock = threading.Lock()
        self._latest: Optional[np.ndarray] = None
        self._latest_id = 0
        self._served_id = 0
        self._stopped = False
        self._thread: Optional[threading.Thread] = None
        if self.is_live:
            self._thread = threading.Thread(target=self._reader, daemon=True)
            self._thread.start()

    # ------------------------------------------------------------------
    def _open(self) -> None:
        if isinstance(self.source, str) and self.source.lower().startswith("rtsp://"):
            # prefer TCP for RTSP: far fewer corrupted frames than UDP
            os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp")
        backend = cv2.CAP_DSHOW if (_is_camera(self.source) and os.name == "nt") else cv2.CAP_ANY
        self.cap = cv2.VideoCapture(self.source, backend)
        if _is_camera(self.source):
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open video source: {self.source}")

    def _reader(self) -> None:
        reconnects = 0
        while not self._stopped:
            ok, frame = self.cap.read()
            if not ok or frame is None:
                if self.max_reconnects and reconnects >= self.max_reconnects:
                    print("[VideoSource] Max reconnect attempts reached, stopping.")
                    self._stopped = True
                    break
                reconnects += 1
                print(f"[VideoSource] Frame read failed - reconnecting ({reconnects}) ...")
                time.sleep(self.reconnect_delay)
                try:
                    self.cap.release()
                    self._open()
                except Exception as e:  # noqa: BLE001
                    print(f"[VideoSource] Reconnect failed: {e}")
                continue
            reconnects = 0
            with self._lock:
                self._latest = frame
                self._latest_id += 1

    # ------------------------------------------------------------------
    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Files  : next frame (sequential).
        Live   : newest frame; blocks briefly until a *new* frame exists.
        """
        if self.is_file:
            ok, frame = self.cap.read()
            return (ok and frame is not None), frame

        deadline = time.time() + 5.0
        while not self._stopped:
            with self._lock:
                if self._latest_id != self._served_id and self._latest is not None:
                    self._served_id = self._latest_id
                    return True, self._latest.copy()
            if time.time() > deadline:
                return False, None
            time.sleep(0.002)
        return False, None

    def release(self) -> None:
        self._stopped = True
        if self._thread is not None:
            self._thread.join(timeout=2)
        if self.cap is not None:
            self.cap.release()
