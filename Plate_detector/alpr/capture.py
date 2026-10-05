"""
Threaded video source.

Why: cv2.VideoCapture.read() blocks. If inference is slower than the camera,
frames pile up in the driver buffer and you end up several seconds behind
"real time". This reader runs in its own thread and, for live streams, always
hands the pipeline the *newest* frame, silently dropping the ones it couldn't
keep up with. For video files it keeps every frame (bounded queue) so nothing
is skipped.
"""
import queue
import threading
import time

import cv2

LIVE_PREFIXES = ("rtsp://", "rtmp://", "http://", "https://", "udp://", "tcp://")


class VideoStream:
    def __init__(self, source, drop_frames="auto", reconnect=True, queue_size=4):
        self.source = int(source) if str(source).isdigit() else str(source)
        self.is_file = isinstance(self.source, str) and not self.source.lower().startswith(LIVE_PREFIXES)
        if drop_frames == "auto":
            drop_frames = not self.is_file
        self.drop = bool(drop_frames)
        self.reconnect = reconnect and not self.is_file

        self.cap = None
        self.stopped = False
        self.frame_id = 0
        self.fps = 0.0
        self.width = 0
        self.height = 0
        self.total_frames = 0  # 0 for live sources - no fixed length to report

        self._latest = None          # (frame_id, frame) for drop mode
        self._latest_lock = threading.Lock()
        self._new = threading.Event()
        self._q = queue.Queue(maxsize=queue_size)  # for keep-all mode
        self._thread = threading.Thread(target=self._run, daemon=True)

    # ------------------------------------------------------------------ #
    def _open(self):
        cap = cv2.VideoCapture(self.source)
        if not cap.isOpened():
            return None
        try:
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # smallest driver buffer -> lowest latency
        except Exception:  # noqa: BLE001
            pass
        self.fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
        self.width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if self.is_file:
            self.total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
        return cap

    def start(self):
        self.cap = self._open()
        if self.cap is None:
            raise RuntimeError(f"Could not open video source: {self.source}")
        self._thread.start()
        return self

    def _run(self):
        while not self.stopped:
            ok, frame = self.cap.read()
            if not ok:
                if self.is_file:
                    self.stopped = True
                    self._new.set()
                    break
                if self.reconnect:
                    print("[capture] stream lost, reconnecting in 2s ...")
                    self.cap.release()
                    time.sleep(2)
                    self.cap = self._open()
                    while self.cap is None and not self.stopped:
                        time.sleep(2)
                        self.cap = self._open()
                    continue
                self.stopped = True
                break

            self.frame_id += 1
            if self.drop:
                with self._latest_lock:
                    self._latest = (self.frame_id, frame)
                self._new.set()
            else:
                # block until the pipeline has consumed frames (keeps all frames for files)
                while not self.stopped:
                    try:
                        self._q.put((self.frame_id, frame), timeout=0.5)
                        break
                    except queue.Full:
                        continue

    # ------------------------------------------------------------------ #
    def read(self, timeout=1.0):
        """Returns (frame_id, frame) or (None, None) when the stream has ended."""
        if self.drop:
            if not self._new.wait(timeout):
                return (None, None) if self.stopped else (-1, None)
            with self._latest_lock:
                self._new.clear()
                if self._latest is None:
                    return None, None
                fid, frame = self._latest
            return fid, frame
        else:
            try:
                return self._q.get(timeout=timeout)
            except queue.Empty:
                return (None, None) if self.stopped else (-1, None)

    def stop(self):
        self.stopped = True
        if self.cap is not None:
            self.cap.release()
