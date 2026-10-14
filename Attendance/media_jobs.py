"""Background jobs for uploaded media (image/video) recognition passes.

Why this exists: the original /api/process_media was one blocking request that
returned the whole result at once, so there was no id for the frontend to
re-attach to. Navigating to another page threw the result away even though the
work was effectively done. These jobs give each upload an id that can be polled
and stopped, matching what Guard/Kitchen/ALPR already expose.

State is deliberately in-memory rather than SQLite: a media job only needs to
outlive a page navigation in the browser, not a backend restart (and Attendance
has no database of its own). A restart loses in-flight jobs, which the frontend
handles by falling back to its demo feed.
"""
from __future__ import annotations

import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

from . import attendance

# queued/processing are live; completed/failed/stopped are terminal. The names
# match the other modules' job statuses so the frontend poll helper is identical.
LIVE_STATUSES = {"queued", "processing"}


@dataclass
class _MediaJob:
    job_id: str
    source_filename: str
    status: str = "queued"
    processed_frames: int = 0
    total_frames: int | None = None
    error: str | None = None
    media_type: str | None = None
    result: bytes | None = None
    summary: dict | None = None
    lock: threading.Lock = field(default_factory=threading.Lock)
    stop_event: threading.Event = field(default_factory=threading.Event)

    def public(self) -> dict:
        with self.lock:
            return {
                "job_id": self.job_id,
                "status": self.status,
                "processed_frames": self.processed_frames,
                "total_frames": self.total_frames,
                "error": self.error,
                "source_filename": self.source_filename,
            }


class MediaJobService:
    def __init__(self, max_workers: int = 1):
        self._lock = threading.RLock()
        self._jobs: dict[str, _MediaJob] = {}
        # One at a time: the recognition model is shared and GPU-bound, so
        # running several uploads concurrently would only make each slower.
        self._executor = ThreadPoolExecutor(
            max_workers=max_workers, thread_name_prefix="attendance-media"
        )

    def submit(self, content: bytes, content_type: str, filename: str) -> dict:
        job = _MediaJob(job_id=uuid.uuid4().hex, source_filename=filename or "upload")
        with self._lock:
            self._jobs[job.job_id] = job
        self._executor.submit(self._run, job, content, content_type)
        return job.public()

    def get(self, job_id: str) -> _MediaJob | None:
        with self._lock:
            return self._jobs.get(job_id)

    def request_stop(self, job_id: str) -> None:
        job = self.get(job_id)
        if job is None:
            raise KeyError(job_id)
        job.stop_event.set()

    def stop_all(self) -> None:
        with self._lock:
            jobs = list(self._jobs.values())
        for job in jobs:
            job.stop_event.set()

    def _run(self, job: _MediaJob, content: bytes, content_type: str) -> None:
        def set_state(**changes) -> None:
            with job.lock:
                for name, value in changes.items():
                    setattr(job, name, value)

        def on_progress(frames_done: int, total: int | None) -> None:
            set_state(processed_frames=frames_done, total_frames=total)

        set_state(status="processing")
        try:
            if content_type.startswith("image"):
                result_bytes, media_type, summary = attendance.process_image_bytes(content)
            elif content_type.startswith("video"):
                result_bytes, media_type, summary = attendance.process_video_bytes(
                    content, stop_event=job.stop_event, progress_cb=on_progress
                )
            else:
                raise ValueError("Unsupported file type - upload an image or video.")
        except Exception as exc:
            set_state(status="failed", error=f"{type(exc).__name__}: {exc}")
            return

        # A stopped job still has a usable partial video, so it is delivered the
        # same way a completed one is - only the status differs.
        set_state(
            status="stopped" if job.stop_event.is_set() else "completed",
            result=result_bytes,
            media_type=media_type,
            summary=summary,
        )


_SERVICE: MediaJobService | None = None
_SERVICE_LOCK = threading.Lock()


def get_service() -> MediaJobService:
    global _SERVICE
    with _SERVICE_LOCK:
        if _SERVICE is None:
            _SERVICE = MediaJobService()
        return _SERVICE
