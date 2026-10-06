from __future__ import annotations

import atexit
import os
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import BoundedSemaphore, Lock

from .storage import JobRecord, SQLiteJobStore


class ALPRQueueFullError(RuntimeError):
    pass


def _runtime_paths() -> dict[str, Path]:
    return {
        "upload_dir": Path(os.getenv("ALPR_UPLOAD_DIR", "local_data/alpr/uploads")),
        "output_root": Path(os.getenv("ALPR_OUTPUT_ROOT", "local_data/alpr/outputs")),
        "db_path": Path(os.getenv("ALPR_JOB_DB", "local_data/alpr/jobs.sqlite3")),
    }


class ALPRJobService:
    """Bounded single-node job service for the ALPR module, mirroring
    guard_monitoring's GuardJobService - heavy imports (ultralytics, cv2
    model loads) happen inside worker execution, not at application startup.

    A "job" is either a file upload (processes to EOF automatically) or a
    live camera source (runs until /jobs/{id}/stop is called) - both go
    through the same run loop, since ALPRPipeline.run() already treats a
    stop_event uniformly for either kind of source."""

    def __init__(self):
        paths = _runtime_paths()
        self.upload_dir = paths["upload_dir"]
        self.output_root = paths["output_root"]
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        self.output_root.mkdir(parents=True, exist_ok=True)
        self.store = SQLiteJobStore(paths["db_path"])
        workers = max(1, int(os.getenv("ALPR_WORKERS", "1")))
        max_pending = max(workers, int(os.getenv("ALPR_MAX_PENDING_JOBS", "20")))
        self._slots = BoundedSemaphore(max_pending)
        self.executor = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="alpr-job")
        atexit.register(self.executor.shutdown, wait=False, cancel_futures=False)

        self._stop_events: dict[str, threading.Event] = {}
        self._stop_events_lock = Lock()

    def submit(self, source_path: str | Path) -> JobRecord:
        """Upload-file job - source_path must already exist on disk, and
        processing stops on its own at EOF (stoppable early too, same as a
        camera job)."""
        source_path = Path(source_path)
        if not source_path.exists():
            raise FileNotFoundError(source_path)
        return self._submit(str(source_path))

    def submit_camera(self, source: str) -> JobRecord:
        """Live job - source is a camera index ("0") or stream URL, not a
        file path. Runs indefinitely until /jobs/{id}/stop is called."""
        return self._submit(source)

    def _submit(self, source: str) -> JobRecord:
        if not self._slots.acquire(blocking=False):
            raise ALPRQueueFullError("ALPR queue is full")

        job_id = uuid.uuid4().hex
        output_dir = self.output_root / job_id
        try:
            record = self.store.create(job_id, source, str(output_dir))
            stop_event = threading.Event()
            with self._stop_events_lock:
                self._stop_events[job_id] = stop_event
            future = self.executor.submit(self._run_job, job_id, source, output_dir, stop_event)
            future.add_done_callback(lambda _future: self._slots.release())
            return record
        except Exception:
            self._slots.release()
            raise

    def request_stop(self, job_id: str) -> None:
        with self._stop_events_lock:
            event = self._stop_events.get(job_id)
        if event is None:
            raise KeyError(job_id)
        event.set()

    def stop_all(self) -> None:
        """Signal every currently-running job to stop - used on app shutdown
        (Ctrl+C) so a long job doesn't keep a non-daemon worker thread alive
        long after the user asked the process to exit."""
        with self._stop_events_lock:
            events = list(self._stop_events.values())
        for event in events:
            event.set()

    def _run_job(self, job_id: str, source: str, output_dir: Path, stop_event: threading.Event) -> None:
        try:
            # Import only when inference starts - existing application startup therefore
            # does not initialize YOLO/torch for this module either.
            from Plate_detector import ALPRModule

            self.store.update(job_id, status="processing", progress=0.0, error=None)
            latest_frame_path = output_dir / "latest.jpg"

            def progress(payload: dict):
                total = payload.get("total_frames")
                processed = int(payload.get("frames_processed", 0))
                pct = round(min(100.0, processed * 100.0 / total), 2) if total else None
                self.store.update(
                    job_id,
                    status="processing",
                    progress=pct,
                    frames_processed=processed,
                    total_frames=total,
                )

            alpr = ALPRModule(output_dir=str(output_dir))
            result = alpr.run(
                source,
                stop_event=stop_event,
                progress_callback=progress,
                latest_frame_path=str(latest_frame_path),
            )
            if not result.get("ok", False):
                self.store.update(job_id, status="failed", error=result.get("error") or "Unknown ALPR failure")
                return

            frames = int(result.get("frames", 0))
            self.store.update(
                job_id,
                status="completed",
                progress=100.0,
                frames_processed=frames,
                total_frames=frames,
                summary=result,
            )
        except Exception as exc:  # noqa: BLE001
            self.store.update(job_id, status="failed", error=f"{type(exc).__name__}: {exc}")
        finally:
            with self._stop_events_lock:
                self._stop_events.pop(job_id, None)

    def read_summary(self, job_id: str) -> dict | None:
        job = self.store.get(job_id)
        return job.summary if job else None


_service: ALPRJobService | None = None
_service_lock = Lock()


def get_service() -> ALPRJobService:
    global _service
    if _service is None:
        with _service_lock:
            if _service is None:
                _service = ALPRJobService()
    return _service
