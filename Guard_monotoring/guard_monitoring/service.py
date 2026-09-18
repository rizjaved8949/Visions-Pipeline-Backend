from __future__ import annotations

import atexit
import json
import os
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import BoundedSemaphore, Lock

from .config import load_config, runtime_paths
from .storage import JobRecord, SQLiteJobStore


class GuardQueueFullError(RuntimeError):
    pass


class GuardJobService:
    """Bounded single-node job service for one FastAPI/GPU host.

    Heavy CV imports happen inside worker execution, not during application startup.
    Job state is persisted in SQLite. For multiple backend nodes, replace this service
    with a shared queue/worker system instead of increasing Uvicorn workers blindly.
    """

    def __init__(self):
        paths = runtime_paths()
        self.upload_dir = paths["upload_dir"]
        self.output_root = paths["output_root"]
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        self.output_root.mkdir(parents=True, exist_ok=True)
        self.store = SQLiteJobStore(paths["db_path"])
        workers = max(1, int(os.getenv("GUARD_WORKERS", "1")))
        max_pending = max(workers, int(os.getenv("GUARD_MAX_PENDING_JOBS", "20")))
        self._slots = BoundedSemaphore(max_pending)
        self.executor = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="guard-monitor")
        atexit.register(self.executor.shutdown, wait=False, cancel_futures=False)

    def submit(self, source_path: str | Path) -> JobRecord:
        source_path = Path(source_path)
        if not source_path.exists():
            raise FileNotFoundError(source_path)
        if not self._slots.acquire(blocking=False):
            raise GuardQueueFullError("Guard monitoring queue is full")

        job_id = uuid.uuid4().hex
        output_dir = self.output_root / job_id
        try:
            record = self.store.create(job_id, str(source_path), str(output_dir))
            future = self.executor.submit(self._run_job, job_id, source_path, output_dir)
            future.add_done_callback(lambda _future: self._slots.release())
            return record
        except Exception:
            self._slots.release()
            raise

    def _run_job(self, job_id: str, source_path: Path, output_dir: Path) -> None:
        try:
            # Import only when inference starts. Existing application startup therefore
            # does not initialize RF-DETR, YOLO, trackers, MediaPipe or OpenCV models.
            from .pipeline import GuardMonitoringPipeline

            self.store.update(job_id, status="processing", progress=0.0, error=None)
            cfg = load_config(source=str(source_path), output_dir=str(output_dir), display=False)

            def progress(payload: dict):
                status = payload.get("status", "processing")
                if status == "completed":
                    status = "processing"
                self.store.update(
                    job_id,
                    status=status,
                    progress=payload.get("progress"),
                    frames_processed=int(payload.get("frames_processed", 0)),
                    total_frames=payload.get("total_frames"),
                )

            summary = GuardMonitoringPipeline(cfg, progress_callback=progress).run()
            self.store.update(
                job_id,
                status="completed",
                progress=100.0,
                frames_processed=int(summary.get("frames_processed", 0)),
                summary=summary,
                error=None,
            )
        except Exception as exc:
            self.store.update(
                job_id,
                status="failed",
                error=f"{type(exc).__name__}: {exc}",
            )

    def read_json_output(self, job_id: str, filename: str):
        job = self.store.get(job_id)
        if job is None:
            return None
        path = Path(job.output_dir) / filename
        if not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8"))

    def read_events(self, job_id: str) -> list[dict] | None:
        job = self.store.get(job_id)
        if job is None:
            return None
        path = Path(job.output_dir) / "events.jsonl"
        if not path.exists():
            return None
        result = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                result.append(json.loads(line))
        return result


_service: GuardJobService | None = None
_service_lock = Lock()


def get_service() -> GuardJobService:
    global _service
    if _service is None:
        with _service_lock:
            if _service is None:
                _service = GuardJobService()
    return _service
