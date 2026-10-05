from __future__ import annotations

import base64
import os
import time
import uuid
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel

from .reporting import build_job_report_csv, build_job_report_pdf
from .service import ALPRQueueFullError, get_service

router = APIRouter()

_ALLOWED_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
_REPORT_MEDIA_TYPES = {
    "csv": "text/csv",
    "pdf": "application/pdf",
}


class HealthResponse(BaseModel):
    enabled: bool
    architecture: str


class JobCreated(BaseModel):
    job_id: str
    status: str


class JobStatus(BaseModel):
    job_id: str
    status: str
    progress: float | None = None
    frames_processed: int = 0
    total_frames: int | None = None
    error: str | None = None


class CameraStartRequest(BaseModel):
    source: str = "0"


def alpr_enabled() -> bool:
    return os.getenv("ALPR_ENABLED", "true").strip().lower() in {"1", "true", "yes", "on"}


def _require_enabled() -> None:
    if not alpr_enabled():
        raise HTTPException(status_code=503, detail="ALPR is disabled")


def _job_or_404(job_id: str):
    job = get_service().store.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="ALPR job not found")
    return job


@router.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse(enabled=alpr_enabled(), architecture="uploaded-video job queue")


@router.post("/jobs", response_model=JobCreated, status_code=202)
async def create_job(video: UploadFile = File(...)):
    _require_enabled()
    suffix = Path(video.filename or "video.mp4").suffix.lower()
    if suffix not in _ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=400, detail="Unsupported video extension")

    service = get_service()
    target = service.upload_dir / f"{uuid.uuid4().hex}{suffix}"
    max_bytes = max(1, int(os.getenv("ALPR_MAX_UPLOAD_MB", "500"))) * 1024 * 1024
    written = 0

    try:
        with target.open("wb") as out:
            while True:
                chunk = await video.read(1024 * 1024)
                if not chunk:
                    break
                written += len(chunk)
                if written > max_bytes:
                    raise HTTPException(status_code=413, detail="Uploaded video exceeds ALPR_MAX_UPLOAD_MB")
                out.write(chunk)
    except Exception:
        target.unlink(missing_ok=True)
        raise
    finally:
        await video.close()

    if written == 0:
        target.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail="Uploaded video is empty")

    try:
        job = service.submit(target)
    except ALPRQueueFullError:
        target.unlink(missing_ok=True)
        raise HTTPException(status_code=503, detail="ALPR queue is full")
    except Exception as exc:
        target.unlink(missing_ok=True)
        raise HTTPException(status_code=500, detail=f"Could not queue ALPR job: {type(exc).__name__}")

    return JobCreated(job_id=job.job_id, status=job.status)


@router.post("/jobs/camera", response_model=JobCreated, status_code=202)
def start_camera_job(body: CameraStartRequest):
    """Live job - source is a camera index ("0") or stream URL, not an
    upload. Runs until /jobs/{job_id}/stop is called."""
    _require_enabled()
    service = get_service()
    try:
        job = service.submit_camera(body.source)
    except ALPRQueueFullError:
        raise HTTPException(status_code=503, detail="ALPR queue is full")
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Could not start ALPR camera job: {type(exc).__name__}")
    return JobCreated(job_id=job.job_id, status=job.status)


@router.post("/jobs/{job_id}/stop")
def stop_job(job_id: str):
    """Stop a running job (live camera, or an upload still processing) -
    the annotated video is still produced and replayable, same as letting
    an upload reach EOF on its own."""
    _require_enabled()
    _job_or_404(job_id)
    try:
        get_service().request_stop(job_id)
    except KeyError:
        raise HTTPException(status_code=409, detail="Job already finished")
    return {"job_id": job_id, "status": "stopping"}


@router.get("/jobs/{job_id}", response_model=JobStatus)
def get_job(job_id: str):
    _require_enabled()
    job = _job_or_404(job_id)
    return JobStatus(
        job_id=job.job_id,
        status=job.status,
        progress=job.progress,
        frames_processed=job.frames_processed,
        total_frames=job.total_frames,
        error=job.error,
    )


@router.get("/jobs/{job_id}/summary")
def get_summary(job_id: str):
    _require_enabled()
    job = _job_or_404(job_id)
    if job.summary is None:
        raise HTTPException(status_code=409, detail=f"Summary not ready; status={job.status}")
    return {"job_id": job_id, "summary": job.summary}


@router.get("/jobs/{job_id}/report")
def get_job_report(job_id: str, format: str = "pdf"):
    _require_enabled()
    job = _job_or_404(job_id)
    if format not in _REPORT_MEDIA_TYPES:
        raise HTTPException(status_code=400, detail="format must be 'csv' or 'pdf'.")
    if job.summary is None:
        raise HTTPException(status_code=409, detail=f"Report not ready; status={job.status}")

    builder = build_job_report_pdf if format == "pdf" else build_job_report_csv
    report_bytes = builder(job.summary)
    if report_bytes is None:
        raise HTTPException(status_code=404, detail="No job data to report.")

    return {
        "media_type": _REPORT_MEDIA_TYPES[format],
        "report_base64": base64.b64encode(report_bytes).decode("ascii"),
        "report_filename": f"{job_id}-report.{format}",
    }


@router.get("/jobs/{job_id}/video")
def get_annotated_video(job_id: str):
    _require_enabled()
    job = _job_or_404(job_id)
    video_path = (job.summary or {}).get("video") if job.summary else None
    if not video_path:
        raise HTTPException(status_code=409, detail=f"Annotated video not ready; status={job.status}")
    path = Path(video_path)
    if not path.exists():
        raise HTTPException(status_code=404, detail="Annotated video file is missing")
    return FileResponse(path, media_type="video/mp4", filename=f"{job_id}-annotated.mp4")


@router.get("/jobs/{job_id}/plates")
def get_plates(job_id: str):
    _require_enabled()
    job = _job_or_404(job_id)
    if job.summary is None:
        raise HTTPException(status_code=409, detail=f"Plates not ready; status={job.status}")
    plates = []
    for p in job.summary.get("plates", []):
        plates.append({
            **p,
            "image_url": f"/api/alpr/jobs/{job_id}/plates/{Path(p['image']).name}" if p.get("image") else None,
            "raw_image_url": (
                f"/api/alpr/jobs/{job_id}/plates/{Path(p['raw_image']).name}" if p.get("raw_image") else None
            ),
        })
    return {"job_id": job_id, "plates": plates, "counts": job.summary.get("counts", {})}


@router.get("/jobs/{job_id}/plates/{filename}")
def get_plate_image(job_id: str, filename: str):
    _require_enabled()
    job = _job_or_404(job_id)
    # filename must be a bare basename - no path traversal out of this job's own output dir.
    if "/" in filename or "\\" in filename or filename in (".", ".."):
        raise HTTPException(status_code=400, detail="Invalid filename")
    output_dir = Path(job.output_dir)
    for sub in ("plates", "plates_lowconf"):
        candidate = output_dir / sub / filename
        if candidate.exists():
            return FileResponse(candidate)
    raise HTTPException(status_code=404, detail="Plate image not found")


def _job_stream_generator(job_id: str):
    """Live preview of a job's annotated frames while it's still processing -
    mirrors Guard's/Kitchen's job stream, reading the same write-then-rename
    latest.jpg a running job keeps updating."""
    while True:
        job = get_service().store.get(job_id)
        if job is None:
            break

        frame_path = Path(job.output_dir) / "latest.jpg"
        if frame_path.exists():
            data = None
            for attempt in range(5):
                try:
                    with open(frame_path, "rb") as f:
                        data = f.read()
                    break
                except (PermissionError, FileNotFoundError):
                    if attempt == 4:
                        data = None
                        break
                    time.sleep(0.01)
            if data:
                yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + data + b"\r\n")

        if job.status in {"completed", "failed"}:
            break
        time.sleep(0.1)


@router.get("/jobs/{job_id}/stream")
def get_job_stream(job_id: str):
    _require_enabled()
    _job_or_404(job_id)
    return StreamingResponse(
        _job_stream_generator(job_id),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )
