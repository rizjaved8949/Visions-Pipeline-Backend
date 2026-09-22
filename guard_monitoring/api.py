from __future__ import annotations

import base64
import os
import time
import uuid
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel

from . import report
from .config import guard_enabled
from .service import GuardQueueFullError, get_service
from .live_service import get_live_service

router = APIRouter()

_ALLOWED_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}


class HealthResponse(BaseModel):
    enabled: bool
    architecture: str
    models: str


class JobCreated(BaseModel):
    job_id: str
    status: str




class LiveStartRequest(BaseModel):
    source: str
    camera_id: str | None = None
    analysis_fps: float | None = None
    reconnect_seconds: float | None = None
    sleep_seconds: float | None = None
    phone_seconds: float | None = None
    stationary_seconds: float | None = None
    absence_seconds: float | None = None

    def rule_settings(self) -> dict[str, float]:
        values = {
            "sleep": self.sleep_seconds,
            "phone": self.phone_seconds,
            "stationary": self.stationary_seconds,
            "absence": self.absence_seconds,
        }
        return {name: float(value) for name, value in values.items() if value is not None}


class LiveSettingsRequest(BaseModel):
    sleep_seconds: float | None = None
    phone_seconds: float | None = None
    stationary_seconds: float | None = None
    absence_seconds: float | None = None

    def rule_settings(self) -> dict[str, float]:
        values = {
            "sleep": self.sleep_seconds,
            "phone": self.phone_seconds,
            "stationary": self.stationary_seconds,
            "absence": self.absence_seconds,
        }
        return {name: float(value) for name, value in values.items() if value is not None}


class LiveSessionCreated(BaseModel):
    session_id: str
    camera_id: str
    source: str
    status: str
    analysis_fps: float
    frames_processed: int
    reconnect_count: int
    created_at: str
    last_frame_at: str | None = None
    uptime_seconds: float
    settings: dict[str, float]
    error: str | None = None


class JobStatus(BaseModel):
    job_id: str
    status: str
    progress: float | None = None
    frames_processed: int = 0
    total_frames: int | None = None
    error: str | None = None


def _require_enabled() -> None:
    if not guard_enabled():
        raise HTTPException(status_code=503, detail="Guard monitoring is disabled")


def _job_or_404(job_id: str):
    job = get_service().store.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Guard monitoring job not found")
    return job


@router.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse(
        enabled=guard_enabled(),
        architecture="uploaded-video job queue + in-memory live CCTV sessions",
        models="lazy-loaded on inference jobs",
    )


@router.post("/jobs", response_model=JobCreated, status_code=202)
async def create_job(video: UploadFile = File(...)):
    _require_enabled()
    suffix = Path(video.filename or "video.mp4").suffix.lower()
    if suffix not in _ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=400, detail="Unsupported video extension")

    service = get_service()
    target = service.upload_dir / f"{uuid.uuid4().hex}{suffix}"
    max_bytes = max(1, int(os.getenv("GUARD_MAX_UPLOAD_MB", "500"))) * 1024 * 1024
    written = 0

    try:
        with target.open("wb") as out:
            while True:
                chunk = await video.read(1024 * 1024)
                if not chunk:
                    break
                written += len(chunk)
                if written > max_bytes:
                    raise HTTPException(status_code=413, detail="Uploaded video exceeds GUARD_MAX_UPLOAD_MB")
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
    except GuardQueueFullError:
        target.unlink(missing_ok=True)
        raise HTTPException(status_code=503, detail="Guard monitoring queue is full")
    except Exception as exc:
        target.unlink(missing_ok=True)
        raise HTTPException(status_code=500, detail=f"Could not queue guard job: {type(exc).__name__}")

    return JobCreated(job_id=job.job_id, status=job.status)


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
    if job.summary is not None:
        return {"job_id": job_id, "summary": job.summary}
    summary = get_service().read_json_output(job_id, "summary.json")
    if summary is None:
        raise HTTPException(status_code=409, detail=f"Summary not ready; status={job.status}")
    return {"job_id": job_id, "summary": summary}


@router.get("/jobs/{job_id}/events")
def get_events(job_id: str):
    _require_enabled()
    job = _job_or_404(job_id)
    events = get_service().read_events(job_id)
    if events is None:
        raise HTTPException(status_code=409, detail=f"Events not ready; status={job.status}")
    return {"job_id": job_id, "events": events}


@router.get("/jobs/{job_id}/module-health")
def get_module_health(job_id: str):
    _require_enabled()
    job = _job_or_404(job_id)
    data = get_service().read_json_output(job_id, "module_health.json")
    if data is None:
        raise HTTPException(status_code=409, detail=f"Module health not ready; status={job.status}")
    return data


@router.get("/jobs/{job_id}/video")
def get_annotated_video(job_id: str):
    _require_enabled()
    job = _job_or_404(job_id)
    path = Path(job.output_dir) / "annotated.mp4"
    if not path.exists():
        raise HTTPException(status_code=409, detail=f"Annotated video not ready; status={job.status}")
    return FileResponse(path, media_type="video/mp4", filename=f"{job_id}-annotated.mp4")


def _job_stream_generator(job_id: str):
    """Live preview of a job's annotated frames while it's still processing -
    mirrors Kitchen's /sessions/{id}/stream, reading the same write-then-
    rename latest.jpg a running job keeps updating (see pipeline.py's
    _write_latest_frame). Ends once the job leaves "processing"."""
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


def _build_report_response(
    report_id: str,
    *,
    title: str,
    camera_id: str,
    source: str,
    fps: float,
    duty_seconds: float,
    stats: dict,
    events: list[dict],
    fmt: str,
):
    summary_rows, event_rows = report.build_report_rows(
        camera_id=camera_id,
        source=source,
        fps=fps,
        duty_seconds=duty_seconds,
        stats=stats,
        events=events,
    )
    fmt = (fmt or "pdf").lower()
    if fmt == "xlsx":
        data = report.build_xlsx(summary_rows, event_rows)
        media_type = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        filename = f"{report_id}-report.xlsx"
    elif fmt == "pdf":
        data = report.build_pdf(title, summary_rows, event_rows)
        media_type = "application/pdf"
        filename = f"{report_id}-report.pdf"
    else:
        raise HTTPException(status_code=400, detail="format must be 'pdf' or 'xlsx'")
    return {
        "media_type": media_type,
        "report_base64": base64.b64encode(data).decode("ascii"),
        "report_filename": filename,
    }


@router.get("/jobs/{job_id}/report")
def get_job_report(job_id: str, format: str = "pdf"):
    _require_enabled()
    job = _job_or_404(job_id)
    summary = job.summary or get_service().read_json_output(job_id, "summary.json")
    if summary is None:
        raise HTTPException(status_code=409, detail=f"Summary not ready; status={job.status}")
    events = get_service().read_events(job_id) or []
    stats = {
        "visible_frames": summary.get("guard_visible_frames", 0),
        "stationary_frames": summary.get("stationary_frames", 0),
        "sleep_candidate_frames": summary.get("sleep_candidate_frames", 0),
        "phone_use_frames": summary.get("phone_use_frames", 0),
        "absence_frames": summary.get("absence_frames", 0),
    }
    fps = float(summary.get("fps") or 25.0)
    duty_seconds = float(summary.get("frames_processed", 0)) / fps if fps else 0.0
    return _build_report_response(
        job_id,
        title="Guard Monitoring Job Report",
        camera_id=str(summary.get("camera_id", "camera-01")),
        source=str(summary.get("source", job.source_path)),
        fps=fps,
        duty_seconds=duty_seconds,
        stats=stats,
        events=events,
        fmt=format,
    )


# ---------------------------------------------------------------------------
# Real-time CCTV monitoring endpoints. These are additive and do not replace
# the existing uploaded-video /jobs API above.
# ---------------------------------------------------------------------------


def _live_or_404(session_id: str):
    session = get_live_service().get(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Guard live session not found")
    return session


@router.get("/live")
def list_live_sessions():
    _require_enabled()
    return {"sessions": get_live_service().list()}


@router.get("/live/current/stream")
def get_current_live_stream():
    """Stable URL for whichever live session is currently active - mirrors the
    Attendance module's single fixed /api/video_feed endpoint, so the frontend
    doesn't need a session_id in hand before it can start streaming."""
    _require_enabled()
    session_id = get_live_service().current_session_id()
    if session_id is None:
        raise HTTPException(status_code=404, detail="No guard live session has been started yet")
    return StreamingResponse(
        get_live_service().mjpeg(session_id),
        media_type="multipart/x-mixed-replace; boundary=frame",
        headers={
            "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
            "Pragma": "no-cache",
        },
    )


@router.post("/live/start", response_model=LiveSessionCreated, status_code=202)
def start_live_session(body: LiveStartRequest):
    _require_enabled()
    try:
        data = get_live_service().start(
            source=body.source,
            camera_id=body.camera_id,
            analysis_fps=body.analysis_fps,
            reconnect_seconds=body.reconnect_seconds,
            rule_settings=body.rule_settings(),
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return LiveSessionCreated(**data)


@router.get("/live/{session_id}/state")
def get_live_state(session_id: str):
    _require_enabled()
    _live_or_404(session_id)
    data = get_live_service().state(session_id)
    if data is None:
        raise HTTPException(status_code=404, detail="Guard live session not found")
    return data


@router.get("/live/{session_id}/events")
def get_live_events(session_id: str):
    _require_enabled()
    _live_or_404(session_id)
    events = get_live_service().events(session_id)
    return {"session_id": session_id, "events": events or []}


@router.get("/live/{session_id}/report")
def get_live_report(session_id: str, format: str = "pdf"):
    _require_enabled()
    session = _live_or_404(session_id)
    public = session.public()
    events = get_live_service().events(session_id) or []
    return _build_report_response(
        session_id,
        title="Guard Live Monitoring Report",
        camera_id=public["camera_id"],
        source=public["source"],
        fps=float(public["analysis_fps"] or 3.0),
        duty_seconds=float(public["uptime_seconds"]),
        stats=public["stats"],
        events=events,
        fmt=format,
    )


@router.get("/live/{session_id}/module-health")
def get_live_module_health(session_id: str):
    _require_enabled()
    _live_or_404(session_id)
    data = get_live_service().module_health(session_id)
    return {"session_id": session_id, "module_health": data or {}}


@router.patch("/live/{session_id}/settings")
def update_live_settings(session_id: str, body: LiveSettingsRequest):
    _require_enabled()
    _live_or_404(session_id)
    settings = body.rule_settings()
    if not settings:
        raise HTTPException(status_code=400, detail="Provide at least one rule duration")
    try:
        data = get_live_service().update_settings(session_id, settings)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if data is None:
        raise HTTPException(status_code=404, detail="Guard live session not found")
    return data


@router.post("/live/{session_id}/stop")
def stop_live_session(session_id: str):
    _require_enabled()
    _live_or_404(session_id)
    data = get_live_service().stop(session_id)
    if data is None:
        raise HTTPException(status_code=404, detail="Guard live session not found")
    return data


@router.get("/live/{session_id}/stream")
def get_live_stream(session_id: str):
    _require_enabled()
    _live_or_404(session_id)
    return StreamingResponse(
        get_live_service().mjpeg(session_id),
        media_type="multipart/x-mixed-replace; boundary=frame",
        headers={
            "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
            "Pragma": "no-cache",
        },
    )
