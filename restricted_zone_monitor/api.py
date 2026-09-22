import asyncio
import base64
from typing import List, Optional

from fastapi import APIRouter, File, HTTPException, Request, UploadFile
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from . import monitor, reporting

router = APIRouter()

REPORT_MEDIA_TYPES = {
    "csv": "text/csv",
    "pdf": "application/pdf",
}


class CameraSourceRequest(BaseModel):
    source: str


class ZonePayload(BaseModel):
    name: Optional[str] = None
    points: List[List[float]]
    color: Optional[List[int]] = None


class ZonesRequest(BaseModel):
    zones: List[ZonePayload]


def _zone_set_payload(zone_set):
    return {"zones": [{"name": z.name, "points": z.points, "color": list(z.color)} for z in zone_set.zones]}


@router.get("/cameras")
def cameras():
    """Local camera devices detected on this machine, plus whichever source
    is currently selected (a local index, a network stream URL, or an
    uploaded video file's path)."""
    return {"cameras": monitor.list_available_cameras(), "current_source": monitor.get_video_source()}


@router.post("/camera/source")
def set_camera_source(body: CameraSourceRequest):
    """Change the source used the NEXT time a session starts. Has no effect
    on an already-running session."""
    try:
        parsed = monitor.set_video_source(body.source)
    except ValueError as e:
        raise HTTPException(400, str(e))
    return {"current_source": parsed}


@router.post("/upload")
async def upload_video(file: UploadFile = File(...)):
    """Upload a video file and select it as the source for the next session."""
    content = await file.read()
    try:
        path = monitor.save_uploaded_video(file.filename, content)
        parsed = monitor.set_video_source(path)
    except ValueError as e:
        raise HTTPException(400, str(e))
    return {"current_source": parsed}


@router.get("/zones/frame")
def zones_frame():
    """A fresh snapshot from the current source for the frontend to draw
    zone polygon(s) on. Call this again for every new file/session - nothing
    about a previous run's zones carries over."""
    try:
        jpeg, width, height = monitor.get_zone_frame_jpeg()
    except (RuntimeError, FileNotFoundError) as e:
        raise HTTPException(400, str(e))
    return {"frame_base64": base64.b64encode(jpeg).decode("ascii"), "width": width, "height": height}


@router.get("/zones")
def get_zones():
    return _zone_set_payload(monitor.get_zones())


@router.post("/zones")
def set_zones(body: ZonesRequest):
    """Set the zone(s) to monitor for the next session. Points are
    normalised 0..1 against the frame returned by GET /zones/frame. Kept in
    memory only - never written to disk, and cleared on wipe."""
    try:
        zone_set = monitor.set_zones([z.model_dump() for z in body.zones])
    except ValueError as e:
        raise HTTPException(400, str(e))
    return _zone_set_payload(zone_set)


@router.post("/session/start")
def session_start():
    try:
        return monitor.start_session()
    except ValueError as e:
        raise HTTPException(400, str(e))


@router.post("/session/stop")
def session_stop():
    """Stop the active session. Data is NOT wiped here - the frontend shows
    a download dialog first, and only /session/report (once confirmed) or
    /wipe_data (if the user declines) actually clears it."""
    try:
        return monitor.stop_session()
    except ValueError as e:
        raise HTTPException(400, str(e))


@router.get("/session/report")
def session_report(format: str = "csv"):
    """Build the just-finished session's report in the requested format,
    then wipe all session + zone data. Called once the user confirms via the
    download dialog."""
    if format not in REPORT_MEDIA_TYPES:
        raise HTTPException(400, "format must be 'csv' or 'pdf'.")

    builder = reporting.build_session_report_pdf if format == "pdf" else reporting.build_session_report_csv
    report_bytes = builder()
    monitor.wipe_all_data()

    if report_bytes is None:
        raise HTTPException(404, "No session data to report.")

    return {
        "media_type": REPORT_MEDIA_TYPES[format],
        "report_base64": base64.b64encode(report_bytes).decode("ascii"),
        "report_filename": f"restricted_zone_report.{format}",
    }


@router.post("/wipe_data")
def wipe_data():
    """Discard the finished session's data on demand - used by the frontend
    once a report has been downloaded, or if the user leaves the download
    view without downloading anything."""
    monitor.wipe_all_data()
    return monitor.get_status()


@router.get("/status")
def status():
    return monitor.get_status()


@router.get("/video_feed")
async def video_feed(request: Request):
    async def generate():
        try:
            while True:
                if await request.is_disconnected():
                    break
                jpeg = await asyncio.to_thread(monitor.get_latest_jpeg)
                if jpeg is not None:
                    yield (b"--frame\r\n"
                           b"Content-Type: image/jpeg\r\n\r\n" + jpeg + b"\r\n")
                await asyncio.sleep(0.05)
        except asyncio.CancelledError:
            pass

    return StreamingResponse(generate(), media_type="multipart/x-mixed-replace; boundary=frame")
