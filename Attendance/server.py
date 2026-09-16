import asyncio
import base64
import threading
from typing import Any, Dict

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from . import attendance

REPORT_MEDIA_TYPES = {
    "xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    "pdf": "application/pdf",
}

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_camera_thread = None


@app.on_event("startup")
def _wipe_on_startup():
    """Guarantee a clean slate on every boot, regardless of how the previous
    run ended (a clean Stop, a killed process, a crash) - nothing should
    persist across restarts."""
    attendance.wipe_all_data()


@app.on_event("shutdown")
def _wipe_on_shutdown():
    """Best-effort cleanup on a graceful shutdown (e.g. Ctrl+C). This can't
    catch a hard kill/crash - that's what the startup wipe above is for."""
    attendance.wipe_all_data()


def _ensure_faces_loaded():
    attendance.load_existing_faces_from_disk()


def _ensure_started():
    global _camera_thread
    _ensure_faces_loaded()
    if _camera_thread is None or not _camera_thread.is_alive():
        _camera_thread = threading.Thread(target=attendance.camera_loop, daemon=True)
        _camera_thread.start()


@app.get("/api/cameras")
def cameras():
    """Local camera devices detected on this machine, plus whichever source
    is currently selected (a local index or a network stream URL)."""
    return {
        "cameras": attendance.list_available_cameras(),
        "current_source": attendance.get_video_source(),
    }


class CameraSourceRequest(BaseModel):
    source: str


@app.post("/api/camera/source")
def set_camera_source(body: CameraSourceRequest):
    """Change the video source used the NEXT time a session starts - a
    local device index ("0", "1", ...) or a network stream URL
    (rtsp://... or http://... from a phone/laptop acting as an IP camera).
    Has no effect on an already-running session."""
    if _camera_thread is not None and _camera_thread.is_alive():
        raise HTTPException(400, "Stop the current session before changing the camera source.")
    try:
        parsed = attendance.set_video_source(body.source)
    except ValueError as e:
        raise HTTPException(400, str(e))
    return {"current_source": parsed}


@app.post("/api/session/start")
def session_start():
    _ensure_faces_loaded()
    if not attendance.all_required_persons:
        raise HTTPException(400, "Register at least one person before starting a session.")
    _ensure_started()
    attendance.start_session()
    return attendance.get_status()


@app.post("/api/session/stop")
def session_stop():
    attendance.stop_session()
    attendance.stop_camera_loop()
    # Wait for the camera thread to actually finish (cap.release()) before
    # responding, so the caller can be sure the physical device is free.
    if _camera_thread is not None:
        _camera_thread.join(timeout=3)

    # Data is NOT wiped here anymore - the frontend shows a download dialog
    # first (format choice, etc.) and only /api/session/report (below)
    # actually builds the report and wipes, once the user confirms.
    return attendance.get_status()


@app.get("/api/session/report")
def session_report(format: str = "xlsx"):
    """Build the just-finished session's report in the requested format,
    then wipe all data. Called only once the user confirms via the download
    dialog - nothing is wiped until this actually runs."""
    if format not in REPORT_MEDIA_TYPES:
        raise HTTPException(400, "format must be 'xlsx' or 'pdf'.")

    builder = attendance.build_session_report_pdf if format == "pdf" else attendance.build_session_report_xlsx
    report_bytes = builder()
    attendance.wipe_all_data()

    if report_bytes is None:
        raise HTTPException(404, "No session data to report.")

    return {
        "media_type": REPORT_MEDIA_TYPES[format],
        "report_base64": base64.b64encode(report_bytes).decode("ascii"),
        "report_filename": f"attendance_report.{format}",
    }


@app.post("/api/wipe_data")
def wipe_data():
    """Delete all registered people and session history on demand - used by
    the frontend once a processed upload's result/report has been
    downloaded, or the user leaves that view."""
    attendance.wipe_all_data()
    return attendance.get_status()


@app.get("/api/status")
def status():
    return attendance.get_status()


@app.get("/api/people")
def people():
    _ensure_faces_loaded()
    return {"people": attendance.get_registered_people()}


@app.post("/api/people/register")
async def register_person(name: str = Form(...), file: UploadFile = File(...)):
    content = await file.read()
    try:
        safe_name = attendance.register_person(name, content)
    except ValueError as e:
        raise HTTPException(400, str(e))
    return {"name": safe_name, "people": attendance.get_registered_people()}


@app.post("/api/process_media")
async def process_media(file: UploadFile = File(...)):
    _ensure_faces_loaded()
    if not attendance.all_required_persons:
        raise HTTPException(400, "Register at least one person before uploading media.")

    content = await file.read()
    content_type = file.content_type or ""

    try:
        if content_type.startswith("image"):
            result_bytes, media_type, summary = attendance.process_image_bytes(content)
        elif content_type.startswith("video"):
            result_bytes, media_type, summary = attendance.process_video_bytes(content)
        else:
            raise HTTPException(400, "Unsupported file type - upload an image or video.")
    except ValueError as e:
        raise HTTPException(400, str(e))

    return {
        "media_type": media_type,
        "media_base64": base64.b64encode(result_bytes).decode("ascii"),
        "summary": summary,
        "source_filename": file.filename or "upload",
    }


class MediaReportRequest(BaseModel):
    filename: str
    summary: Dict[str, Any]
    format: str = "xlsx"


@app.post("/api/media_report")
def media_report(body: MediaReportRequest):
    """Build a report for an already-processed upload in the requested
    format. Takes the summary the frontend already has (from
    /api/process_media) rather than caching anything server-side. Does NOT
    wipe - that's still triggered separately by downloading the result/report
    or leaving the media view (see /api/wipe_data)."""
    if body.format not in REPORT_MEDIA_TYPES:
        raise HTTPException(400, "format must be 'xlsx' or 'pdf'.")

    builder = attendance.build_media_report_pdf if body.format == "pdf" else attendance.build_media_report_xlsx
    report_bytes = builder(body.filename, body.summary)

    return {
        "media_type": REPORT_MEDIA_TYPES[body.format],
        "report_base64": base64.b64encode(report_bytes).decode("ascii"),
        "report_filename": f"media_report.{body.format}",
    }


@app.get("/api/video_feed")
async def video_feed(request: Request):
    async def generate():
        try:
            while True:
                if await request.is_disconnected():
                    break
                # get_latest_jpeg() does cv2.imencode, which is CPU-bound and
                # synchronous - run it in a worker thread so it never blocks
                # the event loop (which also has to service /api/status etc.
                # and would otherwise fight camera_loop's own thread for CPU).
                jpeg = await asyncio.to_thread(attendance.get_latest_jpeg)
                if jpeg is not None:
                    yield (b"--frame\r\n"
                           b"Content-Type: image/jpeg\r\n\r\n" + jpeg + b"\r\n")
                await asyncio.sleep(0.05)
        except asyncio.CancelledError:
            pass

    return StreamingResponse(generate(), media_type="multipart/x-mixed-replace; boundary=frame")


def run(host="0.0.0.0", port=8000):
    import uvicorn
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    run()
