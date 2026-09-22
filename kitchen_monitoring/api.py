import shutil
import time

from pathlib import Path
from typing import Literal

import cv2
import numpy as np

from fastapi import (
    APIRouter,
    File,
    Form,
    HTTPException,
    Query,
    UploadFile,
)

from fastapi.responses import (
    FileResponse,
    StreamingResponse,
)

from pydantic import (
    BaseModel,
)

from .config import (
    UPLOAD_DIR,
    DB_PATH,
)

from .model_registry import (
    MODELS,
)

from .reporting import (
    create_csv,
    create_pdf,
    create_xlsx,
)

from .service import (
    SERVICE,
)

from .storage import (
    STORE,
)


router = APIRouter()


# ============================================================
# SCHEMAS
# ============================================================

class CameraRequest(
    BaseModel
):

    source: str

    camera_id: str = "CAM-03"


# ============================================================
# HEALTH
# ============================================================

@router.get(
    "/health"
)
def health():

    return {
        "status": "ready",
        "models":
            MODELS.health(),

        "storage":
            str(DB_PATH),

        "supported_ppe": [
            "mask",
            "gloves",
            "hair_cover",
        ],

        "unsupported_ppe": [
            "apron",
        ],
    }


# ============================================================
# SINGLE IMAGE PPE TEST
# ============================================================

@router.post(
    "/detect/image"
)
async def detect_image(
    file: UploadFile = File(...),
):

    content = await file.read()


    data = np.frombuffer(
        content,
        dtype=np.uint8,
    )


    frame = cv2.imdecode(
        data,
        cv2.IMREAD_COLOR,
    )


    if frame is None:

        raise HTTPException(
            status_code=400,
            detail="Invalid image",
        )


    try:

        result = MODELS.predict_ppe(
            frame
        )

    except FileNotFoundError as exc:

        raise HTTPException(
            status_code=503,
            detail=str(exc),
        )


    detections = []


    if result.boxes is not None:

        for box in result.boxes:

            class_id = int(
                box.cls[0]
                .detach()
                .cpu()
                .item()
            )


            detections.append(
                {
                    "class_id":
                        class_id,

                    "class_name":
                        result.names[
                            class_id
                        ],

                    "confidence":
                        float(
                            box.conf[0]
                            .detach()
                            .cpu()
                            .item()
                        ),

                    "bbox":
                        [
                            float(v)
                            for v
                            in box.xyxy[0]
                            .detach()
                            .cpu()
                            .tolist()
                        ],
                }
            )


    return {
        "filename":
            file.filename,

        "count":
            len(
                detections
            ),

        "detections":
            detections,
    }


# ============================================================
# UPLOAD VIDEO
# ============================================================

@router.post(
    "/sessions/upload"
)
async def upload_video(
    file: UploadFile = File(...),

    camera_id: str = Form(
        "CAM-03"
    ),
):

    extension = (
        Path(
            file.filename
            or "video.mp4"
        )
        .suffix
        .lower()
    )


    if extension not in {
        ".mp4",
        ".avi",
        ".mov",
        ".mkv",
        ".mpeg",
        ".mpg",
    }:

        raise HTTPException(
            status_code=400,
            detail=(
                "Unsupported video format"
            ),
        )


    destination = (
        UPLOAD_DIR
        /
        (
            f"{int(time.time() * 1000)}"
            f"{extension}"
        )
    )


    with open(
        destination,
        "wb",
    ) as output:

        shutil.copyfileobj(
            file.file,
            output,
        )


    session_id = SERVICE.start(
        source=str(
            destination
        ),
        source_type="upload",
        camera_id=camera_id,
    )


    return {
        "session_id":
            session_id,

        "status":
            "queued",
    }


# ============================================================
# CAMERA / RTSP
# ============================================================

@router.post(
    "/sessions/camera"
)
def start_camera(
    request: CameraRequest,
):

    source = request.source


    # Allows:
    # "0" -> local camera index 0
    # rtsp://...
    if source.isdigit():

        source_value = int(
            source
        )

    else:

        source_value = source


    session_id = SERVICE.start(
        source=source_value,
        source_type="camera",
        camera_id=
            request.camera_id,
    )


    return {
        "session_id":
            session_id,

        "status":
            "queued",

        "camera_id":
            request.camera_id,
    }


# ============================================================
# STOP
# ============================================================

@router.post(
    "/sessions/{session_id}/stop"
)
def stop_session(
    session_id: str,
):

    session = STORE.get_session(
        session_id
    )


    if session is None:

        raise HTTPException(
            404,
            "Session not found",
        )


    stopped = SERVICE.stop(
        session_id
    )


    return {
        "session_id":
            session_id,

        "stop_requested":
            stopped,
    }


# ============================================================
# SESSION STATUS
# ============================================================

@router.get(
    "/sessions/{session_id}"
)
def session_status(
    session_id: str,
):

    session = STORE.get_session(
        session_id
    )


    if session is None:

        raise HTTPException(
            404,
            "Session not found",
        )


    return session


# ============================================================
# DASHBOARD
# ============================================================

@router.get(
    "/sessions/{session_id}/dashboard"
)
def dashboard(
    session_id: str,
):

    session = STORE.get_session(
        session_id
    )


    if session is None:

        raise HTTPException(
            404,
            "Session not found",
        )


    summary = (
        session.get(
            "summary"
        )
        or {}
    )


    violations = (
        STORE.list_violations(
            session_id,
            active_only=True,
        )
    )


    trend = STORE.get_metrics(
        session_id,
        limit=100,
    )


    return {
        "session_id":
            session_id,

        "status":
            session[
                "status"
            ],

        "camera_id":
            session.get(
                "camera_id"
            ),

        "summary": {
            "staff_detected":
                summary.get(
                    "staff_detected",
                    0,
                ),

            "fully_compliant":
                summary.get(
                    "fully_compliant",
                    0,
                ),

            "open_violations":
                summary.get(
                    "open_violations",
                    0,
                ),

            "compliance_score":
                summary.get(
                    "compliance_score"
                ),
        },

        "requirements":
            summary.get(
                "requirements",
                {},
            ),

        "persons":
            summary.get(
                "persons",
                [],
            ),

        "active_violations":
            violations,

        "compliance_trend":
            trend,
    }


# ============================================================
# PERSONS
# ============================================================

@router.get(
    "/sessions/{session_id}/persons"
)
def persons(
    session_id: str,
):

    session = STORE.get_session(
        session_id
    )


    if session is None:

        raise HTTPException(
            404,
            "Session not found",
        )


    return {
        "session_id":
            session_id,

        "persons":
            (
                session
                .get(
                    "summary",
                    {}
                )
                .get(
                    "persons",
                    [],
                )
            ),
    }


# ============================================================
# VIOLATIONS
# ============================================================

@router.get(
    "/sessions/{session_id}/violations"
)
def violations(
    session_id: str,

    active_only: bool = Query(
        False
    ),
):

    if STORE.get_session(
        session_id
    ) is None:

        raise HTTPException(
            404,
            "Session not found",
        )


    return {
        "session_id":
            session_id,

        "violations":
            STORE.list_violations(
                session_id,
                active_only=
                    active_only,
            ),
    }


# ============================================================
# LATEST ANNOTATED FRAME
# ============================================================

@router.get(
    "/sessions/{session_id}/frame"
)
def latest_frame(
    session_id: str,
):

    session = STORE.get_session(
        session_id
    )


    if session is None:

        raise HTTPException(
            404,
            "Session not found",
        )


    frame_path = session.get(
        "latest_frame"
    )


    if (
        not frame_path
        or
        not Path(
            frame_path
        ).exists()
    ):

        raise HTTPException(
            404,
            "Frame not ready",
        )


    return FileResponse(
        frame_path,
        media_type="image/jpeg",
    )


# ============================================================
# MJPEG STREAM
# ============================================================

def _mjpeg_generator(session_id):

    while True:

        session = (
            STORE.get_session(
                session_id
            )
        )


        if session is None:

            break


        frame_path = (
            session.get(
                "latest_frame"
            )
        )


        if (
            frame_path
            and
            Path(
                frame_path
            ).exists()
        ):

            # The pipeline thread writes this same file via a write-then-
            # atomic-rename (see pipeline.py) so a reader never sees a
            # half-written frame - but the rename itself can transiently
            # deny access to a concurrent reader on Windows. That used to
            # be an uncaught PermissionError here, which crashed this whole
            # streaming response (visible as the video dying mid-stream).
            # Skip this one polling tick and pick up the next frame instead.
            data = None

            for attempt in range(5):

                try:

                    with open(
                        frame_path,
                        "rb",
                    ) as f:

                        data = f.read()

                    break

                except (
                    PermissionError,
                    FileNotFoundError,
                ):

                    if attempt == 4:
                        data = None
                        break

                    time.sleep(0.01)

            if data:

                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n"
                    +
                    data
                    +
                    b"\r\n"
                )


        if (
            session[
                "status"
            ]
            in {
                "completed",
                "stopped",
                "failed",
            }
        ):

            break


        time.sleep(
            0.10
        )


@router.get(
    "/sessions/current/stream"
)
def stream_current():
    """Stable URL for whichever session is currently active - mirrors the
    Guard module's /live/current/stream, so the frontend doesn't need a
    session_id in hand before it can start streaming."""

    session_id = (
        SERVICE.current_session_id()
    )

    if session_id is None:

        raise HTTPException(
            404,
            "No kitchen session has been started yet",
        )

    return StreamingResponse(
        _mjpeg_generator(session_id),
        media_type=(
            "multipart/x-mixed-replace;"
            " boundary=frame"
        ),
    )


@router.get(
    "/sessions/{session_id}/stream"
)
def stream(
    session_id: str,
):

    if STORE.get_session(
        session_id
    ) is None:

        raise HTTPException(
            404,
            "Session not found",
        )

    return StreamingResponse(
        _mjpeg_generator(session_id),
        media_type=(
            "multipart/x-mixed-replace;"
            " boundary=frame"
        ),
    )


# ============================================================
# ANNOTATED VIDEO
# ============================================================

@router.get(
    "/sessions/{session_id}/video"
)
def video(
    session_id: str,
):

    session = STORE.get_session(
        session_id
    )


    if session is None:

        raise HTTPException(
            404,
            "Session not found",
        )


    path = session.get(
        "output_video"
    )


    if (
        not path
        or
        not Path(path).exists()
    ):

        raise HTTPException(
            404,
            "Annotated video not ready",
        )


    return FileResponse(
        path,
        media_type="video/mp4",
        filename=(
            f"{session_id}_annotated.mp4"
        ),
    )


# ============================================================
# REPORT EXPORT
# ============================================================

@router.get(
    "/sessions/{session_id}/report"
)
def report(
    session_id: str,

    format: Literal[
        "pdf",
        "xlsx",
        "csv",
    ] = Query(
        "pdf"
    ),
):

    if STORE.get_session(
        session_id
    ) is None:

        raise HTTPException(
            404,
            "Session not found",
        )


    if format == "pdf":

        path = create_pdf(
            session_id
        )

        media_type = (
            "application/pdf"
        )


    elif format == "xlsx":

        path = create_xlsx(
            session_id
        )

        media_type = (
            "application/"
            "vnd.openxmlformats-"
            "officedocument."
            "spreadsheetml.sheet"
        )


    else:

        path = create_csv(
            session_id
        )

        media_type = (
            "text/csv"
        )


    return FileResponse(
        path,
        media_type=
            media_type,

        filename=
            Path(path).name,
    )