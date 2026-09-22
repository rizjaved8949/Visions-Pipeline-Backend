# Restricted Zone Monitor

Detects objects (YOLO + ByteTrack) crossing into user-drawn restricted zones
and raises one alert per (object, zone) entry. Mounted at `/api/restricted-zone`.

Session model mirrors `Attendance`: draw zones -> start a session -> stop it
-> the frontend shows a download dialog (CSV or PDF) -> the report is built
and **all session data (including breach snapshot images) is deleted the
moment the report bytes are returned**. Nothing here is meant to persist
between sessions - not even the zones, which are drawn fresh every time.

## Setup

Weights (`weights/yolov8n.pt`) are tracked with Git LFS - run `git lfs install`
once per machine, then a normal `git clone` / `git pull` fetches them. If
they're ever missing, Ultralytics will fall back to downloading `yolov8n.pt`
itself on first use.

## Typical flow (what the frontend should call)

1. `POST /api/restricted-zone/camera/source` `{"source": "0"}` (webcam index,
   an `rtsp://`/`http://` stream URL) - or `POST /upload` with a video file
   instead.
2. `GET /api/restricted-zone/zones/frame` - a fresh snapshot (base64 JPEG) to
   draw on. Call this again for every new session; nothing is cached.
3. `POST /api/restricted-zone/zones` - the polygon(s) the user drew, points
   normalised 0..1 against that frame's width/height.
4. `POST /api/restricted-zone/session/start`
5. `GET /api/restricted-zone/video_feed` - live MJPEG stream with zones,
   boxes and the breach banner drawn in; `GET /status` for JSON state/poll.
6. `POST /api/restricted-zone/session/stop` - ends monitoring, does **not**
   wipe anything yet (report is still buildable).
7. Show the download dialog. On confirm: `GET /session/report?format=csv`
   or `?format=pdf` - returns `{media_type, report_base64, report_filename}`
   and wipes all session + zone data as a side effect.
   If the user closes the dialog / navigates away without downloading,
   call `POST /wipe_data` instead so nothing lingers.

A video file source ends the session automatically at EOF (same wipe-on-
report rule applies) - `stop_session` is only needed to end a live
camera/stream session early.
