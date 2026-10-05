import cv2
import numpy as np

CLS_COLORS = {
    "car": (60, 200, 60),
    "motorcycle": (255, 160, 0),
    "bicycle": (255, 220, 0),
    "bus": (0, 140, 255),
    "truck": (180, 60, 255),
}
PLATE_COLOR = (0, 255, 255)
DONE_COLOR = (255, 255, 255)


def _label(img, text, x, y, color, scale=0.55, thick=1):
    (tw, th), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)
    y0 = max(th + 4, y)
    cv2.rectangle(img, (x, y0 - th - 4), (x + tw + 4, y0 + base - 2), color, -1)
    cv2.putText(img, text, (x + 2, y0 - 2), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thick, cv2.LINE_AA)


def draw(frame, tracks: dict, fps: float | None = None, pending: int = 0):
    # Fixed sizes on purpose. Scaling these with frame height was tried and reverted:
    # on a 2160x3840 source it multiplied everything by 3.5x and the labels swamped the
    # picture. Legibility of the saved 4K video is not worth making the live view unusable.
    s_lbl, s_small, th = 0.55, 0.45, 1
    box_th = 2

    for tid, t in tracks.items():
        if t.last_vehicle_box is None:
            continue
        x1, y1, x2, y2 = map(int, t.last_vehicle_box)
        color = CLS_COLORS.get(t.cls_name, (200, 200, 200))
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, box_th)

        status = ""
        if t.result_status == "done":
            status = " | saved"
        elif t.result_status == "pending":
            status = " | ..."
        elif t.result_status == "lowconf":
            status = " | low-conf (quarantined)"
        elif t.result_status == "skipped":
            status = " | no plate"
        _label(frame, f"#{tid} {t.cls_name}{status}", x1, y1,
               color if not status.endswith("saved") else DONE_COLOR, scale=s_lbl, thick=th)

        if t.last_plate_box is not None:
            px1, py1, px2, py2 = map(int, t.last_plate_box)
            if t.finalized:
                box_col, thick = DONE_COLOR, max(1, box_th // 2)
            else:
                box_col, thick = PLATE_COLOR, box_th
            cv2.rectangle(frame, (px1, py1), (px2, py2), box_col, thick)
            if t.last_plate_conf:
                # confidence is the single most useful number when diagnosing a miss or a
                # false positive, since plate.conf is what decides whether it is kept
                _label(frame, f"{t.last_plate_conf:.2f}", px1, max(12, py1 - 2),
                       box_col, scale=s_small, thick=th)

    hud = "ALPR detect"
    if fps is not None:
        hud += f" | {fps:.1f} FPS"
    if pending:
        hud += f" | queue: {pending}"
    _label(frame, hud, 8, 24, (255, 255, 255), scale=0.6, thick=th)
    return frame


def screen_size(default=(1920, 1080)) -> tuple[int, int]:
    """Usable screen size, so the window can be as large as actually fits."""
    try:
        import ctypes
        u = ctypes.windll.user32
        u.SetProcessDPIAware()
        w, h = u.GetSystemMetrics(0), u.GetSystemMetrics(1)
        if w > 0 and h > 0:
            return w, h
    except Exception:  # noqa: BLE001 - not Windows, or no desktop
        pass
    return default


def fit_scale(w: int, h: int, max_width: int | None, max_height: int | None) -> float:
    """Shrink factor so the frame fits inside BOTH limits (never enlarges).

    Height matters as much as width: a portrait 4K source (2160x3840) capped on width
    alone is still 2275 px tall, so the window runs off-screen and only the top of the
    frame is visible - which looks like the video has been cropped or zoomed in.
    """
    r = 1.0
    if max_width and w > max_width:
        r = min(r, max_width / w)
    if max_height and h > max_height:
        r = min(r, max_height / h)
    return r


def resize_for_display(frame, max_width, max_height=None):
    h, w = frame.shape[:2]
    r = fit_scale(w, h, max_width, max_height)
    if r >= 1.0:
        return frame
    return cv2.resize(frame, (max(1, int(w * r)), max(1, int(h * r))), interpolation=cv2.INTER_AREA)
