from pathlib import Path

import torch

PACKAGE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_DIR.parent

WEIGHTS_DIR = PACKAGE_DIR / "weights"

DATA_DIR = PROJECT_ROOT / "local_data" / "restricted_zone"
UPLOAD_DIR = DATA_DIR / "uploads"
SNAPSHOT_DIR = DATA_DIR / "snapshots"
OUTPUT_DIR = DATA_DIR / "outputs"

for directory in (WEIGHTS_DIR, DATA_DIR, UPLOAD_DIR, SNAPSHOT_DIR, OUTPUT_DIR):
    directory.mkdir(parents=True, exist_ok=True)


# ============================================================
# PERSON / OBJECT DETECTOR
# ============================================================
#
# yolov8n.pt lives in weights/ and is tracked with Git LFS, so a
# `git clone` / `git lfs pull` gets it automatically - no first-run
# download needed. Falls back to letting Ultralytics fetch it if it's
# ever missing (e.g. LFS not pulled yet).
# ============================================================

LOCAL_PERSON_MODEL = WEIGHTS_DIR / "yolov8n.pt"
PERSON_MODEL_PATH = str(LOCAL_PERSON_MODEL) if LOCAL_PERSON_MODEL.exists() else "yolov8n.pt"

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
HALF = torch.cuda.is_available()

DETECTION_CONF = 0.40
DETECTION_IOU = 0.50
DETECTION_IMGSZ = 640
DETECTION_CLASSES = None  # None = every COCO class Ultralytics knows
TRACKER = "bytetrack.yaml"


# ============================================================
# BREACH LOGIC
# ============================================================

# "bottom" (feet) matches a floor-plane zone polygon under a high,
# downward-looking camera far better than the raw bbox centre.
ANCHOR = "bottom"
ENTER_FRAMES = 3       # anchor must be inside this many consecutive frames -> entry
EXIT_FRAMES = 10        # ... and outside this many -> exit (anti-flicker)
TRACK_TTL_S = 5.0       # forget a track not seen for this long
DUP_RADIUS_PX = 60.0    # suppress a re-alert from a tracker id-swap within this radius...
DUP_WINDOW_S = 2.0      # ...and this many seconds of a previous alert
RE_ALERT_COOLDOWN_S = 0.0  # 0 = alert again on every genuine re-entry
