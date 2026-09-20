from pathlib import Path
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]

PACKAGE_DIR = Path(__file__).resolve().parent

WEIGHTS_DIR = PACKAGE_DIR / "weights"

DATA_DIR = PROJECT_ROOT / "local_data" / "kitchen"

UPLOAD_DIR = DATA_DIR / "uploads"

SESSION_DIR = DATA_DIR / "sessions"

REPORT_DIR = DATA_DIR / "reports"

DB_PATH = DATA_DIR / "kitchen.db"


for directory in [
    WEIGHTS_DIR,
    DATA_DIR,
    UPLOAD_DIR,
    SESSION_DIR,
    REPORT_DIR,
]:
    directory.mkdir(
        parents=True,
        exist_ok=True,
    )


# ============================================================
# TRAINED PPE MODEL
# ============================================================

PPE_MODEL_PATH = (
    WEIGHTS_DIR /
    "best.pt"
)


# ============================================================
# PERSON DETECTOR
# ============================================================
#
# If yolov8n.pt is placed in weights/, it will use the local
# file. Otherwise Ultralytics will try to obtain yolov8n.pt.
#
# Production recommendation:
# keep yolov8n.pt locally after first successful test.
# ============================================================

LOCAL_PERSON_MODEL = (
    WEIGHTS_DIR /
    "yolov8n.pt"
)

PERSON_MODEL_PATH = (
    str(LOCAL_PERSON_MODEL)
    if LOCAL_PERSON_MODEL.exists()
    else "yolov8n.pt"
)


# ============================================================
# DEVICE
# ============================================================

DEVICE = (
    0
    if torch.cuda.is_available()
    else "cpu"
)


# ============================================================
# PPE MODEL SETTINGS
# ============================================================

PPE_IMAGE_SIZE = 768

PPE_CONFIDENCE = 0.25

PPE_IOU = 0.50


# ============================================================
# PERSON MODEL SETTINGS
# ============================================================

PERSON_IMAGE_SIZE = 640

PERSON_CONFIDENCE = 0.35

PERSON_IOU = 0.50


# ============================================================
# TEMPORAL FILTERING
# ============================================================

TEMPORAL_WINDOW = 7

TEMPORAL_MIN_VOTES = 3

CONFLICT_CONFIDENCE_MARGIN = 0.05


# Close tracks after this many missing processed frames.
TRACK_STALE_FRAMES = 30


# Dashboard trend sample frequency
METRIC_SAMPLE_SECONDS = 5.0


# One inference worker initially.
# Increase only after GPU memory/concurrency testing.
MAX_WORKERS = 1


# ============================================================
# EXPECTED TRAINED CLASSES
# ============================================================

EXPECTED_CLASSES = {
    0: "glove",
    1: "hairnet",
    2: "incorrect_mask",
    3: "mask",
    4: "no_glove",
    5: "no_hairnet",
    6: "no_mask",
}


# ============================================================
# DEFAULT BUSINESS SEVERITY
# ============================================================
#
# These are NOT learned by YOLO.
# Change them if your SOP policy differs.
# ============================================================

SEVERITY_MAP = {
    "no_glove": "critical",
    "no_mask": "warning",
    "incorrect_mask": "warning",
    "no_hairnet": "warning",
}