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
# TRAINED PPE MODEL  (mask / gloves / hairnet — 7 classes)
# ============================================================

PPE_MODEL_PATH = (
    WEIGHTS_DIR /
    "best.pt"
)


# ============================================================
# APRON MODEL  (apron / no_apron — 2 classes)
# Download apron_detector_best.pt from Kaggle and place it here.
# If the file is absent the pipeline falls back gracefully:
# apron status is reported as "unsupported" instead of
# raising an error, so the rest of the pipeline keeps running.
# ============================================================

APRON_MODEL_PATH = (
    WEIGHTS_DIR /
    "apron_detector_best.pt"
)


# ============================================================
# PERSON DETECTOR
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
# APRON MODEL SETTINGS
# Tune APRON_CONFIDENCE to the value recommended in
# apron_model_metadata.json (threshold_analysis output).
# ============================================================

APRON_IMAGE_SIZE = 640

APRON_CONFIDENCE = 0.25

APRON_IOU = 0.50


# ============================================================
# PERSON MODEL SETTINGS
# ============================================================

PERSON_IMAGE_SIZE = 640

PERSON_CONFIDENCE = 0.10

PERSON_TRACKER_PATH = PACKAGE_DIR / "person_tracker.yaml"

PERSON_IOU = 0.50


# ============================================================
# TEMPORAL FILTERING
# ============================================================

TEMPORAL_WINDOW = 7

TEMPORAL_MIN_VOTES = 3

CONFLICT_CONFIDENCE_MARGIN = 0.05


# Source-time evidence/card retention and admission of new staff identities.
PPE_EVIDENCE_SECONDS = 1.0
# Count actual observations within this window, not empty video frames.
PPE_CONFIRM_SECONDS = 1.0
# Slow streams need enough time for three distinct observations; never keep
# an arbitrarily old vote to manufacture confirmation.
PPE_CONFIRM_MAX_SECONDS = 6.0
# Clothing stays on through brief missed detections; contradictory evidence
# still clears a held decision immediately. Mask retains the shorter timeout.
PPE_HOLD_SECONDS = {"mask": 1.0, "gloves": 3.0, "hair_cover": 3.0, "apron": 3.0}
PERSON_HOLD_SECONDS = 2.0
PERSON_CONFIRM_FRAMES = 3
PERSON_CONFIRM_CONFIDENCE = 0.60


# Dashboard trend sample frequency
METRIC_SAMPLE_SECONDS = 5.0


# One inference worker initially.
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

EXPECTED_APRON_CLASSES = {
    0: "apron",
    1: "no_apron",
}


# ============================================================
# DEFAULT BUSINESS SEVERITY
# ============================================================

SEVERITY_MAP = {
    "no_glove"      : "critical",
    "no_mask"       : "warning",
    "incorrect_mask": "warning",
    "no_hairnet"    : "warning",
    "no_apron"      : "warning",
}

# Bounded rechecks for unresolved PPE, across all four requirements.
PPE_RECHECK_ENABLED = True
PPE_RECHECK_IMAGE_SIZE = 1024
APRON_RECHECK_IMAGE_SIZE = 512
PPE_RECHECK_MAX_PEOPLE = 2
PPE_RECHECK_MAX_HEIGHT_RATIO = 0.70

# Small people can be retried immediately; nearby people are retried only
# after an unresolved interval, avoiding extra work on a one-frame miss.
PPE_RECHECK_NEAR_DELAY_SECONDS = 0.5
