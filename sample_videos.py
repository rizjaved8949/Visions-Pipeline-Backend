"""Lists the demo clips in "Sample videos SOPs/" so any pipeline's frontend
page can offer them as a stand-in for a real upload when no camera or footage
of its own is available. Read-only - nothing here writes to that folder."""

import os

from fastapi import APIRouter

SAMPLE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Sample videos SOPs")
VIDEO_EXTENSIONS = (".mp4", ".mov", ".avi", ".mkv")

router = APIRouter()


@router.get("/list")
def list_sample_videos():
    if not os.path.isdir(SAMPLE_DIR):
        return {"videos": []}
    files = sorted(f for f in os.listdir(SAMPLE_DIR) if f.lower().endswith(VIDEO_EXTENSIONS))
    return {"videos": [{"name": f, "url": f"/sample-videos/{f}"} for f in files]}
