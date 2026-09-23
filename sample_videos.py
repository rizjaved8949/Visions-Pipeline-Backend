"""Lists the demo clips in "Sample videos SOPs/" so any pipeline's frontend
page can offer them as a stand-in for a real upload when no camera or footage
of its own is available. Read-only - nothing here writes to that folder.

A clip belongs to a module if the module's keyword appears anywhere in its
filename (case-insensitive) - e.g. "Guard Sleeping.mp4" matches module
"Guard", "Kitchen_1.mp4" matches module "Kitchen". This keeps the mapping
purely a naming convention: dropping in "Restricted Zone - hallway.mp4"
makes it show up for that module automatically, no code change needed."""

import os

from fastapi import APIRouter

SAMPLE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Sample videos SOPs")
VIDEO_EXTENSIONS = (".mp4", ".mov", ".avi", ".mkv")

router = APIRouter()


@router.get("/list")
def list_sample_videos(module: str | None = None):
    """module: restricts the list to filenames containing this keyword
    (case-insensitive). Omit it to list everything (used for admin/debug
    purposes only - frontend pages always pass their own module)."""
    if not os.path.isdir(SAMPLE_DIR):
        return {"videos": []}
    files = sorted(f for f in os.listdir(SAMPLE_DIR) if f.lower().endswith(VIDEO_EXTENSIONS))
    if module:
        keyword = module.strip().lower()
        files = [f for f in files if keyword in f.lower()]
    return {"videos": [{"name": f, "url": f"/sample-videos/{f}"} for f in files]}
