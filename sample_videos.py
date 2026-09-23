"""Lists the demo clips in "Sample videos SOPs/" so any pipeline's frontend
page can offer them as a stand-in for a real upload when no camera or footage
of its own is available. Read-only - nothing here writes to that folder.

Each module gets its own subfolder, named after the module (e.g.
"Sample videos SOPs/Guard/", "Sample videos SOPs/Kitchen/") - a clip only
ever shows up for the module whose subfolder it's dropped into, with no
naming convention to get right. Matching a subfolder to a module keyword is
case-insensitive so the frontend doesn't have to send an exact-case match."""

import os
import urllib.parse

from fastapi import APIRouter

SAMPLE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Sample videos SOPs")
VIDEO_EXTENSIONS = (".mp4", ".mov", ".avi", ".mkv")

router = APIRouter()


def _find_module_dir(module: str) -> str | None:
    if not os.path.isdir(SAMPLE_DIR):
        return None
    wanted = module.strip().lower()
    for name in os.listdir(SAMPLE_DIR):
        if name.lower() == wanted and os.path.isdir(os.path.join(SAMPLE_DIR, name)):
            return name
    return None


@router.get("/list")
def list_sample_videos(module: str | None = None):
    """module: the subfolder to list (case-insensitive, e.g. "Guard"). Omit
    it to list nothing - frontend pages always pass their own module, and
    there is no cross-module fallback."""
    if not module:
        return {"videos": []}
    module_dir = _find_module_dir(module)
    if module_dir is None:
        return {"videos": []}
    folder_path = os.path.join(SAMPLE_DIR, module_dir)
    files = sorted(f for f in os.listdir(folder_path) if f.lower().endswith(VIDEO_EXTENSIONS))
    return {
        "videos": [
            {
                "name": f,
                "url": f"/sample-videos/{urllib.parse.quote(module_dir)}/{urllib.parse.quote(f)}",
            }
            for f in files
        ]
    }
