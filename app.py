import os
import subprocess
import sys
import threading


MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
REQUIREMENTS_FILE = os.path.join(MODULE_DIR, "requirements.txt")
DEPS_MARKER = os.path.join(MODULE_DIR, ".deps_installed")


def ensure_python_deps():
    """Install everything in requirements.txt if it hasn't been installed yet
    (or requirements.txt changed since the last install), so a plain
    `python app.py` works on a fresh machine with nothing pre-installed."""

    up_to_date = (
        os.path.exists(DEPS_MARKER)
        and os.path.getmtime(DEPS_MARKER) >= os.path.getmtime(REQUIREMENTS_FILE)
    )

    if up_to_date:
        print("[INFO] Python dependencies already installed.")
        return

    print("[INFO] Installing Python dependencies from requirements.txt...")

    subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "-r",
            REQUIREMENTS_FILE,
        ],
        check=True,
    )

    with open(DEPS_MARKER, "w") as f:
        f.write("installed")


def ensure_frontend_deps(frontend_dir):
    node_modules = os.path.join(frontend_dir, "node_modules")

    if not os.path.isdir(node_modules):
        print("[INFO] node_modules not found, running 'npm install'...")

        subprocess.run(
            ["npm", "install"],
            cwd=frontend_dir,
            shell=True,
            check=True,
        )
    else:
        print("[INFO] Frontend dependencies already installed.")


def main():

    # Existing dependency setup
    ensure_python_deps()

    from dotenv import load_dotenv

    load_dotenv()

    frontend_dir = os.environ["FRONTEND_DIR"]

    # Existing backend
    from Attendance import server


    # ========================================================
    # GUARD MONITORING ROUTER
    # ========================================================

    from guard_monitoring.api import (
        router as guard_monitoring_router,
    )

    server.app.include_router(
        guard_monitoring_router,
        prefix="/api/guard",
        tags=["Guard Monitoring"],
    )

    print(
        "[INFO] Guard Monitoring API added at /api/guard"
    )


    # ========================================================
    # NEW - KITCHEN HYGIENE ROUTER
    # ========================================================

    from kitchen_monitoring.api import (
        router as kitchen_router,
    )

    server.app.include_router(
        kitchen_router,
        prefix="/api/kitchen",
        tags=["Kitchen Hygiene"],
    )

    print(
        "[INFO] Kitchen Hygiene API added at /api/kitchen"
    )


    # ========================================================
    # NEW - RESTRICTED ZONE MONITOR ROUTER
    # ========================================================

    from restricted_zone_monitor.api import (
        router as restricted_zone_router,
    )
    from restricted_zone_monitor import (
        monitor as restricted_zone_monitor_state,
    )

    server.app.include_router(
        restricted_zone_router,
        prefix="/api/restricted-zone",
        tags=["Restricted Zone Monitor"],
    )

    # Session + zone data is scoped to a single run (drawn fresh each time,
    # discarded once the report is downloaded) - guarantee a clean slate no
    # matter how the previous run ended, same as Attendance does.
    # (Using the same on_event(...)(fn) form Attendance/server.py itself
    # relies on - add_event_handler() was removed in newer FastAPI/Starlette.)
    server.app.on_event("startup")(restricted_zone_monitor_state.wipe_all_data)
    server.app.on_event("shutdown")(restricted_zone_monitor_state.wipe_all_data)

    print(
        "[INFO] Restricted Zone Monitor API added at /api/restricted-zone"
    )


    # ========================================================
    # NEW - SAMPLE VIDEOS (shared across every pipeline's frontend)
    # ========================================================

    from fastapi.staticfiles import StaticFiles

    from sample_videos import router as sample_videos_router, SAMPLE_DIR

    server.app.include_router(
        sample_videos_router,
        prefix="/api/sample-videos",
        tags=["Sample Videos"],
    )

    if os.path.isdir(SAMPLE_DIR):
        server.app.mount(
            "/sample-videos",
            StaticFiles(directory=SAMPLE_DIR),
            name="sample-videos-files",
        )

    print(
        "[INFO] Sample Videos API added at /api/sample-videos"
    )


    # ========================================================
    # EXISTING CODE - UNCHANGED
    # ========================================================

    ensure_frontend_deps(frontend_dir)

    backend_thread = threading.Thread(
        target=server.run,
        daemon=True,
    )

    backend_thread.start()

    print("[INFO] Starting frontend ('npm run dev')...")

    frontend_process = subprocess.Popen(
        ["npm", "run", "dev"],
        cwd=frontend_dir,
        shell=True,
    )

    try:
        frontend_process.wait()

    except KeyboardInterrupt:
        print("\n[INFO] Shutting down frontend...")
        frontend_process.terminate()


if __name__ == "__main__":
    main()