import os
import subprocess
import sys
import threading
import time


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


def create_app():
    from dotenv import load_dotenv
    load_dotenv()
    from Attendance import server

    if getattr(server.app, "_pipeline_routes_registered", False):
        return server.app


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
    # NEW - ALPR (AUTOMATIC LICENSE PLATE RECOGNITION) ROUTER
    # ========================================================

    from Plate_detector.api import (
        router as alpr_router,
    )

    server.app.include_router(
        alpr_router,
        prefix="/api/alpr",
        tags=["ALPR"],
    )

    print(
        "[INFO] ALPR API added at /api/alpr"
    )


    # ========================================================
    # NEW - SYSTEM SETTINGS (pipeline on/off, notification/display prefs)
    # ========================================================

    from system_settings import router as system_settings_router

    server.app.include_router(
        system_settings_router,
        prefix="/api/system",
        tags=["System Settings"],
    )

    print(
        "[INFO] System Settings API added at /api/system"
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

    server.app._pipeline_routes_registered = True
    return server.app


def main():
    ensure_python_deps()
    create_app()
    frontend_dir = os.environ["FRONTEND_DIR"]
    from Attendance import server

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
        print("\n[INFO] Shutting down...")
        frontend_process.terminate()
        stop_all_jobs()
        _wait_for_jobs_to_stop(timeout=8.0)
        # Force-exit rather than falling through to normal interpreter
        # shutdown: any job thread that didn't notice its stop_event in time
        # is non-daemon and would otherwise keep the process alive
        # indefinitely. Everything reachable has already been asked to stop
        # above - this is just the guaranteed upper bound on how long
        # Ctrl+C can take.
        os._exit(0)


def _wait_for_jobs_to_stop(timeout: float) -> None:
    """Give signalled jobs a bounded window to actually exit their worker
    threads before the caller force-exits regardless."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        lingering = [
            t for t in threading.enumerate()
            if t is not threading.main_thread() and not t.daemon
        ]
        if not lingering:
            return
        time.sleep(0.2)


def stop_all_jobs():
    """Signal every currently-running backend job/session across all
    modules to stop immediately, instead of letting their worker threads
    (non-daemon - concurrent.futures.ThreadPoolExecutor's defaults) keep
    processing to completion after Ctrl+C. Without this, the process used
    to stay alive - invisibly, with the frontend already gone - until
    whatever video was mid-processing finished on its own, sometimes
    minutes later, and could even crash on the way out (a model's internal
    executor torn down by interpreter shutdown while a frame was still
    in flight). Each call is best-effort and independent so one module's
    failure can't block the others from being signalled."""
    print("[INFO] Stopping any in-progress jobs...")

    try:
        from guard_monitoring.service import get_service as get_guard_service
        get_guard_service().stop_all()
    except Exception as exc:  # noqa: BLE001
        print(f"[WARN] Could not stop Guard jobs: {exc}")

    try:
        from kitchen_monitoring.service import SERVICE as kitchen_service
        kitchen_service.stop_all()
    except Exception as exc:  # noqa: BLE001
        print(f"[WARN] Could not stop Kitchen sessions: {exc}")

    try:
        from restricted_zone_monitor import monitor as restricted_zone_monitor_state
        restricted_zone_monitor_state.stop_all()
    except Exception as exc:  # noqa: BLE001
        print(f"[WARN] Could not stop Restricted Zone session: {exc}")

    try:
        from Plate_detector.service import get_service as get_alpr_service
        get_alpr_service().stop_all()
    except Exception as exc:  # noqa: BLE001
        print(f"[WARN] Could not stop ALPR jobs: {exc}")


if __name__ == "__main__":
    main()