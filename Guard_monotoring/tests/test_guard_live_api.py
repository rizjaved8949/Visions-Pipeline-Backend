import importlib
import sys


def test_live_routes_are_additive_and_import_stays_lightweight():
    for name in ["rfdetr", "ultralytics", "mediapipe", "trackers"]:
        sys.modules.pop(name, None)
    sys.modules.pop("guard_monitoring.api", None)

    api = importlib.import_module("guard_monitoring.api")
    paths = {route.path for route in api.router.routes}

    # Existing uploaded-video API remains present.
    assert "/jobs" in paths
    assert "/jobs/{job_id}/summary" in paths
    assert "/jobs/{job_id}/video" in paths

    # New real-time API is additive.
    assert "/live/start" in paths
    assert "/live/{session_id}/state" in paths
    assert "/live/{session_id}/events" in paths
    assert "/live/{session_id}/module-health" in paths
    assert "/live/{session_id}/settings" in paths
    assert "/live/{session_id}/stop" in paths
    assert "/live/{session_id}/stream" in paths

    for name in ["rfdetr", "ultralytics", "mediapipe", "trackers"]:
        assert name not in sys.modules


def test_live_request_maps_rule_durations():
    from guard_monitoring.api import LiveStartRequest, LiveSettingsRequest

    start = LiveStartRequest(
        source="0",
        sleep_seconds=300,
        phone_seconds=600,
        stationary_seconds=3600,
        absence_seconds=120,
    )
    assert start.rule_settings() == {
        "sleep": 300.0,
        "phone": 600.0,
        "stationary": 3600.0,
        "absence": 120.0,
    }

    patch = LiveSettingsRequest(phone_seconds=900)
    assert patch.rule_settings() == {"phone": 900.0}
