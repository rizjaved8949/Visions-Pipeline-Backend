import importlib
import sys


def test_api_import_does_not_import_heavy_cv_models():
    for name in ["rfdetr", "ultralytics", "mediapipe", "trackers"]:
        sys.modules.pop(name, None)
    sys.modules.pop("guard_monitoring.api", None)

    importlib.import_module("guard_monitoring.api")

    for name in ["rfdetr", "ultralytics", "mediapipe", "trackers"]:
        assert name not in sys.modules
