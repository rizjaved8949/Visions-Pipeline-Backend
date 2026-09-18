import importlib
import sys
from pathlib import Path


def test_nested_guard_package_imports_without_heavy_models():
    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    for name in ["rfdetr", "ultralytics", "mediapipe", "trackers"]:
        sys.modules.pop(name, None)

    module = importlib.import_module("Guard_monotoring.guard_monitoring.api")
    assert module.router is not None

    for name in ["rfdetr", "ultralytics", "mediapipe", "trackers"]:
        assert name not in sys.modules
