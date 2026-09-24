from __future__ import annotations

from copy import deepcopy
from importlib.metadata import PackageNotFoundError, version
import platform

BUILD_ID = 'guard-motion-sleep-fix-2026-09-24'


def effective_config(cfg):
    # Deliberately exclude camera input URLs and unrelated application settings.
    return {key: deepcopy(cfg[key]) for key in (
        'models', 'tracker', 'guard_selection', 'camera_motion', 'movement',
        'pose_logic', 'phone_logic', 'sleep_logic', 'rules', 'activity',
        'visualization', 'modules') if key in cfg}


def runtime_info(registry):
    packages = {}
    for name in ('opencv-python', 'opencv-python-headless', 'numpy', 'torch',
                 'rfdetr', 'ultralytics', 'mediapipe', 'trackers'):
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = None
    try:
        devices = registry.device_summary()
    except Exception:
        devices = {'status': 'not_reported'}
    return dict(python=platform.python_version(), platform=platform.platform(),
                packages=packages, model_devices=devices)
