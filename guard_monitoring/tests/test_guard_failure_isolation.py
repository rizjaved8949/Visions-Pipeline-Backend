from pathlib import Path

import cv2
import numpy as np

from guard_monitoring.config import load_config
from guard_monitoring.contracts import ok
from guard_monitoring.pipeline import GuardMonitoringPipeline
from guard_monitoring.types import EyeState


class FakeDetections:
    def __init__(self):
        self.xyxy = np.asarray([[10.0, 10.0, 50.0, 90.0]], dtype=float)
        self.confidence = np.asarray([0.95], dtype=float)
        self.tracker_id = np.asarray([1], dtype=int)


class FakeDetector:
    def predict(self, frame):
        return object()


class FakeTracker:
    def update(self, detections, timestamp):
        return FakeDetections()


class BrokenPose:
    def predict(self, frame, guard_box):
        raise RuntimeError("simulated pose failure")


class FakePhone:
    def predict(self, frame, guard_box):
        return []


class FakeEyes:
    def analyze(self, frame, guard_box, keypoints, timestamp_ms=0):
        return EyeState(
            available=True,
            quality_ok=True,
            eyes_closed=False,
            ear_mean=0.3,
            reason="ok",
        )


class FakeRegistry:
    def guard_detector(self):
        return ok(FakeDetector())

    def tracker(self, frame_rate=30.0):
        return ok(FakeTracker())

    def pose(self):
        return ok(BrokenPose())

    def phone(self):
        return ok(FakePhone())

    def eyes(self):
        return ok(FakeEyes())


def _make_video(path: Path):
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 5.0, (96, 96))
    assert writer.isOpened()
    for _ in range(4):
        writer.write(np.zeros((96, 96, 3), dtype=np.uint8))
    writer.release()


def test_pose_failure_does_not_stop_independent_modules(tmp_path, monkeypatch):
    monkeypatch.setenv("GUARD_TEST_MODE", "true")
    video = tmp_path / "input.mp4"
    _make_video(video)

    cfg = load_config(source=str(video), output_dir=str(tmp_path / "out"), max_frames=4)
    cfg["guard_selection"]["confirm_seconds"] = 0.0
    cfg["models"]["pose"]["every_n_frames"] = 1
    cfg["models"]["phone"]["every_n_frames"] = 1
    cfg["models"]["eyes"]["every_n_frames"] = 1

    summary = GuardMonitoringPipeline(cfg, registry=FakeRegistry()).run()

    assert summary["frames_processed"] == 4
    assert summary["guard_visible_frames"] == 4
    assert Path(summary["output_video"]).exists()

    health_text = Path(summary["module_health"]).read_text(encoding="utf-8")
    assert '"pose"' in health_text
    assert '"errors": 4' in health_text
    assert '"phone_detection"' in health_text
    assert '"eyes"' in health_text
