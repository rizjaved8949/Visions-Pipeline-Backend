"""Re-run fresh status models on a video using recorded guard/camera observations.

This skips reloading RF-DETR for status tuning, not video frames. Pose, phone,
eyes, frame motion, composition, and alert logic use the current real pipeline.
It is explicitly a replay of guard tracking, not a new guard-detector benchmark.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np

from guard_monitoring.config import load_config
from guard_monitoring.contracts import ok
from guard_monitoring.model_registry import LazyModelRegistry
from guard_monitoring.pipeline import GuardMonitoringPipeline


class RecordedGuardRegistry:
    def __init__(self, cfg, rows):
        self.models = LazyModelRegistry(cfg)
        self.rows = rows
        self.index = -1

    def predict_guard(self, frame):
        self.index += 1
        if self.index >= len(self.rows):
            raise ValueError("Input video exceeds recorded frames")
        row = self.rows[self.index]
        observed = row.get("guard_seen_now") and row.get("guard_bbox") is not None
        return SimpleNamespace(
            xyxy=np.asarray([row["guard_bbox"]] if observed else [], dtype=float).reshape(-1, 4),
            confidence=np.asarray([row["guard_confidence"]] if observed else [], dtype=float),
            tracker_id=np.asarray([row["guard_track_id"]] if observed else [], dtype=int),
        )

    def guard_detector(self):
        return ok(SimpleNamespace(predict=self.predict_guard))

    def tracker(self, frame_rate=30):
        return ok(SimpleNamespace(update=lambda detections, timestamp: detections))

    def reset_tracker(self):
        pass

    def pose(self):
        return self.models.pose()

    def phone(self):
        return self.models.phone()

    def eyes(self):
        return self.models.eyes()

    def device_summary(self):
        return self.models.device_summary()

    def close(self):
        self.models.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--frames", required=True)
    parser.add_argument("--output", required=True, help="New output directory")
    args = parser.parse_args()
    rows = [json.loads(line) for line in Path(args.frames).read_text(encoding="utf-8").splitlines()]
    if [row["frame"] for row in rows] != list(range(len(rows))):
        raise ValueError("Frame log must cover every frame in order")
    source = cv2.VideoCapture(args.source)
    count = int(source.get(cv2.CAP_PROP_FRAME_COUNT))
    source.release()
    if count != len(rows):
        raise ValueError("Input video and recorded frame counts differ")
    if Path(args.output).exists():
        raise FileExistsError("Use a new output directory")
    cfg = load_config(source=args.source, output_dir=args.output)
    cfg["api"]["progress_every_frames"] = 24
    registry = RecordedGuardRegistry(cfg, rows)
    pipeline = GuardMonitoringPipeline(cfg, registry=registry)
    pipeline.camera_motion.update = lambda frame, boxes, now: registry.rows[registry.index]["camera_motion"]
    pipeline.progress_callback = lambda data: print(
        f"{data['status']}: {data['frames_processed']}/{data['total_frames']} frames", flush=True)
    summary = pipeline.run()
    if summary["frame_errors"] or summary["frames_processed"] != len(rows):
        raise RuntimeError("Replay did not successfully process every input frame")
    cap = cv2.VideoCapture(summary["output_video"])
    decoded = 0
    try:
        while cap.read()[0]:
            decoded += 1
    finally:
        cap.release()
    if decoded != len(rows):
        raise RuntimeError("Output video lost frames")
    summary.update(validation_scope="Fresh pose/phone/eyes/status pipeline; recorded selected guard and camera observations",
                   recorded_guard_source=args.frames, output_decoded_frames=decoded)
    Path(args.output, "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({key: summary[key] for key in ("frames_processed", "output_decoded_frames", "frame_errors",
                                                   "validation_scope", "display_status_counts")}, indent=2))


if __name__ == "__main__":
    main()
