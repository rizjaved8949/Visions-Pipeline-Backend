"""Verify sleep changes on video using recorded guard/phone/movement inference.

Re-runs real eye and posture analysis on every recorded guard frame. This is
explicitly a sensor replay, not a fresh run of all detection models. Outputs a
separate annotated video and an audit of displayed labels and frame counts.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np

from guard_monitoring.config import load_config
from guard_monitoring.diagnostics import BUILD_ID
from guard_monitoring.io import make_writer, open_capture, video_metadata
from guard_monitoring.model_registry import LazyModelRegistry
from guard_monitoring.monitoring.posture import PostureAnalyzer
from guard_monitoring.monitoring.sleep import SleepAnalyzer, SleepStatusStabilizer
from guard_monitoring.pipeline import GuardMonitoringPipeline
from guard_monitoring.types import EyeState, GuardTrack, MovementState, PhoneState, PostureState, SleepState
from guard_monitoring.visualization.overlay import draw_guard_box_and_status


def verify(source, frames, output):
    rows = [json.loads(line) for line in Path(frames).read_text(encoding="utf-8").splitlines()]
    if [row["frame"] for row in rows] != list(range(len(rows))):
        raise ValueError("Frame log must contain every frame in order")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    cfg = load_config(source=source, output_dir=str(output))
    registry = LazyModelRegistry(cfg)
    eye_result = registry.eyes()
    if not eye_result.ok:
        raise RuntimeError(f"Eye model unavailable: {eye_result.detail}")
    sleep_cfg = dict(cfg["sleep_logic"], torso_lean_deg=cfg["pose_logic"]["torso_lean_deg"])
    analyzer = SleepAnalyzer(sleep_cfg)
    posture_analyzer = PostureAnalyzer(cfg["pose_logic"])
    display = SleepStatusStabilizer(sleep_cfg)
    renderer = SimpleNamespace(sleep_status=display)
    cap = open_capture(str(source))
    width, height, fps = video_metadata(cap)
    video = output / "annotated.mp4"
    writer = make_writer(video, width, height, fps)
    decisions = []
    try:
        for row in rows:
            decoded, frame = cap.read()
            if not decoded:
                raise ValueError("Video has fewer frames than the sensor log")
            t = row["time_seconds"]
            track_id = row.get("guard_track_id")
            if row.get("camera_motion", {}).get("scene_change"):
                analyzer.reset(track_id)
                display.reset(track_id)
            eye, posture, sleep = EyeState(), PostureState(), SleepState()
            phone = PhoneState(**row.get("phone", {}))
            movement = MovementState(**row.get("movement", {}))
            if row.get("guard_seen_now"):
                raw_pose = row.get("pose_keypoints")
                keypoints = np.asarray(raw_pose) if raw_pose is not None else None
                posture = posture_analyzer.analyze(keypoints, row["guard_bbox"])
                eye = eye_result.value.analyze(frame, row["guard_bbox"], keypoints, round(t * 1000))
                eye.observed_at = t
                modules = row["module_status"]
                sleep = analyzer.update(track_id, t, eye, posture, movement, phone, dict(
                    eyes=True, posture=True, movement=modules["movement"]["status"] == "ok",
                    phone=modules["phone_use"]["status"] == "ok"))
            else:
                analyzer.pause(t)
            shown = display.update(sleep, t, track_id=track_id, present=row.get("present"),
                                   observed_at=eye.observed_at if sleep.evidence_quality == "high" else None)
            labels = []
            if row.get("guard_bbox") and row.get("present") is True:
                labels = GuardMonitoringPipeline._active_statuses(renderer, phone, sleep, movement, posture)
                guard = GuardTrack(track_id, tuple(row["guard_bbox"]), row.get("guard_confidence") or 0,
                                   seen_now=row.get("guard_seen_now", False))
                draw_guard_box_and_status(frame, guard, labels)
            writer.write(frame)
            decisions.append(dict(frame=row["frame"], time_seconds=t, eyes=asdict(eye),
                                  posture=asdict(posture), sleep=asdict(sleep), sleep_display=shown,
                                  display_statuses=labels, original_display_statuses=row.get("display_statuses", [])))
        if cap.read()[0]:
            raise ValueError("Video has more frames than the sensor log; complete the original run first")
    finally:
        cap.release()
        writer.release()
        registry.close()
    decoded_count = 0
    encoded = open_capture(str(video))
    try:
        while encoded.read()[0]:
            decoded_count += 1
    finally:
        encoded.release()
    if decoded_count != len(rows):
        raise ValueError("Output video lost frames")
    sleep_rows = [row for row in decisions if "Sleeping" in row["display_statuses"]]
    first = sleep_rows[0]["frame"] if sleep_rows else None
    other_labels = lambda labels: [label for label in labels if label not in {"Sleeping", "Possible Sleep"}]
    changed = [r["frame"] for r in decisions
               if other_labels(r["display_statuses"]) != other_labels(r["original_display_statuses"])]
    report = dict(build_id=BUILD_ID, scope="Fresh eyes/posture/sleep; recorded guard/phone/movement sensors",
                  source=str(source), sensor_log=str(frames), frames=len(rows), output_decoded_frames=decoded_count,
                  first_sleep_seconds=sleep_rows[0]["time_seconds"] if sleep_rows else None,
                  sleep_display_frames=len(sleep_rows), non_sleep_label_changes=changed,
                  gaps_after_first_sleep=[r["frame"] for r in decisions if first is not None
                                          and r["frame"] >= first and "Sleeping" not in r["display_statuses"]],
                  sleep_config=sleep_cfg, decisions=decisions)
    (output / "verification.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return {key: value for key, value in report.items() if key != "decisions"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--frames", required=True)
    parser.add_argument("--output", required=True, help="New output directory; existing directories are refused")
    args = parser.parse_args()
    print(json.dumps(verify(args.source, args.frames, args.output), indent=2))
