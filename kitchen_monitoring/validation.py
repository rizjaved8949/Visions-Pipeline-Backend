"""Repeatable, isolated video smoke tests (not an accuracy benchmark).

Usage: python -m kitchen_monitoring.validation --manifest cases.json --output review_dir
Manifest: [{"name": "camera_day", "video": "clip.mp4", "start_frame": 0, "frames": 90}]
Video paths are relative to the manifest. No production sessions/violations are written.
"""
import argparse
import json
import math
import re
import threading
import time
from pathlib import Path
from unittest.mock import Mock, patch

import cv2

from guard_monitoring.io import make_writer
from .pipeline import KitchenPipeline, REQUIREMENTS
from .visualization import annotation_size


def load_cases(manifest):
    manifest = Path(manifest).resolve()
    cases = json.loads(manifest.read_text(encoding="utf-8"))
    if not isinstance(cases, list) or not cases:
        raise ValueError("Manifest must be a nonempty list of video cases")
    resolved = []
    for index, case in enumerate(cases):
        video = (manifest.parent / case["video"]).resolve()
        start, frames = case.get("start_frame", 0), case.get("frames", 90)
        if not video.is_file():
            raise ValueError(f"Video does not exist: {video}")
        if type(start) is not int or type(frames) is not int or start < 0 or frames < 1:
            raise ValueError("start_frame must be nonnegative and frames must be positive integers")
        name = re.sub(r"[^A-Za-z0-9_-]", "_", str(case.get("name", video.stem)))[:80]
        resolved.append({"name": f"{index:03d}_{name}", "video": str(video),
                         "start_frame": start, "frames": frames})
    return resolved


def run_case(case, output):
    cap = cv2.VideoCapture(case["video"])
    writer = None
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open {case['video']}")
    started = time.perf_counter()
    rows = []
    try:
        fps = cap.get(cv2.CAP_PROP_FPS)
        if not math.isfinite(fps) or fps <= 0:
            fps = 30.0
        cap.set(cv2.CAP_PROP_POS_FRAMES, case["start_frame"])
        with patch("kitchen_monitoring.pipeline.STORE", Mock()), patch("kitchen_monitoring.pipeline.SESSION_DIR", output):
            pipeline = KitchenPipeline(case["name"], threading.Event())
            pipeline.source_fps = fps
            pipeline.show_labels = True
            pipeline.show_detection_boxes = True
            for number in range(case["frames"]):
                ok, frame = cap.read()
                if not ok:
                    break
                if writer is None:
                    size = annotation_size(frame.shape[1], frame.shape[0])
                    writer = make_writer(pipeline.output_video, *size, fps)
                rendered, summary = pipeline.process_frame(frame, number)
                rows.append({"source_frame": case["start_frame"] + number, **summary})
                writer.write(rendered)
                if number in {0, case["frames"] // 2, case["frames"] - 1}:
                    cv2.imwrite(str(pipeline.output_dir / f"frame_{number:06d}.png"), rendered)
                (output / "progress.json").write_text(json.dumps({"case": case["name"], "frames": len(rows)}))
        if not rows:
            raise RuntimeError(f"No frames decoded for {case['name']}")
    finally:
        cap.release()
        if writer is not None:
            writer.release()
    counts = {r: {s: 0 for s in ("compliant", "violation", "unknown")} for r in REQUIREMENTS}
    for row in rows:
        for person in row["persons"]:
            for requirement in REQUIREMENTS:
                counts[requirement][person[requirement]["state"]] += 1
    report = {**case, "processed_frames": len(rows), "complete": len(rows) == case["frames"],
              "elapsed_seconds": round(time.perf_counter() - started, 3),
              "unique_track_labels": len(pipeline.display_ids),
              "max_visible_people": max(row["staff_detected"] for row in rows),
              "requirement_observations": counts, "accuracy": None,
              "note": "Unlabelled smoke test. Coverage and ID counts are not accuracy or unique-person ground truth."}
    (pipeline.output_dir / "results.json").write_text(json.dumps(rows))
    (pipeline.output_dir / "report.json").write_text(json.dumps(report, indent=2))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    cases = load_cases(args.manifest)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    reports = []
    for case in cases:
        reports.append(run_case(case, output))
        (output / "report.json").write_text(json.dumps(reports, indent=2))
        print(json.dumps(reports[-1]), flush=True)
    return 0 if all(report["complete"] for report in reports) else 1


if __name__ == "__main__":
    raise SystemExit(main())
