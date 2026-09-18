from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Guard_monotoring.guard_monitoring.config import load_config
from Guard_monotoring.guard_monitoring.pipeline import GuardMonitoringPipeline


def parse_args():
    parser = argparse.ArgumentParser(description="Run guard monitoring on one local video")
    parser.add_argument("--source", required=True, help="Path to input CCTV/video file")
    parser.add_argument("--output", required=True, help="Directory for annotated video and JSON outputs")
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--display", action="store_true")
    parser.add_argument(
        "--test-mode",
        action="store_true",
        help="Use short rule durations from GUARD_TEST_* environment variables",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    source = Path(args.source)
    if not source.exists():
        raise FileNotFoundError(source)
    if args.test_mode:
        os.environ["GUARD_TEST_MODE"] = "true"

    cfg = load_config(
        source=str(source),
        output_dir=args.output,
        max_frames=args.max_frames,
        display=args.display,
    )
    summary = GuardMonitoringPipeline(cfg).run()
    print("Guard monitoring finished")
    print(f"Frames processed: {summary['frames_processed']}")
    print(f"Annotated video: {summary['output_video']}")
    print(f"Summary: {Path(args.output) / 'summary.json'}")
    print(f"Module health: {summary['module_health']}")


if __name__ == "__main__":
    main()
