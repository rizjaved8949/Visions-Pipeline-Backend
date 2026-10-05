"""Standalone CLI, so the module still runs on its own:

    python -m Plate_detector --source traffic.mp4
    python -m Plate_detector --source 0 --show
    python -m Plate_detector --images photos/ --output-dir runs/stills
    python -m Plate_detector --preflight
"""
import argparse
import json
import sys


def main() -> int:
    ap = argparse.ArgumentParser(prog="Plate_detector", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", help="video file, rtsp url, or webcam index")
    ap.add_argument("--images", help="image file or folder (instead of --source)")
    ap.add_argument("--config")
    ap.add_argument("--output-dir")
    ap.add_argument("--show", action="store_true", help="open a preview window")
    ap.add_argument("--save-video", dest="save_video", action="store_true", default=None)
    ap.add_argument("--no-save-video", dest="save_video", action="store_false")
    ap.add_argument("--device", help="0 for first GPU, or cpu")
    ap.add_argument("--plate-conf", type=float)
    ap.add_argument("--preflight", action="store_true", help="check deps/weights and exit")
    ap.add_argument("--json", action="store_true", help="print the result dict as JSON")
    a = ap.parse_args()

    # run as a script OR as -m from the parent dir
    try:
        from Plate_detector import ALPRModule
    except ImportError:
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
        from Plate_detector import ALPRModule

    mod = ALPRModule(config=a.config, output_dir=a.output_dir,
                     show=a.show, save_video=a.save_video)

    if a.preflight:
        info = mod.preflight()
        print(json.dumps(info, indent=2))
        return 0 if info["ok"] else 1

    extra = {}
    if a.device:
        extra["device"] = a.device
    if a.plate_conf is not None:
        extra["plate.conf"] = a.plate_conf

    if a.images:
        result = mod.process_images(a.images, **extra)
    elif a.source is not None:
        src = int(a.source) if str(a.source).isdigit() else a.source
        result = mod.run(src, **extra)
    else:
        ap.error("one of --source or --images is required (or --preflight)")
        return 2

    if a.json:
        print(json.dumps(result, indent=2, default=str))
    else:
        c = result.get("counts", {})
        print(f"\n{result['module']}: ok={result['ok']} "
              f"plates={c.get('plates_saved', 0)} "
              f"lowconf={c.get('lowconf_quarantined', 0)} "
              f"-> {result['output_dir']}")
        if result.get("error"):
            print(f"error: {result['error']}")
    return 0 if result.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
