"""
Public API for the ALPR module. DETECTION ONLY - it produces plate IMAGES, not text.

Designed to be driven by an external orchestrator, so:
  * nothing happens at import time (no model loading, no argparse, no windows);
  * every path is resolved relative to THIS folder, so the host's working directory
    does not matter;
  * `run()` returns structured results rather than only writing files;
  * it is headless by default - an orchestrated pipeline should not pop up a window;
  * a `stop_event` allows a clean shutdown without a GUI keypress.

Typical host usage:

    from Plate_detector import ALPRModule

    alpr = ALPRModule(output_dir="runs/job123/alpr")
    result = alpr.run("traffic.mp4")
    for p in result["plates"]:
        print(p["text"], p["image"])
"""
from __future__ import annotations

import csv
import time
from pathlib import Path
from typing import Any, Iterable

HERE = Path(__file__).resolve().parent
DEFAULT_CONFIG = HERE / "config.yaml"


class ALPRModule:
    """Vehicle detection -> plate detection -> enhancement -> (optional) OCR.

    Parameters
    ----------
    config : path to a YAML config. Defaults to the one shipped with the module.
    output_dir : where crops / csv / video go. Relative paths resolve against the
        module folder; pass an absolute path to direct output into the host's run dir.
    show : open a preview window. Off by default; an orchestrated run should stay headless.
    save_video : write an annotated review video. Defaults to the config value.
    overrides : any dotted config key, e.g. {"plate.conf": 0.6, "device": "cpu"}.
    """

    name = "alpr"
    description = ("Detects vehicles, locates their licence plates, and saves enhanced plate "
                   "crops. Detection only - it does not read plate text.")

    def __init__(self, config: str | Path | None = None, output_dir: str | Path | None = None,
                 show: bool = False, save_video: bool | None = None,
                 overrides: dict[str, Any] | None = None):
        self.config_path = Path(config) if config else DEFAULT_CONFIG
        if not self.config_path.exists():
            raise FileNotFoundError(f"config not found: {self.config_path}")

        self._overrides: dict[str, Any] = dict(overrides or {})
        if output_dir is not None:
            self._overrides["output.dir"] = str(Path(output_dir).resolve())
        self._overrides["display.show"] = bool(show)
        if save_video is not None:
            self._overrides["output.save_video"] = bool(save_video)

    # ------------------------------------------------------------------ #
    def _load(self, extra: dict[str, Any] | None = None):
        from .alpr.config import load_config
        merged = {**self._overrides, **(extra or {})}
        return load_config(self.config_path, merged)

    def preflight(self) -> dict[str, Any]:
        """Check the module can actually run, WITHOUT loading models.

        Worth calling once at host start-up: a missing weight file or a broken OpenCV
        build should surface before a long job begins, not twenty minutes in.
        """
        problems: list[str] = []
        cfg = self._load()
        for label, path in (("vehicle weights", cfg.vehicle.weights),
                            ("plate weights", cfg.plate.weights)):
            if not Path(path).exists():
                problems.append(f"missing {label}: {path}")
        if not Path(cfg.enhance.model_dir).exists() and cfg.enhance.method != "classic":
            problems.append(f"missing SR models: {cfg.enhance.model_dir}")
        try:
            import cv2  # noqa: F401
            import ultralytics  # noqa: F401
        except Exception as e:  # noqa: BLE001
            problems.append(f"import failed: {e}")
        return {"module": self.name, "ok": not problems, "problems": problems,
                "config": str(self.config_path)}

    # ------------------------------------------------------------------ #
    def run(self, source: str | int | None = None, stop_event=None,
            progress_callback=None, latest_frame_path: str | Path | None = None,
            **overrides: Any) -> dict[str, Any]:
        """Process a video / stream / webcam index. Blocking.

        `stop_event` is any object with `.is_set()` (threading.Event), letting the host
        stop a long job. `progress_callback`, if given, is called periodically with
        {"frames_processed", "total_frames"}. `latest_frame_path`, if given, gets the
        current annotated frame written to it continuously for a live preview during
        a long job. Returns a summary dict; see `plates` for the per-plate records.
        """
        from .alpr.pipeline import ALPRPipeline

        extra = dict(overrides)
        if source is not None:
            extra["source"] = source
        cfg = self._load(extra)

        t0 = time.time()
        error = None
        stats: dict[str, Any] = {}
        try:
            stats = ALPRPipeline(cfg).run(
                stop_event=stop_event,
                progress_callback=progress_callback,
                latest_frame_path=latest_frame_path,
            ) or {}
        except Exception as e:  # noqa: BLE001 - a module must not take the host down
            error = f"{type(e).__name__}: {e}"

        out_dir = Path(stats.get("output_dir") or cfg.output.dir)
        result = {
            "module": self.name,
            "ok": error is None,
            "error": error,
            "source": str(cfg.source),
            "elapsed_s": round(time.time() - t0, 2),
            "output_dir": str(out_dir),
            "plates": self._read_plates(out_dir / cfg.output.csv),
            **{k: v for k, v in stats.items() if k != "output_dir"},
        }
        result["counts"] = {
            "plates_saved": len(result["plates"]),
            "lowconf_quarantined": stats.get("lowconf", 0),
            "rejected_adframe": stats.get("rejected_adframe", 0),
        }
        return result

    def process_images(self, path: str | Path, **overrides: Any) -> dict[str, Any]:
        """Stills instead of video: a single image file or a directory of them."""
        import cv2
        from .alpr.enhancer import PlateEnhancer
        from .alpr.plate_detector import PlateDetector
        from .alpr.vehicle_detector import VehicleDetector

        cfg = self._load(overrides)
        src = Path(path)
        files: Iterable[Path] = ([src] if src.is_file() else
                                 sorted(f for f in src.iterdir()
                                        if f.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp")))

        veh = VehicleDetector(cfg.vehicle, cfg.device, cfg.half)
        pl = PlateDetector(cfg.plate, cfg.device, cfg.half)
        enh = PlateEnhancer(cfg.enhance)
        out_dir = Path(cfg.output.dir) / "plates"
        out_dir.mkdir(parents=True, exist_ok=True)

        plates: list[dict[str, Any]] = []
        for f in files:
            img = cv2.imread(str(f))
            if img is None:
                continue
            res = veh.model.predict(img, imgsz=cfg.vehicle.imgsz, conf=cfg.vehicle.conf,
                                    classes=list(cfg.vehicle.classes), device=cfg.device,
                                    half=veh.half, verbose=False)[0]
            boxes = res.boxes.xyxy.cpu().numpy().astype(int) if res.boxes is not None else []
            for i, det in enumerate(pl.detect(img, list(boxes))):
                if det is None:
                    continue
                dst = out_dir / f"{f.stem}_v{i}_enhanced.png"
                cv2.imwrite(str(dst), enh.enhance(det.crop))
                plates.append({"source_image": str(f), "vehicle_index": i,
                               "conf": round(float(det.conf), 3), "image": str(dst),
                               "low_conf": bool(getattr(det, "low_conf", False))})
        return {"module": self.name, "ok": True, "error": None,
                "output_dir": str(cfg.output.dir), "plates": plates,
                "counts": {"plates_saved": len(plates)}}

    # ------------------------------------------------------------------ #
    @staticmethod
    def _read_plates(csv_path: Path) -> list[dict[str, Any]]:
        if not csv_path.exists():
            return []
        out = []
        with open(csv_path, newline="", encoding="utf-8") as fh:
            for r in csv.DictReader(fh):
                out.append({
                    "track_id": r.get("track_id"),
                    "vehicle": r.get("vehicle"),
                    "plate_conf": _f(r.get("plate_conf")),
                    "quality": _f(r.get("quality_score")),
                    "image": r.get("enhanced_path"),
                    "raw_image": r.get("raw_path") or None,
                    # a quarantined crop had no confident detection - treat as needs-review
                    "low_conf": "plates_lowconf" in (r.get("raw_path") or ""),
                    "timestamp": r.get("timestamp"),
                })
        return out


def _f(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def run(source, **kwargs) -> dict[str, Any]:
    """One-liner for hosts that do not want to hold an instance."""
    opts = {k: kwargs.pop(k) for k in ("config", "output_dir", "show", "save_video")
            if k in kwargs}
    return ALPRModule(**opts).run(source, **kwargs)
