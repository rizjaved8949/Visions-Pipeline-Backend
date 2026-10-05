# ALPR module

Self-contained licence-plate pipeline: **vehicle detection → plate detection → enhancement**.

> **Detection only.** This module produces clean plate *images*. It does not read plate text,
> and has no OCR dependency. Reading is a separate concern, to be planned later.

Drop this folder into a host project and call it from a single top-level `main.py` alongside your other modules.

## Layout

```
Plate_detector/
  __init__.py        public API (import is side-effect free)
  __main__.py        standalone CLI: python -m Plate_detector ...
  module.py          ALPRModule - the class a host talks to
  config.yaml        all tunables
  alpr/              implementation (capture, detectors, tracker, enhancer, ocr, visualize)
  weights/           yolo11m.pt, plate_detector_finetuned.pt, sr/
  requirements.txt
```

## Contract

```python
from Plate_detector import ALPRModule

alpr   = ALPRModule(output_dir="runs/job1/alpr")   # nothing loads yet
check  = alpr.preflight()                          # verify deps/weights, no models loaded
result = alpr.run("traffic.mp4")                   # blocking
```

`run()` returns:

```python
{
  "module": "alpr",
  "ok": True,                 # False => see "error"; never raises into the host
  "error": None,
  "source": "traffic.mp4",
  "frames": 825,
  "elapsed_s": 56.6,
  "fps": 14.57,
  "output_dir": ".../runs/job1/alpr",
  "csv": ".../results.csv",
  "video": ".../annotated_traffic.mp4",   # or None
  "counts": {"plates_saved": 1, "lowconf_quarantined": 0},
  "plates": [
    {"track_id": "1", "vehicle": "truck", "plate_conf": 0.642, "quality": 0.629,
     "image": ".../x_enhanced.png", "raw_image": ".../x_raw.png",
     "low_conf": False, "timestamp": "20261002_220745_186"}
  ],
}
```

Design points that matter when embedding it:

- **Import is side-effect free.** No model loading, no argparse, no windows at import.
- **Never raises into the host.** Failures come back as `ok: False` + `error`.
- **Headless by default.** Pass `show=True` only for interactive debugging.
- **CWD-independent.** Weights and defaults resolve relative to this folder. Pass an absolute `output_dir` to send results into the host's run directory.
- **Cancellable.** Pass `stop_event` (a `threading.Event`); the run stops at the next frame, flushes queued work, and still returns valid partial results.

## Wiring into a host `main.py`

```python
import threading
from Plate_detector import ALPRModule
# from other_module import OtherModule

MODULES = [
    ALPRModule(output_dir="runs/job1/alpr"),
    # OtherModule(output_dir="runs/job1/other"),
]

def main(source):
    for m in MODULES:                       # fail fast before any long job
        chk = m.preflight()
        if not chk["ok"]:
            raise SystemExit(f"{m.name} not ready: {chk['problems']}")

    stop = threading.Event()                # share across modules for global cancel
    results = {}
    for m in MODULES:
        results[m.name] = m.run(source, stop_event=stop)
        if not results[m.name]["ok"]:
            print(f"{m.name} failed: {results[m.name]['error']}")
    return results
```

To run modules **concurrently**, give each its own thread and `output_dir`. Note they will
contend for the GPU — on a 4 GB card, run GPU-heavy modules sequentially.

## Options

| Constructor arg | Meaning |
|---|---|
| `output_dir` | absolute path recommended; where crops / csv / video go |
| `show` | preview window (default off) |
| `save_video` | annotated review video (default from config) |
| `overrides` | any dotted config key, e.g. `{"plate.conf": 0.6, "device": "cpu"}` |

Per-call overrides work too: `alpr.run(src, **{"plate.conf": 0.6})`.

Stills instead of video: `alpr.process_images("photos/")`.

## CLI

```bash
python -m Plate_detector --preflight
python -m Plate_detector --source traffic.mp4
python -m Plate_detector --source 0 --show
python -m Plate_detector --images photos/ --output-dir runs/stills --json
```

## Tuning notes (measured, not guessed)

- **`plate.conf: 0.50`** is the main quality lever. Raising it from 0.15 took crop purity
  from 81% to 97% *and increased* the number of genuine plates found, because suppressing a
  weak junk box lets the tracker keep looking and catch the real plate later in the track.
  At 0.65 recall collapses. Re-measure purity before changing it.
- **`plate.conf_floor`** is disabled by default. Lowering it feeds more boxes into NMS and
  measurably *suppressed* correct high-confidence detections. Enable only for diagnostics.

## Known limits

- **Ad/dealer frames are NOT filtered.** A plate-shaped plaque carrying a brand name
  (e.g. "SELENIA") is saved as if it were a plate. The filter that caught these used OCR to
  spot lettering-without-digits, and was removed with the rest of the OCR code. Measured
  cost on the test clips: crop purity 100% -> ~96.6% (one such plaque reappears). These score
  high confidence, so no `plate.conf` threshold removes them.

- **Degraded conditions.** On a darkened clip ~71% of plates were missed; the detector
  produces no boxes at all for them, so no threshold recovers them.
- **False positives return under blur.** A sun-glare region scores 0.34 in clean footage
  (filtered) but 0.66 when blurred (kept). A fixed threshold does not hold as conditions change.
- **Training data is single-country (CCPD) with ~0 plate-free images**, which is why the
  detector nominates mirrors/glare when no plate is present.

Full evidence lives in the development workspace (`alpr-pipeline/TECHNICAL_REPORT.md`), which is NOT part of this module.

## Copying this folder to another machine

The folder is self-contained **as code and weights** (verified: copied to another drive,
run from an unrelated working directory, and alongside a host package also named `alpr`).
Three things do **not** travel with it:

1. **Python dependencies.** Install PyTorch (matching that machine's CUDA) then
   `pip install -r requirements.txt`. No OCR dependency.
2. **A CUDA GPU.** `config.yaml` has `device: 0`. On a CPU-only machine set `device: cpu`
   (or pass `overrides={"device": "cpu"}`) — it works, just slower.

Run `python -m Plate_detector --preflight` on the target machine; it reports exactly what is
missing without loading any models.

## Requirements

Install PyTorch with CUDA first, then `pip install -r requirements.txt`.
Verified on torch 2.5.1+cu121 / ultralytics 8.4.155, Python 3.12.

