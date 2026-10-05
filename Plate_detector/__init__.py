"""ALPR module - vehicle detection, licence-plate detection, enhancement, optional OCR.

Import is side-effect free: no models are loaded and no windows are opened until you
call `run()`. Drop this folder into a host project as-is.

    from Plate_detector import ALPRModule
    result = ALPRModule(output_dir="runs/job1/alpr").run("traffic.mp4")

All internal imports are relative, so a host package that also happens to be named
`alpr` cannot shadow this module's implementation.
"""
from .module import ALPRModule, run

__all__ = ["ALPRModule", "run"]
__version__ = "1.0.0"
