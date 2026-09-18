from __future__ import annotations

from pathlib import Path

import numpy as np


class RFDETRPersonDetector:
    """RF-DETR adapter used by the guard pipeline.

    When ``weights`` is configured, RF-DETR loads that local fine-tuned checkpoint
    with checkpoint metadata so the model variant does not have to be guessed. The
    project deliberately does not invent a checkpoint filename or class label.
    """

    def __init__(
        self,
        size: str = "medium",
        weights: str = "",
        confidence: float = 0.35,
        target_label: str = "",
        require_finetuned: bool = True,
    ):
        self.confidence = float(confidence)
        self.target_label = target_label.strip().lower()
        self.using_finetuned_weights = bool(weights.strip())

        if require_finetuned and not self.using_finetuned_weights:
            raise FileNotFoundError(
                "GUARD_RFDETR_WEIGHTS is empty. Copy the fine-tuned RF-DETR checkpoint "
                "into Guard_monotoring/weights/guard/ and set its exact path in .env."
            )

        if self.using_finetuned_weights:
            checkpoint = self._resolve_checkpoint(weights)
            # RFDETR.from_checkpoint reads checkpoint metadata and infers the correct
            # RF-DETR variant, avoiding a guessed Nano/Small/Medium/Large architecture.
            from rfdetr import RFDETR

            self.model = RFDETR.from_checkpoint(str(checkpoint))
            return

        # Generic COCO fallback is available only when explicitly allowed by
        # GUARD_REQUIRE_FINETUNED_GUARD_MODEL=false.
        from rfdetr import RFDETRLarge, RFDETRMedium, RFDETRNano, RFDETRSmall

        models = {
            "nano": RFDETRNano,
            "small": RFDETRSmall,
            "medium": RFDETRMedium,
            "large": RFDETRLarge,
        }
        if size not in models:
            raise ValueError(f"Unsupported RF-DETR size: {size}")
        self.model = models[size]()
        if not self.target_label:
            self.target_label = "person"

    @staticmethod
    def _resolve_checkpoint(weights: str) -> Path:
        path = Path(weights).expanduser()
        if not path.is_absolute():
            # detector_rfdetr.py -> models -> guard_monitoring -> Guard_monotoring -> repo root
            repo_root = Path(__file__).resolve().parents[3]
            path = repo_root / path
        path = path.resolve()
        if not path.is_file():
            raise FileNotFoundError(f"RF-DETR guard checkpoint not found: {path}")
        return path

    def predict(self, frame_bgr):
        frame_rgb = np.ascontiguousarray(frame_bgr[:, :, ::-1])
        detections = self.model.predict(frame_rgb, threshold=self.confidence)
        if len(detections) == 0 or not self.target_label:
            return detections

        data = getattr(detections, "data", None)
        if isinstance(data, dict) and "class_name" in data:
            names = np.asarray(data["class_name"], dtype=object)
            mask = np.asarray(
                [str(name).strip().lower() == self.target_label for name in names],
                dtype=bool,
            )
            return detections[mask]

        # For a fine-tuned checkpoint, do not map custom class IDs through COCO.
        # If a label filter was requested but class names are unavailable, fail
        # explicitly instead of silently filtering against the wrong taxonomy.
        if self.using_finetuned_weights:
            raise RuntimeError(
                "GUARD_RFDETR_TARGET_LABEL is set, but RF-DETR detections did not expose "
                "class_name metadata. Leave the target label blank for a single-class "
                "guard checkpoint or verify the checkpoint class metadata."
            )

        # Pretrained fallback: map COCO IDs only when no fine-tuned checkpoint is used.
        from rfdetr.assets.coco_classes import COCO_CLASSES

        if getattr(detections, "class_id", None) is None:
            return detections
        names = np.asarray([COCO_CLASSES[int(i)] for i in detections.class_id], dtype=object)
        mask = np.asarray(
            [str(name).strip().lower() == self.target_label for name in names],
            dtype=bool,
        )
        return detections[mask]
