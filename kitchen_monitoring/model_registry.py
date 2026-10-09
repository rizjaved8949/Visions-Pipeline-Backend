import threading

from .tracking import suppress_nested_person_detections

import torch
from ultralytics import YOLO

from .config import (
    PPE_MODEL_PATH,
    APRON_MODEL_PATH,
    PERSON_MODEL_PATH,
    PPE_IMAGE_SIZE,
    PPE_CONFIDENCE,
    PPE_IOU,
    APRON_IMAGE_SIZE,
    APRON_CONFIDENCE,
    APRON_IOU,
    DEVICE,
    EXPECTED_CLASSES,
    EXPECTED_APRON_CLASSES,
)


class KitchenModelRegistry:

    def __init__(self):

        self._ppe_model   = None
        self._apron_model = None

        self._ppe_lock   = threading.Lock()
        self._apron_lock = threading.Lock()


    # --------------------------------------------------------
    # PPE model  (mask / gloves / hairnet — existing)
    # --------------------------------------------------------

    def get_ppe_model(self):

        if self._ppe_model is None:

            if not PPE_MODEL_PATH.exists():

                raise FileNotFoundError(
                    f"PPE model not found: "
                    f"{PPE_MODEL_PATH}"
                )

            self._ppe_model = YOLO(
                str(PPE_MODEL_PATH)
            )

            actual_names = {
                int(k): str(v)
                for k, v
                in self._ppe_model.names.items()
            }

            if actual_names != EXPECTED_CLASSES:

                raise RuntimeError(
                    "Unexpected PPE class mapping.\n"
                    f"Expected: {EXPECTED_CLASSES}\n"
                    f"Model:    {actual_names}"
                )

        return self._ppe_model


    def predict_ppe(
        self,
        frame,
        image_size=None,
    ):

        model = self.get_ppe_model()

        with self._ppe_lock:

            result = model.predict(
                source=frame,
                imgsz=image_size or PPE_IMAGE_SIZE,
                conf=PPE_CONFIDENCE,
                iou=PPE_IOU,
                device=DEVICE,
                verbose=False,
            )[0]

        return result


    # --------------------------------------------------------
    # Apron model  (apron / no_apron — new)
    # --------------------------------------------------------

    def apron_model_available(self) -> bool:
        """Returns True only when apron_detector_best.pt exists on disk."""
        return APRON_MODEL_PATH.exists()


    def get_apron_model(self):
        """
        Lazy-loads the apron model.
        Raises FileNotFoundError when the weight file is absent —
        callers should check apron_model_available() first.
        """

        if self._apron_model is None:

            if not APRON_MODEL_PATH.exists():

                raise FileNotFoundError(
                    f"Apron model not found: {APRON_MODEL_PATH}\n"
                    "Train it via the Kaggle notebook and copy "
                    "apron_detector_best.pt into kitchen_monitoring/weights/"
                )

            self._apron_model = YOLO(
                str(APRON_MODEL_PATH)
            )

            actual_names = {
                int(k): str(v)
                for k, v
                in self._apron_model.names.items()
            }

            if actual_names != EXPECTED_APRON_CLASSES:

                raise RuntimeError(
                    "Unexpected apron class mapping.\n"
                    f"Expected: {EXPECTED_APRON_CLASSES}\n"
                    f"Model:    {actual_names}"
                )

        return self._apron_model


    def predict_apron(
        self,
        frame,
        image_size=None,
    ):
        """
        Runs the apron model on a frame.
        Returns the Ultralytics result object (same shape as predict_ppe).
        Returns None when the model file is absent — pipeline handles
        this gracefully by skipping apron detections for the frame.
        """

        if not self.apron_model_available():
            return None

        model = self.get_apron_model()

        with self._apron_lock:

            result = model.predict(
                source=frame,
                imgsz=image_size or APRON_IMAGE_SIZE,
                conf=APRON_CONFIDENCE,
                iou=APRON_IOU,
                device=DEVICE,
                verbose=False,
            )[0]

        return result


    # --------------------------------------------------------
    # Person model
    # --------------------------------------------------------

    def create_person_tracker(self):

        # Separate instance per session because tracking state
        # must NOT leak between different camera sessions.
        model = YOLO(PERSON_MODEL_PATH)
        model.add_callback(
            "on_predict_postprocess_end",
            suppress_nested_person_detections,
        )
        return model


    # --------------------------------------------------------
    # Health
    # --------------------------------------------------------

    def health(self):

        return {

            "ppe_model_path":
                str(PPE_MODEL_PATH),

            "ppe_model_exists":
                PPE_MODEL_PATH.exists(),

            "expected_classes":
                EXPECTED_CLASSES,

            "apron_model_path":
                str(APRON_MODEL_PATH),

            "apron_model_exists":
                APRON_MODEL_PATH.exists(),

            "expected_apron_classes":
                EXPECTED_APRON_CLASSES,

            "device":
                str(DEVICE),

            "cuda_available":
                torch.cuda.is_available(),

            "gpu":
                (
                    torch.cuda.get_device_name(0)
                    if torch.cuda.is_available()
                    else None
                ),

            "person_model":
                str(PERSON_MODEL_PATH),

        }


MODELS = KitchenModelRegistry()
