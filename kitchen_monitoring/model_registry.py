import threading

import torch
from ultralytics import YOLO

from .config import (
    PPE_MODEL_PATH,
    PERSON_MODEL_PATH,
    PPE_IMAGE_SIZE,
    PPE_CONFIDENCE,
    PPE_IOU,
    DEVICE,
    EXPECTED_CLASSES,
)


class KitchenModelRegistry:

    def __init__(self):

        self._ppe_model = None

        self._ppe_lock = threading.Lock()


    # --------------------------------------------------------
    # PPE model
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
    ):

        model = self.get_ppe_model()

        # Ultralytics model objects should not be hit by
        # multiple application threads simultaneously.
        with self._ppe_lock:

            result = model.predict(
                source=frame,
                imgsz=PPE_IMAGE_SIZE,
                conf=PPE_CONFIDENCE,
                iou=PPE_IOU,
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
        return YOLO(
            PERSON_MODEL_PATH
        )


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