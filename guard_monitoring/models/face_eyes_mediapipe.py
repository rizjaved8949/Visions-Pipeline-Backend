from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from ..geometry import clamp_bbox
from ..types import EyeState

LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]


class MediaPipeEyeAnalyzer:
    """Eye geometry adapter with compatibility fallbacks.

    Preferred path is MediaPipe Face Landmarker Tasks when a task model path is
    configured. If no task model is configured, it uses Face Mesh when the installed
    MediaPipe build still exposes that API. If neither path is available, model loading
    fails only for the eye module; the rest of the guard pipeline continues.
    """

    def __init__(
        self,
        ear_closed_threshold: float = 0.20,
        min_face_pixels: int = 48,
        min_eye_width_pixels: int = 4,
        min_eye_symmetry_ratio: float = 0.30,
        face_landmarker_model: str = "",
    ):
        import mediapipe as mp

        self.mp = mp
        self.ear_closed_threshold = float(ear_closed_threshold)
        self.min_face_pixels = int(min_face_pixels)
        self.min_eye_width_pixels = int(min_eye_width_pixels)
        self.min_eye_symmetry_ratio = float(min_eye_symmetry_ratio)
        self.backend = None
        self.face_mesh = None
        self.landmarker = None

        model_path = Path(face_landmarker_model).expanduser() if face_landmarker_model else None
        if model_path and model_path.exists():
            base_options = mp.tasks.BaseOptions(model_asset_path=str(model_path))
            options = mp.tasks.vision.FaceLandmarkerOptions(
                base_options=base_options,
                running_mode=mp.tasks.vision.RunningMode.VIDEO,
                num_faces=1,
                min_face_detection_confidence=0.5,
                min_face_presence_confidence=0.5,
                min_tracking_confidence=0.5,
            )
            self.landmarker = mp.tasks.vision.FaceLandmarker.create_from_options(options)
            self.backend = "tasks"
        elif getattr(mp, "solutions", None) is not None and hasattr(mp.solutions, "face_mesh"):
            self.face_mesh = mp.solutions.face_mesh.FaceMesh(
                static_image_mode=True,
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.5,
            )
            self.backend = "solutions"
        else:
            raise RuntimeError(
                "MediaPipe face landmarks are unavailable. Set GUARD_FACE_LANDMARKER_MODEL "
                "to a valid Face Landmarker .task file, or use a MediaPipe build exposing face_mesh."
            )

    @staticmethod
    def _ear(points: np.ndarray) -> float:
        p1, p2, p3, p4, p5, p6 = points
        a = np.linalg.norm(p2 - p6)
        b = np.linalg.norm(p3 - p5)
        c = np.linalg.norm(p1 - p4)
        return float((a + b) / (2.0 * c + 1e-9))

    def analyze(
        self,
        frame_bgr,
        guard_box,
        pose_keypoints: np.ndarray | None = None,
        timestamp_ms: int = 0,
    ) -> EyeState:
        h, w = frame_bgr.shape[:2]
        head_box = self._head_box(guard_box, pose_keypoints, w, h)
        x1, y1, x2, y2 = [int(v) for v in clamp_bbox(head_box, w, h)]
        crop = frame_bgr[y1:y2, x1:x2]
        if crop.size == 0 or min(crop.shape[:2]) < self.min_face_pixels:
            return EyeState(available=False, quality_ok=False, reason="face_too_small_or_empty")

        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        landmarks = self._landmarks(rgb, timestamp_ms)
        if landmarks is None:
            return EyeState(available=False, quality_ok=False, reason="face_not_found")

        ch, cw = crop.shape[:2]

        def pts(indices):
            return np.asarray([(landmarks[i].x * cw, landmarks[i].y * ch) for i in indices], dtype=float)

        left = pts(LEFT_EYE)
        right = pts(RIGHT_EYE)
        left_width = float(np.linalg.norm(left[0] - left[3]))
        right_width = float(np.linalg.norm(right[0] - right[3]))
        if min(left_width, right_width) < self.min_eye_width_pixels:
            return EyeState(available=True, quality_ok=False, reason="eyes_too_small")

        symmetry = min(left_width, right_width) / max(left_width, right_width)
        if symmetry < self.min_eye_symmetry_ratio:
            return EyeState(available=True, quality_ok=False, reason="extreme_side_view_or_occlusion")

        ear_left = self._ear(left)
        ear_right = self._ear(right)
        ear_mean = (ear_left + ear_right) / 2.0
        return EyeState(
            available=True,
            quality_ok=True,
            eyes_closed=bool(ear_mean < self.ear_closed_threshold),
            ear_left=ear_left,
            ear_right=ear_right,
            ear_mean=ear_mean,
            reason="ok",
        )

    def _landmarks(self, rgb, timestamp_ms: int):
        if self.backend == "solutions":
            result = self.face_mesh.process(rgb)
            if not result.multi_face_landmarks:
                return None
            return result.multi_face_landmarks[0].landmark

        image = self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=rgb)
        result = self.landmarker.detect_for_video(image, int(max(0, timestamp_ms)))
        if not result.face_landmarks:
            return None
        return result.face_landmarks[0]

    def _head_box(self, guard_box, keypoints, width: int, height: int):
        gx1, gy1, gx2, gy2 = guard_box
        gw, gh = gx2 - gx1, gy2 - gy1
        if keypoints is not None and len(keypoints) >= 5:
            valid = []
            for idx in [0, 1, 2, 3, 4]:
                x, y, c = keypoints[idx]
                if c >= 0.20:
                    valid.append((x, y))
            if len(valid) >= 2:
                arr = np.asarray(valid, dtype=float)
                cx, cy = arr.mean(axis=0)
                span_x = max(float(np.ptp(arr[:, 0])), gw * 0.12)
                span_y = max(float(np.ptp(arr[:, 1])), gh * 0.08)
                half_w = max(span_x * 1.7, gw * 0.16)
                half_h = max(span_y * 2.2, gh * 0.14)
                return clamp_bbox(
                    (cx - half_w, cy - half_h, cx + half_w, cy + half_h),
                    width,
                    height,
                )
        return clamp_bbox((gx1, gy1, gx2, gy1 + gh * 0.38), width, height)

    def close(self) -> None:
        if self.landmarker is not None:
            self.landmarker.close()
        if self.face_mesh is not None and hasattr(self.face_mesh, "close"):
            self.face_mesh.close()
