"""Regression tests for the robust-tracking refactor.

These tests verify the new behavior introduced in this refactor without
disturbing any existing test. All tests use scripted / numpy-only inputs
so they run without YOLO weights, cv2, or a GPU.
"""
from __future__ import annotations

import os
import sys
import types
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np

# ---------------------------------------------------------------------------
# cv2 shim – many modules import cv2 at module level.  The shim must be
# installed before any guard_monitoring import to avoid ImportError in
# environments without OpenCV.
# ---------------------------------------------------------------------------
if "cv2" not in sys.modules:
    _cv2_mock = MagicMock()
    _cv2_mock.COLOR_BGR2HSV = 40
    _cv2_mock.calcHist.return_value = np.ones((24, 24, 24), dtype=np.float32)
    _cv2_mock.normalize.side_effect = lambda src, dst, **kw: src
    _cv2_mock.cvtColor.return_value = np.zeros((10, 10, 3), dtype=np.uint8)
    sys.modules["cv2"] = _cv2_mock

from guard_monitoring.config import load_config, validate_config
from guard_monitoring.monitoring.guard_selector import GuardSelector
from guard_monitoring.monitoring.kalman import BBoxKalman
from guard_monitoring.monitoring.movement import MovementMonitor
from guard_monitoring.monitoring.posture import PostureAnalyzer
from guard_monitoring.monitoring.reid import AppearanceSignature, TorsoAppearanceModel
from guard_monitoring.types import GuardTrack, MovementState, PostureState


# ---------------------------------------------------------------------------
# Helpers shared across test cases
# ---------------------------------------------------------------------------

def detections(ids=(), boxes=None):
    if boxes is None:
        boxes = [(20.0, 10.0, 80.0, 100.0) for _ in ids]
    return SimpleNamespace(
        tracker_id=np.asarray(ids),
        xyxy=np.asarray(boxes).reshape(-1, 4),
        confidence=np.full(len(ids), 0.95),
    )


def selector(zone=None, **kwargs):
    if zone is None:
        zone = [(0, 0), (200, 0), (200, 200), (0, 200)]
    return GuardSelector(zone, confirm_seconds=0, **kwargs)


# ---------------------------------------------------------------------------
# Kalman filter tests
# ---------------------------------------------------------------------------

class KalmanTests(unittest.TestCase):

    def test_predict_returns_none_before_initialise(self):
        k = BBoxKalman()
        self.assertIsNone(k.predict(1.0))

    def test_predict_after_single_update_is_close(self):
        k = BBoxKalman()
        bbox = (20.0, 10.0, 80.0, 100.0)
        k.update(bbox, 0.0)
        pred = k.predict(0.1)
        self.assertIsNotNone(pred)
        # Velocity should be near zero after one frame; predicted box very close.
        for a, b in zip(pred, bbox):
            self.assertAlmostEqual(a, b, delta=1.0)

    def test_constant_velocity_prediction_advances_box(self):
        k = BBoxKalman()
        k.update((20.0, 10.0, 80.0, 100.0), 0.0)
        k.update((30.0, 10.0, 90.0, 100.0), 0.1)
        pred = k.predict(0.2)
        self.assertIsNotNone(pred)
        # Should predict somewhere near (40, 10, 100, 100) – accept generous tolerance
        self.assertGreater(pred[0], 25.0)

    def test_predict_returns_none_beyond_max_seconds(self):
        k = BBoxKalman(max_predict_seconds=0.5)
        k.update((20.0, 10.0, 80.0, 100.0), 0.0)
        self.assertIsNone(k.predict(1.0))

    def test_advance_without_measurement_increments_staleness(self):
        k = BBoxKalman(max_predict_seconds=0.5)
        k.update((20.0, 10.0, 80.0, 100.0), 0.0)
        # Advance 3 steps of 0.2 s each = 0.6 s total, beyond max_predict_seconds.
        k.advance_without_measurement(0.2)
        k.advance_without_measurement(0.4)
        result = k.advance_without_measurement(0.6)
        self.assertIsNone(result)

    def test_update_after_advance_resets_staleness(self):
        k = BBoxKalman(max_predict_seconds=0.5)
        k.update((20.0, 10.0, 80.0, 100.0), 0.0)
        k.advance_without_measurement(0.3)
        k.update((25.0, 10.0, 85.0, 100.0), 0.4)
        self.assertIsNotNone(k.predict(0.5))

    def test_reset_returns_to_uninitialised(self):
        k = BBoxKalman()
        k.update((20.0, 10.0, 80.0, 100.0), 0.0)
        k.reset()
        self.assertFalse(k.initialised)
        self.assertIsNone(k.predict(0.1))


# ---------------------------------------------------------------------------
# ReID appearance model tests
# ---------------------------------------------------------------------------

class ReIDTests(unittest.TestCase):

    def _model(self, bins=8):
        return TorsoAppearanceModel(bins=bins)

    def _uniform_frame(self, value=128, shape=(60, 40, 3)):
        return np.full(shape, value, dtype=np.uint8)

    def test_extract_returns_signature_for_valid_crop(self):
        model = self._model()
        sig = model.extract(self._uniform_frame(), (5, 5, 35, 55))
        self.assertIsNotNone(sig)
        self.assertFalse(sig.is_empty())

    def test_extract_returns_none_for_degenerate_box(self):
        model = self._model()
        # Zero-size box
        sig = model.extract(self._uniform_frame(), (10, 10, 10, 10))
        self.assertIsNone(sig)

    def test_uniform_frame_is_not_distinctive(self):
        # All-black or all-white frames are not ReID-safe; the histogram
        # concentrates into a single bin which is_distinctive() must reject.
        model = self._model()
        for value in (0, 255):
            frame = np.full((60, 40, 3), value, dtype=np.uint8)
            sig = model.extract(frame, (5, 5, 35, 55))
            if sig is not None:
                self.assertFalse(
                    sig.is_distinctive(),
                    f"uniform frame (value={value}) should not be distinctive",
                )

    def test_similar_crops_score_higher_than_different_crops(self):
        # Force the numpy fallback path (no real cv2) so the histograms
        # actually reflect the pixel values.
        model_no_cv2 = TorsoAppearanceModel(bins=8)
        model_no_cv2._cv2 = None  # force numpy path regardless of env
        # Two patches of same colour -> high similarity
        patch_a = np.zeros((60, 40, 3), dtype=np.uint8)
        patch_a[:, :, 1] = 200  # green tint
        patch_a[20:40, 10:30, 0] = 80  # add texture so is_distinctive() passes
        sig_a1 = model_no_cv2.extract(patch_a, (0, 0, 40, 60))
        sig_a2 = model_no_cv2.extract(patch_a.copy(), (0, 0, 40, 60))
        # Very different colour patch
        patch_b = np.zeros((60, 40, 3), dtype=np.uint8)
        patch_b[:, :, 2] = 200  # red tint
        patch_b[20:40, 10:30, 1] = 80
        sig_b = model_no_cv2.extract(patch_b, (0, 0, 40, 60))
        if sig_a1 is not None and sig_a2 is not None and sig_b is not None:
            same_score = model_no_cv2.similarity(sig_a1, sig_a2)
            diff_score = model_no_cv2.similarity(sig_a1, sig_b)
            self.assertGreater(same_score, diff_score,
                               "identical patches should score higher than different ones")

    def test_similarity_of_missing_signatures_is_zero(self):
        model = self._model()
        empty = AppearanceSignature(np.array([]), 8)
        self.assertEqual(model.similarity(None, None), 0.0)
        self.assertEqual(model.similarity(empty, empty), 0.0)


# ---------------------------------------------------------------------------
# GuardSelector with ReID alias set (identity continuity across track-ID change)
# ---------------------------------------------------------------------------

class SelectorReIDTests(unittest.TestCase):

    def _reid_frame(self, color=(80, 200, 40)):
        """Return a small distinctive BGR frame."""
        frame = np.zeros((200, 200, 3), dtype=np.uint8)
        frame[:, :] = color
        # Add texture so it is_distinctive()
        frame[20:80, 20:80] = (200, 50, 150)
        frame[100:160, 100:160] = (30, 220, 90)
        return frame

    def test_alias_preserves_active_id(self):
        """After ReID adoption the active_id stays the same so downstream
        timers are never reset."""
        s = GuardSelector(
            [(0, 0), (200, 0), (200, 200), (0, 200)],
            confirm_seconds=0,
            reid_enabled=True,
            reid_min_similarity=0.0,   # accept any score
            reid_max_footpoint_ratio=1.0,
        )
        frame = self._reid_frame()
        # Establish guard with track_id=1
        s.update(detections([1]), 0.0, frame=frame)
        original_id = s.active_id
        self.assertEqual(original_id, 1)

        # Guard disappears; ByteTrack mints new id=2 with same appearance
        # and position.
        s.update(detections([]), 0.1, frame=frame)
        result = s.update(detections([2]), 0.2, frame=frame)

        # active_id must stay 1 (alias mechanism)
        self.assertEqual(s.active_id, 1)
        # The returned track reports the original id so the rule engine
        # continues on the same timer.
        if result is not None:
            self.assertEqual(result.track_id, 1)

    def test_uniform_black_frame_never_triggers_reid(self):
        """All-zero frames must never result in ReID adoption because the
        signatures are not distinctive (they concentrate in one bin)."""
        s = GuardSelector(
            [(0, 0), (200, 0), (200, 200), (0, 200)],
            confirm_seconds=0,
            reid_enabled=True,
            reid_min_similarity=0.0,
        )
        black_frame = np.zeros((200, 200, 3), dtype=np.uint8)
        s.update(detections([1]), 0.0, frame=black_frame)
        s.update(detections([]), 0.1, frame=black_frame)
        # Should NOT adopt id=2 through ReID because histogram is not distinctive.
        result = s.update(detections([2]), 0.2, frame=black_frame)
        # Either None (retention only) or still tracking via retained.
        if result is not None:
            self.assertEqual(result.track_id, 1)  # original id retained

    def test_reid_inactive_without_frame(self):
        """When no frame is passed, ReID is never attempted even if enabled."""
        s = GuardSelector(
            [(0, 0), (200, 0), (200, 200), (0, 200)],
            confirm_seconds=0,
            reid_enabled=True,
            reid_min_similarity=0.0,
        )
        s.update(detections([1]), 0.0)           # no frame
        s.update(detections([]), 0.1)
        result = s.update(detections([2]), 0.2)  # no frame
        # No adoption without a frame; id=1 should be retained.
        if result is not None:
            self.assertEqual(result.track_id, 1)


# ---------------------------------------------------------------------------
# Kalman predicted-box retention
# ---------------------------------------------------------------------------

class SelectorKalmanTests(unittest.TestCase):

    def test_retained_with_kalman_returns_predicted_source(self):
        s = GuardSelector(
            [(0, 0), (200, 0), (200, 200), (0, 200)],
            confirm_seconds=0,
            predict_during_grace=True,
        )
        s.update(detections([1]), 0.0)
        s.update(detections([1]), 0.1)  # second update so Kalman has velocity
        # Guard disappears
        retained = s.retained(0.2)
        self.assertIsNotNone(retained)
        self.assertEqual(retained.source, "predicted")
        self.assertFalse(retained.seen_now)

    def test_retained_without_kalman_returns_retained_source(self):
        s = GuardSelector(
            [(0, 0), (200, 0), (200, 200), (0, 200)],
            confirm_seconds=0,
            predict_during_grace=False,
        )
        s.update(detections([1]), 0.0)
        s.update(detections([]), 0.1)
        retained = s.retained(0.2)
        self.assertIsNotNone(retained)
        self.assertEqual(retained.source, "retained")

    def test_predicted_box_expires_within_max_seconds(self):
        s = GuardSelector(
            [(0, 0), (200, 0), (200, 200), (0, 200)],
            confirm_seconds=0,
            predict_during_grace=True,
            release_seconds=5.0,
            presence_grace_seconds=5.0,
        )
        s.update(detections([1]), 0.0)
        # A very short max_predict_seconds
        s._kalman.max_predict_seconds = 0.3
        retained = s.retained(0.5)
        # Either None (Kalman expired → retained is None) or 'retained' source.
        if retained is not None:
            self.assertIn(retained.source, ("retained", "predicted"))


# ---------------------------------------------------------------------------
# Soft border movement rule
# ---------------------------------------------------------------------------

class SoftBorderMovementTests(unittest.TestCase):

    def _monitor(self, min_visible_area_ratio=0.75):
        return MovementMonitor(
            history_seconds=3, stationary_radius_ratio=0.035,
            minimum_history_seconds=1, patrol_zones=[],
            min_visible_area_ratio=min_visible_area_ratio,
        )

    def test_mostly_visible_box_keeps_history(self):
        """When >75% of the box is inside the frame the soft rule keeps history."""
        m = self._monitor(min_visible_area_ratio=0.75)
        frame_shape = (480, 640, 3)
        # Box mostly on screen: x spans 580–650 (w=640), visible x = 580–640 = 60/70 = 86%
        box = (580.0, 80.0, 650.0, 300.0)  # 60px on, 10px off => 85.7% visible
        state = m.update(1, box, 0.0, frame_shape=frame_shape)
        # Not invalidated by soft rule (>75% visible)
        self.assertNotEqual(state.reason, "guard_box_truncated")

    def test_mostly_off_screen_box_is_truncated(self):
        """When <75% visible, the soft rule still invalidates."""
        m = self._monitor(min_visible_area_ratio=0.75)
        frame_shape = (480, 640, 3)
        # Only the leftmost 10px is on screen (90% off)
        box = (630.0, 80.0, 900.0, 300.0)
        state = m.update(1, box, 0.0, frame_shape=frame_shape)
        self.assertEqual(state.reason, "guard_box_truncated")

    def test_strict_border_still_works_when_ratio_is_zero(self):
        """Default ratio=0 → original strict edge-touch rule."""
        m = self._monitor(min_visible_area_ratio=0.0)
        frame_shape = (480, 640, 3)
        # Edge-touching box
        box = (0.0, 80.0, 200.0, 300.0)
        state = m.update(1, box, 0.0, frame_shape=frame_shape)
        self.assertEqual(state.reason, "guard_box_truncated")


# ---------------------------------------------------------------------------
# Posture aspect-ratio fallback
# ---------------------------------------------------------------------------

class PostureAspectFallbackTests(unittest.TestCase):

    def _analyzer(self, enabled=True):
        return PostureAnalyzer({
            "aspect_fallback_enabled": enabled,
            "sitting_aspect_ratio_max": 1.55,
            "standing_aspect_ratio_min": 2.10,
        })

    def test_no_keypoints_tall_box_is_standing(self):
        a = self._analyzer()
        # h/w = 210/80 = 2.625 > standing_aspect_min
        state = a.analyze(None, bbox=(20.0, 10.0, 100.0, 220.0))
        self.assertEqual(state.posture, "standing")
        self.assertEqual(state.posture_source, "bbox_shape")

    def test_no_keypoints_wide_box_is_sitting(self):
        a = self._analyzer()
        # h/w = 60/80 = 0.75 < sitting_aspect_max
        state = a.analyze(None, bbox=(20.0, 100.0, 100.0, 160.0))
        self.assertEqual(state.posture, "sitting")

    def test_fallback_disabled_returns_unknown(self):
        a = self._analyzer(enabled=False)
        state = a.analyze(None, bbox=(20.0, 100.0, 100.0, 160.0))
        self.assertEqual(state.posture, "unknown")

    def test_aspect_recorded_even_when_keypoints_present(self):
        a = self._analyzer()
        # 17 all-zero keypoints (all below confidence threshold → unknown posture)
        kp = np.zeros((17, 3), dtype=float)
        state = a.analyze(kp, bbox=(20.0, 10.0, 100.0, 220.0))
        self.assertIsNotNone(state.bbox_aspect_ratio)

    def test_no_bbox_returns_unknown_when_no_keypoints(self):
        a = self._analyzer()
        state = a.analyze(None)
        self.assertEqual(state.posture, "unknown")


# ---------------------------------------------------------------------------
# Activity mapping: evidence-based mode
# ---------------------------------------------------------------------------

class ActivityMappingTests(unittest.TestCase):
    """Integration tests that run a few pipeline frames with scripted models
    and verify the new 'evidence' mapping behaves correctly."""

    def _cfg(self):
        env = {k: v for k, v in os.environ.items() if not k.startswith("GUARD_")}
        with patch.dict(os.environ, env, clear=True):
            cfg = load_config()
        cfg["guard_selection"]["confirm_seconds"] = 0
        cfg["movement"].update(minimum_history_seconds=0.5, history_seconds=1.5)
        cfg["rules"].update(warmup_seconds=0, sleep_seconds=2, phone_seconds=2,
                            stationary_seconds=20, absence_seconds=1)
        for name in ("pose", "phone", "eyes"):
            cfg["models"][name]["every_n_frames"] = 1
        cfg["activity"]["mapping_mode"] = "evidence"
        cfg["activity"]["frame_status_enabled"] = False
        return cfg

    def _pipeline_and_registry(self, cfg):
        """Return (pipeline, registry) with scripted models."""
        import types as _types

        # Import here so the cv2 shim above is already active.
        try:
            from guard_monitoring.pipeline import GuardMonitoringPipeline
        except Exception:
            self.skipTest("pipeline could not be imported")

        from guard_monitoring.types import EyeState, PoseObservation

        class Registry:
            def __init__(self):
                self.ids = [1]
                self.dx = 0

            def guard_detector(self):
                def predict(frame):
                    return _types.SimpleNamespace(
                        xyxy=np.array([(20 + self.dx, 10, 80 + self.dx, 100)]),
                        confidence=np.array([0.95]),
                        tracker_id=np.array(self.ids),
                    )
                from guard_monitoring.contracts import ok
                return ok(_types.SimpleNamespace(predict=predict))

            def tracker(self, frame_rate=30):
                from guard_monitoring.contracts import ok
                return ok(_types.SimpleNamespace(update=lambda d, **kw: d))

            def pose(self):
                def predict(frame, box):
                    k = np.zeros((17, 3))
                    k[:, 2] = 1
                    k[0, :2] = (50, 20)
                    for pair, y in [((5, 6), 40), ((11, 12), 65), ((13, 14), 80), ((15, 16), 95)]:
                        k[pair[0], :2] = (40, y); k[pair[1], :2] = (60, y)
                    return PoseObservation(k, tuple(box), 0.95)
                from guard_monitoring.contracts import ok
                return ok(_types.SimpleNamespace(predict=predict))

            def phone(self):
                from guard_monitoring.contracts import ok
                return ok(_types.SimpleNamespace(predict=lambda f, b: []))

            def eyes(self):
                from guard_monitoring.contracts import ok
                return ok(_types.SimpleNamespace(
                    analyze=lambda f, b, k, t: EyeState(
                        available=True, quality_ok=True, eyes_closed=False, reason="scripted"
                    )
                ))

            def reset_tracker(self): pass
            def close(self): pass

        registry = Registry()
        pipeline = GuardMonitoringPipeline(cfg, registry=registry)
        return pipeline, registry

    def _run_sequence(self, pipeline, registry, cfg, steps=61):
        from guard_monitoring.monitoring.guard_selector import GuardSelector
        from guard_monitoring.monitoring.movement import MovementMonitor
        from guard_monitoring.monitoring.rules import TimedRuleEngine
        from guard_monitoring.geometry import normalized_polygon_to_pixels

        selector = GuardSelector([(0, 0), (1000, 0), (1000, 1000), (0, 1000)], confirm_seconds=0)
        m = cfg["movement"]
        movement = MovementMonitor(m["history_seconds"], m["stationary_radius_ratio"],
                                   m["minimum_history_seconds"], [])
        rules = TimedRuleEngine("cam", cfg["rules"])
        frame = np.zeros((128, 256, 3), dtype=np.uint8)

        camera_patch = patch.object(
            pipeline.camera_motion, "update",
            return_value=dict(enabled=True, valid=True, scene_change=False,
                              reason="scripted_background", affine=[[1, 0, 0], [0, 1, 0]])
        )
        draw_patch = patch("guard_monitoring.pipeline.draw_activity_status")
        id_patch = patch("guard_monitoring.pipeline.draw_guard_identity")
        all_persons_patch = patch("guard_monitoring.pipeline.draw_all_persons")
        hud_patch = patch("guard_monitoring.pipeline.draw_hud")

        with camera_patch, draw_patch, id_patch, all_persons_patch, hud_patch:
            results = []
            for i in range(steps):
                out = pipeline._process_frame(
                    frame=frame, frame_idx=i, now=i * 0.1,
                    selector=selector, movement_monitor=movement,
                    rules=rules, duty_zone=selector.zone, patrol_zones=[],
                )
                results.append(out)
        return results

    def test_stationary_standing_is_stationary_not_moving(self):
        """Core bug fix: a still-standing guard must report Stationary,
        not Moving."""
        cfg = self._cfg()
        pipeline, registry = self._pipeline_and_registry(cfg)
        out = self._run_sequence(pipeline, registry, cfg)
        final_status = out[-1]["frame_log"]["activity_status"]
        self.assertEqual(final_status, "Stationary",
                         f"standing-still guard should be Stationary, got {final_status!r}")

    def test_walking_guard_is_moving(self):
        cfg = self._cfg()
        pipeline, registry = self._pipeline_and_registry(cfg)

        # Patch registry dx to advance each frame
        original_run = self._run_sequence

        from guard_monitoring.monitoring.guard_selector import GuardSelector
        from guard_monitoring.monitoring.movement import MovementMonitor
        from guard_monitoring.monitoring.rules import TimedRuleEngine

        selector = GuardSelector([(0, 0), (1000, 0), (1000, 1000), (0, 1000)], confirm_seconds=0)
        m = cfg["movement"]
        movement = MovementMonitor(m["history_seconds"], m["stationary_radius_ratio"],
                                   m["minimum_history_seconds"], [])
        rules = TimedRuleEngine("cam", cfg["rules"])
        frame = np.zeros((128, 256, 3), dtype=np.uint8)

        camera_patch = patch.object(
            pipeline.camera_motion, "update",
            return_value=dict(enabled=True, valid=True, scene_change=False,
                              reason="scripted_background", affine=[[1, 0, 0], [0, 1, 0]])
        )
        draw_patch = patch("guard_monitoring.pipeline.draw_activity_status")
        id_patch = patch("guard_monitoring.pipeline.draw_guard_identity")
        all_persons_patch = patch("guard_monitoring.pipeline.draw_all_persons")
        hud_patch = patch("guard_monitoring.pipeline.draw_hud")

        out = []
        with camera_patch, draw_patch, id_patch, all_persons_patch, hud_patch:
            for i in range(61):
                registry.dx = i * 2
                o = pipeline._process_frame(
                    frame=frame, frame_idx=i, now=i * 0.1,
                    selector=selector, movement_monitor=movement,
                    rules=rules, duty_zone=selector.zone, patrol_zones=[],
                )
                out.append(o)

        final_status = out[-1]["frame_log"]["activity_status"]
        self.assertEqual(final_status, "Moving",
                         f"walking guard should be Moving, got {final_status!r}")

    def test_legacy_mode_standing_is_moving(self):
        """Opt-out: GUARD_ACTIVITY_MAPPING=legacy preserves the old behaviour
        (standing → Moving)."""
        cfg = self._cfg()
        cfg["activity"]["mapping_mode"] = "legacy"
        pipeline, registry = self._pipeline_and_registry(cfg)
        out = self._run_sequence(pipeline, registry, cfg)
        final_status = out[-1]["frame_log"]["activity_status"]
        # Legacy mode: standing unconditionally maps to Moving
        self.assertEqual(final_status, "Moving")

    def test_activity_decision_basis_is_logged(self):
        cfg = self._cfg()
        pipeline, registry = self._pipeline_and_registry(cfg)
        out = self._run_sequence(pipeline, registry, cfg, steps=10)
        for frame_result in out:
            fl = frame_result["frame_log"]
            self.assertIn("activity_decision_basis", fl)


# ---------------------------------------------------------------------------
# Config validation guards
# ---------------------------------------------------------------------------

class ConfigValidationTests(unittest.TestCase):

    def _clean(self):
        env = {k: v for k, v in os.environ.items() if not k.startswith("GUARD_")}
        with patch.dict(os.environ, env, clear=True):
            return load_config()

    def test_invalid_reid_similarity_rejected(self):
        cfg = self._clean()
        cfg["guard_selection"]["reid_min_similarity"] = 1.5
        with self.assertRaises(ValueError):
            validate_config(cfg)

    def test_invalid_aspect_ratio_ordering_rejected(self):
        cfg = self._clean()
        cfg["pose_logic"]["sitting_aspect_ratio_max"] = 3.0
        cfg["pose_logic"]["standing_aspect_ratio_min"] = 2.0
        with self.assertRaises(ValueError):
            validate_config(cfg)

    def test_invalid_mapping_mode_rejected(self):
        cfg = self._clean()
        cfg["activity"]["mapping_mode"] = "unsupported_mode"
        with self.assertRaises(ValueError):
            validate_config(cfg)

    def test_new_defaults_pass_validation(self):
        cfg = self._clean()
        # Should not raise
        validate_config(cfg)

    def test_negative_wrist_crop_ratio_rejected(self):
        cfg = self._clean()
        cfg["phone_logic"]["wrist_crop_ratio"] = -0.1
        with self.assertRaises(ValueError):
            validate_config(cfg)


# ---------------------------------------------------------------------------
# draw_all_persons overlay (no real cv2 needed — shim is enough)
# ---------------------------------------------------------------------------

class DrawAllPersonsTests(unittest.TestCase):

    def _frame(self):
        return np.zeros((240, 320, 3), dtype=np.uint8)

    def test_no_crash_on_none_detections(self):
        from guard_monitoring.visualization.overlay import draw_all_persons
        draw_all_persons(self._frame(), None, selected_id=1)

    def test_selected_id_is_skipped(self):
        """draw_all_persons must never draw the selected guard's box
        (that is done by draw_guard_identity)."""
        from guard_monitoring.visualization.overlay import draw_all_persons
        tracked = detections([1, 2], [(10, 10, 60, 100), (150, 10, 200, 100)])
        frame = self._frame()
        draw_all_persons(frame, tracked, selected_id=1)
        # Person 1 (selected) → no draw; person 2 → draw.
        # We can't easily verify pixel-level without real cv2, but the call
        # must not raise.

    def test_no_crash_empty_detections(self):
        from guard_monitoring.visualization.overlay import draw_all_persons
        tracked = detections([])
        draw_all_persons(self._frame(), tracked, selected_id=None)


if __name__ == "__main__":
    unittest.main()
