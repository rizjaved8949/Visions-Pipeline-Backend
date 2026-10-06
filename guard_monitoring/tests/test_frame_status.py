"""Current-frame behavior, independent classes, and camera-aware motion tests."""
from types import SimpleNamespace
from unittest.mock import patch

import cv2
import numpy as np
import pytest

from guard_monitoring.config import load_config
from guard_monitoring.contracts import ok
from guard_monitoring.monitoring.frame_status import FrameMovementAnalyzer, compose_frame_statuses, PRIMARY_STATUSES
from guard_monitoring.monitoring.posture import PostureAnalyzer
from guard_monitoring.monitoring.sleep import SleepAnalyzer
from guard_monitoring.pipeline import GuardMonitoringPipeline
from guard_monitoring.models.pose_yolo import YOLOPoseEstimator
from guard_monitoring.types import EyeState, MovementState, PhoneState, PostureState, SleepState
from test_sleep_continuity import pipeline_fixture


def compose(*, eye=None, phone=None, posture=None, movement=None, sleep=None, seen_now=True):
    return compose_frame_statuses(phone or PhoneState(), sleep or SleepState(), eye or EyeState(),
                                  posture or PostureState(), movement or MovementState(), seen_now=seen_now)


def test_six_primary_classes_have_distinct_meanings():
    assert set(PRIMARY_STATUSES) == {"Sleeping", "Using Mobile", "Moving", "Stationary", "Sitting", "Standing"}


@pytest.mark.parametrize("usage", ["visible", "screen_use", "call"])
def test_phone_in_hand_cannot_hide_sleep(usage):
    labels = compose(eye=EyeState(quality_ok=True, eyes_closed=True, head_roll_degrees=25),
                     phone=PhoneState(detected=True, usage=usage))
    assert "Sleeping" in labels and "Phone Visible" in labels
    assert "Using Mobile" not in labels


def test_sleep_clears_on_first_fresh_open_eye_frame():
    raw = SleepState(candidate=True, evidence_quality="high")
    assert "Sleeping" in compose(eye=EyeState(quality_ok=True, eyes_closed=True), sleep=raw)
    assert "Sleeping" not in compose(eye=EyeState(quality_ok=True, eyes_closed=False), sleep=raw)


def test_closed_eyes_without_sleep_posture_are_explicitly_uncertain():
    assert compose(eye=EyeState(quality_ok=True, eyes_closed=True)) == ["Possible Sleep"]


@pytest.mark.parametrize("source", ["bbox_shape", "occlusion_inferred", "unknown"])
def test_cropped_walking_guard_cannot_be_called_sitting_by_box_shape(source):
    labels = compose(movement=MovementState(reliable=True, stationary=False),
                     posture=PostureState(posture="sitting", posture_source=source))
    assert labels == ["Moving"]


def test_stationary_and_sitting_are_both_shown():
    assert compose(movement=MovementState(reliable=True, stationary=True),
                   posture=PostureState(posture="sitting")) == ["Stationary", "Sitting"]


def test_phone_use_and_body_movement_are_both_shown():
    labels = compose(phone=PhoneState(detected=True, usage="screen_use"),
                     movement=MovementState(reliable=True, stationary=False),
                     posture=PostureState(posture="standing"))
    assert labels == ["Using Mobile", "Moving", "Standing"]


def test_unobserved_guard_does_not_reuse_last_frame_labels():
    assert compose(eye=EyeState(quality_ok=True, eyes_closed=True, head_roll_degrees=25),
                   seen_now=False) == ["Status Unknown"]


def skeleton(dx=0):
    points = np.zeros((17, 3))
    points[:, 2] = 1
    for i, x, y in ((0, 100, 60), (1, 95, 55), (2, 105, 55), (3, 85, 60), (4, 115, 60),
                    (5, 70, 90), (6, 130, 90), (7, 65, 125), (8, 135, 125), (9, 70, 150),
                    (10, 130, 150), (11, 80, 140), (12, 120, 140), (13, 80, 175),
                    (14, 120, 175), (15, 80, 205), (16, 120, 205)):
        points[i, :2] = x + dx, y
    return points


def camera(dx=0, known=True):
    return dict(valid=known, affine=[[1, 0, dx], [0, 1, 0]] if known else None, scene_change=False)


def texture(dx=0):
    image = np.zeros((240, 320, 3), dtype=np.uint8)
    patch_image = np.random.default_rng(4).integers(0, 255, (170, 100, 3), dtype=np.uint8)
    image[40:210, 50 + dx:150 + dx] = patch_image
    return image


def test_movement_start_and_stop_change_on_the_next_observed_frame():
    m = FrameMovementAnalyzer({})
    m.update(texture(), 1, (50, 40, 150, 210), skeleton(), 0, camera())
    assert m.update(texture(), 1, (50, 40, 150, 210), skeleton(), .1, camera()).stationary
    moving = m.update(texture(6), 1, (56, 40, 156, 210), skeleton(6), .2, camera())
    assert moving.reliable and not moving.stationary
    stopped = m.update(texture(6), 1, (56, 40, 156, 210), skeleton(6), .3, camera())
    assert stopped.reliable and stopped.stationary


def test_camera_pan_does_not_become_guard_movement():
    m = FrameMovementAnalyzer({})
    m.update(texture(), 1, (50, 40, 150, 210), skeleton(), 0, camera())
    state = m.update(texture(6), 1, (56, 40, 156, 210), skeleton(6), .1, camera(6))
    assert state.reliable and state.stationary


def test_unknown_camera_cannot_establish_whole_body_stationarity():
    m = FrameMovementAnalyzer({})
    m.update(texture(), 1, (50, 40, 150, 210), skeleton(), 0, camera(known=False))
    state = m.update(texture(6), 1, (56, 40, 156, 210), skeleton(6), .1, camera(known=False))
    assert not state.reliable


def test_arm_movement_is_detected_even_when_seated_and_camera_unknown():
    m = FrameMovementAnalyzer({})
    m.update(texture(), 1, (50, 40, 150, 210), skeleton(), 0, camera(known=False))
    changed = skeleton()
    changed[9:11, 0] += 8
    state = m.update(texture(), 1, (50, 40, 150, 210), changed, .1, camera(known=False))
    assert state.reliable and not state.stationary


def test_small_consistent_articulation_survives_a_failed_background_fit():
    m = FrameMovementAnalyzer({})
    frame = cv2.resize(texture(), None, fx=3, fy=3)
    pose = skeleton(); pose[:, :2] *= 3
    m.update(frame, 1, (150, 120, 450, 630), pose, 0, camera(known=False))
    changed = pose.copy(); changed[9:11, 0] += 3
    state = m.update(frame, 1, (150, 120, 450, 630), changed, 1 / 24, camera(known=False))
    assert state.reliable and not state.stationary


def test_subpixel_pose_jitter_with_unknown_camera_does_not_assert_movement():
    m = FrameMovementAnalyzer({})
    pose = skeleton(); pose[:, :2] *= 3
    frame = cv2.resize(texture(), None, fx=3, fy=3)
    m.update(frame, 1, (150, 120, 450, 630), pose, 0, camera(known=False))
    changed = pose.copy(); changed[9:11, 0] += .5
    state = m.update(frame, 1, (150, 120, 450, 630), changed, 1 / 24, camera(known=False))
    assert not state.reliable


def test_walking_bent_knee_is_standing_not_sitting():
    pose = skeleton()
    pose[15, :2] = (105, 200)  # A bent walking knee, still with vertical thighs.
    state = PostureAnalyzer(load_config()["pose_logic"]).analyze(pose, (50, 40, 150, 210))
    assert state.posture == "standing"


def test_pose_is_associated_with_guard_instead_of_highest_confidence_bystander():
    boxes = np.array([(50, 20, 150, 220), (120, 20, 180, 220)])
    assert YOLOPoseEstimator.select_guard_pose(boxes, [.7, .99], (50, 20, 150, 220)) == 0
    assert YOLOPoseEstimator.select_guard_pose(boxes, [.7, .99], (200, 20, 250, 220)) is None


def test_bent_knees_and_horizontal_thighs_are_sitting():
    pose = skeleton()
    pose[13, :2], pose[14, :2] = (125, 145), (165, 145)
    pose[15, :2], pose[16, :2] = (125, 205), (165, 205)
    assert PostureAnalyzer(load_config()["pose_logic"]).analyze(pose).posture == "sitting"


def test_sitting_is_detected_when_ankles_are_out_of_frame():
    pose = skeleton()
    pose[13, :2], pose[14, :2] = (125, 145), (165, 145)
    pose[15:17, 2] = 0
    assert PostureAnalyzer(load_config()["pose_logic"]).analyze(pose).posture == "sitting"


def test_reading_phone_with_occluded_eyes_does_not_imply_sleep():
    labels = compose(phone=PhoneState(detected=True, usage="screen_use"),
                     posture=PostureState(posture="sitting", head_down=True, head_down_known=True))
    assert "Using Mobile" in labels and "Possible Sleep" not in labels


def test_confirmed_sleep_evidence_can_coexist_with_a_held_phone():
    analyzer = SleepAnalyzer({})
    for i in range(31):
        state = analyzer.update(1, i / 10, EyeState(quality_ok=True, eyes_closed=True, head_roll_degrees=25),
                                PostureState(), MovementState(), PhoneState(detected=True, usage="screen_use"))
    assert state.candidate and state.evidence_quality == "high"


@pytest.fixture
def frame_pipeline(pipeline_fixture):
    f = pipeline_fixture
    f.pipeline.frame_status_enabled = True
    f.cfg["activity"]["frame_status_enabled"] = True
    f.pipeline.sleep_analyzer.suppress_phone = False
    return f


def test_pipeline_displays_sleep_and_wake_on_consecutive_frames(frame_pipeline):
    f = frame_pipeline
    eye = EyeState(quality_ok=True, eyes_closed=False, head_roll_degrees=25)
    with patch.object(f.registry, "eyes", return_value=ok(SimpleNamespace(analyze=lambda *args: eye))):
        assert "Sleeping" not in f.tick(0)["frame_log"]["display_statuses"]
        eye.eyes_closed = True
        sleeping = f.tick(1 / 24)
        assert sleeping["frame_log"]["activity_status"] == "Sleeping"
        assert "Sleeping" in sleeping["frame_log"]["display_statuses"]
        assert not sleeping["sleep"].candidate  # No inflated alert evidence.
        assert not sleeping["frame_log"]["sleep_display"]["held"]
        eye.eyes_closed = False
        assert "Sleeping" not in f.tick(2 / 24)["frame_log"]["display_statuses"]


def test_frame_mode_runs_all_sensors_despite_configured_strides(pipeline_fixture):
    f = pipeline_fixture
    f.cfg["activity"]["frame_status_enabled"] = True
    for name in ("pose", "phone", "eyes"):
        f.cfg["models"][name]["every_n_frames"] = 10
    f.pipeline = GuardMonitoringPipeline(f.cfg, registry=f.registry)
    with patch.object(f.pipeline.camera_motion, "update", return_value=camera()):
        for i in range(8):
            f.tick(i / 24)
    assert f.registry.pose_calls == 8
    assert len(f.registry.eye_timestamps) == 8


def test_pipeline_move_then_stop_has_no_presentation_wait(frame_pipeline):
    f = frame_pipeline
    f.tick(0)
    f.registry.dx = 6
    assert "Moving" in f.tick(1 / 24)["frame_log"]["display_statuses"]
    labels = f.tick(2 / 24)["frame_log"]["display_statuses"]
    assert "Moving" not in labels and "Stationary" in labels


def test_retained_guard_gets_fresh_status_analysis_without_advancing_alerts(frame_pipeline):
    f = frame_pipeline
    eye = EyeState(quality_ok=True, eyes_closed=True, head_pitch_degrees=-10)
    with patch.object(f.registry, "eyes", return_value=ok(SimpleNamespace(analyze=lambda *args: eye))):
        f.tick(0)
        f.registry.ids = []
        out = f.tick(1 / 24)
        assert not out["guard_seen_now"]
        assert out["frame_log"]["status_observed_now"]
        assert out["frame_log"]["sleep_display"]["fresh"]
        assert "Sleeping" in out["frame_log"]["display_statuses"]
        assert out["frame_log"]["rule_conditions"]["sleep"] is None
        assert f.registry.pose_calls == 2


def test_unverified_retained_guard_is_not_assigned_fresh_statuses(frame_pipeline):
    f = frame_pipeline
    f.tick(0)
    f.registry.ids = []
    f.registry.pose_failure = True
    out = f.tick(1 / 24)
    assert not out["frame_log"]["status_observed_now"]
    assert out["frame_log"]["display_statuses"] == ["Status Unknown"]
