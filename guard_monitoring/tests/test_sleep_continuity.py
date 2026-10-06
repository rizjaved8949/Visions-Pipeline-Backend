"""Sleep sensor and actual multi-status overlay regressions, without AI weights."""
from unittest.mock import patch
import json
from types import SimpleNamespace

import pytest
import numpy as np

from guard_monitoring.config import validate_config
from guard_monitoring.contracts import ok
from guard_monitoring.monitoring.sleep import SleepAnalyzer, SleepStatusStabilizer
from guard_monitoring.monitoring.posture import PostureAnalyzer
from guard_monitoring.models.face_eyes_mediapipe import MediaPipeEyeAnalyzer, LEFT_EYE, RIGHT_EYE
from guard_monitoring.types import EyeState, MovementState, PhoneState, PostureState, SleepState
from test_regression_stability import clean_config
import test_regression_pipeline as pipeline_tests


def observation(analyzer, t, closed=True, *, ear=None, posture=None, movement=None, phone=None, available=None):
    return analyzer.update(
        1, t, EyeState(quality_ok=closed is not None, eyes_closed=closed, ear_mean=ear, observed_at=t),
        posture or PostureState(), movement or MovementState(stationary=True, reliable=True),
        phone or PhoneState(), available,
    )


@pytest.mark.parametrize("fps", [3, 10, 24, 30])
@pytest.mark.parametrize("duration", [10, 15, 30])
def test_every_frame_after_qualification_survives_brief_eye_dropout(fps, duration):
    analyzer = SleepAnalyzer({})
    display = SleepStatusStabilizer({})
    first = None
    for i in range(duration * fps):
        t = i / fps
        # Two bad samples every three seconds, also at the live service's 3 FPS.
        phase = i % (3 * fps)
        closed = None if i >= 3 * fps and phase == 0 else not (i >= 3 * fps and phase == 1)
        state = observation(analyzer, t, closed)
        shown = display.update(state, t, track_id=1, present=True)
        if state.candidate and first is None:
            first = t
            assert shown["label"] == "Sleeping"  # No second entry timer.
        if first is not None:
            assert shown["label"] == "Sleeping", (fps, t, state)
    assert first is not None and 2 <= first <= 2 + 1 / fps


def test_ear_jitter_is_filtered_but_sustained_open_eyes_clear_sleep():
    analyzer = SleepAnalyzer({})
    display = SleepStatusStabilizer({})
    for i in range(101):
        t = i / 10
        ear = (0.19 if i % 2 == 0 else 0.21) if t < 6 else 0.30
        state = observation(analyzer, t, ear < .20, ear=ear)
        shown = display.update(state, t, track_id=1, present=True)
        if 2.1 <= t < 6:
            assert shown["label"] == "Sleeping"
        if t >= 7.2:
            assert shown["label"] is None


def test_blinks_and_open_eye_head_down_do_not_qualify():
    analyzer = SleepAnalyzer({})
    posture = PostureState(posture="sitting", head_down=True, head_down_known=True)
    for i in range(300):
        state = observation(analyzer, i / 30, i % 60 < 5, posture=posture)
        assert not state.candidate


def test_blink_after_waking_cannot_reuse_old_sleep_history():
    analyzer = SleepAnalyzer({})
    display = SleepStatusStabilizer({})
    for i in range(71):
        t = i / 10
        closed = t < 5 or 6.5 <= t <= 6.6
        state = observation(analyzer, t, closed)
        shown = display.update(state, t, track_id=1, present=True)
        if 2.1 <= t < 5:
            assert shown["label"] == "Sleeping"
        if t >= 6:
            assert not state.candidate
            assert shown["label"] is None


def test_posture_fallback_qualifies_in_short_clip_and_pauses_unknown_pose():
    analyzer = SleepAnalyzer({"fallback_confirm_seconds": 4})
    display = SleepStatusStabilizer({})
    posture = PostureState(posture="sitting", head_down=True, head_down_known=True)
    last_duration = 0
    for i in range(100):
        t = i / 10
        missing_pose = i in (20, 21, 60)
        state = observation(analyzer, t, None, posture=posture,
                            available=dict(movement=True, posture=not missing_pose, eyes=False, phone=True))
        shown = display.update(state, t, track_id=1, present=True)
        if missing_pose:
            assert state.fallback_seconds == last_duration
        last_duration = state.fallback_seconds
        if t < 4:
            assert shown["label"] is None
        if t >= 4.5:
            assert shown["label"] == "Possible Sleep"


def test_posture_fallback_resets_on_awake_eyes_and_long_gap():
    analyzer = SleepAnalyzer({"fallback_confirm_seconds": 4})
    posture = PostureState(posture="sitting", head_down=True, head_down_known=True)
    for i in range(30):
        observation(analyzer, i / 10, None, posture=posture)
    assert observation(analyzer, 3, False, posture=posture).fallback_seconds == 0
    for i in range(31, 50):
        observation(analyzer, i / 10, None, posture=posture)
    observation(analyzer, 5, None, available=dict(movement=False, posture=False, eyes=False, phone=True))
    assert observation(analyzer, 6, None, posture=posture).fallback_seconds == 0


@pytest.mark.parametrize("reason", ["moving", "phone_active", "phone_association_uncertain"])
def test_conflicting_activity_removes_held_sleep_immediately(reason):
    display = SleepStatusStabilizer({})
    display.update(SleepState(candidate=True, evidence_quality="high"), 0, track_id=1, present=True)
    assert display.update(SleepState(reason=reason), .1, track_id=1, present=True)["label"] is None


def test_hold_expires_and_does_not_transfer_to_another_guard():
    display = SleepStatusStabilizer({})
    positive = SleepState(candidate=True, evidence_quality="high")
    display.update(positive, 0, track_id=1, present=True)
    assert display.update(SleepState(), .5, track_id=1, present=None)["held"]
    assert display.update(SleepState(), 1.1, track_id=1, present=True)["label"] is None
    display.update(positive, 2, track_id=1, present=True)
    assert display.update(SleepState(), 2.1, track_id=2, present=True)["label"] is None
    display.update(positive, 3, track_id=2, present=True)
    assert display.update(SleepState(), 3.1, track_id=2, present=False)["label"] is None


def test_cached_eye_sample_does_not_extend_hold_from_processing_time():
    display = SleepStatusStabilizer({})
    positive = SleepState(candidate=True, evidence_quality="high")
    display.update(positive, 0, track_id=1, present=True, observed_at=0)
    display.update(positive, .7, track_id=1, present=True, observed_at=0)
    assert display.update(SleepState(), 1.1, track_id=1, present=True)["label"] is None


def test_fresh_posture_supports_existing_sleep_for_a_bounded_period_only():
    display = SleepStatusStabilizer({})
    support = SleepState(posture_support=True)
    assert display.update(support, 0, track_id=1, present=True)["label"] is None
    display.update(SleepState(candidate=True, evidence_quality="high"), .1, track_id=1, present=True)
    for t in (.5, 1, 1.5, 2, 2.5, 3):
        shown = display.update(support, t, track_id=1, present=True)
        assert shown["label"] == "Sleeping" and shown["held"]
    assert display.update(support, 3.2, track_id=1, present=True)["label"] is None


def test_cropped_torso_head_drop_does_not_change_sitting_classification():
    analyzer = PostureAnalyzer(clean_config()["pose_logic"])
    keypoints = np.zeros((17, 3))
    keypoints[0] = (50, 60, 1)
    keypoints[1] = (60, 35, 1)
    keypoints[3] = (85, 30, 1)
    posture = analyzer.analyze(keypoints, (0, 0, 100, 120))
    assert posture.head_down_known and posture.head_down
    assert posture.posture == "sitting"
    keypoints[0] = (50, 20, 1)
    posture = analyzer.analyze(keypoints, (0, 0, 100, 120))
    assert posture.head_down_known and not posture.head_down
    assert posture.posture == "sitting"
    # Low confidence head landmarks cannot supply fallback evidence.
    keypoints[3, 2] = .2
    assert not analyzer.analyze(keypoints, (0, 0, 100, 120)).head_down_known


def test_posture_to_eye_handoff_still_requires_sustained_wake_evidence():
    display = SleepStatusStabilizer({})
    display.update(SleepState(candidate=True, evidence_quality="high"), 0, track_id=1, present=True)
    for t in (.5, 1, 1.5, 2):
        display.update(SleepState(posture_support=True), t, track_id=1, present=True)
    awake = SleepState(reason="awake_eye_evidence", evidence_quality="high")
    for t in (2.1, 2.3, 2.5, 2.7):
        assert display.update(awake, t, track_id=1, present=True)["label"] == "Sleeping"
    assert display.update(awake, 2.9, track_id=1, present=True)["label"] is None


@pytest.mark.parametrize("depth,valid", [(0, True), (.24, False)])
def test_profile_eye_geometry_is_unknown_not_awake(depth, valid):
    # Exercise adapter quality gating with known 3-D geometry, without loading AI.
    analyzer = MediaPipeEyeAnalyzer.__new__(MediaPipeEyeAnalyzer)
    analyzer.min_face_pixels = 48
    analyzer.min_eye_width_pixels = 4
    analyzer.min_eye_symmetry_ratio = .3
    analyzer.ear_closed_threshold = .2
    analyzer.max_eye_yaw_degrees = 45
    points = [SimpleNamespace(x=.5, y=.5, z=0) for _ in range(478)]
    offsets = [(-.05, 0), (-.025, -.005), (.025, -.005), (.05, 0), (.025, .005), (-.025, .005)]
    for indices, cx, z in ((LEFT_EYE, .4, 0), (RIGHT_EYE, .6, depth)):
        for idx, (dx, dy) in zip(indices, offsets):
            points[idx] = SimpleNamespace(x=cx + dx, y=.4 + dy, z=z)
    analyzer._head_box = lambda *args: (0, 0, 200, 200)
    analyzer._landmarks = lambda *args: points
    eye = analyzer.analyze(np.zeros((200, 200, 3), dtype=np.uint8), (0, 0, 200, 200))
    assert eye.quality_ok is valid
    assert eye.eyes_closed is (True if valid else None)
    if not valid:
        assert eye.reason == "extreme_side_view_or_occlusion"


def test_eye_only_and_combined_posture_routes_are_preserved():
    analyzer = SleepAnalyzer({})
    posture = PostureState(posture="sitting", head_down=True, head_down_known=True)
    for i in range(31):
        state = observation(analyzer, i / 10, posture=posture,
                            movement=MovementState(reliable=False))
    assert state.candidate and state.decision_basis == "eyes_and_head_posture"


@pytest.mark.parametrize("field,value", [("ear_open_threshold", .1), ("wake_confirm_seconds", 2),
                                          ("display_hold_seconds", 3), ("ear_smoothing_seconds", -1)])
def test_invalid_sleep_tuning_rejected(field, value):
    cfg = clean_config()
    cfg["sleep_logic"][field] = value
    with pytest.raises(ValueError):
        validate_config(cfg)


@pytest.fixture
def pipeline_fixture():
    # Reuse scripted sensors, without collecting the base TestCase a second time.
    fixture = pipeline_tests.PipelineRegressionTests()
    fixture.setUp()
    try:
        yield fixture
    finally:
        fixture.doCleanups()


def test_actual_overlay_and_frame_log_hold_sleep_without_advancing_alerts(pipeline_fixture):
    f = pipeline_fixture
    f.registry.closed = True
    f.sequence(stop=3)
    with patch("guard_monitoring.pipeline.draw_guard_box_and_status") as draw:
        f.registry.eyes_available = False
        f.tick(3.1)
        before = f.rules.active_duration("sleep", 3.1)
        out = f.tick(3.4)
        assert "Sleeping" in draw.call_args.args[2]
        assert draw.call_args.args[2] == out["frame_log"]["display_statuses"]
        assert "Stationary" in out["frame_log"]["display_statuses"]
        assert out["frame_log"]["sleep_display"]["held"]
        assert out["frame_log"]["rule_conditions"]["sleep"] is None
        assert f.rules.active_duration("sleep", 3.4) == before
        assert "Sleeping" not in f.tick(4.2)["frame_log"]["display_statuses"]


def test_actual_overlay_keeps_sleep_during_short_track_gap(pipeline_fixture):
    f = pipeline_fixture
    f.registry.closed = True
    f.sequence(stop=3)
    f.registry.ids = []
    assert "Sleeping" in f.tick(3.2)["frame_log"]["display_statuses"]
    f.registry.ids = [1]
    assert "Sleeping" in f.tick(3.3)["frame_log"]["display_statuses"]


def test_phone_sitting_and_movement_statuses_still_render(pipeline_fixture):
    f = pipeline_fixture
    f.registry.closed = True
    f.sequence(stop=3)
    f.registry.phone_active = True
    statuses = f.tick(3.1)["frame_log"]["display_statuses"]
    assert "Using Mobile" in statuses and "Stationary" in statuses
    assert "Sleeping" not in statuses
    with patch.object(f.pipeline.posture_analyzer, "analyze", return_value=PostureState(posture="sitting")):
        statuses = f.tick(3.2)["frame_log"]["display_statuses"]
        assert "Using Mobile" in statuses and "Sitting" in statuses
    for i in range(33, 60):
        f.registry.dx = (i - 32) * 2
        statuses = f.tick(i / 10)["frame_log"]["display_statuses"]
    assert "Using Mobile" in statuses and "Moving" in statuses


def test_posture_only_overlay_is_qualified_and_does_not_trigger_alert(pipeline_fixture):
    f = pipeline_fixture
    f.registry.eyes_available = False
    posture = PostureState(posture="sitting", head_down=True, head_down_known=True)
    with patch.object(f.pipeline.posture_analyzer, "analyze", return_value=posture):
        out = f.sequence(stop=12)
    assert "Possible Sleep" in out[-1]["frame_log"]["display_statuses"]
    assert "Sitting" in out[-1]["frame_log"]["display_statuses"]
    assert not any(e["rule"] == "sleep" for r in out for e in r["events"])


def test_ten_second_video_preserves_every_decoded_frame(pipeline_fixture, tmp_path):
    cv2 = pytest.importorskip("cv2")
    f = pipeline_fixture
    source = tmp_path / "input.avi"
    codec = cv2.VideoWriter_fourcc(*"MJPG")
    writer = cv2.VideoWriter(str(source), codec, 24, (256, 128))
    assert writer.isOpened()
    for _ in range(240):
        writer.write(f.frame)
    writer.release()
    f.cfg["input"]["source"] = str(source)
    f.cfg["output"].update(directory=str(tmp_path / "output"), annotated_video="annotated.avi")

    def eyes(frame, box, keypoints, timestamp_ms):
        t = timestamp_ms / 1000
        visible = not 4 <= t < 4.3
        return EyeState(available=visible, quality_ok=visible,
                        eyes_closed=t < 8 if visible else None)

    with patch.object(f.registry, "eyes", return_value=ok(SimpleNamespace(analyze=eyes))), \
            patch("guard_monitoring.pipeline.make_writer",
                  side_effect=lambda path, w, h, fps: cv2.VideoWriter(str(path), codec, fps, (w, h))):
        summary = f.pipeline.run()
    rows = [json.loads(line) for line in (tmp_path / "output" / "frames.jsonl").read_text().splitlines()]
    assert summary["frames_processed"] == 240
    assert summary["frame_errors"] == 0
    assert [r["frame"] for r in rows] == list(range(240))
    first = next(r["frame"] for r in rows if "Sleeping" in r["display_statuses"])
    assert first <= 50  # Two seconds at 24 FPS; entry has no added display delay.
    assert all("Sleeping" in r["display_statuses"] for r in rows[first:192])
    assert all("Sleeping" not in r["display_statuses"] for r in rows[216:])
    decoded = 0
    cap = cv2.VideoCapture(summary["output_video"])
    try:
        while cap.read()[0]:
            decoded += 1
    finally:
        cap.release()
    assert decoded == 240
