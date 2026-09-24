"""Pipeline wiring regression tests with scripted models.

When OpenCV is unavailable, only its import/render boundary is mocked. Real
OpenCV rendering/camera-estimation checks are explicitly skipped in that case.
No test in this file claims to validate trained-model detection accuracy.
"""
from __future__ import annotations

import importlib.util
import os
import sys
import types
import unittest
from unittest.mock import MagicMock, patch

import numpy as np

HAS_CV2 = importlib.util.find_spec('cv2') is not None
if HAS_CV2:
    from guard_monitoring.pipeline import GuardMonitoringPipeline
else:
    sys.modules['cv2'] = MagicMock()
    try:
        from guard_monitoring.pipeline import GuardMonitoringPipeline
    finally:
        sys.modules.pop('cv2', None)

from guard_monitoring.config import load_config
from guard_monitoring.contracts import ok, error_result
from guard_monitoring.geometry import normalized_polygon_to_pixels
from guard_monitoring.live_service import GuardLiveService, _LiveSession
from guard_monitoring.monitoring.guard_selector import GuardSelector
from guard_monitoring.monitoring.movement import MovementMonitor
from guard_monitoring.monitoring.rules import TimedRuleEngine
from guard_monitoring.types import EyeState, PoseObservation, PostureState


class ScriptedRegistry:
    def __init__(self):
        self.ids=[1]
        self.dx=0
        self.closed=False
        self.eyes_available=True
        self.phone_active=False
        self.pose_failure=False
        self.detection_failure=False
        self.eye_timestamps=[]
        self.pose_calls=0
        self.reset_calls=0
        self.close_calls=0

    def guard_detector(self):
        def predict(frame):
            if self.detection_failure: raise RuntimeError('scripted detector failure')
            return types.SimpleNamespace(
                xyxy=np.asarray([(20+self.dx,10,80+self.dx,100) for _ in self.ids]).reshape(-1,4),
                confidence=np.full(len(self.ids),.95),tracker_id=np.asarray(self.ids))
        return ok(types.SimpleNamespace(predict=predict))

    def tracker(self, frame_rate=30):
        return ok(types.SimpleNamespace(update=lambda detections,timestamp:detections))

    def pose(self):
        def predict(frame,box):
            self.pose_calls+=1
            if self.pose_failure: raise RuntimeError('scripted pose failure')
            k=np.zeros((17,3));k[:,2]=1
            k[0,:2]=(50+self.dx,20)
            k[3,:2]=(44+self.dx,20);k[4,:2]=(56+self.dx,20)
            for ids,y in [((5,6),40),((11,12),65),((13,14),80),((15,16),95)]:
                k[ids[0],:2]=(40+self.dx,y);k[ids[1],:2]=(60+self.dx,y)
            k[9,:2]=(48+self.dx,55);k[10,:2]=(52+self.dx,55)
            return PoseObservation(k,tuple(box),.95)
        return ok(types.SimpleNamespace(predict=predict))

    def phone(self):
        def predict(frame,box):
            return ([dict(bbox=(45+self.dx,51,51+self.dx,57),confidence=.95)]
                    if self.phone_active else [])
        return ok(types.SimpleNamespace(predict=predict))

    def eyes(self):
        def analyze(frame,box,keypoints,timestamp):
            self.eye_timestamps.append(timestamp)
            return EyeState(available=self.eyes_available,quality_ok=self.eyes_available,
                            eyes_closed=self.closed if self.eyes_available else None,reason='scripted')
        return ok(types.SimpleNamespace(analyze=analyze))

    def reset_tracker(self): self.reset_calls+=1
    def close(self): self.close_calls+=1


class PipelineRegressionTests(unittest.TestCase):
    def setUp(self):
        env={k:v for k,v in os.environ.items() if not k.startswith('GUARD_')}
        with patch.dict(os.environ,env,clear=True): self.cfg=load_config()
        self.cfg['camera_motion']['enabled']=True
        self.cfg['guard_selection']['confirm_seconds']=0
        self.cfg['movement'].update(minimum_history_seconds=.5,history_seconds=1.5)
        self.cfg['rules'].update(warmup_seconds=0,sleep_seconds=2,phone_seconds=2,
                                 stationary_seconds=20,absence_seconds=1)
        for name in ('pose','phone','eyes'):self.cfg['models'][name]['every_n_frames']=1
        self.registry=ScriptedRegistry()
        self.pipeline=GuardMonitoringPipeline(self.cfg,registry=self.registry)
        # Scripted pixel frames have no real background. Supply known camera
        # evidence here; explicit unknown/cut tests override this fixture.
        self.camera_patch=patch.object(self.pipeline.camera_motion,'update',return_value=dict(
            enabled=True,valid=True,scene_change=False,reason='scripted_background',
            affine=[[1,0,0],[0,1,0]]))
        self.camera_mock=self.camera_patch.start();self.addCleanup(self.camera_patch.stop)
        self.selector=GuardSelector([(0,0),(1000,0),(1000,1000),(0,1000)],confirm_seconds=0)
        m=self.cfg['movement']
        self.movement=MovementMonitor(m['history_seconds'],m['stationary_radius_ratio'],
                                     m['minimum_history_seconds'],[])
        self.rules=TimedRuleEngine('cam',self.cfg['rules'])
        self.frame=np.zeros((128,256,3),dtype=np.uint8)
        self.draw=patch('guard_monitoring.pipeline.draw_activity_status')
        self.draw_mock=self.draw.start()
        self.addCleanup(self.draw.stop)
        self.identity_draw=patch('guard_monitoring.pipeline.draw_guard_identity')
        self.identity_draw_mock=self.identity_draw.start();self.addCleanup(self.identity_draw.stop)
        # Expected failures are assertions, not useful stack traces in test output.
        self.logs=patch('guard_monitoring.health.logger.exception')
        self.logs.start();self.addCleanup(self.logs.stop)
        self.index=0

    def tick(self,t):
        out=self.pipeline._process_frame(frame=self.frame,frame_idx=self.index,now=t,
            selector=self.selector,movement_monitor=self.movement,rules=self.rules,
            duty_zone=self.selector.zone,patrol_zones=[])
        self.index+=1
        return out

    def sequence(self,start=0,stop=6):
        return [self.tick(i*.1) for i in range(round(start*10),round(stop*10)+1)]

    def test_normal_standing_is_stationary_without_sleep_event(self):
        out=self.sequence()
        self.assertEqual(out[-1]['frame_log']['activity_status'],'Stationary')
        self.assertFalse(any(e['rule']=='sleep' for r in out for e in r['events']))

    def test_walking_is_moving_without_sleep_event(self):
        out=[]
        for i in range(61):
            self.registry.dx=i*2;out.append(self.tick(i*.1))
        self.assertEqual(out[-1]['frame_log']['activity_status'],'Moving')
        self.assertFalse(any(e['rule']=='sleep' for r in out for e in r['events']))

    def test_sleeping_emits_once_after_positive_evidence_threshold(self):
        self.registry.closed=True
        out=self.sequence()
        events=[e for r in out for e in r['events'] if e['rule']=='sleep']
        self.assertEqual(len(events),1)
        self.assertGreaterEqual(events[0]['triggered_at'],4)
        self.assertEqual(out[-1]['frame_log']['activity_status'],'Sleeping')

    def test_phone_use_preserves_phone_event_and_suppresses_sleep(self):
        self.registry.phone_active=True;self.registry.closed=True
        out=self.sequence()
        self.assertEqual(out[-1]['frame_log']['activity_status'],'Using Mobile')
        self.assertEqual(len([e for r in out for e in r['events'] if e['rule']=='phone']),1)
        self.assertFalse(any(e['rule']=='sleep' for r in out for e in r['events']))

    def test_posture_only_sleep_alert_requires_explicit_opt_in(self):
        self.registry.eyes_available=False
        posture=PostureState(posture='sitting',head_down=True,head_down_known=True)
        with patch.object(self.pipeline.posture_analyzer,'analyze',return_value=posture):
            before=self.sequence(stop=12)
            self.assertFalse(any(e['rule']=='sleep' for r in before for e in r['events']))
            self.assertNotEqual(before[-1]['frame_log']['activity_status'],'Sleeping')
            self.cfg['sleep_logic']['allow_degraded_rule_trigger']=True
            after=self.sequence(start=12.1,stop=16)
        self.assertEqual(len([e for r in after for e in r['events'] if e['rule']=='sleep']),1)
        self.assertEqual(after[-1]['frame_log']['activity_status'],'Sleeping')

    def test_short_obstruction_retains_identity_but_pauses_rules_and_crops(self):
        self.registry.phone_active=True
        self.sequence(stop=1)
        calls=self.registry.pose_calls
        self.registry.ids=[]
        gap=self.tick(1.1)['frame_log']
        self.assertEqual(gap['guard_track_id'],1)
        self.assertFalse(gap['guard_seen_now'])
        self.assertIsNone(gap['rule_conditions']['phone'])
        self.assertEqual(self.registry.pose_calls,calls)
        self.registry.ids=[1]
        out=self.tick(1.4)
        self.assertEqual(out['frame_log']['guard_track_id'],1)
        self.assertEqual(self.selector.generation,1)
        self.assertFalse(out['events'])

    def test_real_absence_removes_status_and_starts_absence_event(self):
        self.sequence(stop=1);self.registry.ids=[]
        out=self.sequence(start=1.1,stop=7)
        self.assertIsNone(out[-1]['frame_log']['activity_status'])
        self.assertEqual(len([e for r in out for e in r['events'] if e['rule']=='absence']),1)

    def test_low_quality_no_face_or_pose_does_not_fabricate_sleep(self):
        self.registry.eyes_available=False;self.registry.pose_failure=True
        out=self.sequence()
        self.assertFalse(any(e['rule']=='sleep' for r in out for e in r['events']))
        self.assertEqual(out[-1]['frame_log']['module_status']['pose']['status'],'error')
        self.assertEqual(out[-1]['frame_log']['module_status']['phone_detection']['status'],'ok')

    def test_detection_module_failure_is_unknown_not_absent(self):
        self.sequence(stop=1);self.registry.detection_failure=True
        out=self.sequence(start=1.1,stop=7)
        self.assertTrue(all(r['frame_log']['present'] is None for r in out))
        self.assertFalse(any(e['rule']=='absence' for r in out for e in r['events']))
        self.assertIsNone(out[-1]['frame_log']['activity_status'])

    def test_multiple_people_never_transfer_phone_timer_to_replacement(self):
        self.registry.phone_active=True;self.sequence(stop=1)
        self.registry.ids=[1,2];self.tick(1.1)
        self.assertEqual(self.selector.active_id,1)
        self.registry.ids=[2]
        out=self.tick(7)
        self.assertEqual(out['frame_log']['guard_track_id'],2)
        self.assertEqual(self.rules.active_duration('phone',7),0)
        self.assertFalse(out['events'])

    def test_visualization_has_no_effect_on_event_logic_or_input_pixels(self):
        before=self.frame.copy();self.sequence(stop=2)
        self.assertTrue(np.array_equal(before,self.frame))
        for call in self.draw_mock.call_args_list:
            self.assertIn(call.args[1],(None,'Sleeping','Moving','Using Mobile','Stationary'))
        self.assertEqual(self.rules.is_triggered('sleep'),False)

    def test_rendering_failure_cannot_drop_a_real_phone_event(self):
        self.registry.phone_active=True
        self.draw_mock.side_effect=RuntimeError('renderer unavailable')
        out=self.sequence(stop=3)
        self.assertEqual(len([e for r in out for e in r['events'] if e['rule']=='phone']),1)
        self.assertEqual(out[-1]['frame_log']['module_status']['visualization']['status'],'error')

    def test_eye_timestamps_follow_elapsed_time_and_are_strictly_increasing(self):
        self.tick(0);self.tick(.25);self.tick(1.7);self.tick(1.7001)
        self.assertEqual(self.registry.eye_timestamps[:3],[0,250,1700])
        self.assertGreater(self.registry.eye_timestamps[3],1700)

    def test_stride_cache_never_reuses_expired_data(self):
        self.pipeline.pose_stride=100
        self.tick(0);count=self.registry.pose_calls
        self.tick(.2);self.assertEqual(self.registry.pose_calls,count)
        self.tick(1);self.assertEqual(self.registry.pose_calls,count+1)

    def test_reconnect_resets_tracking_perception_and_activity(self):
        self.sequence(stop=2)
        self.pipeline.reset_stream()
        self.assertIsNone(self.pipeline.last_identity)
        self.assertIsNone(self.pipeline.activity.label)
        self.assertEqual(self.registry.reset_calls,1)
        self.assertEqual(self.registry.close_calls,1)

    def test_live_runtime_has_same_selector_and_rules_as_uploads(self):
        runtime=GuardLiveService._build_runtime(self.pipeline,self.cfg,256,128,3,
            GuardSelector,MovementMonitor,TimedRuleEngine,normalized_polygon_to_pixels)
        self.assertEqual(runtime['rules'].thresholds,self.rules.thresholds)
        self.assertEqual(runtime['selector'].presence_grace_seconds,2)
        self.assertGreaterEqual(runtime['selector'].candidate_gap_seconds,.5)

    def test_live_preview_encoding_failure_does_not_lose_phone_alert(self):
        self.registry.phone_active=True
        svc=GuardLiveService()
        session=_LiveSession('test','cam','0',3,2,80,settings={'phone':0})
        session.latest_jpeg=b'old-preview'
        capture=MagicMock()
        def read_once():
            session.stop_event.set()
            return True,self.frame
        capture.read.side_effect=read_once
        cv=MagicMock()
        cv.imencode.side_effect=RuntimeError('encoder failure')
        runtime=dict(selector=self.selector,movement_monitor=self.movement,
                     rules=self.rules,duty_zone=self.selector.zone,patrol_zones=[])
        with patch.dict(sys.modules,{'cv2':cv}), \
             patch('guard_monitoring.io.open_capture',return_value=capture), \
             patch('guard_monitoring.io.video_metadata',return_value=(256,128,30)), \
             patch('guard_monitoring.pipeline.GuardMonitoringPipeline',return_value=self.pipeline), \
             patch.object(svc,'_build_runtime',return_value=runtime):
            svc._run_impl(session)
        self.assertEqual([e['rule'] for e in session.events],['phone'])
        self.assertEqual(session.frames_processed,1)
        self.assertIsNone(session.latest_jpeg)


@unittest.skipUnless(HAS_CV2, 'Real OpenCV unavailable: pixel rendering and optical flow not executed')
class RealOpenCVRegressionTests(unittest.TestCase):
    def test_overlay_draws_only_the_requested_text(self):
        import cv2
        from guard_monitoring.visualization.overlay import draw_activity_status
        original=cv2.putText
        for label in ['Sleeping','Moving','Using Mobile','Stationary']:
            with patch.object(cv2,'putText',wraps=original) as draw:
                frame=np.zeros((240,320,3),dtype=np.uint8)
                draw_activity_status(frame,label)
                self.assertEqual([c.args[1] for c in draw.call_args_list],[f'GUARD STATUS: {label}'])
                self.assertTrue(frame.any())

    def test_no_status_is_drawn_without_evidence(self):
        from guard_monitoring.visualization.overlay import draw_activity_status
        frame=np.full((96,96,3),128,dtype=np.uint8)
        original=frame.copy();draw_activity_status(frame,None)
        self.assertTrue(np.array_equal(frame,original))

    def test_background_translation_is_recovered(self):
        import cv2
        from guard_monitoring.monitoring.camera_motion import CameraMotionEstimator
        random=np.random.default_rng(27)
        base=random.integers(0,255,size=(240,320,3),dtype=np.uint8)
        moved=cv2.warpAffine(base,np.float32([[1,0,3],[0,1,2]]),(320,240))
        estimator=CameraMotionEstimator({'enabled':True})
        self.assertFalse(estimator.update(base,[])['valid'])
        state=estimator.update(moved,[])
        self.assertTrue(state['valid'])
        np.testing.assert_allclose(np.asarray(state['affine'])[:,2],[3,2],atol=.6)


if __name__=='__main__': unittest.main()
