"""Correction tests: scripted models and recorded sensors, not model accuracy."""
from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np

from guard_monitoring.config import validate_config
from guard_monitoring.diagnostics import BUILD_ID, effective_config
from guard_monitoring.health import ModuleHealthRegistry, safe_run
from guard_monitoring.monitoring.camera_motion import CameraMotionEstimator
from guard_monitoring.monitoring.movement import MovementMonitor
from guard_monitoring.monitoring.sleep import SleepAnalyzer
from guard_monitoring.monitoring.activity import ActivityStabilizer
from guard_monitoring.types import EyeState, MovementState, PhoneState, PostureState, GuardTrack
from guard_monitoring.tests import test_regression_pipeline as wiring
from guard_monitoring.tests.test_regression_stability import clean_config


class IndependentEyeTests(unittest.TestCase):
    def sequence(self, *, closed=True, phone=None, available=None, roll=None,
                 movement=None, cfg=None, duration=6):
        analyzer=SleepAnalyzer(cfg or clean_config()['sleep_logic'])
        return [analyzer.update(0,i*.1,
            EyeState(available=closed is not None,quality_ok=closed is not None,
                     eyes_closed=closed,observed_at=i*.1,head_roll_degrees=roll),
            PostureState(),movement or MovementState(),phone or PhoneState(),available)
            for i in range(round(duration*10)+1)]

    def test_unknown_motion_needs_stronger_eyes(self):
        states=self.sequence()
        self.assertFalse(any(s.candidate for s in states[:40]))
        self.assertTrue(states[40].candidate)
        self.assertEqual(states[40].decision_basis,'sustained_eyes')
        self.assertIn('movement',states[40].missing_modules)

    def test_awake_stationary_never_means_sleep(self):
        self.assertFalse(any(s.candidate for s in self.sequence(closed=False)))

    def test_missing_face_cannot_supply_independent_sleep(self):
        self.assertFalse(any(s.candidate for s in self.sequence(closed=None,roll=35)))

    def test_reliable_walking_still_vetoes_closed_eyes(self):
        self.assertFalse(any(s.candidate for s in self.sequence(movement=MovementState(reliable=True,stationary=False))))

    def test_phone_active_still_vetoes_closed_eyes(self):
        for usage in ('call','screen_use','visible'):
            self.assertFalse(any(s.candidate for s in self.sequence(phone=PhoneState(detected=True,usage=usage))))

    def test_missing_phone_blocks_independent_eye_route(self):
        states=self.sequence(available=dict(movement=True,eyes=True,posture=True,phone=False))
        self.assertFalse(any(s.candidate for s in states))

    def test_head_tilt_supports_sustained_eyes_not_instant_sleep(self):
        states=self.sequence(roll=30)
        self.assertFalse(any(s.candidate for s in states[:20]))
        self.assertTrue(states[20].candidate)
        self.assertEqual(states[20].decision_basis,'eyes_and_head_posture')
        self.assertFalse(any(s.candidate for s in self.sequence(closed=False,roll=30)))

    def test_eye_only_can_be_disabled(self):
        cfg=clean_config()['sleep_logic'];cfg['allow_eye_only_sleep']=False
        self.assertFalse(any(s.candidate for s in self.sequence(cfg=cfg)))

    def test_stale_eyes_cannot_keep_sleep_alive(self):
        a=SleepAnalyzer(clean_config()['sleep_logic'])
        for i in range(51):
            t=i*.1
            state=a.update(0,t,EyeState(quality_ok=True,eyes_closed=True,observed_at=0),
                           PostureState(),MovementState(),PhoneState())
            self.assertFalse(state.candidate)

    def test_recorded_sensor_fixture_produces_evidence_based_status(self):
        rows=json.loads((Path(__file__).parent/'fixtures/sleep_eye_evidence.json').read_text())['frames']
        analyzer=SleepAnalyzer(clean_config()['sleep_logic']);activity=ActivityStabilizer({})
        candidates=[];display=[]
        for row in rows:
            t=row['time_seconds']
            state=analyzer.update(row['guard_track_id'],t,EyeState(**row['eyes']),
                                 PostureState(**row['posture']),MovementState(),PhoneState(**row['phone']))
            candidates.append(state)
            display.append(activity.update('Sleeping' if state.candidate else None,t,present=True)['label'])
        self.assertEqual(len(rows),215)
        self.assertTrue(any(s.candidate for s in candidates))
        self.assertIn('Sleeping',display)
        self.assertNotIn('Moving',display)
        self.assertFalse(candidates[0].candidate)


class CameraQualityTests(unittest.TestCase):
    def setUp(self):
        self.estimator=CameraMotionEstimator({})
        self.points=np.array([(x,y) for x in (10,80,150,230,300) for y in (10,90,160,220)],dtype=float)
        self.matrix=np.array([[1.,0.,3.],[0.,1.,2.]])
        self.inliers=np.ones((len(self.points),1),dtype=np.uint8)

    def fit(self, difference=.02):
        return self.estimator.assess_fit(self.matrix,self.inliers,self.points,(240,320),difference)

    def test_supported_background_translation_accepted(self):
        state=self.fit();self.assertTrue(state['valid']);self.assertFalse(state['scene_change'])

    def test_low_consensus_is_unknown(self):
        self.inliers[:10]=0
        state=self.fit();self.assertFalse(state['valid']);self.assertFalse(state['scene_change'])

    def test_failed_consensus_and_large_difference_signal_cut(self):
        self.inliers[:10]=0
        self.assertTrue(self.fit(.4)['scene_change'])

    def test_zoom_discontinuity_invalidates_identity(self):
        self.matrix[:,:2]*=1.4
        state=self.fit();self.assertFalse(state['valid']);self.assertTrue(state['scene_change'])

    def test_localized_features_are_not_global_camera_evidence(self):
        self.points*=.01
        self.assertEqual(self.fit()['reason'],'localized_background')

    def test_disabled_is_explicitly_not_a_valid_camera_estimate(self):
        state=CameraMotionEstimator({'enabled':False}).update(None,[])
        self.assertFalse(state['enabled']);self.assertFalse(state['valid'])
        self.assertEqual(state['reason'],'disabled')


class MovementQualityTests(unittest.TestCase):
    def setUp(self):
        self.m=MovementMonitor(3,.035,1,[])
        self.camera=dict(valid=True,scene_change=False)

    def sample(self, t, box=(100,80,200,300), camera=None, static=False):
        return self.m.update(0,box,t,frame_shape=(480,640,3),
                             camera_state=self.camera if camera is None else camera,
                             assume_static_camera=static)

    def test_cropped_box_is_not_a_footpoint(self):
        for box in [(0,80,200,300),(100,0,200,300),(100,80,640,300),(100,80,200,480)]:
            state=self.sample(0,box)
            self.assertFalse(state.reliable);self.assertEqual(state.reason,'guard_box_truncated')

    def test_unknown_camera_never_becomes_moving(self):
        for i in range(31): state=self.sample(i*.1,camera=dict(valid=False))
        self.assertFalse(state.reliable);self.assertEqual(state.reason,'camera_motion_unresolved')

    def test_explicit_fixed_camera_allows_history(self):
        for i in range(31): state=self.sample(i*.1,camera=dict(valid=False),static=True)
        self.assertTrue(state.reliable);self.assertTrue(state.stationary)
        self.assertEqual(state.motion_source,'configured_static_camera')

    def test_compensated_camera_pan_keeps_stationary_guard(self):
        for i in range(21):
            if i:self.m.compensate([[1,0,3],[0,1,0]])
            state=self.sample(i*.1,(100+3*i,80,200+3*i,300))
        self.assertTrue(state.reliable);self.assertTrue(state.stationary)

    def test_walking_relative_to_background_is_moving(self):
        for i in range(21):
            if i:self.m.compensate([[1,0,2],[0,1,0]])
            state=self.sample(i*.1,(100+7*i,80,200+7*i,300))
        self.assertTrue(state.reliable);self.assertFalse(state.stationary)

    def test_box_scale_jump_invalidates_old_motion_history(self):
        for i in range(21):self.sample(i*.1)
        state=self.sample(2.1,(50,50,300,400))
        self.assertFalse(state.reliable);self.assertEqual(state.reason,'box_scale_discontinuity')
        self.assertFalse(self.m.history)


class PipelineCorrectionTests(unittest.TestCase):
    setUp=wiring.PipelineRegressionTests.setUp
    tick=wiring.PipelineRegressionTests.tick
    sequence=wiring.PipelineRegressionTests.sequence

    def unknown_camera(self):
        self.camera_mock.return_value=self.pipeline.camera_motion.state('background_occluded')

    def test_unknown_motion_does_not_default_to_moving(self):
        self.unknown_camera();out=self.sequence()
        self.assertTrue(all(r['frame_log']['rule_conditions']['stationary'] is None for r in out))
        self.assertTrue(all(r['frame_log']['activity_status'] is None for r in out))
        self.assertEqual(out[-1]['frame_log']['module_status']['camera_motion']['status'],'unknown')

    def test_independent_eyes_reach_status_and_event_once(self):
        self.unknown_camera();self.registry.closed=True
        out=self.sequence(stop=9)
        self.assertEqual(out[-1]['frame_log']['activity_status'],'Sleeping')
        events=[e for r in out for e in r['events'] if e['rule']=='sleep']
        self.assertEqual(len(events),1);self.assertGreaterEqual(events[0]['triggered_at'],6)

    def test_camera_cut_resets_person_timers_without_absence_alert(self):
        self.registry.phone_active=True;self.sequence(stop=1)
        self.assertGreater(self.rules.active_duration('phone',1),0)
        self.camera_mock.return_value=self.pipeline.camera_motion.state('scene_change',scene_change=True)
        out=self.tick(1.1)
        self.assertEqual(self.registry.reset_calls,1)
        self.assertIsNone(out['frame_log']['present'])
        self.assertIsNone(out['frame_log']['rule_conditions']['absence'])
        self.assertEqual(self.rules.active_duration('phone',1.1),0)
        self.assertFalse(out['events'])

    def test_zero_guard_id_box_and_build_are_logged_and_drawn(self):
        self.registry.ids=[0];out=self.tick(0)
        self.assertEqual(out['frame_log']['guard_track_id'],0)
        self.assertEqual(out['frame_log']['guard_bbox'],[20,10,80,100])
        self.assertEqual(out['frame_log']['build_id'],BUILD_ID)
        self.assertEqual(self.identity_draw_mock.call_args.args[1].track_id,0)
        self.assertIn('guard_detection',out['frame_log']['timings_ms'])
        self.assertEqual(len(out['frame_log']['pose_keypoints']),17)

    def test_retained_identity_overlay_expires_with_presence(self):
        self.tick(0);self.registry.ids=[];self.tick(.2)
        self.assertFalse(self.identity_draw_mock.call_args.args[1].seen_now)
        self.identity_draw_mock.reset_mock();self.tick(2.5)
        self.identity_draw_mock.assert_not_called()


class DiagnosticTests(unittest.TestCase):
    def test_timing_counters_cover_success_and_failure(self):
        h=ModuleHealthRegistry();safe_run('ok',lambda:42,health=h)
        with patch('guard_monitoring.health.logger.exception'):
            safe_run('bad',lambda:1/0,health=h)
        for name in ('ok','bad'):
            self.assertEqual(h.snapshot()[name]['timed_calls'],1)
            self.assertGreaterEqual(h.snapshot()[name]['total_ms'],0)

    def test_effective_config_omits_input_url(self):
        cfg=clean_config();cfg['input']['source']='rtsp://user:password@example/camera'
        self.assertNotIn('password',json.dumps(effective_config(cfg)))
        self.assertNotIn('input',effective_config(cfg))

    def test_weaker_independent_threshold_rejected(self):
        for key,value in [('eye_only_min_seconds',.1),('eye_only_perclos_threshold',.2),('eye_only_min_coverage',.1)]:
            cfg=clean_config();cfg['sleep_logic'][key]=value
            with self.assertRaises(ValueError):validate_config(cfg)


@unittest.skipUnless(wiring.HAS_CV2,'Real OpenCV unavailable: box/ID pixel tests not executed')
class IdentityPixelTests(unittest.TestCase):
    def test_zero_identity_is_drawn_without_confidence(self):
        import cv2
        from guard_monitoring.visualization.overlay import draw_guard_identity
        frame=np.zeros((240,320,3),dtype=np.uint8)
        with patch.object(cv2,'putText',wraps=cv2.putText) as text:
            draw_guard_identity(frame,GuardTrack(0,(40,90,140,210),.93))
            self.assertEqual([c.args[1] for c in text.call_args_list],['GUARD ID: 0'])
        self.assertTrue(frame[100,40].any())

    def test_retained_box_has_last_seen_label(self):
        import cv2
        from guard_monitoring.visualization.overlay import draw_guard_identity
        frame=np.zeros((240,320,3),dtype=np.uint8)
        with patch.object(cv2,'putText',wraps=cv2.putText) as text:
            draw_guard_identity(frame,GuardTrack(2,(40,90,140,210),.93,False))
            self.assertIn('(last seen)',text.call_args.args[1])


if __name__=='__main__':unittest.main()
