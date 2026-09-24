"""Deterministic regression tests: no model weights, GPU, or camera required.

These verify the decision/state logic, not trained-model accuracy. Run from the
backend root: python -m unittest discover -s guard_monitoring/tests -p 'test_regression_*.py' -v
"""
from __future__ import annotations

import importlib
import math
import os
import runpy
import sys
import threading
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from guard_monitoring.config import load_config, validate_config
from guard_monitoring.monitoring.activity import ActivityStabilizer
from guard_monitoring.monitoring.guard_selector import GuardSelector
from guard_monitoring.monitoring.movement import MovementMonitor
from guard_monitoring.monitoring.rules import TimedRuleEngine
from guard_monitoring.monitoring.sleep import SleepAnalyzer
from guard_monitoring.model_registry import LazyModelRegistry
from guard_monitoring.types import EyeState, MovementState, PhoneState, PostureState


def clean_config():
    env = {k:v for k,v in os.environ.items() if not k.startswith('GUARD_')}
    with patch.dict(os.environ, env, clear=True):
        return load_config(source='test.mp4', output_dir='unused-test-output')


def rule_config(**changes):
    cfg = dict(warmup_seconds=0, sleep_seconds=5, phone_seconds=3,
               stationary_seconds=10, absence_seconds=2,
               grace_seconds=dict(sleep=1, phone=1, stationary=1, absence=0),
               unknown_reset_seconds=5, max_evidence_gap_seconds=2)
    cfg.update(changes)
    return cfg


def detections(ids=(), boxes=None):
    if boxes is None:
        boxes = [(20.,10.,80.,100.) for _ in ids]
    return SimpleNamespace(tracker_id=np.asarray(ids),
                           xyxy=np.asarray(boxes).reshape(-1,4),
                           confidence=np.full(len(ids), .95))


class RuleRegressionTests(unittest.TestCase):
    def test_full_300_second_sleep_threshold_and_one_alert(self):
        rules = TimedRuleEngine('cam', rule_config(sleep_seconds=300))
        emitted=[]
        for t in range(310):
            event=rules.update('sleep',True,t,1,{})
            if event: emitted.append(event)
        self.assertEqual([e.triggered_at for e in emitted],[300])
        self.assertEqual(emitted[0].episode_started_at,0)

    def test_unknown_false_true_never_counts_unknown_time(self):
        rules=TimedRuleEngine('cam',rule_config(unknown_reset_seconds=20))
        for value,t in [(True,0),(None,1),(False,10)]:
            self.assertIsNone(rules.update('sleep',value,t,1,{}))
        self.assertIsNone(rules.update('sleep',True,10.5,1,{}))
        self.assertAlmostEqual(rules.active_duration('sleep',10.5),1)

    def test_false_grace_preserves_episode_but_not_duration(self):
        rules=TimedRuleEngine('cam',rule_config(sleep_seconds=2))
        for value,t in [(True,0),(False,1),(True,1.5),(True,2)]:
            self.assertIsNone(rules.update('sleep',value,t,1,{}))
        self.assertIsNotNone(rules.update('sleep',True,2.5,1,{}))

    def test_false_grace_expiry_checked_when_positive_returns(self):
        rules=TimedRuleEngine('cam',rule_config(sleep_seconds=2))
        rules.update('sleep',True,0,1,{})
        rules.update('sleep',False,1,1,{})
        self.assertIsNone(rules.update('sleep',True,4,1,{}))
        self.assertEqual(rules.active_duration('sleep',4),0)

    def test_different_identity_starts_own_timer(self):
        rules=TimedRuleEngine('cam',rule_config(sleep_seconds=2))
        rules.update('sleep',True,0,1,{})
        rules.update('sleep',True,1,1,{})
        self.assertIsNone(rules.update('sleep',True,2,2,{}))
        self.assertEqual(rules.active_duration('sleep',2),0)

    def test_unknown_gap_is_paused_and_bounded(self):
        rules=TimedRuleEngine('cam',rule_config(sleep_seconds=2))
        rules.update('sleep',True,0,1,{})
        rules.update('sleep',None,1,1,{})
        self.assertIsNone(rules.update('sleep',True,3,1,{}))
        self.assertIsNotNone(rules.update('sleep',True,4,1,{}))
        rules.update('sleep',None,4.1,1,{})
        rules.update('sleep',None,10,1,{})
        self.assertFalse(rules.is_triggered('sleep'))

    def test_no_frame_over_processing_stall_counts_as_evidence(self):
        rules=TimedRuleEngine('cam',rule_config(sleep_seconds=2))
        rules.update('sleep',True,0,1,{})
        self.assertIsNone(rules.update('sleep',True,4,1,{}))
        self.assertEqual(rules.active_duration('sleep',4),0)

    def test_short_gap_after_alert_does_not_duplicate(self):
        rules=TimedRuleEngine('cam',rule_config(phone_seconds=1))
        rules.update('phone',True,0,1,{})
        self.assertIsNotNone(rules.update('phone',True,1,1,{}))
        rules.update('phone',None,1.2,1,{})
        self.assertIsNone(rules.update('phone',True,1.5,1,{}))

    def test_absence_is_not_transferred_or_reset_with_guard_timers(self):
        rules=TimedRuleEngine('cam',rule_config())
        rules.update('absence',True,0,1,{})
        rules.reset_guard(2)
        self.assertIsNotNone(rules.update('absence',True,2,2,{}))

    def test_pause_all_excludes_failed_frames(self):
        rules=TimedRuleEngine('cam',rule_config(phone_seconds=2))
        rules.update('phone',True,0,1,{})
        rules.pause_all(1)
        self.assertIsNone(rules.update('phone',True,2,1,{}))
        self.assertIsNotNone(rules.update('phone',True,3,1,{}))

    def test_nonfinite_or_backwards_timestamps_rejected(self):
        rules=TimedRuleEngine('cam',rule_config())
        with self.assertRaises(ValueError): rules.update('sleep',True,float('nan'),1,{})
        rules.update('sleep',True,2,1,{})
        with self.assertRaises(ValueError): rules.update('sleep',True,1,1,{})

    def test_original_rule_tests_still_pass(self):
        original=runpy.run_path(str(Path(__file__).with_name('test_guard_rules.py')))
        original['test_phone_rule_triggers_once']()
        original['test_unknown_time_pauses_positive_timer']()


class SelectionRegressionTests(unittest.TestCase):
    def selector(self, **kwargs):
        return GuardSelector([(0,0),(200,0),(200,200),(0,200)],confirm_seconds=0,**kwargs)

    def test_short_occlusion_retains_identity_without_fabricating_detection(self):
        s=self.selector()
        self.assertTrue(s.update(detections([0]),0).seen_now)
        held=s.update(detections(),.3)
        self.assertEqual(held.track_id,0)
        self.assertFalse(held.seen_now)
        self.assertTrue(s.is_present(.3))
        self.assertEqual(s.update(detections([0]),1).track_id,0)
        self.assertEqual(s.generation,1)

    def test_long_absence_expires_box_and_presence(self):
        s=self.selector(); s.update(detections([1]),0)
        self.assertIsNotNone(s.update(detections(),3))
        self.assertFalse(s.is_present(3))
        self.assertIsNone(s.update(detections(),6))
        self.assertIsNone(s.active_id)

    def test_other_person_cannot_steal_identity_during_grace(self):
        s=self.selector(); s.update(detections([1]),0)
        for t in [.2,.4,.6,1.,2.]:
            observed=s.update(detections([2]),t)
            self.assertEqual(observed.track_id,1)
            self.assertFalse(observed.seen_now)
        self.assertEqual(s.update(detections([2]),6).track_id,2)
        self.assertEqual(s.generation,2)

    def test_multiple_people_do_not_change_visible_selected_guard(self):
        s=self.selector();s.update(detections([1]),0)
        self.assertEqual(s.update(detections([1,2]),.2).track_id,1)

    def test_reused_id_after_long_gap_starts_new_generation(self):
        s=self.selector();s.update(detections([1]),0)
        s.update(detections([1]),10)
        self.assertEqual(s.generation,2)

    def test_isolated_false_detections_do_not_accumulate_dwell(self):
        s=GuardSelector([(0,0),(200,0),(200,200),(0,200)],confirm_seconds=1)
        for t in [0,2,4,6]: self.assertIsNone(s.update(detections([1]),t))

    def test_unconfirmed_tracker_ids_and_invalid_boxes_are_ignored(self):
        s=self.selector()
        self.assertIsNone(s.update(detections([-1]),0))
        self.assertIsNone(s.update(detections([1],[(0,0,float('nan'),30)]),1))

    def test_new_guard_outside_duty_zone_cannot_be_selected(self):
        self.assertIsNone(self.selector().update(detections([1],[(250,10,300,80)]),0))

    def test_guard_touching_bottom_edge_of_duty_zone_is_selected(self):
        self.assertIsNotNone(self.selector().update(detections([1],[(20,10,80,200)]),0))


class MovementRegressionTests(unittest.TestCase):
    def monitor(self):
        return MovementMonitor(3,.035,1,[],smoothing_seconds=.2)

    def test_standing_with_small_box_jitter_is_stationary(self):
        m=self.monitor()
        for i in range(31):
            jitter=0.5 if i%2 else -0.5
            state=m.update(1,(20+jitter,10,80+jitter,100),i*.1)
        self.assertTrue(state.reliable)
        self.assertTrue(state.stationary)

    def test_walking_is_moving(self):
        m=self.monitor()
        for i in range(31): state=m.update(1,(20+i*2,10,80+i*2,100),i*.1)
        self.assertTrue(state.reliable)
        self.assertFalse(state.stationary)

    def test_single_bbox_outlier_does_not_create_long_movement_episode(self):
        m=self.monitor()
        for i in range(31):
            dx=40 if i==25 else 0
            state=m.update(1,(20+dx,10,80+dx,100),i*.1)
        self.assertTrue(state.stationary)

    def test_camera_translation_is_not_guard_walking(self):
        m=self.monitor()
        for i in range(31):
            if i: m.compensate([[1,0,2],[0,1,0]])
            state=m.update(1,(20+i*2,10,80+i*2,100),i*.1)
        self.assertTrue(state.stationary)

    def test_walking_survives_camera_motion_compensation(self):
        m=self.monitor()
        for i in range(31):
            if i: m.compensate([[1,0,2],[0,1,0]])
            state=m.update(1,(20+i*4,10,80+i*4,100),i*.1)
        self.assertFalse(state.stationary)

    def test_long_evidence_gap_requires_fresh_movement_history(self):
        m=self.monitor()
        for i in range(31): m.update(1,(20,10,80,100),i*.1)
        self.assertFalse(m.update(1,(20,10,80,100),10).reliable)


class SleepRegressionTests(unittest.TestCase):
    def run_sequence(self, *, closed=None, head_down=False, phone=None, duration=10,
                     movement=True, available=None, cached=False):
        analyzer=SleepAnalyzer(clean_config()['sleep_logic'])
        states=[]
        for i in range(round(duration*10)+1):
            t=i*.1
            eyes=EyeState(quality_ok=closed is not None,eyes_closed=closed,
                          observed_at=0 if cached else t)
            posture=PostureState(posture='sitting',head_down=head_down,head_down_known=True)
            states.append(analyzer.update(1,t,eyes,posture,
                          MovementState(stationary=movement,reliable=True),
                          phone or PhoneState(),available))
        return states

    def test_sitting_stationary_awake_is_not_sleeping(self):
        self.assertFalse(any(s.candidate for s in self.run_sequence(closed=False)))

    def test_stationary_without_eye_or_head_evidence_is_not_sleeping(self):
        self.assertFalse(any(s.candidate for s in self.run_sequence()))

    def test_closed_eyes_require_sustained_evidence(self):
        states=self.run_sequence(closed=True,duration=4)
        self.assertFalse(any(s.candidate for s in states[:20]))
        self.assertTrue(states[-1].candidate)
        self.assertEqual(states[-1].evidence_quality,'high')

    def test_one_cached_closed_eye_sample_is_not_repeated_evidence(self):
        states=self.run_sequence(closed=True,cached=True)
        self.assertFalse(any(s.candidate for s in states))
        self.assertEqual(states[-1].eye_samples,1)

    def test_short_blink_does_not_trigger_sleep(self):
        analyzer=SleepAnalyzer(clean_config()['sleep_logic'])
        for i in range(61):
            t=i*.1
            state=analyzer.update(1,t,EyeState(quality_ok=True,eyes_closed=10<=i<=12,observed_at=t),
                PostureState(),MovementState(stationary=True,reliable=True),PhoneState())
            self.assertFalse(state.candidate)

    def test_occluded_eyes_need_sustained_head_down_and_posture(self):
        states=self.run_sequence(head_down=True)
        self.assertFalse(any(s.candidate for s in states[:80]))
        self.assertTrue(states[-1].candidate)
        self.assertEqual(states[-1].evidence_quality,'degraded')

    def test_phone_use_suppresses_sleep_even_with_closed_eye_estimate(self):
        states=self.run_sequence(closed=True,head_down=True,phone=PhoneState(detected=True,usage='screen_use'))
        self.assertFalse(any(s.candidate for s in states))

    def test_ambiguous_visible_phone_does_not_produce_sleep_alert(self):
        states=self.run_sequence(head_down=True,phone=PhoneState(detected=True,usage='visible'))
        self.assertFalse(any(s.candidate for s in states))

    def test_missing_phone_module_blocks_posture_only_sleep(self):
        states=self.run_sequence(head_down=True,available=dict(movement=True,posture=True,eyes=False,phone=False))
        self.assertFalse(any(s.candidate for s in states))

    def test_moving_guard_is_not_sleeping(self):
        self.assertFalse(any(s.candidate for s in self.run_sequence(closed=True,movement=False)))

    def test_unknown_movement_with_sustained_eyes_and_posture_can_qualify(self):
        states=self.run_sequence(closed=True,available=dict(movement=False,posture=True,eyes=True,phone=True))
        self.assertFalse(any(s.candidate for s in states[:20]))
        self.assertTrue(states[-1].candidate)
        self.assertEqual(states[-1].reason,'eye_evidence_motion_unknown')


class ActivityRegressionTests(unittest.TestCase):
    def test_short_label_flips_are_debounced_and_real_changes_are_prompt(self):
        s=ActivityStabilizer({})
        s.update('Stationary',0,present=True)
        self.assertEqual(s.update('Stationary',.4,present=True)['label'],'Stationary')
        self.assertEqual(s.update('Moving',.5,present=True)['label'],'Stationary')
        self.assertEqual(s.update('Stationary',.6,present=True)['label'],'Stationary')
        s.update('Using Mobile',.7,present=True)
        self.assertEqual(s.update('Using Mobile',1.1,present=True)['label'],'Using Mobile')

    def test_unknown_evidence_has_bounded_hold(self):
        s=ActivityStabilizer({'confirm_seconds':0})
        s.update('Moving',0,present=True)
        self.assertEqual(s.update(None,.3,present=None)['label'],'Moving')
        self.assertIsNone(s.update(None,1.1,present=None)['label'])

    def test_confirmed_absence_clears_status_immediately(self):
        s=ActivityStabilizer({'confirm_seconds':0})
        s.update('Stationary',0,present=True)
        self.assertIsNone(s.update(None,.1,present=False)['label'])

    def test_alternating_conflicting_candidates_cannot_keep_old_status_forever(self):
        s=ActivityStabilizer({'sleep_confirm_seconds':0})
        s.update('Sleeping',0,present=True)
        for i in range(1,8):
            result=s.update('Moving' if i%2 else 'Using Mobile',i*.2,present=True)
        self.assertIsNone(result['label'])

    def test_invalid_labels_rejected(self):
        with self.assertRaises(ValueError): ActivityStabilizer({}).update('unknown',0,present=True)


class ConfigurationAndResourceTests(unittest.TestCase):
    def test_nan_thresholds_rejected(self):
        cfg=clean_config();cfg['rules']['sleep_seconds']=float('nan')
        with self.assertRaises(ValueError): validate_config(cfg)

    def test_unsafe_grace_relationship_rejected(self):
        cfg=clean_config();cfg['guard_selection']['presence_grace_seconds']=10
        with self.assertRaises(ValueError): validate_config(cfg)

    def test_confidence_defaults_unchanged(self):
        cfg=clean_config()
        self.assertEqual(cfg['models']['guard_detector']['confidence'],.25)
        self.assertEqual(cfg['tracker']['track_activation_threshold'],.45)
        self.assertTrue(cfg['models']['guard_detector']['require_finetuned'])
        self.assertFalse(cfg['sleep_logic']['allow_degraded_rule_trigger'])
        self.assertTrue(cfg['camera_motion']['enabled'])

    def test_production_and_explicit_test_durations_remain_separate(self):
        cfg=clean_config();self.assertEqual(cfg['rules']['sleep_seconds'],30)
        with patch.dict(os.environ,{'GUARD_TEST_MODE':'true','GUARD_TEST_SLEEP_SECONDS':'4'}):
            self.assertEqual(load_config()['rules']['sleep_seconds'],4)

    def test_original_persistent_storage_test(self):
        original=runpy.run_path(str(Path(__file__).with_name('test_guard_storage.py')))
        with TemporaryDirectory() as tmp:
            original['test_sqlite_job_store_persists'](Path(tmp))

    def test_shared_models_are_loaded_once_and_inference_is_serialized(self):
        counter={'loads':0,'active':0,'peak':0}
        class Model:
            def predict(self):
                counter['active']+=1
                counter['peak']=max(counter['peak'],counter['active'])
                time.sleep(.002)
                counter['active']-=1
                return 1
        def build():
            counter['loads']+=1;time.sleep(.005);return Model()
        key=('regression',id(counter))
        registries=[LazyModelRegistry({}) for _ in range(8)]
        with ThreadPoolExecutor(8) as pool:
            results=list(pool.map(lambda r:r._get('test_model',build,cache_key=key),registries))
            self.assertEqual(list(pool.map(lambda r:r.value.predict(),results)),[1]*8)
        self.assertEqual(counter['loads'],1)
        self.assertEqual(counter['peak'],1)

    def test_live_initialization_errors_are_reported_as_failed(self):
        from guard_monitoring.live_service import GuardLiveService,_LiveSession
        svc=GuardLiveService()
        session=_LiveSession('id','cam','0',3,2,80)
        with patch.object(svc,'_run_impl',side_effect=ImportError('missing CV package')):
            svc._run(session)
        self.assertEqual(session.status,'failed')
        self.assertIn('ImportError',session.error)
        self.assertIsNotNone(session.ended_monotonic)

    def test_live_nan_duration_rejected(self):
        from guard_monitoring.live_service import _validate_rule_settings
        with self.assertRaises(ValueError): _validate_rule_settings({'sleep':float('nan')})


if __name__ == '__main__':
    unittest.main()
