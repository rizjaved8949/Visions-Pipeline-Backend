"""Scenario logic tests: synthetic observations, not model-accuracy claims."""
import unittest
from unittest.mock import Mock, patch
import numpy as np
from kitchen_monitoring.pipeline import KitchenPipeline
from kitchen_monitoring.person_state import TemporalState, TrackAdmission, PersonPresence
from kitchen_monitoring.refinement import PPERefiner


def person(track_id=1):
    return {"track_id": track_id, "staff_label": f"STAFF-{track_id:02d}",
            "person_confidence": .9, "bbox": [10, 10, 100, 200]}


def pipeline_for_test():
    p = KitchenPipeline.__new__(KitchenPipeline)
    p.session_id = "scenario-test"
    p.source_type = "camera"
    p.started_monotonic = 100
    p.temporal = TemporalState()
    p.presence = PersonPresence()
    p.track_last_seen = {}
    p.requirement_frame_counts = {r: {s: 0 for s in ("unknown", "compliant", "violation")} for r in ("mask", "gloves", "hair_cover", "apron")}
    p._persons = Mock(return_value=[person()])
    p._ppe = Mock(return_value=[{"class_name": "mask", "confidence": .9, "bbox": [30, 20, 60, 50]}])
    p._apron = Mock(return_value=[])
    p._annotate = Mock(return_value=None)
    p.refiner = Mock()
    p.refiner.refine.return_value = []
    return p


class GeneralScenarioTests(unittest.TestCase):
    def test_confirmation_at_multiple_source_rates(self):
        for fps in (0.5, 1, 5, 15, 30, 60):
            with self.subTest(fps=fps):
                t = TemporalState()
                for n in range(3):
                    result = t.update(1, "mask", "compliant", .9, "mask", n / fps)
                self.assertEqual(result["state"], "compliant")

    def test_long_gaps_do_not_pool_stale_votes(self):
        t = TemporalState()
        for now in (0, 7, 14):
            result = t.update(1, "apron", "compliant", .9, "apron", now)
            self.assertEqual(result["state"], "unknown")

    def test_clock_restart_cannot_mix_future_evidence(self):
        t = TemporalState()
        for now in (10, 10.1, 0):
            result = t.update(1, "mask", "compliant", .9, "mask", now)
        self.assertEqual(result["state"], "unknown")
        self.assertEqual(len(t.history[1]["mask"]), 1)

    def test_continuously_seen_track_is_not_reset_by_slow_processing(self):
        admission = TrackAdmission()
        for now in (0, 3, 6):
            admitted = admission.update([person()], now)
        self.assertEqual(admitted[0]["track_id"], 1)
        self.assertEqual(admission.update([dict(person(), person_confidence=.2)], 9)[0]["track_id"], 1)

    def test_reappearing_track_after_real_gap_requires_admission_again(self):
        admission = TrackAdmission()
        for now in (0, .1, .2):
            admission.update([person()], now)
        admission.update([], .3)
        self.assertEqual(admission.update([person()], 3), [])

    def test_slow_camera_processing_preserves_votes_and_identity(self):
        p = pipeline_for_test()
        with patch("kitchen_monitoring.pipeline.STORE") as store, patch("kitchen_monitoring.pipeline.time.monotonic", side_effect=[100, 103, 106]):
            for n in range(3):
                _, summary = p.process_frame(np.zeros((240, 120, 3), np.uint8), n)
        self.assertEqual(summary["persons"][0]["mask"]["state"], "compliant")
        store.close_track_violations.assert_not_called()
        self.assertEqual(p.observation_period, 3)

    def test_pipeline_forgets_evidence_after_missing_person_expires(self):
        p = pipeline_for_test()
        p.source_type = "upload";p.source_fps = 10
        p._persons = Mock(side_effect=[[person()], [person()], [person()], [], [person()]])
        with patch("kitchen_monitoring.pipeline.STORE") as store:
            for n in (0, 1, 2, 30, 31):
                _, summary = p.process_frame(np.zeros((240, 120, 3), np.uint8), n)
        self.assertEqual(summary["persons"][0]["mask"]["state"], "unknown")
        store.close_track_violations.assert_called_once_with("scenario-test", 1)

    def test_long_source_disconnect_does_not_reuse_an_old_identity(self):
        p = pipeline_for_test()
        p.display_ids = {1: 1, 2: 2};p.next_display_id = 3;p.track_id_offset = 0
        p.track_last_seen = {1: 0, 2: 0}
        with patch("kitchen_monitoring.pipeline.STORE") as store, patch("kitchen_monitoring.pipeline.MODELS.create_person_tracker", return_value=Mock()):
            p._reset_after_source_gap()
        self.assertEqual(store.close_track_violations.call_count, 2)
        new_id = 1 + p.track_id_offset
        self.assertGreater(new_id, 2)
        self.assertEqual(p._staff_label(new_id), "STAFF-03")
        self.assertEqual(p.track_last_seen, {})
        self.assertEqual(p.presence.records, {})

    def test_missing_mask_and_hairnet_can_be_rechecked_too(self):
        r = PPERefiner()
        people = [person()]
        frame = np.zeros((400, 200, 3), np.uint8)
        ppe = Mock(return_value=[{"class_name": "mask", "bbox": [30, 25, 60, 50]}, {"class_name": "hairnet", "bbox": [30, 10, 60, 30]}])
        output = r.refine(frame, people, {1: []}, {}, 0, ppe, Mock(return_value=[]), KitchenPipeline.__new__(KitchenPipeline)._associate)
        self.assertEqual({d["class_name"] for d in output}, {"mask", "hairnet"})

    def test_nearby_people_are_retried_after_unresolved_interval(self):
        r = PPERefiner();people = [person()]
        frame = np.zeros((240, 120, 3), np.uint8)
        ppe, apron = Mock(return_value=[]), Mock(return_value=[])
        associate = KitchenPipeline.__new__(KitchenPipeline)._associate
        r.refine(frame, people, {1: []}, {}, 0, ppe, apron, associate)
        ppe.assert_not_called()
        r.refine(frame, people, {1: []}, {}, .6, ppe, apron, associate)
        self.assertEqual(ppe.call_count, 1)
        self.assertEqual(apron.call_count, 1)

    def test_rechecks_are_fair_when_person_order_changes(self):
        r = PPERefiner();people = [person(i) for i in range(1, 6)]
        frame = np.zeros((400, 200, 3), np.uint8)
        associate = KitchenPipeline.__new__(KitchenPipeline)._associate
        visited = set()
        for n in range(3):
            before = dict(r.last_attempt)
            r.refine(frame, list(reversed(people)) if n % 2 else people,
                     {p["track_id"]: [] for p in people}, {}, n / 30,
                     Mock(return_value=[]), Mock(return_value=[]), associate)
            visited.update(i for i, v in r.last_attempt.items() if v > before.get(i, 0))
        self.assertEqual(visited, {1, 2, 3, 4, 5})


if __name__ == "__main__":
    unittest.main()
