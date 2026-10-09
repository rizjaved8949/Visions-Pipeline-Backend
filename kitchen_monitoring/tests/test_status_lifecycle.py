import unittest
from unittest.mock import Mock, patch
import numpy as np

from kitchen_monitoring.person_state import TemporalState, TrackAdmission, PersonPresence
from kitchen_monitoring.pipeline import KitchenPipeline, _choose_evidence
from kitchen_monitoring.visualization import annotate_people, status_rows


def person(track_id=1, confidence=.9):
    return {"track_id": track_id, "staff_label": f"STAFF-{track_id:02d}",
            "bbox": [10, 10, 60, 110], "person_confidence": confidence,
            **{r: {"state": "unknown"} for r in ("mask", "gloves", "hair_cover", "apron")}}


class ConfirmedStatusTests(unittest.TestCase):
    def test_exact_status_vocabulary_and_fixed_order(self):
        mappings = {"mask": {"mask": "Worn", "no_mask": "Not worn", "incorrect_mask": "Incorrect mask"},
                    "gloves": {"glove": "Worn", "no_glove": "Missing"},
                    "hair_cover": {"hairnet": "Worn", "no_hairnet": "Not worn"},
                    "apron": {"apron": "Apron", "no_apron": "No apron"}}
        for i, (requirement, classes) in enumerate(mappings.items()):
            for evidence, label in classes.items():
                p = person()
                p[requirement] = {"state": "violation" if evidence.startswith("no_") or evidence == "incorrect_mask" else "compliant", "evidence_type": evidence}
                rows = status_rows(p)
                self.assertEqual([r[0] for r in rows], ["Mask", "Gloves", "Hairnet", "Apron"])
                self.assertEqual(rows[i][1], label)
        self.assertEqual([r[1] for r in status_rows(person())], ["-"] * 4)

    def test_sparse_real_observations_survive_empty_frames(self):
        t = TemporalState()
        for n in range(21):
            known = n in (0, 10, 20)
            result = t.update(1, "gloves", "compliant" if known else "unknown", .8 if known else 0, "glove" if known else None, n / 30)
        self.assertEqual(result["evidence_type"], "glove")
        self.assertEqual(len(t.history[1]["gloves"]), 3)

    def test_clothing_hold_is_bounded_and_not_refreshed_by_empty_frames(self):
        t = TemporalState()
        for now in (0, .1, .2):
            t.update(1, "apron", "compliant", .8, "apron", now)
        for now in (1, 2, 3):
            result = t.update(1, "apron", "unknown", 0, None, now)
            self.assertEqual(result["evidence_type"], "apron")
            self.assertEqual(result["evidence_timestamp"], .2)
        self.assertEqual(t.update(1, "apron", "unknown", 0, None, 3.3)["state"], "unknown")

    def test_multiple_views_of_one_frame_cannot_confirm_a_status(self):
        t = TemporalState()
        for _ in range(3):
            result = t.update(1, "apron", "compliant", .8, "apron", 0)
        self.assertEqual(result["state"], "unknown")
        self.assertEqual(len(t.history[1]["apron"]), 1)

    def test_negative_classes_cannot_pool_confirmation_votes(self):
        t = TemporalState()
        for n, name in enumerate(("no_mask", "incorrect_mask", "no_mask")):
            self.assertEqual(t.update(1, "mask", "violation", .9, name, n / 30)["state"], "unknown")

    def test_contradiction_resets_status_and_missing_does_not_restore_it(self):
        t = TemporalState()
        for n in range(3):
            confirmed = t.update(1, "mask", "compliant", .9, "mask", n / 30)
        self.assertEqual(confirmed["state"], "compliant")
        self.assertEqual(t.update(1, "mask", "violation", .9, "no_mask", .1)["state"], "unknown")
        self.assertEqual(t.update(1, "mask", "unknown", 0, None, .2)["state"], "unknown")
        t.update(1, "mask", "violation", .9, "no_mask", .3)
        self.assertEqual(t.update(1, "mask", "violation", .9, "no_mask", .4)["evidence_type"], "no_mask")

    def test_missing_frames_do_not_refresh_evidence_expiry(self):
        t = TemporalState()
        for n in range(3):
            t.update(1, "mask", "compliant", .8, "mask", n / 30)
        self.assertEqual(t.update(1, "mask", "unknown", 0, None, .5)["state"], "compliant")
        self.assertEqual(t.update(1, "mask", "unknown", 0, None, 1.1)["state"], "unknown")
        self.assertEqual(t.update(1, "mask", "compliant", .8, "mask", 1.2)["state"], "unknown")

    def test_exact_mask_conflict_clears_previous_result(self):
        conflict = _choose_evidence([
            {"state": "violation", "evidence_type": "no_mask", "confidence": .8},
            {"state": "violation", "evidence_type": "incorrect_mask", "confidence": .78}])
        self.assertEqual(conflict["evidence_type"], "conflicting_evidence")
        t = TemporalState()
        for n in range(3):
            t.update(1, "mask", "compliant", .8, "mask", n / 30)
        self.assertEqual(t.update(1, "mask", **conflict, timestamp=.2)["state"], "unknown")

    def test_duplicate_positive_boxes_cannot_hide_conflicting_class(self):
        result = _choose_evidence([
            {"state": "compliant", "evidence_type": "mask", "confidence": .85},
            {"state": "compliant", "evidence_type": "mask", "confidence": .84},
            {"state": "violation", "evidence_type": "no_mask", "confidence": .83},
        ])
        self.assertEqual(result["evidence_type"], "conflicting_evidence")

    def test_expired_track_has_no_old_confirmation(self):
        t = TemporalState()
        for n in range(3):
            t.update(1, "mask", "compliant", .8, "mask", n / 30)
        t.forget(1)
        self.assertEqual(t.update(1, "mask", "compliant", .8, "mask", .2)["state"], "unknown")


class IdentityLifecycleTests(unittest.TestCase):
    def test_admission_rejects_brief_candidates_and_keeps_recovered_id(self):
        a = TrackAdmission()
        self.assertEqual(a.update([person()], 0), [])
        self.assertEqual(a.update([person()], .03), [])
        self.assertEqual(a.update([person()], .06)[0]["track_id"], 1)
        a.update([person(2)], .1)
        a.update([person(2)], .13)
        self.assertEqual(a.update([person(2, .4)], .16), [])
        a.update([], .2)
        self.assertEqual(a.update([person(1, .2)], 1.8)[0]["track_id"], 1)

    def test_real_new_people_are_not_capped_at_two(self):
        a = TrackAdmission()
        for n in range(3):
            result = a.update([person(i) for i in (1, 2, 3)], n / 30)
        self.assertEqual([p["track_id"] for p in result], [1, 2, 3])

    def test_missing_frame_resets_new_candidate_confirmation(self):
        a = TrackAdmission()
        a.update([person()], 0)
        a.update([person()], .03)
        a.update([], .06)
        self.assertEqual(a.update([person()], .09), [])

    def test_cached_thumbnail_no_ghost_box_and_no_hidden_text(self):
        registry = PersonPresence()
        frame = np.full((120, 80, 3), 170, dtype=np.uint8)
        p = person()
        p["apron"] = {"state": "compliant", "evidence_type": "apron", "evidence_timestamp": 0}
        registry.update([p], frame, 0)
        blank = np.zeros_like(frame)
        cards, _ = registry.update([], blank, .5)
        self.assertFalse(cards[0]["visible"])
        self.assertTrue(registry.thumbnails[1].any())
        with patch("kitchen_monitoring.visualization._text") as draw:
            rendered = annotate_people(blank, cards, [], thumbnails=registry.thumbnails)
        self.assertFalse(rendered[:, :480].any())
        self.assertEqual([c.args[1] for c in draw.call_args_list], ["STAFF-01", "Mask", "-", "Gloves", "-", "Hairnet", "-", "Apron", "Apron"])
        cards, _ = registry.update([], blank, 1.1)
        self.assertEqual(cards[0]["apron"]["state"], "compliant")
        cards, expired = registry.update([], blank, 2.1)
        self.assertEqual((cards, expired), ([], [1]))
        self.assertEqual(registry.thumbnails, {})

    def test_pipeline_retention_uses_video_time_and_does_not_count_hidden_cards(self):
        p = KitchenPipeline.__new__(KitchenPipeline)
        p.source_type = "upload"
        p.source_fps = 30
        p.session_id = "unit-test"
        p.track_last_seen = {}
        p.temporal = TemporalState()
        p.presence = PersonPresence()
        p.requirement_frame_counts = {r: {s: 0 for s in ("unknown", "compliant", "violation")} for r in ("mask", "gloves", "hair_cover", "apron")}
        p._persons = Mock(side_effect=[[person()], [], [person()]])
        p._ppe = Mock(return_value=[])
        p._apron = Mock(return_value=[])
        p._annotate = Mock(return_value=None)
        frame = np.zeros((120, 80, 3), dtype=np.uint8)
        with patch("kitchen_monitoring.pipeline.STORE"):
            p.process_frame(frame, 0)
            _, summary = p.process_frame(frame, 45)
            self.assertEqual(summary["staff_detected"], 0)
            self.assertEqual(p._annotate.call_args.args[1][0]["track_id"], 1)
            _, summary = p.process_frame(frame, 47)
            self.assertEqual(summary["persons"][0]["track_id"], 1)
        self.assertEqual(p.frame_time_seconds, 47 / 30)


if __name__ == "__main__":
    unittest.main()
