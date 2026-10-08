import unittest
from unittest.mock import patch

import numpy as np

from kitchen_monitoring.pipeline import KitchenPipeline, TemporalState
from kitchen_monitoring.visualization import annotate_people, detected_class_rows


class DetectedClassTests(unittest.TestCase):
    def setUp(self):
        self.pipeline = KitchenPipeline.__new__(KitchenPipeline)
        self.pipeline.temporal = TemporalState()
        self.person = {
            "track_id": 1, "staff_label": "STAFF-01",
            "bbox": [10, 10, 100, 200], "person_confidence": 0.9,
        }

    def detection(self, name, confidence=0.82):
        return {"class_name": name, "confidence": confidence}

    def test_class_visible_before_compliance_is_confirmed(self):
        result = self.pipeline._person_state(self.person, [self.detection("mask")])
        self.assertEqual(detected_class_rows(result), [("mask", 0.82)])
        self.assertEqual(result["mask"]["state"], "unknown")
        self.assertEqual(result["overall"], "unknown")

    def test_no_detection_does_not_invent_a_negative_class(self):
        result = self.pipeline._person_state(self.person, [])
        self.assertEqual(detected_class_rows(result), [])

    def test_conflicting_evidence_is_not_presented_as_a_class(self):
        result = self.pipeline._person_state(self.person, [
            self.detection("apron", 0.80), self.detection("no_apron", 0.78),
        ])
        self.assertEqual(detected_class_rows(result), [])

    def test_negative_class_requires_actual_model_evidence(self):
        result = self.pipeline._person_state(self.person, [self.detection("no_apron")])
        self.assertEqual(detected_class_rows(result), [("no_apron", 0.82)])

    def test_current_empty_result_does_not_reuse_old_compliance(self):
        for _ in range(3):
            result = self.pipeline._person_state(self.person, [self.detection("apron")])
        self.assertEqual(result["apron"]["state"], "compliant")
        result = self.pipeline._person_state(self.person, [])
        self.assertEqual(result["apron"]["state"], "compliant")
        self.assertEqual(detected_class_rows(result), [])

    def test_older_saved_results_preserve_exact_class(self):
        result = {**self.person, "mask": {
            "state": "violation", "evidence_type": "incorrect_mask", "confidence": 0.75}}
        self.assertEqual(detected_class_rows(result), [("incorrect_mask", 0.75)])

    def test_card_renders_classes_and_respects_confidence_preference(self):
        result = self.pipeline._person_state(self.person, [self.detection("incorrect_mask")])
        frame = np.zeros((360, 202, 3), dtype=np.uint8)
        for show_confidence, expected in ((True, "incorrect_mask  82%"), (False, "incorrect_mask")):
            with patch("kitchen_monitoring.visualization._text") as draw:
                annotate_people(frame, [result], [], show_confidence=show_confidence)
                texts = [call.args[1] for call in draw.call_args_list]
            self.assertIn(expected, texts)
            self.assertFalse(any("UNKNOWN" in text or "MISSING / INCORRECT" in text for text in texts))


if __name__ == "__main__":
    unittest.main()
