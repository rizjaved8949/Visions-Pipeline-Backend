import unittest
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np

from kitchen_monitoring.pipeline import KitchenPipeline
from kitchen_monitoring.tracking import distinct_person_indices
from kitchen_monitoring.visualization import annotate_people, annotation_size, person_color


def person(track_id=1, bbox=(10, 10, 100, 200)):
    return {
        "track_id": track_id, "staff_label": f"STAFF-{track_id:02d}",
        "bbox": bbox, "person_confidence": 0.9, "overall": "unknown",
        **{key: {"state": "unknown"} for key in ("mask", "gloves", "hair_cover", "apron")},
    }


class PersonPresentationTests(unittest.TestCase):
    def test_nested_upper_body_does_not_create_another_person(self):
        boxes = [(10, 10, 110, 310), (35, 12, 85, 100), (85, 15, 155, 300)]
        self.assertEqual(distinct_person_indices(boxes), [0, 2])
        self.assertEqual(distinct_person_indices([]), [])
        self.assertEqual(distinct_person_indices([(91, 95, 132, 269), (89, 87, 118, 141)]), [0])

    def test_overlapping_people_do_not_steal_ppe(self):
        pipeline = KitchenPipeline.__new__(KitchenPipeline)
        people = [person(1, (0, 0, 100, 200)), person(2, (20, 20, 80, 180))]
        detection = {"bbox": (30, 40, 50, 60)}
        self.assertEqual(pipeline._associate(people, [detection]), {1: [], 2: []})
        self.assertEqual(pipeline._associate(people[:1], [detection]), {1: [detection]})

    def test_unconfirmed_detection_does_not_get_a_fake_id(self):
        pipeline = KitchenPipeline.__new__(KitchenPipeline)
        box = SimpleNamespace(id=None, xyxy=np.array([[0, 0, 10, 20]]))
        # _extract_xyxy expects tensor methods.
        box.xyxy = Mock()
        box.xyxy.__getitem__ = Mock(return_value=SimpleNamespace(
            detach=lambda: SimpleNamespace(cpu=lambda: SimpleNamespace(
                tolist=lambda: [0, 0, 10, 20]))))
        pipeline.person_model = Mock()
        pipeline.person_model.track.return_value = [SimpleNamespace(boxes=[box])]
        self.assertEqual(pipeline._persons(np.zeros((30, 30, 3), dtype=np.uint8)), [])

    def test_encoder_dimensions_stay_fixed_as_people_enter(self):
        frame = np.zeros((360, 202, 3), dtype=np.uint8)
        for count in (0, 1, 5, 8):
            result = annotate_people(frame, [person(i + 1) for i in range(count)], [])
            self.assertEqual(result.shape, (720, 824, 3))
        self.assertEqual(annotation_size(202, 360), (824, 720))

    def test_identity_color_survives_status_and_order_changes(self):
        frame = np.zeros((360, 202, 3), dtype=np.uint8)
        people = [person(1), person(2, (110, 30, 190, 320))]
        a = annotate_people(frame, people, [])
        b = annotate_people(frame, list(reversed(people)), [])
        np.testing.assert_array_equal(a, b)
        color = person_color(1)
        people[0]["overall"] = "violation"
        people[0]["apron"]["state"] = "violation"
        c = annotate_people(frame, people, [])
        self.assertEqual(tuple(c[100, 20]), color)

    def test_display_preferences_and_source_coordinates(self):
        frame = np.zeros((360, 202, 3), dtype=np.uint8)
        people = [person()]
        before = frame.copy()
        output = annotate_people(frame, people, [], show_labels=False, show_boxes=False)
        self.assertEqual(output.shape, (720, 404, 3))
        self.assertFalse(output.any())
        self.assertEqual(people[0]["bbox"], (10, 10, 100, 200))
        np.testing.assert_array_equal(frame, before)


if __name__ == "__main__":
    unittest.main()
