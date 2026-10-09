import unittest
from unittest.mock import Mock, patch
import numpy as np
from kitchen_monitoring.pipeline import KitchenPipeline
from kitchen_monitoring.refinement import PPERefiner


def person(i, box):
    return {"track_id": i, "bbox": box}


def det(name, box):
    return {"class_name": name, "class_id": 0, "confidence": .8, "bbox": box, "source": "test"}


class RefinementTests(unittest.TestCase):
    def setUp(self):
        self.frame = np.zeros((240, 200, 3), np.uint8)
        self.people = [person(1, [50, 20, 100, 140]), person(2, [125, 20, 180, 140])]
        self.associate = KitchenPipeline.__new__(KitchenPipeline)._associate
        self.refiner = PPERefiner()

    def test_crop_coordinates_are_reprojected_and_neighbour_is_not_stolen(self):
        assignments = {1: [det("glove", [60, 80, 70, 90])], 2: [det("glove", [140, 80, 150, 90]), det("apron", [130, 50, 170, 110])]}
        for items in assignments.values():
            items.extend([det("mask", [60, 30, 90, 50]), det("hairnet", [60, 20, 90, 35])])
        own = det("apron", [30, 45, 70, 100])
        neighbour = det("apron", [120, 40, 150, 95])
        apron = Mock(return_value=[own, neighbour])
        ppe = Mock()
        output = self.refiner.refine(self.frame, self.people, assignments, {}, 0, ppe, apron, self.associate)
        self.assertEqual(len(output), 1)
        self.assertEqual(output[0]["bbox"], [55, 53, 95, 108])
        self.assertEqual(own["bbox"], [30, 45, 70, 100])
        self.assertEqual(apron.call_args.kwargs["image_size"], 512)
        ppe.assert_not_called()

    def test_ambiguous_overlap_does_not_gain_ownership_from_crop(self):
        people = [person(1, [50, 20, 100, 140]), person(2, [55, 20, 105, 140])]
        assignments = {1: [det("glove", [60, 80, 70, 90])], 2: [det("glove", [60, 80, 70, 90])]}
        for items in assignments.values():
            items.extend([det("mask", [60, 30, 90, 50]), det("hairnet", [60, 20, 90, 35])])
        output = self.refiner.refine(self.frame, people, assignments, {}, 0, Mock(), Mock(return_value=[det("apron", [30, 45, 70, 100])]), self.associate)
        self.assertEqual(output, [])

    def test_glove_recheck_is_shared_and_cannot_replace_mask(self):
        assignments = {i: [det("apron", [60, 50, 90, 110]), det("mask", [60, 30, 90, 50]), det("hairnet", [60, 20, 90, 35])] for i in (1, 2)}
        ppe = Mock(return_value=[det("glove", [60, 80, 70, 90]), det("glove", [140, 80, 150, 90]), det("no_mask", [60, 30, 90, 50])])
        apron = Mock()
        output = self.refiner.refine(self.frame, self.people, assignments, {}, 0, ppe, apron, self.associate)
        self.assertEqual([d["class_name"] for d in output], ["glove", "glove"])
        self.assertEqual(ppe.call_count, 1)
        self.assertEqual(ppe.call_args.kwargs["image_size"], 1024)
        apron.assert_not_called()

    def test_recent_confirmed_evidence_skips_recheck_without_refreshing_it(self):
        confirmed = {(1, req): {"evidence_timestamp": 0} for req in ("mask", "gloves", "hair_cover", "apron")}
        ppe, apron = Mock(), Mock()
        result = self.refiner.refine(self.frame, self.people[:1], {1: []}, confirmed, .25, ppe, apron, self.associate)
        self.assertEqual(result, [])
        ppe.assert_not_called();apron.assert_not_called()
        self.assertEqual(confirmed[(1, "apron")]["evidence_timestamp"], 0)

    def test_feature_can_be_disabled_without_running_models(self):
        ppe, apron = Mock(), Mock()
        with patch("kitchen_monitoring.refinement.PPE_RECHECK_ENABLED", False):
            self.assertEqual(self.refiner.refine(self.frame, self.people, {1: [], 2: []}, {}, 0, ppe, apron, self.associate), [])
        ppe.assert_not_called();apron.assert_not_called()


if __name__ == "__main__":
    unittest.main()
