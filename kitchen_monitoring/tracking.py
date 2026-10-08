"""Suppress nested head/torso duplicates before they receive tracker IDs."""
import numpy as np


def distinct_person_indices(boxes):
    """Prefer a full body over a small, aligned upper-body box inside it."""
    boxes = np.asarray(boxes)
    if len(boxes) == 0:
        return []
    areas = np.maximum(0, boxes[:, 2] - boxes[:, 0]) * np.maximum(0, boxes[:, 3] - boxes[:, 1])
    kept = []
    for index in np.argsort(-areas, kind="stable"):
        small = boxes[index]
        duplicate = False
        for other in kept:
            large = boxes[other]
            intersection = max(0, min(small[2], large[2]) - max(small[0], large[0])) * max(
                0, min(small[3], large[3]) - max(small[1], large[1]))
            width, height = large[2] - large[0], large[3] - large[1]
            aligned = abs((small[0] + small[2] - large[0] - large[2]) / 2) <= width * 0.35
            near_top = abs(small[1] - large[1]) <= height * 0.2
            if (areas[index] > 0 and intersection / areas[index] >= 0.75
                    and areas[index] < areas[other] * 0.5 and aligned and near_top):
                duplicate = True
                break
        if not duplicate:
            kept.append(int(index))
    return sorted(kept)


def suppress_nested_person_detections(predictor):
    # This callback is registered before Ultralytics installs its tracking hook.
    for index, result in enumerate(predictor.results):
        if result.boxes is None or len(result.boxes) < 2:
            continue
        kept = distinct_person_indices(result.boxes.xyxy.detach().cpu().numpy())
        if len(kept) != len(result.boxes):
            predictor.results[index] = result[kept]
