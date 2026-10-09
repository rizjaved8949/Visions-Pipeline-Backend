"""Targeted second inference views; every result must retain source-space ownership."""
from .config import (
    PPE_RECHECK_ENABLED, PPE_RECHECK_IMAGE_SIZE, APRON_RECHECK_IMAGE_SIZE,
    PPE_RECHECK_MAX_PEOPLE, PPE_RECHECK_MAX_HEIGHT_RATIO, PPE_HOLD_SECONDS,
    PPE_RECHECK_NEAR_DELAY_SECONDS,
)

CLASSES = {
    "mask": {"mask", "no_mask", "incorrect_mask"},
    "gloves": {"glove", "no_glove"},
    "hair_cover": {"hairnet", "no_hairnet"},
    "apron": {"apron", "no_apron"},
}


class PPERefiner:
    def __init__(self):
        self.turn = 0
        self.last_attempt = {}
        self.unresolved_since = {}

    def refine(self, frame, persons, assignments, confirmed, now, ppe, apron, associate):
        if not PPE_RECHECK_ENABLED:
            return []
        height, width = frame.shape[:2]
        active_ids = {p["track_id"] for p in persons}
        self.last_attempt = {k: v for k, v in self.last_attempt.items() if k in active_ids}
        self.unresolved_since = {k: v for k, v in self.unresolved_since.items() if k[0] in active_ids}
        pending = []
        for person in persons:
            track_id = person["track_id"]
            x1, y1, x2, y2 = person["bbox"]
            small = y2 - y1 <= height * PPE_RECHECK_MAX_HEIGHT_RATIO
            requirements = []
            for requirement, classes in CLASSES.items():
                key = (track_id, requirement)
                if any(d["class_name"] in classes for d in assignments[track_id]):
                    self.unresolved_since.pop(key, None)
                    continue
                recent = confirmed.get((track_id, requirement))
                if recent and now - recent["evidence_timestamp"] < PPE_HOLD_SECONDS[requirement] / 2:
                    continue
                since = self.unresolved_since.setdefault(key, now)
                if not small and now - since < PPE_RECHECK_NEAR_DELAY_SECONDS:
                    continue
                requirements.append(requirement)
            if requirements:
                pending.append((person, requirements))
        if not pending:
            return []
        # Fairness is keyed by identity, not by position in a changing list.
        for person, _ in pending:
            self.last_attempt.setdefault(person["track_id"], self.turn)
        selected = sorted(pending, key=lambda item: (
            self.last_attempt[item[0]["track_id"]], item[0]["track_id"]))[:PPE_RECHECK_MAX_PEOPLE]
        for person, _ in selected:
            self.turn += 1
            self.last_attempt[person["track_id"]] = self.turn
        refined = []
        ppe_targets = {p["track_id"]: set(reqs) - {"apron"} for p, reqs in selected}
        if any(ppe_targets.values()):
            candidates = [dict(d, source="ppe_recheck")
                          for d in ppe(frame, image_size=PPE_RECHECK_IMAGE_SIZE)]
            owned = associate(persons, candidates)
            for track_id, requirements in ppe_targets.items():
                allowed = set().union(*(CLASSES[r] for r in requirements))
                refined.extend(d for d in owned[track_id] if d["class_name"] in allowed)
        for person, requirements in selected:
            if "apron" not in requirements:
                continue
            x1, y1, x2, y2 = person["bbox"]
            person_width, person_height = x2 - x1, y2 - y1
            left = max(0, int(x1 - person_width * .5))
            right = min(width, int(x2 + person_width * .5))
            top = max(0, int(y1 - person_height * .1))
            bottom = min(height, int(y2 + person_height * .1))
            if right <= left or bottom <= top:
                continue
            crop = frame[top:bottom, left:right]
            candidates = []
            for detection in apron(crop, image_size=APRON_RECHECK_IMAGE_SIZE):
                if detection["class_name"] not in CLASSES["apron"]:
                    continue
                # YOLO returns coordinates relative to the crop. Reproject
                # before ownership checks; the crop may include a neighbour.
                box = [value + (left, top)[i % 2] for i, value in enumerate(detection["bbox"])]
                candidates.append(dict(detection, bbox=box, source="apron_recheck"))
            owned = associate(persons, candidates)
            refined.extend(owned[person["track_id"]])
        return refined
