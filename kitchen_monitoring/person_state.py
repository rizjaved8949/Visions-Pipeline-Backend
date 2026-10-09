"""Confirmed PPE evidence and short-lived presentation records per native track."""
from collections import defaultdict, deque
from copy import deepcopy
from statistics import median

import cv2

from .config import (
    TEMPORAL_WINDOW, TEMPORAL_MIN_VOTES, PPE_EVIDENCE_SECONDS,
    PERSON_HOLD_SECONDS, PERSON_CONFIRM_FRAMES, PERSON_CONFIRM_CONFIDENCE,
    PPE_CONFIRM_SECONDS, PPE_CONFIRM_MAX_SECONDS, PPE_HOLD_SECONDS,
)

REQUIREMENTS = ("mask", "gloves", "hair_cover", "apron")


def unknown():
    return {"state": "unknown", "confidence": 0.0, "evidence_type": None}


class TemporalState:
    def __init__(self):
        self.history = defaultdict(lambda: defaultdict(lambda: deque(maxlen=TEMPORAL_WINDOW)))
        self.confirmed = {}
        self.clocks = {}
        self.intervals = {}

    def forget(self, track_id):
        self.history.pop(track_id, None)
        for key in list(self.confirmed):
            if key[0] == track_id:
                self.confirmed.pop(key, None)
        for key in list(self.clocks):
            if key[0] == track_id:
                self.clocks.pop(key, None)
                self.intervals.pop(key, None)

    def update(self, track_id, requirement, state, confidence, evidence_type, timestamp=None):
        key = (track_id, requirement)
        now = float(timestamp) if timestamp is not None else self.clocks.get(key, 0.0) + 1 / 30
        history = self.history[track_id][requirement]
        previous_time = self.clocks.get(key)
        if previous_time is not None and now < previous_time:
            history.clear()
            self.confirmed.pop(key, None)
            self.intervals.pop(key, None)
        intervals = self.intervals.setdefault(key, deque(maxlen=TEMPORAL_WINDOW))
        if previous_time is not None and now > previous_time:
            intervals.append(now - previous_time)
        self.clocks[key] = now
        cadence_window = median(intervals) * TEMPORAL_MIN_VOTES if intervals else PPE_CONFIRM_SECONDS
        vote_seconds = min(PPE_CONFIRM_MAX_SECONDS, max(PPE_CONFIRM_SECONDS, cadence_window))
        while history and now - history[0]["timestamp"] > vote_seconds:
            history.popleft()
        previous = self.confirmed.get(key)
        if previous and now - previous["evidence_timestamp"] > PPE_HOLD_SECONDS.get(requirement, PPE_EVIDENCE_SECONDS):
            self.confirmed.pop(key, None)
            previous = None
        if evidence_type == "conflicting_evidence":
            history.clear()
            self.confirmed.pop(key, None)
            return unknown()
        known = state in {"compliant", "violation"} and evidence_type is not None
        if known:
            # Different negative classes are different decisions too.
            last = next((item for item in reversed(history) if item["evidence_type"]), None)
            prior_class = last["evidence_type"] if last else (previous or {}).get("evidence_type")
            if prior_class and prior_class != evidence_type:
                history.clear()
                self.confirmed.pop(key, None)
                previous = None
        if known:
            # One vote per source observation, even if multiple inference
            # views of the same frame find the item. Empty frames are not votes.
            if history and history[-1]["timestamp"] == now:
                result = self.confirmed.get(key)
                return dict(result) if result else unknown()
            history.append({"state": state, "confidence": confidence,
                            "evidence_type": evidence_type, "timestamp": now})
            votes = [item for item in history if item["evidence_type"] == evidence_type]
            if len(votes) >= TEMPORAL_MIN_VOTES or (previous and previous["evidence_type"] == evidence_type):
                self.confirmed[key] = {"state": state, "confidence": round(sum(item["confidence"] for item in votes) / len(votes), 4),
                                       "evidence_type": evidence_type, "evidence_timestamp": now}
        result = self.confirmed.get(key)
        if not result:
            return unknown()
        return {**result, "evidence_age_seconds": round(now - result["evidence_timestamp"], 3)}


class TrackAdmission:
    """Require sustained confidence for a new ID; keep low-confidence established tracks."""
    def __init__(self):
        self.records = {}

    def update(self, persons, now):
        seen = {p["track_id"] for p in persons}
        for track_id in list(self.records):
            record = self.records[track_id]
            interrupted = track_id not in seen or record.get("missing", False)
            if interrupted and now - record["last_seen"] > PERSON_HOLD_SECONDS:
                del self.records[track_id]
            elif track_id not in seen:
                record["hits"] = 0
                record["missing"] = True
        accepted = []
        for person in persons:
            record = self.records.setdefault(person["track_id"], {"hits": 0, "accepted": False, "last_seen": now})
            record["last_seen"] = now
            record["missing"] = False
            if not record["accepted"]:
                record["hits"] = record["hits"] + 1 if person["person_confidence"] >= PERSON_CONFIRM_CONFIDENCE else 0
                record["accepted"] = record["hits"] >= PERSON_CONFIRM_FRAMES
            if record["accepted"]:
                accepted.append(person)
        return accepted


class PersonPresence:
    def __init__(self):
        self.records = {}
        self.thumbnails = {}

    def update(self, persons, frame, now):
        expired = []
        # Expire before reacquisition, so old evidence cannot become current.
        for track_id, (last_seen, _) in list(self.records.items()):
            if now - last_seen > PERSON_HOLD_SECONDS:
                expired.append(track_id)
                del self.records[track_id]
                self.thumbnails.pop(track_id, None)
        visible = set()
        height, width = frame.shape[:2]
        for person in persons:
            track_id = person["track_id"]
            visible.add(track_id)
            self.records[track_id] = (now, deepcopy(person))
            x1, y1, x2, y2 = person["bbox"]
            crop = frame[max(0, int(y1)):min(height, int(y1 + (y2 - y1) * 0.45)), max(0, int(x1)):min(width, int(x2))]
            if crop.size:
                ratio = min(54 / crop.shape[1], 66 / crop.shape[0])
                self.thumbnails[track_id] = cv2.resize(crop, (max(1, round(crop.shape[1] * ratio)), max(1, round(crop.shape[0] * ratio))))
        display = []
        for track_id, (last_seen, saved) in self.records.items():
            person = deepcopy(saved)
            person["visible"] = track_id in visible
            if not person["visible"]:
                person["detected_classes"] = []
                for requirement in REQUIREMENTS:
                    item = person[requirement]
                    supported_at = item.get("evidence_timestamp", last_seen)
                    if now - supported_at > PPE_HOLD_SECONDS.get(requirement, PPE_EVIDENCE_SECONDS):
                        person[requirement] = unknown()
            display.append(person)
        return display, expired
