import os
import time

from collections import defaultdict

import cv2

from guard_monitoring.io import make_writer
from system_settings import get_display_prefs

from .config import (
    DEVICE,
    PERSON_IMAGE_SIZE,
    PERSON_CONFIDENCE,
    PERSON_TRACKER_PATH,
    PERSON_IOU,
    CONFLICT_CONFIDENCE_MARGIN,
    PERSON_HOLD_SECONDS,
    METRIC_SAMPLE_SECONDS,
    SESSION_DIR,
    SEVERITY_MAP,
)

from .model_registry import MODELS
from .visualization import annotate_people, annotation_size
from .tracking import distinct_person_indices
from .person_state import TemporalState, TrackAdmission, PersonPresence
from .refinement import PPERefiner

from .storage import (
    STORE,
    utc_now,
)


# ============================================================
# PPE SEMANTIC MAPPING
# ============================================================

PPE_RULES = {

    "glove":
        (
            "gloves",
            "compliant",
            "glove",
        ),

    "no_glove":
        (
            "gloves",
            "violation",
            "no_glove",
        ),

    "hairnet":
        (
            "hair_cover",
            "compliant",
            "hairnet",
        ),

    "no_hairnet":
        (
            "hair_cover",
            "violation",
            "no_hairnet",
        ),

    "mask":
        (
            "mask",
            "compliant",
            "mask",
        ),

    "no_mask":
        (
            "mask",
            "violation",
            "no_mask",
        ),

    "incorrect_mask":
        (
            "mask",
            "violation",
            "incorrect_mask",
        ),

    # ---- apron classes (from separate apron model)
    "apron":
        (
            "apron",
            "compliant",
            "apron",
        ),

    "no_apron":
        (
            "apron",
            "violation",
            "no_apron",
        ),
}


# Requirements that are always tracked.
# "apron" is included here; when the apron model weight file is
# absent, every frame emits state="unknown" for it so the
# compliance score is unaffected (unknown frames are excluded
# from the percentage calculation).
REQUIREMENTS = [
    "mask",
    "gloves",
    "hair_cover",
    "apron",
]


# ============================================================
# HELPERS
# ============================================================

def _extract_xyxy(
    box,
):

    values = (
        box.xyxy[0]
        .detach()
        .cpu()
        .tolist()
    )

    return [
        float(v)
        for v in values
    ]


def _center(
    bbox,
):

    x1, y1, x2, y2 = bbox

    return (
        (x1 + x2) / 2,
        (y1 + y2) / 2,
    )


def _bbox_area(
    bbox,
):

    x1, y1, x2, y2 = bbox

    return max(
        0,
        x2 - x1,
    ) * max(
        0,
        y2 - y1,
    )


def _contains(
    person_bbox,
    point,
):

    x1, y1, x2, y2 = (
        person_bbox
    )

    px, py = point

    width = x2 - x1

    height = y2 - y1


    # Small padding because gloves can sit close to
    # the outer body/person rectangle.
    x_pad = width * 0.08

    y_pad = height * 0.05


    return (
        x1 - x_pad <= px <= x2 + x_pad
        and
        y1 - y_pad <= py <= y2 + y_pad
    )


def _choose_evidence(
    detections,
):

    if not detections:

        return {
            "state":
                "unknown",

            "confidence":
                0.0,

            "evidence_type":
                None,
        }


    ordered = sorted(
        detections,
        key=lambda item:
            item[
                "confidence"
            ],
        reverse=True,
    )


    best = ordered[0]


    # Compare the strongest different class, even if duplicate boxes
    # for the best class come before it.
    if len(ordered) >= 2:

        second = next(
            (item for item in ordered[1:]
             if item["evidence_type"] != best["evidence_type"]),
            best,
        )

        if (
            second["evidence_type"]
            != best["evidence_type"]
            and
            abs(
                second[
                    "confidence"
                ]
                -
                best[
                    "confidence"
                ]
            )
            <= CONFLICT_CONFIDENCE_MARGIN
        ):

            return {
                "state":
                    "unknown",

                "confidence":
                    max(
                        best[
                            "confidence"
                        ],
                        second[
                            "confidence"
                        ],
                    ),

                "evidence_type":
                    "conflicting_evidence",
            }


    return best


# ============================================================
# MAIN PIPELINE
# ============================================================

class KitchenPipeline:

    def __init__(
        self,
        session_id,
        stop_event,
        source_type="upload",
    ):

        self.session_id = (
            session_id
        )

        self.stop_event = (
            stop_event
        )

        self.source_type = (
            source_type
        )

        self.person_model = (
            MODELS
            .create_person_tracker()
        )

        self.temporal = (
            TemporalState()
        )

        # Workspace-wide Display preferences (Settings page) - read once per
        # session, not per frame.
        display_prefs = get_display_prefs()
        self.show_detection_boxes = bool(display_prefs.get("show_detection_boxes", True))
        self.show_labels = bool(display_prefs.get("show_labels", True))
        self.show_confidence = bool(display_prefs.get("show_confidence", True))


        self.display_ids = {}

        self.next_display_id = 1
        self.track_id_offset = 0


        self.track_last_seen = {}
        self.admission = TrackAdmission()
        self.presence = PersonPresence()
        self.refiner = PPERefiner()
        self.source_fps = 30.0
        self.frame_time_seconds = 0.0
        self.started_monotonic = time.monotonic()

        # Cumulative, session-wide per-requirement frame tallies.
        self.requirement_frame_counts = {
            requirement: {
                "compliant": 0,
                "violation": 0,
                "unknown": 0,
            }
            for requirement in REQUIREMENTS
        }


        self.output_dir = (
            SESSION_DIR /
            session_id
        )

        self.output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )


        self.output_video = (
            self.output_dir /
            "annotated.mp4"
        )

        self.latest_frame = (
            self.output_dir /
            "latest.jpg"
        )


    # --------------------------------------------------------
    # Display labels
    # --------------------------------------------------------

    def _staff_label(
        self,
        track_id,
    ):

        if (
            track_id
            not in self.display_ids
        ):

            self.display_ids[
                track_id
            ] = self.next_display_id

            self.next_display_id += 1


        return (
            f"STAFF-"
            f"{self.display_ids[track_id]:02d}"
        )


    # --------------------------------------------------------
    # Person tracking
    # --------------------------------------------------------

    def _persons(
        self,
        frame,
    ):

        result = (
            self.person_model.track(
                source=frame,
                persist=True,
                classes=[0],
                conf=PERSON_CONFIDENCE,
                iou=PERSON_IOU,
                imgsz=PERSON_IMAGE_SIZE,
                tracker=str(PERSON_TRACKER_PATH),
                device=DEVICE,
                verbose=False,
            )[0]
        )


        people = []


        if result.boxes is None:

            if hasattr(self, "admission"):
                self.admission.update([], getattr(self, "frame_time_seconds", 0.0))
            return people


        for box in result.boxes:

            bbox = _extract_xyxy(
                box
            )


            if box.id is not None:

                track_id = int(
                    box.id[0]
                    .detach()
                    .cpu()
                    .item()
                )

            else:

                # Detection order is not a stable person identity.
                # Wait for the tracker to confirm an ID.
                continue


            track_id += getattr(self, "track_id_offset", 0)

            confidence = float(
                box.conf[0]
                .detach()
                .cpu()
                .item()
            )


            people.append(
                {
                    "track_id":
                        track_id,

                    "bbox":
                        bbox,

                    "person_confidence":
                        round(
                            confidence,
                            4,
                        ),
                }
            )


        kept = distinct_person_indices([person["bbox"] for person in people])
        if not hasattr(self, "admission"):
            self.admission = TrackAdmission()
        accepted = self.admission.update(
            [people[index] for index in kept], getattr(self, "frame_time_seconds", 0.0))
        for person in accepted:
            person["staff_label"] = self._staff_label(person["track_id"])
        # Align the lost-track frame budget with source/processing cadence.
        fps = (getattr(self, "source_fps", 30.0) if getattr(self, "source_type", "upload") == "upload"
               else 1 / max(.001, getattr(self, "observation_period", 1 / 30)))
        trackers = getattr(getattr(self.person_model, "predictor", None), "trackers", [])
        for tracker in trackers if isinstance(trackers, (list, tuple)) else []:
            tracker.max_time_lost = max(1, round(fps * PERSON_HOLD_SECONDS))
        return accepted


    # --------------------------------------------------------
    # PPE detection  (mask / gloves / hairnet model)
    # --------------------------------------------------------

    def _ppe(
        self,
        frame,
        image_size=None,
    ):

        result = MODELS.predict_ppe(
            frame, image_size=image_size
        )


        detections = []


        if result.boxes is None:

            return detections


        for box in result.boxes:

            class_id = int(
                box.cls[0]
                .detach()
                .cpu()
                .item()
            )


            class_name = str(
                result.names[
                    class_id
                ]
            )


            confidence = float(
                box.conf[0]
                .detach()
                .cpu()
                .item()
            )


            detections.append(
                {
                    "class_id":
                        class_id,

                    "class_name":
                        class_name,

                    "confidence":
                        round(
                            confidence,
                            4,
                        ),

                    "bbox":
                        _extract_xyxy(
                            box
                        ),

                    "source":
                        "ppe_model",
                }
            )


        return detections


    # --------------------------------------------------------
    # Apron detection  (separate apron model)
    # --------------------------------------------------------

    def _apron(
        self,
        frame,
        image_size=None,
    ):
        """
        Runs the apron model and returns detections in the same
        format as _ppe() so they can be merged into one list
        before association.

        Returns an empty list when the apron model is not
        available (weight file absent) — the pipeline continues
        normally and apron status stays "unknown".
        """

        result = MODELS.predict_apron(
            frame, image_size=image_size
        )

        detections = []

        # predict_apron() returns None when model file is absent.
        if result is None or result.boxes is None:
            return detections


        for box in result.boxes:

            class_id = int(
                box.cls[0]
                .detach()
                .cpu()
                .item()
            )


            class_name = str(
                result.names[
                    class_id
                ]
            )


            confidence = float(
                box.conf[0]
                .detach()
                .cpu()
                .item()
            )


            detections.append(
                {
                    "class_id":
                        class_id,

                    "class_name":
                        class_name,

                    "confidence":
                        round(
                            confidence,
                            4,
                        ),

                    "bbox":
                        _extract_xyxy(
                            box
                        ),

                    "source":
                        "apron_model",
                }
            )


        return detections


    # --------------------------------------------------------
    # PPE -> person association
    # --------------------------------------------------------

    def _associate(
        self,
        persons,
        detections,
    ):

        assigned = {
            person[
                "track_id"
            ]: []
            for person
            in persons
        }


        for detection in detections:

            point = _center(
                detection[
                    "bbox"
                ]
            )


            candidates = [
                person
                for person
                in persons
                if _contains(
                    person[
                        "bbox"
                    ],
                    point,
                )
            ]


            # Overlapping boxes do not establish which person owns PPE.
            # Leave ambiguous evidence unassigned instead of guessing.
            if len(candidates) != 1:
                continue

            person = candidates[0]

            assigned[
                person[
                    "track_id"
                ]
            ].append(
                detection
            )


        return assigned


    # --------------------------------------------------------
    # Raw PPE state
    # --------------------------------------------------------

    def _raw_state(
        self,
        detections,
    ):

        evidence = {
            requirement: []
            for requirement
            in REQUIREMENTS
        }


        for detection in detections:

            rule = PPE_RULES.get(
                detection[
                    "class_name"
                ]
            )


            if rule is None:

                continue


            (
                requirement,
                state,
                evidence_type,
            ) = rule


            evidence[
                requirement
            ].append(
                {
                    "state":
                        state,

                    "confidence":
                        detection[
                            "confidence"
                        ],

                    "evidence_type":
                        evidence_type,
                }
            )


        return {
            requirement:
                _choose_evidence(
                    values
                )
            for (
                requirement,
                values
            )
            in evidence.items()
        }


    # --------------------------------------------------------
    # Stable person state
    # --------------------------------------------------------

    def _person_state(
        self,
        person,
        detections,
    ):

        track_id = person[
            "track_id"
        ]


        raw = self._raw_state(
            detections
        )


        stable = {}


        for requirement in REQUIREMENTS:

            item = raw[
                requirement
            ]


            stable[
                requirement
            ] = (
                self.temporal.update(
                    track_id,
                    requirement,
                    item[
                        "state"
                    ],
                    item[
                        "confidence"
                    ],
                    item[
                        "evidence_type"
                    ],
                    timestamp=getattr(self, "frame_time_seconds", None),
                )
            )


        states = [
            stable[
                requirement
            ][
                "state"
            ]
            for requirement
            in REQUIREMENTS
        ]


        if (
            "violation"
            in states
        ):

            overall = (
                "violation"
            )

        elif all(
            state == "compliant"
            for state
            in states
        ):

            overall = (
                "compliant"
            )

        else:

            overall = (
                "unknown"
            )


        result = {
            **person,

            # Preserve raw model evidence for diagnostics; cards use stable state.
            "detected_classes": [
                {
                    "class_name": item["evidence_type"],
                    "confidence": item["confidence"],
                }
                for item in raw.values()
                if item["state"] != "unknown"
                and item["evidence_type"] in PPE_RULES
            ],

            "mask":
                stable[
                    "mask"
                ],

            "gloves":
                stable[
                    "gloves"
                ],

            "hair_cover":
                stable[
                    "hair_cover"
                ],

            "apron":
                stable[
                    "apron"
                ],

            "overall":
                overall,
        }


        return result


    # --------------------------------------------------------
    # Violation persistence
    # --------------------------------------------------------

    def _update_violations(
        self,
        persons,
    ):

        for person in persons:

            track_id = person[
                "track_id"
            ]

            staff_label = person[
                "staff_label"
            ]


            for requirement in REQUIREMENTS:

                item = person[
                    requirement
                ]


                if (
                    item[
                        "state"
                    ]
                    ==
                    "violation"
                ):

                    violation_type = (
                        item[
                            "evidence_type"
                        ]
                        or requirement
                    )


                    severity = (
                        SEVERITY_MAP
                        .get(
                            violation_type,
                            "warning",
                        )
                    )


                    STORE.touch_violation(
                        session_id=
                            self.session_id,

                        track_id=
                            track_id,

                        staff_label=
                            staff_label,

                        requirement=
                            requirement,

                        violation_type=
                            violation_type,

                        severity=
                            severity,

                        confidence=
                            item[
                                "confidence"
                            ],
                    )

                else:

                    STORE.close_violation(
                        self.session_id,
                        track_id,
                        requirement,
                    )


    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    def _summary(
        self,
        persons,
    ):

        staff_detected = len(
            persons
        )


        fully_compliant = sum(
            1
            for person in persons
            if person[
                "overall"
            ]
            ==
            "compliant"
        )


        open_violations = sum(
            1
            for person in persons
            for requirement in REQUIREMENTS
            if person[requirement]["state"] == "violation"
        )


        requirements = {}

        known = 0

        compliant = 0


        for requirement in REQUIREMENTS:

            counts = self.requirement_frame_counts[
                requirement
            ]

            compliant_count = counts["compliant"]

            violation_count = counts["violation"]

            unknown_count = counts["unknown"]


            requirement_known = (
                compliant_count
                +
                violation_count
            )

            known += requirement_known

            compliant += compliant_count


            percentage = (
                round(
                    (
                        compliant_count
                        /
                        requirement_known
                    )
                    *
                    100,
                    2,
                )
                if requirement_known
                else None
            )

            # Apron is "supported" only when the model weight is present.
            is_supported = (
                True
                if requirement != "apron"
                else MODELS.apron_model_available()
            )

            requirements[
                requirement
            ] = {

                "supported":
                    is_supported,

                "compliant":
                    compliant_count,

                "violation":
                    violation_count,

                "unknown":
                    unknown_count,

                "known":
                    requirement_known,

                "percentage":
                    percentage,
            }

            if not is_supported:
                requirements[requirement]["status"] = "model_not_loaded"


        score = (
            round(
                (
                    compliant
                    /
                    known
                )
                *
                100,
                2,
            )
            if known
            else None
        )


        return {

            "staff_detected":
                staff_detected,

            "fully_compliant":
                fully_compliant,

            "open_violations":
                open_violations,

            "compliance_score":
                score,

            "requirements":
                requirements,

            "persons":
                persons,
        }


    # --------------------------------------------------------
    # Visualization
    # --------------------------------------------------------

    def _annotate(
        self,
        frame,
        persons,
        ppe,
    ):

        return annotate_people(
            frame,
            persons,
            ppe,
            show_boxes=self.show_detection_boxes,
            show_labels=self.show_labels,
            show_confidence=self.show_confidence,
            thumbnails=getattr(getattr(self, "presence", None), "thumbnails", None),
            show_raw_ppe=os.getenv("KITCHEN_SHOW_RAW_PPE_BOXES", "").lower()
            in {"1", "true", "yes", "on"},
        )


    # --------------------------------------------------------
    # Process frame
    # --------------------------------------------------------

    def process_frame(
        self,
        frame,
        frame_number,
    ):

        self.frame_time_seconds = (
            frame_number / self.source_fps if self.source_type == "upload"
            else time.monotonic() - self.started_monotonic
        )
        previous_time = getattr(self, "last_observation_time", None)
        if previous_time is not None and self.frame_time_seconds > previous_time:
            delta = self.frame_time_seconds - previous_time
            self.observation_period = .5 * getattr(self, "observation_period", delta) + .5 * delta
        self.last_observation_time = self.frame_time_seconds
        persons = self._persons(frame)
        visible_ids = {person["track_id"] for person in persons}
        previous_ids = getattr(self, "previous_visible_ids", set())
        # A slow inference call is not a disappearance. Only expire tracks
        # that are missing now or have actually missed an observation.
        for track_id, last_seen in list(self.track_last_seen.items()):
            interrupted = track_id not in visible_ids or track_id not in previous_ids
            if interrupted and self.frame_time_seconds - last_seen > PERSON_HOLD_SECONDS:
                STORE.close_track_violations(self.session_id, track_id)
                self.temporal.forget(track_id)
                self.track_last_seen.pop(track_id, None)
        self.previous_visible_ids = visible_ids

        # Run both models and merge detections into one list.
        # Assign evidence only when its centre belongs to one person.
        ppe_detections   = self._ppe(frame)
        apron_detections = self._apron(frame)
        all_detections   = ppe_detections + apron_detections


        assignments = self._associate(
            persons,
            all_detections,
        )


        if not hasattr(self, "refiner"):
            self.refiner = PPERefiner()
        extra = self.refiner.refine(
            frame, persons, assignments, self.temporal.confirmed,
            self.frame_time_seconds, self._ppe, self._apron, self._associate)
        if extra:
            all_detections.extend(extra)
            assignments = self._associate(persons, all_detections)

        processed_people = []


        for person in persons:

            track_id = person[
                "track_id"
            ]

            self.track_last_seen[
                track_id
            ] = self.frame_time_seconds


            state = self._person_state(
                person,
                assignments.get(
                    track_id,
                    [],
                ),
            )


            processed_people.append(
                state
            )

            for requirement in REQUIREMENTS:

                req_state = state[
                    requirement
                ][
                    "state"
                ]

                self.requirement_frame_counts[
                    requirement
                ][
                    req_state
                ] += 1


        display_people, _ = self.presence.update(
            processed_people, frame, self.frame_time_seconds)

        self._update_violations(
            processed_people
        )


        summary = self._summary(
            processed_people
        )


        # Keep raw detections available for optional diagnostic boxes.
        annotated = self._annotate(
            frame,
            display_people,
            all_detections,
        )


        return (
            annotated,
            summary,
        )


    def _reset_after_source_gap(self):
        """A long disconnected source cannot establish person continuity."""
        for track_id in self.track_last_seen:
            STORE.close_track_violations(self.session_id, track_id)
        self.track_id_offset = max(self.display_ids, default=self.track_id_offset) + 1
        self.person_model = MODELS.create_person_tracker()
        self.temporal = TemporalState()
        self.admission = TrackAdmission()
        self.presence = PersonPresence()
        self.refiner = PPERefiner()
        self.track_last_seen.clear()
        self.previous_visible_ids = set()
        self.last_observation_time = None

    # --------------------------------------------------------
    # Main source processing
    # --------------------------------------------------------

    def run(
        self,
        source,
    ):

        STORE.update_session(
            self.session_id,
            status="running",
            started_at=utc_now(),
        )


        cap = cv2.VideoCapture(
            source
        )


        if not cap.isOpened():

            raise RuntimeError(
                f"Cannot open source: "
                f"{source}"
            )


        fps = cap.get(
            cv2.CAP_PROP_FPS
        )


        if (
            not fps
            or fps <= 1
        ):

            fps = 25.0


        self.source_fps = fps

        total_frames = int(
            cap.get(
                cv2.CAP_PROP_FRAME_COUNT
            )
            or 0
        )


        width = int(
            cap.get(
                cv2.CAP_PROP_FRAME_WIDTH
            )
        )


        height = int(
            cap.get(
                cv2.CAP_PROP_FRAME_HEIGHT
            )
        )


        output_width, output_height = annotation_size(
            width, height, self.show_labels
        )
        writer = make_writer(
            self.output_video,
            output_width,
            output_height,
            fps,
        )


        STORE.update_session(
            self.session_id,
            fps=fps,
            total_frames=total_frames,
            output_video=str(
                self.output_video
            ),
            latest_frame=str(
                self.latest_frame
            ),
        )


        frame_number = 0

        last_metric_time = 0.0

        last_summary = {}

        reconnect_attempts = 0

        MAX_RECONNECT_ATTEMPTS = 20

        RECONNECT_DELAY_SECONDS = 1.0


        try:

            while (
                not self.stop_event.is_set()
            ):

                ok, frame = cap.read()


                if not ok:

                    if self.source_type == "upload":
                        break

                    if reconnect_attempts == 0:
                        capture_failed_at = time.monotonic()
                    reconnect_attempts += 1

                    if (
                        reconnect_attempts
                        > MAX_RECONNECT_ATTEMPTS
                    ):
                        break

                    cap.release()

                    if self.stop_event.wait(
                        RECONNECT_DELAY_SECONDS
                    ):
                        break

                    cap = cv2.VideoCapture(
                        source
                    )

                    continue


                reconnect_attempts = 0

                frame_number += 1


                (
                    annotated,
                    summary,
                ) = self.process_frame(
                    frame,
                    frame_number,
                )


                last_summary = summary


                writer.write(
                    annotated
                )


                tmp_frame_path = str(
                    self.latest_frame.with_name(
                        self.latest_frame.stem
                        + ".tmp"
                        + self.latest_frame.suffix
                    )
                )

                if not cv2.imwrite(
                    tmp_frame_path,
                    annotated,
                ):
                    raise RuntimeError(
                        f"Failed to write frame preview: {tmp_frame_path}"
                    )

                for attempt in range(5):

                    try:

                        os.replace(
                            tmp_frame_path,
                            str(
                                self.latest_frame
                            ),
                        )

                        break

                    except PermissionError:

                        if attempt == 4:
                            raise

                        time.sleep(0.01)


                now = time.monotonic()


                if (
                    now
                    -
                    last_metric_time
                    >=
                    METRIC_SAMPLE_SECONDS
                ):

                    STORE.add_metric(
                        self.session_id,
                        summary,
                    )

                    last_metric_time = now


                STORE.update_session(
                    self.session_id,
                    processed_frames=
                        frame_number,

                    summary_json=
                        summary,
                )


        finally:

            cap.release()

            writer.release()


        STORE.close_all_violations(
            self.session_id
        )


        final_status = (
            "stopped"
            if self.stop_event.is_set()
            else "completed"
        )


        STORE.update_session(
            self.session_id,
            status=final_status,
            ended_at=utc_now(),
            processed_frames=
                frame_number,
            summary_json=
                last_summary,
        )


        return last_summary
