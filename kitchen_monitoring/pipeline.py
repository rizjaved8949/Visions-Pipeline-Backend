import os
import time

from collections import (
    Counter,
    defaultdict,
    deque,
)

from pathlib import Path

import cv2

from guard_monitoring.io import make_writer

from .config import (
    DEVICE,
    PERSON_IMAGE_SIZE,
    PERSON_CONFIDENCE,
    PERSON_IOU,
    TEMPORAL_WINDOW,
    TEMPORAL_MIN_VOTES,
    CONFLICT_CONFIDENCE_MARGIN,
    TRACK_STALE_FRAMES,
    METRIC_SAMPLE_SECONDS,
    SESSION_DIR,
    SEVERITY_MAP,
)

from .model_registry import MODELS

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
}


REQUIREMENTS = [
    "mask",
    "gloves",
    "hair_cover",
]


# ============================================================
# TEMPORAL SMOOTHER
# ============================================================

class TemporalState:

    def __init__(self):

        self.history = defaultdict(
            lambda: defaultdict(
                lambda: deque(
                    maxlen=TEMPORAL_WINDOW
                )
            )
        )


    def update(
        self,
        track_id,
        requirement,
        state,
        confidence,
        evidence_type,
    ):

        history = (
            self.history[
                track_id
            ][
                requirement
            ]
        )

        history.append(
            {
                "state":
                    state,

                "confidence":
                    confidence,

                "evidence_type":
                    evidence_type,
            }
        )


        known = [
            item
            for item in history
            if item["state"]
            != "unknown"
        ]


        if len(
            known
        ) < TEMPORAL_MIN_VOTES:

            return {
                "state": "unknown",
                "confidence": 0.0,
                "evidence_type": None,
            }


        counts = Counter(
            item["state"]
            for item in known
        )


        state, votes = (
            counts.most_common(
                1
            )[0]
        )


        if votes < TEMPORAL_MIN_VOTES:

            return {
                "state": "unknown",
                "confidence": 0.0,
                "evidence_type": None,
            }


        selected = [
            item
            for item in known
            if item["state"]
            == state
        ]


        confidence = (
            sum(
                item[
                    "confidence"
                ]
                for item
                in selected
            )
            /
            len(selected)
        )


        evidence_types = Counter(
            item[
                "evidence_type"
            ]
            for item
            in selected
            if item[
                "evidence_type"
            ]
        )


        evidence_type = (
            evidence_types
            .most_common(1)[0][0]
            if evidence_types
            else None
        )


        return {
            "state":
                state,

            "confidence":
                round(
                    confidence,
                    4,
                ),

            "evidence_type":
                evidence_type,
        }


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


    # If model gives two opposite states at nearly
    # the same confidence, don't invent certainty.
    if len(ordered) >= 2:

        second = ordered[1]

        if (
            second["state"]
            != best["state"]
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


        self.display_ids = {}

        self.next_display_id = 1


        self.track_last_seen = {}

        # Cumulative, session-wide per-requirement frame tallies. _summary()
        # used to compute the "compliance by requirement" breakdown purely
        # from the *current* frame's tracked persons, which meant it briefly
        # reverted to "no data" every time temporal smoothing re-evaluated a
        # track (or a track was lost/reacquired) even though real violations
        # had already been confirmed and logged. Tallying every frame here
        # instead gives a stable, session-long view.
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
                tracker="bytetrack.yaml",
                device=DEVICE,
                verbose=False,
            )[0]
        )


        people = []


        if result.boxes is None:

            return people


        for index, box in enumerate(
            result.boxes
        ):

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

                # Fallback only if ByteTrack has not yet
                # assigned an ID.
                track_id = (
                    100000 + index
                )


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

                    "staff_label":
                        self._staff_label(
                            track_id
                        ),

                    "bbox":
                        bbox,

                    "person_confidence":
                        round(
                            confidence,
                            4,
                        ),
                }
            )


        return people


    # --------------------------------------------------------
    # PPE detection
    # --------------------------------------------------------

    def _ppe(
        self,
        frame,
    ):

        result = MODELS.predict_ppe(
            frame
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


            if not candidates:

                continue


            # In overlapping people, choose smallest
            # containing person bbox.
            person = min(
                candidates,
                key=lambda item:
                    _bbox_area(
                        item[
                            "bbox"
                        ]
                    ),
            )


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


        # "Open violations" is deliberately about right now, from the
        # current persons snapshot (matches its "needs supervisor" framing).
        open_violations = sum(
            1
            for person in persons
            for requirement in REQUIREMENTS
            if person[requirement]["state"] == "violation"
        )


        # "Compliance by requirement" and the overall score, on the other
        # hand, use the session-wide cumulative tallies (self.requirement_
        # frame_counts, updated every frame in process_frame) rather than
        # just this instant's persons - otherwise a track reset or a brief
        # temporal-smoothing gap makes an already-confirmed violation revert
        # to "no data yet" even though it's still sitting in the violations
        # log.
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


            requirements[
                requirement
            ] = {

                "supported":
                    True,

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


        # Your current trained model has no apron class.
        requirements[
            "apron"
        ] = {
            "supported": False,
            "status": "unavailable",
        }


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

        output = frame.copy()


        # PPE detections
        for detection in ppe:

            x1, y1, x2, y2 = [
                int(v)
                for v in detection[
                    "bbox"
                ]
            ]


            cv2.rectangle(
                output,
                (
                    x1,
                    y1,
                ),
                (
                    x2,
                    y2,
                ),
                (
                    230,
                    180,
                    50,
                ),
                1,
            )


            text = (
                f"{detection['class_name']} "
                f"{detection['confidence']:.2f}"
            )


            cv2.putText(
                output,
                text,
                (
                    x1,
                    max(
                        18,
                        y1 - 5,
                    ),
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.42,
                (
                    230,
                    180,
                    50,
                ),
                1,
                cv2.LINE_AA,
            )


        # Person states
        for person in persons:

            x1, y1, x2, y2 = [
                int(v)
                for v in person[
                    "bbox"
                ]
            ]


            overall = person[
                "overall"
            ]


            if overall == "compliant":

                color = (
                    70,
                    180,
                    70,
                )

            elif overall == "violation":

                color = (
                    50,
                    50,
                    230,
                )

            else:

                color = (
                    0,
                    180,
                    230,
                )


            cv2.rectangle(
                output,
                (
                    x1,
                    y1,
                ),
                (
                    x2,
                    y2,
                ),
                color,
                2,
            )


            label = (
                f"{person['staff_label']} "
                f"- {overall.upper()}"
            )


            cv2.putText(
                output,
                label,
                (
                    x1,
                    max(
                        22,
                        y1 - 26,
                    ),
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                color,
                2,
                cv2.LINE_AA,
            )


            def symbol(
                requirement,
            ):

                state = (
                    person[
                        requirement
                    ][
                        "state"
                    ]
                )

                if state == "compliant":
                    return "OK"

                if state == "violation":
                    return "NO"

                return "?"


            details = (
                f"M:{symbol('mask')} "
                f"G:{symbol('gloves')} "
                f"H:{symbol('hair_cover')}"
            )


            cv2.putText(
                output,
                details,
                (
                    x1,
                    max(
                        42,
                        y1 - 7,
                    ),
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                color,
                1,
                cv2.LINE_AA,
            )


        return output


    # --------------------------------------------------------
    # Process frame
    # --------------------------------------------------------

    def process_frame(
        self,
        frame,
        frame_number,
    ):

        persons = self._persons(
            frame
        )


        ppe = self._ppe(
            frame
        )


        assignments = self._associate(
            persons,
            ppe,
        )


        processed_people = []


        current_ids = set()


        for person in persons:

            track_id = person[
                "track_id"
            ]

            current_ids.add(
                track_id
            )

            self.track_last_seen[
                track_id
            ] = frame_number


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


        # Clean stale tracks
        for track_id in list(
            self.track_last_seen
        ):

            last_seen = (
                self.track_last_seen[
                    track_id
                ]
            )


            if (
                frame_number
                -
                last_seen
                >
                TRACK_STALE_FRAMES
            ):

                STORE.close_track_violations(
                    self.session_id,
                    track_id,
                )

                self.track_last_seen.pop(
                    track_id,
                    None,
                )


        self._update_violations(
            processed_people
        )


        summary = self._summary(
            processed_people
        )


        annotated = self._annotate(
            frame,
            processed_people,
            ppe,
        )


        return (
            annotated,
            summary,
        )


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


        # mp4v (MPEG-4 Part 2) is not decodable by Chrome, Edge or Firefox -
        # the browser <video> element shows a black frame with 0:00 duration
        # even though the file itself is valid. make_writer() tries real
        # H.264 first (falling back to mp4v only if that's unavailable),
        # same fix already applied for Guard's job output.
        writer = make_writer(
            self.output_video,
            width,
            height,
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
                        # End of file - the correct, expected way an
                        # uploaded video finishes.
                        break

                    # Live camera/RTSP: a single failed read is usually
                    # transient (USB hiccup, exposure change, brief signal
                    # drop) - reconnect instead of ending the whole session
                    # on the first glitch.
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


                # Write-then-rename so the stream endpoint (reading this same
                # path from a different thread, on a timer) never opens a
                # half-written file. cv2.imwrite() writing the real path
                # directly let the reader occasionally catch a truncated
                # JPEG mid-write, which fails to decode in the browser and
                # kills the whole <img> stream even though this loop and the
                # backend session keep running fine underneath.
                # Must still end in .jpg - cv2.imwrite picks its encoder from
                # the file extension, so a plain "+.tmp" suffix (ending in
                # ".tmp", not ".jpg") makes it silently fail to write.
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

                # On Windows, os.replace() can transiently fail with
                # PermissionError if the stream endpoint's reader has the
                # destination file open at that exact instant (its read is a
                # brief open+read+close, not a long-held lock). Retry a few
                # times rather than letting one unlucky collision fail the
                # whole session.
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