from __future__ import annotations

import json
import os
import time
from dataclasses import asdict
from pathlib import Path
from typing import Callable

import cv2

from .contracts import ModuleResult, ModuleStatus, disabled, error_result, ok, unknown
from .events import JsonlWriter
from .diagnostics import BUILD_ID, effective_config, runtime_info
from .geometry import normalized_polygon_to_pixels
from .health import ModuleHealthRegistry, safe_run
from .io import make_writer, open_capture, video_metadata
from .model_registry import LazyModelRegistry
from .monitoring.guard_selector import GuardSelector
from .monitoring.activity import ActivityStabilizer
from .monitoring.camera_motion import CameraMotionEstimator
from .monitoring.movement import MovementMonitor
from .monitoring.phone_use import PhoneUseAnalyzer
from .monitoring.posture import PostureAnalyzer
from .monitoring.presence import PresenceState, presence_from_tracker
from .monitoring.rules import TimedRuleEngine
from .monitoring.sleep import SleepAnalyzer
from .types import EyeState, MovementState, PhoneState, PostureState, SleepState
from .visualization.overlay import draw_activity_status, draw_guard_identity

ProgressCallback = Callable[[dict], None]


class GuardMonitoringPipeline:
    """Fault-isolated guard monitoring orchestrator.

    Heavy AI models are lazy-loaded through LazyModelRegistry. Optional monitor errors
    become ERROR/UNKNOWN module states and do not terminate unrelated monitors. Guard
    detection/tracking remain foundational: if they are unavailable, guard-specific
    downstream evidence is marked UNKNOWN rather than fabricated.
    """

    def __init__(
        self,
        cfg: dict,
        *,
        registry: LazyModelRegistry | None = None,
        progress_callback: ProgressCallback | None = None,
    ):
        self.cfg = cfg
        self.camera_id = str(cfg["project"].get("camera_id", "camera-01"))
        self.health = ModuleHealthRegistry()
        self.registry = registry or LazyModelRegistry(cfg, health=self.health)
        self.progress_callback = progress_callback

        pose_cfg = cfg["models"]["pose"]
        self.pose_stride = max(1, int(pose_cfg.get("every_n_frames", 2)))
        self.pose_cache_max = max(0, int(pose_cfg.get("cache_max_frames", self.pose_stride + 1)))

        phone_cfg = cfg["models"]["phone"]
        self.phone_stride = max(1, int(phone_cfg.get("every_n_frames", 2)))
        self.phone_cache_max = max(0, int(phone_cfg.get("cache_max_frames", self.phone_stride + 1)))

        eyes_cfg = cfg["models"]["eyes"]
        self.eye_stride = max(1, int(eyes_cfg.get("every_n_frames", 2)))
        self.eye_cache_max = max(0, int(eyes_cfg.get("cache_max_frames", self.eye_stride + 1)))

        self.posture_analyzer = PostureAnalyzer(cfg["pose_logic"])
        self.phone_analyzer = PhoneUseAnalyzer(
            cfg["phone_logic"],
            keypoint_confidence=cfg["pose_logic"].get("keypoint_confidence", 0.25),
        )
        sleep_cfg = dict(cfg["sleep_logic"])
        sleep_cfg.setdefault("torso_lean_deg", cfg["pose_logic"].get("torso_lean_deg", 28.0))
        self.sleep_analyzer = SleepAnalyzer(sleep_cfg)

        self.activity = ActivityStabilizer(cfg.get("activity", {}))
        self.camera_motion = CameraMotionEstimator(cfg.get("camera_motion", {}))
        self.last_identity = None
        self.last_track_id = None
        self.last_pose: ModuleResult = unknown("pose_not_run")
        self.last_pose_frame = -10**9
        self.last_phone_detections: ModuleResult = unknown("phone_not_run")
        self.last_phone_frame = -10**9
        self.last_eye: ModuleResult = unknown("eyes_not_run")
        self.last_eye_frame = -10**9
        self.last_pose_at = self.last_phone_at = self.last_eye_at = None
        self.last_eye_timestamp_ms = -1

    def run(self) -> dict:
        source = str(self.cfg["input"]["source"])
        cap = open_capture(source)
        width, height, fps = video_metadata(cap)
        self.video_fps = fps
        raw_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

        output_cfg = self.cfg["output"]
        out_dir = Path(output_cfg["directory"])
        out_dir.mkdir(parents=True, exist_ok=True)
        video_path = out_dir / output_cfg.get("annotated_video", "annotated.mp4")
        frame_log_path = out_dir / output_cfg.get("frame_log", "frames.jsonl")
        event_log_path = out_dir / output_cfg.get("event_log", "events.jsonl")
        summary_path = out_dir / output_cfg.get("summary", "summary.json")
        health_path = out_dir / output_cfg.get("module_health", "module_health.json")
        display = bool(output_cfg.get("display", False))
        max_frames = self.cfg["input"].get("max_frames")
        total_frames = raw_total
        if max_frames is not None:
            total_frames = min(raw_total, int(max_frames)) if raw_total > 0 else int(max_frames)

        config_path = out_dir / "runtime_config.json"
        config_path.write_text(json.dumps({"build_id": BUILD_ID, "effective_config": effective_config(self.cfg)}, indent=2), encoding="utf-8")
        writer = make_writer(video_path, width, height, fps)
        self.latest_frame = out_dir / "latest.jpg"

        duty_zone = normalized_polygon_to_pixels(self.cfg["guard_selection"]["duty_zone"], width, height)
        patrol_zones = [
            (z["name"], normalized_polygon_to_pixels(z["polygon"], width, height))
            for z in self.cfg["movement"].get("patrol_zones", [])
        ]
        sel_cfg = self.cfg["guard_selection"]
        selector = GuardSelector(
            duty_zone,
            confirm_seconds=sel_cfg.get("confirm_seconds", 1.0),
            release_seconds=sel_cfg.get("release_seconds", 5.0),
            presence_grace_seconds=sel_cfg.get("presence_grace_seconds", 2.0),
            manual_track_id=sel_cfg.get("manual_track_id"),
            candidate_gap_seconds=max(sel_cfg.get("candidate_gap_seconds", 0.5), 1.5 / fps),
        )
        move_cfg = self.cfg["movement"]
        movement_monitor = MovementMonitor(
            history_seconds=move_cfg.get("history_seconds", 5.0),
            stationary_radius_ratio=move_cfg.get("stationary_radius_ratio", 0.035),
            minimum_history_seconds=move_cfg.get("minimum_history_seconds", 2.0),
            patrol_zones=patrol_zones,
            smoothing_seconds=move_cfg.get("smoothing_seconds", 0.25),
            radius_quantile=move_cfg.get("radius_quantile", 0.90),
            max_gap_seconds=move_cfg.get("max_gap_seconds", 2.0),
            border_margin_ratio=move_cfg.get("border_margin_ratio", 0.015),
            max_box_scale_change=move_cfg.get("max_box_scale_change", 0.25),
        )
        rules = TimedRuleEngine(self.camera_id, self.cfg["rules"])

        frame_count = 0
        guard_visible_frames = 0
        phone_detected_frames = 0
        phone_use_frames = 0
        sleep_candidate_frames = 0
        stationary_frames = 0
        absence_frames = 0
        events_count = 0
        frame_errors = 0
        started_wall = time.time()
        continue_on_frame_error = bool(
            self.cfg.get("fault_tolerance", {}).get("continue_on_frame_error", True)
        )

        self._progress("starting", 0, total_frames)

        with JsonlWriter(frame_log_path) as frame_log, JsonlWriter(event_log_path) as event_log:
            try:
                while True:
                    decode_started = time.perf_counter()
                    ok_read, frame = cap.read()
                    self.health.record("video_decode", ok(ok_read), (time.perf_counter()-decode_started)*1000.0)
                    if not ok_read:
                        break
                    if max_frames is not None and frame_count >= int(max_frames):
                        break

                    now = frame_count / fps
                    try:
                        result = self._process_frame(
                            frame=frame,
                            frame_idx=frame_count,
                            now=now,
                            selector=selector,
                            movement_monitor=movement_monitor,
                            rules=rules,
                            duty_zone=duty_zone,
                            patrol_zones=patrol_zones,
                        )
                    except Exception as exc:
                        # Last-resort frame isolation. Normal module errors should already
                        # be contained by safe_run; this catches orchestration bugs.
                        frame_errors += 1
                        rules.pause_all(now)
                        self.sleep_analyzer.pause(now)
                        self.activity.update(None, now, present=None)
                        fatal_result = error_result(exc)
                        self.health.record("frame_pipeline", fatal_result)
                        if not continue_on_frame_error:
                            raise
                        writer.write(frame)
                        self._write_latest_frame(frame)
                        frame_log.write(
                            {
                                "frame": frame_count,
                                "time_seconds": now,
                                "frame_status": "error",
                                "error": type(exc).__name__,
                                "detail": str(exc),
                            }
                        )
                        frame_count += 1
                        self._maybe_progress(frame_count, total_frames)
                        continue

                    guard_visible_frames += int(result["guard_seen_now"])
                    phone_detected_frames += int(result["phone"].detected)
                    phone_use_frames += int(result["phone"].usage in {"call", "screen_use"})
                    sleep_candidate_frames += int(result["sleep"].candidate)
                    movement_data = result["frame_log"].get("movement") or {}
                    stationary_frames += int(bool(movement_data.get("stationary")))
                    absence_frames += int(result["frame_log"].get("present") is False)
                    events_count += result["events_count"]

                    for event in result["events"]:
                        event_log.write(event)

                    encode_started = time.perf_counter()
                    writer.write(result["annotated"])
                    self.health.record("video_encode", ok(None), (time.perf_counter()-encode_started)*1000.0)
                    preview_started = time.perf_counter()
                    self._write_latest_frame(result["annotated"])
                    self.health.record("preview_write", ok(None), (time.perf_counter()-preview_started)*1000.0)
                    frame_log.write(result["frame_log"])

                    if display:
                        cv2.imshow("Guard Monitoring", result["annotated"])
                        if cv2.waitKey(1) & 0xFF == ord("q"):
                            frame_count += 1
                            break

                    frame_count += 1
                    self._maybe_progress(frame_count, total_frames)
            finally:
                cap.release()
                writer.release()
                self.close()
                self.latest_frame.unlink(missing_ok=True)
                if display:
                    cv2.destroyAllWindows()

        health_snapshot = self.health.snapshot()
        health_path.write_text(json.dumps(health_snapshot, indent=2), encoding="utf-8")

        wall_seconds = time.time() - started_wall
        summary = {
            "build_id": BUILD_ID,
            "processing_fps": round(frame_count / max(wall_seconds, 1e-9), 4),
            "video_seconds": round(frame_count / fps, 3),
            "effective_config": effective_config(self.cfg),
            "runtime": runtime_info(self.registry),
            "stage_timings_ms": self.health.timing_totals(),
            "runtime_config": str(config_path),
            "camera_id": self.camera_id,
            "source": source,
            "fps": fps,
            "resolution": [width, height],
            "frames_processed": frame_count,
            "frame_errors": frame_errors,
            "guard_visible_frames": guard_visible_frames,
            "phone_detected_frames": phone_detected_frames,
            "phone_use_frames": phone_use_frames,
            "sleep_candidate_frames": sleep_candidate_frames,
            "stationary_frames": stationary_frames,
            "absence_frames": absence_frames,
            "events_triggered": events_count,
            "processing_wall_seconds": round(wall_seconds, 3),
            "output_video": str(video_path),
            "frame_log": str(frame_log_path),
            "event_log": str(event_log_path),
            "module_health": str(health_path),
        }
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        self._progress("completed", frame_count, total_frames, summary=summary)
        return summary

    def _write_latest_frame(self, frame) -> None:
        """Write-then-rename so a concurrent MJPEG reader (see
        /jobs/{job_id}/stream) never opens a half-written JPEG - the same
        race fixed for Kitchen's live preview. Must still end in .jpg since
        cv2.imwrite picks its encoder from the file extension."""
        tmp_path = str(self.latest_frame.with_name(self.latest_frame.stem + ".tmp" + self.latest_frame.suffix))
        if not cv2.imwrite(tmp_path, frame):
            return
        for attempt in range(5):
            try:
                os.replace(tmp_path, str(self.latest_frame))
                break
            except PermissionError:
                if attempt == 4:
                    return
                time.sleep(0.01)

    def _process_frame(
        self,
        *,
        frame,
        frame_idx: int,
        now: float,
        selector: GuardSelector,
        movement_monitor: MovementMonitor,
        rules: TimedRuleEngine,
        duty_zone,
        patrol_zones,
    ) -> dict:
        frame_started = time.perf_counter()
        timings_before = self.health.timing_totals()
        module_results: dict[str, ModuleResult] = {}

        # STEP 1 - detector
        det_model = self.registry.guard_detector()
        if det_model.ok:
            detections = safe_run(
                "guard_detection",
                det_model.value.predict,
                frame,
                health=self.health,
            )
        else:
            detections = self._propagate("guard_detection", det_model)
        module_results["guard_detection"] = detections

        boxes = getattr(detections.value, "xyxy", []) if detections.ok else []
        camera_started = time.perf_counter()
        camera_motion = safe_run("camera_motion", self.camera_motion.update, frame, boxes, now)
        camera_state = camera_motion.value if camera_motion.ok else self.camera_motion.state("error")
        if camera_motion.ok and not camera_state.get("enabled"):
            camera_motion = disabled("camera_motion_disabled")
        elif camera_motion.ok and not camera_state.get("valid"):
            camera_motion = unknown(camera_state.get("reason", "unresolved"))
        self.health.record("camera_motion", camera_motion, (time.perf_counter()-camera_started)*1000.0)
        module_results["camera_motion"] = camera_motion
        scene_change = bool(camera_state.get("scene_change"))
        if scene_change:
            # A cut makes identity across the discontinuity unverified.
            self.registry.reset_tracker()
            selector.reset()
            self._reset_guard_dependent_state(None)
            self.last_identity = None
            movement_monitor.reset(None)
            rules.reset_guard(None)
        elif camera_state.get("valid"):
            movement_monitor.compensate(camera_state["affine"])
        elif not self.cfg["movement"].get("assume_static_camera", False):
            movement_monitor.invalidate()

        # STEP 2 - tracker. Tracking is not run when detection evidence is unavailable.
        if detections.ok:
            tracker_model = self.registry.tracker(frame_rate=getattr(self, "video_fps", 30.0))
            if tracker_model.ok:
                tracked = safe_run(
                    "tracking",
                    tracker_model.value.update,
                    detections.value,
                    timestamp=now,
                    health=self.health,
                )
            else:
                tracked = self._propagate("tracking", tracker_model)
        else:
            tracked = self._unknown_recorded("tracking", "guard_detection_unavailable")
        module_results["tracking"] = tracked

        # Guard selection/presence are only considered known when tracking ran.
        if tracked.ok:
            guard_result = safe_run(
                "guard_selection",
                selector.update,
                tracked.value,
                now,
                health=self.health,
            )
        else:
            guard_result = self._unknown_recorded("guard_selection", "tracking_unavailable")
        module_results["guard_selection"] = guard_result

        if guard_result.ok:
            presence_state = presence_from_tracker(selector, True, now)
            presence_result = ok(presence_state)
            self.health.record("presence", presence_result)
        else:
            presence_result = self._unknown_recorded("presence", "guard_selection_unavailable")
        module_results["presence"] = presence_result

        guard = guard_result.value if guard_result.ok else selector.retained(now)
        present = presence_result.value.present if presence_result.ok else None
        if scene_change:
            # Do not count a camera cut as absence or as observed rule duration.
            present = None

        identity = (guard.track_id, selector.generation) if guard is not None else None
        if identity is not None and identity != self.last_identity:
            self._reset_guard_dependent_state(guard.track_id)
            self.last_identity = identity
            movement_monitor.reset(guard.track_id)
            rules.reset_guard(guard.track_id)
        elif guard is None and selector.active_id is None and self.last_identity is not None:
            self._reset_guard_dependent_state(None)
            self.last_identity = None
            movement_monitor.reset(None)
            rules.reset_guard(None)

        # Defaults are neutral/unknown values. Status metadata determines whether a
        # rule is allowed to consume them.
        movement = MovementState()
        posture = PostureState()
        phone = PhoneState()
        eye = EyeState(reason="no_visible_guard")
        sleep = SleepState()

        movement_result = unknown("no_current_guard")
        pose_result = unknown("no_current_guard")
        posture_result = unknown("pose_unavailable")
        phone_detection_result = unknown("no_current_guard")
        phone_use_result = unknown("no_current_guard")
        eye_result = unknown("no_current_guard")
        sleep_result = unknown("no_current_guard")

        guard_seen_now = bool(guard is not None and guard.seen_now)
        if guard_seen_now:
            # STEP 3 - movement/patrol
            movement_result = safe_run(
                "movement",
                movement_monitor.update,
                guard.track_id,
                guard.bbox,
                now,
                frame_shape=frame.shape,
                camera_state=camera_state,
                assume_static_camera=self.cfg["movement"].get("assume_static_camera", False),
                health=self.health,
            )
            if movement_result.ok:
                movement = movement_result.value

            # STEP 4 - pose. A pose failure does not stop phone detection or eyes.
            pose_result = self._get_pose(frame, guard.bbox, frame_idx, now)
            pose_obs = pose_result.value if pose_result.ok else None
            if pose_result.ok and pose_obs is not None:
                posture_result = safe_run(
                    "posture",
                    self.posture_analyzer.analyze,
                    pose_obs.keypoints,
                    health=self.health,
                )
                if posture_result.ok:
                    posture = posture_result.value
            elif pose_result.status is ModuleStatus.DISABLED:
                posture_result = self._disabled_recorded("posture")
            else:
                posture_result = self._unknown_recorded("posture", "pose_unavailable")

            # STEP 5 - phone detector is independent from pose; pose only improves the
            # call/screen-use association heuristic.
            phone_detection_result = self._get_phone_detections(frame, guard.bbox, frame_idx, now)
            if phone_detection_result.ok:
                phone_use_result = safe_run(
                    "phone_use",
                    self.phone_analyzer.analyze,
                    phone_detection_result.value,
                    guard.bbox,
                    pose_obs.keypoints if pose_obs is not None else None,
                    health=self.health,
                )
                if phone_use_result.ok:
                    phone = phone_use_result.value
            elif phone_detection_result.status is ModuleStatus.DISABLED:
                phone_use_result = self._disabled_recorded("phone_use")
            else:
                phone_use_result = self._unknown_recorded("phone_use", "phone_detector_unavailable")

            # STEP 6 eye evidence can use pose keypoints when available but has a guard
            # box fallback, so pose failure does not automatically disable eye analysis.
            eye_result = self._get_eye_state(
                frame,
                guard.bbox,
                pose_obs.keypoints if pose_obs is not None else None,
                frame_idx,
                now,
            )
            if eye_result.ok:
                eye = eye_result.value

            availability = {
                "movement": movement_result.ok,
                "posture": posture_result.ok,
                "eyes": eye_result.ok,
                "phone": phone_use_result.ok,
            }
            sleep_result = safe_run(
                "sleep",
                self.sleep_analyzer.update,
                guard.track_id,
                now,
                eye,
                posture,
                movement,
                phone,
                availability,
                health=self.health,
            )
            if sleep_result.ok:
                sleep = sleep_result.value

        elif present is False:
            # Do not carry stale per-guard perception into a confirmed absence episode.
            self.last_pose = unknown("guard_absent")
            self.last_phone_detections = unknown("guard_absent")
            self.last_eye = unknown("guard_absent")

        if not guard_seen_now:
            self.sleep_analyzer.pause(now)
            # Re-observation must use a fresh crop, not a crop from before the gap.
            self.last_pose = unknown("tracking_gap")
            self.last_phone_detections = unknown("tracking_gap")
            self.last_eye = unknown("tracking_gap")
            for module_name, module_result in (
                ("movement", movement_result),
                ("pose", pose_result),
                ("posture", posture_result),
                ("phone_detection", phone_detection_result),
                ("phone_use", phone_use_result),
                ("eyes", eye_result),
                ("sleep", sleep_result),
            ):
                self.health.record(module_name, module_result)

        module_results.update(
            {
                "movement": movement_result,
                "pose": pose_result,
                "posture": posture_result,
                "phone_detection": phone_detection_result,
                "phone_use": phone_use_result,
                "eyes": eye_result,
                "sleep": sleep_result,
            }
        )

        # STEP 7 - tri-state conditions. Unknown module evidence pauses a timer and
        # cannot be mistaken for positive evidence or for absence.
        allow_degraded_sleep = bool(
            self.cfg.get("sleep_logic", {}).get("allow_degraded_rule_trigger", False)
        )
        sleep_condition = None
        if present is True and sleep_result.ok:
            if sleep.candidate and (sleep.evidence_quality == "high" or allow_degraded_sleep):
                sleep_condition = True
            elif not sleep.candidate and sleep.evidence_quality == "high":
                sleep_condition = False
            # Insufficient/disabled evidence is unknown, not evidence of wakefulness.
        elif present is False:
            sleep_condition = False

        phone_condition = None
        if present is True and phone_use_result.ok:
            if phone.usage in {"call", "screen_use"}:
                phone_condition = True
            elif phone.detected and phone.usage == "visible" and not posture_result.ok:
                phone_condition = None
            else:
                phone_condition = False
        elif present is False:
            phone_condition = False

        stationary_condition = None
        if present is True and movement_result.ok and movement.reliable:
            stationary_condition = bool(movement.stationary)
        elif present is False:
            stationary_condition = False

        absence_condition = None if present is None else bool(not present)

        rule_payload = self._rule_payload(present, movement, posture, phone, eye, sleep, module_results)
        rule_conditions: dict[str, bool | None] = {
            "sleep": sleep_condition,
            "phone": phone_condition,
            "stationary": stationary_condition,
            "absence": absence_condition,
        }

        events: list[dict] = []
        for name, condition in rule_conditions.items():
            event_result = safe_run(
                f"rule_{name}",
                rules.update,
                name,
                condition,
                now,
                selector.active_id,
                rule_payload,
                health=self.health,
            )
            if event_result.ok and event_result.value is not None:
                events.append(event_result.value.to_dict())

        candidate = None
        if guard_seen_now:
            if phone_condition is True:
                candidate = "Using Mobile"
            elif sleep_condition is True:
                candidate = "Sleeping"
            elif stationary_condition is not None:
                # Activity meaning for a guard is duty behavior, not only box jitter:
                # - standing guard = active duty -> Moving
                # - walking patrol = Moving
                # - sitting/holding position = Stationary
                # Sleep and phone have already been handled above.
                waiting = ((self.activity.label == "Sleeping" and sleep_condition is None
                            and stationary_condition is True)
                           or (self.activity.label == "Using Mobile" and phone_condition is None))
                if not waiting:
                    if posture.posture == "standing":
                        candidate = "Moving"
                    elif posture.posture == "sitting" and movement.reliable:
                        candidate = "Stationary"
                    else:
                        candidate = "Stationary" if stationary_condition else "Moving"
        activity = self.activity.update(candidate, now, present=present)
        annotated = frame.copy()
        rendering = safe_run(
            "visualization", self._draw,
            annotated,
            tracked.value if tracked.ok else None,
            guard,
            duty_zone,
            patrol_zones,
            posture,
            movement,
            phone,
            eye,
            sleep,
            rules,
            module_results,
            present,
            health=self.health,
        )
        module_results["visualization"] = rendering

        frame_log = {
            "build_id": BUILD_ID,
            "guard_bbox": list(guard.bbox) if guard is not None else None,
            "guard_confidence": guard.confidence if guard is not None else None,
            "camera_motion": camera_state,
            "activity_candidate": candidate,
            "sensor_timestamps": {"pose": self.last_pose_at, "phone": self.last_phone_at, "eyes": self.last_eye_at},
            "pose_keypoints": (pose_result.value.keypoints.tolist() if pose_result.ok and pose_result.value is not None else None),
            "timings_ms": self.health.timings_since(timings_before),
            "frame_processing_ms": round((time.perf_counter()-frame_started)*1000.0, 3),
            "frame": frame_idx,
            "time_seconds": now,
            "frame_status": "ok",
            "guard_track_id": selector.active_id,
            "guard_seen_now": guard_seen_now,
            "guard_observation": "observed" if guard_seen_now else ("retained" if guard else "missing"),
            "activity_status": activity["label"],
            "activity": activity,
            "rule_durations_seconds": {
                name: rules.active_duration(name, now) for name in rule_conditions
            },
            "present": present,
            "movement": asdict(movement),
            "posture": asdict(posture),
            "phone": asdict(phone),
            "eyes": asdict(eye),
            "sleep": asdict(sleep),
            "module_status": {
                name: self._status_dict(result) for name, result in module_results.items()
            },
            "rule_conditions": rule_conditions,
            "rules_triggered": {
                name: rules.is_triggered(name) for name in rule_conditions
            },
        }

        return {
            "annotated": annotated,
            "frame_log": frame_log,
            "events": events,
            "events_count": len(events),
            "guard_seen_now": guard_seen_now,
            "phone": phone,
            "sleep": sleep,
        }

    def _reset_guard_dependent_state(self, track_id):
        self.last_track_id = track_id
        self.last_pose = unknown("new_guard")
        self.last_pose_frame = -10**9
        self.last_phone_detections = unknown("new_guard")
        self.last_phone_frame = -10**9
        self.last_eye = unknown("new_guard")
        self.last_eye_frame = -10**9
        self.sleep_analyzer.reset(track_id)
        self.activity.reset()
        self.last_pose_at = self.last_phone_at = self.last_eye_at = None

    def _sample_sensor(self, name, frame_idx, now, invoke) -> ModuleResult:
        cfg_name = "eyes" if name == "eye" else name
        config = self.cfg["models"][cfg_name]
        result_name = {"pose": "last_pose", "phone": "last_phone_detections", "eye": "last_eye"}[name]
        health_name = {"pose": "pose", "phone": "phone_detection", "eye": "eyes"}[name]
        load = getattr(self.registry, cfg_name)()
        if not load.ok:
            return self._propagate(health_name, load)
        previous = getattr(self, result_name)
        last_time = getattr(self, f"last_{name}_at")
        last_frame = getattr(self, f"last_{name}_frame")
        stride = getattr(self, f"{name}_stride")
        max_frames = getattr(self, f"{name}_cache_max")
        expired = (last_time is None or now - last_time > config.get("cache_max_seconds", 0.75)
                   or frame_idx - last_frame > max_frames)
        if frame_idx % stride == 0 or not previous.ok or expired:
            result = safe_run(health_name, invoke, load.value, health=self.health)
            if name == "eye" and result.ok and result.value is not None:
                result.value.observed_at = now
            setattr(self, result_name, result)
            setattr(self, f"last_{name}_frame", frame_idx)
            setattr(self, f"last_{name}_at", now)
            return result
        return previous

    def _get_pose(self, frame, guard_box, frame_idx, now=None) -> ModuleResult:
        now = frame_idx / max(1.0, getattr(self, "video_fps", 30.0)) if now is None else now
        return self._sample_sensor("pose", frame_idx, now,
                                   lambda model: model.predict(frame, guard_box))

    def _get_phone_detections(self, frame, guard_box, frame_idx, now=None) -> ModuleResult:
        now = frame_idx / max(1.0, getattr(self, "video_fps", 30.0)) if now is None else now
        return self._sample_sensor("phone", frame_idx, now,
                                   lambda model: model.predict(frame, guard_box))

    def _get_eye_state(self, frame, guard_box, keypoints, frame_idx, now=None) -> ModuleResult:
        now = frame_idx / max(1.0, getattr(self, "video_fps", 30.0)) if now is None else now
        def invoke(model):
            timestamp = max(self.last_eye_timestamp_ms + 1, int(now * 1000.0))
            self.last_eye_timestamp_ms = timestamp
            return model.analyze(frame, guard_box, keypoints, timestamp)
        return self._sample_sensor("eye", frame_idx, now, invoke)

    def reset_stream(self):
        """A reconnection begins a new observation episode, never a stale one."""
        if hasattr(self.registry, "reset_tracker"):
            self.registry.reset_tracker()
        if hasattr(self.registry, "close"):
            self.registry.close()
        self._reset_guard_dependent_state(None)
        self.last_identity = None
        self.camera_motion.reset()
        self.last_eye_timestamp_ms = -1


    def close(self):
        if hasattr(self.registry, "close"):
            self.registry.close()

    def _propagate(self, module_name: str, source: ModuleResult) -> ModuleResult:
        result = ModuleResult(
            status=source.status,
            value=None,
            error=source.error,
            detail=source.detail,
        )
        self.health.record(module_name, result)
        return result

    def _unknown_recorded(self, name: str, detail: str) -> ModuleResult:
        result = unknown(detail)
        self.health.record(name, result)
        return result

    def _disabled_recorded(self, name: str) -> ModuleResult:
        result = disabled()
        self.health.record(name, result)
        return result

    @staticmethod
    def _status_dict(result: ModuleResult) -> dict:
        return {
            "status": result.status.value,
            "error": result.error,
            "detail": result.detail,
        }

    @staticmethod
    def _rule_payload(present, movement, posture, phone, eye, sleep, module_results):
        return {
            "present": present,
            "movement": asdict(movement),
            "posture": asdict(posture),
            "phone": asdict(phone),
            "eyes": asdict(eye),
            "sleep": asdict(sleep),
            "module_status": {
                name: GuardMonitoringPipeline._status_dict(result)
                for name, result in module_results.items()
            },
        }

    def _draw(self, frame, tracked, guard, duty_zone, patrol_zones, posture,
              movement, phone, eye, sleep, rules, module_results, present):
        # Drawing reads state only; it never changes observations or event timers.
        if guard is not None and present is True:
            viz = self.cfg.get("visualization", {})
            draw_guard_identity(frame, guard, show_box=viz.get("draw_guard_box", True),
                                show_id=viz.get("draw_guard_id", True))
            draw_activity_status(frame, self.activity.label, bbox=guard.bbox)

    def _maybe_progress(self, processed: int, total: int) -> None:
        every = max(1, int(self.cfg.get("api", {}).get("progress_every_frames", 25)))
        if processed % every == 0 or (total and processed >= total):
            self._progress("processing", processed, total)

    def _progress(self, status: str, processed: int, total: int, summary: dict | None = None) -> None:
        if self.progress_callback is None:
            return
        payload = {
            "status": status,
            "frames_processed": processed,
            "total_frames": total if total > 0 else None,
            "progress": (
                round(min(100.0, processed * 100.0 / total), 2) if total > 0 else None
            ),
        }
        if summary is not None:
            payload["summary"] = summary
        try:
            self.progress_callback(payload)
        except Exception:
            # UI/job progress must never break inference.
            pass

    @staticmethod
    def _fmt_ratio(value):
        return "n/a" if value is None else f"{value:.2f}"

    @staticmethod
    def _eye_text(eye):
        if not eye.quality_ok:
            return f"unknown ({eye.reason})"
        if eye.ear_mean is None:
            return f"closed={eye.eyes_closed} EAR=n/a"
        return f"closed={eye.eyes_closed} EAR={eye.ear_mean:.3f}"
