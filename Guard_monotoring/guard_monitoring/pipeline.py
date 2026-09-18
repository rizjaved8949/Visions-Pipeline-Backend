from __future__ import annotations

import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Callable

import cv2

from .contracts import ModuleResult, ModuleStatus, disabled, error_result, ok, unknown
from .events import JsonlWriter
from .geometry import normalized_polygon_to_pixels
from .health import ModuleHealthRegistry, safe_run
from .io import make_writer, open_capture, video_metadata
from .model_registry import LazyModelRegistry
from .monitoring.guard_selector import GuardSelector
from .monitoring.movement import MovementMonitor
from .monitoring.phone_use import PhoneUseAnalyzer
from .monitoring.posture import PostureAnalyzer
from .monitoring.presence import PresenceState, presence_from_tracker
from .monitoring.rules import TimedRuleEngine
from .monitoring.sleep import SleepAnalyzer
from .types import EyeState, MovementState, PhoneState, PostureState, SleepState
from .visualization.overlay import draw_box, draw_polygon, draw_pose, draw_status_panel

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

        self.last_track_id = None
        self.last_pose: ModuleResult = unknown("pose_not_run")
        self.last_pose_frame = -10**9
        self.last_phone_detections: ModuleResult = unknown("phone_not_run")
        self.last_phone_frame = -10**9
        self.last_eye: ModuleResult = unknown("eyes_not_run")
        self.last_eye_frame = -10**9

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

        writer = make_writer(video_path, width, height, fps)

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
        )
        move_cfg = self.cfg["movement"]
        movement_monitor = MovementMonitor(
            history_seconds=move_cfg.get("history_seconds", 5.0),
            stationary_radius_ratio=move_cfg.get("stationary_radius_ratio", 0.035),
            minimum_history_seconds=move_cfg.get("minimum_history_seconds", 2.0),
            patrol_zones=patrol_zones,
        )
        rules = TimedRuleEngine(self.camera_id, self.cfg["rules"])

        frame_count = 0
        guard_visible_frames = 0
        phone_detected_frames = 0
        phone_use_frames = 0
        sleep_candidate_frames = 0
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
                    ok_read, frame = cap.read()
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
                        fatal_result = error_result(exc)
                        self.health.record("frame_pipeline", fatal_result)
                        if not continue_on_frame_error:
                            raise
                        writer.write(frame)
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
                    events_count += result["events_count"]

                    for event in result["events"]:
                        event_log.write(event)

                    writer.write(result["annotated"])
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
                if display:
                    cv2.destroyAllWindows()

        health_snapshot = self.health.snapshot()
        health_path.write_text(json.dumps(health_snapshot, indent=2), encoding="utf-8")

        summary = {
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
            "events_triggered": events_count,
            "processing_wall_seconds": round(time.time() - started_wall, 3),
            "output_video": str(video_path),
            "frame_log": str(frame_log_path),
            "event_log": str(event_log_path),
            "module_health": str(health_path),
        }
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        self._progress("completed", frame_count, total_frames, summary=summary)
        return summary

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

        guard = guard_result.value if guard_result.ok else None
        present = presence_result.value.present if presence_result.ok else None

        if guard is not None and guard.track_id != self.last_track_id:
            self._reset_guard_dependent_state(guard.track_id)
            movement_monitor.reset(guard.track_id)

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
                health=self.health,
            )
            if movement_result.ok:
                movement = movement_result.value

            # STEP 4 - pose. A pose failure does not stop phone detection or eyes.
            pose_result = self._get_pose(frame, guard.bbox, frame_idx)
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
            phone_detection_result = self._get_phone_detections(frame, guard.bbox, frame_idx)
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
            sleep_condition = bool(
                sleep.candidate
                and (sleep.evidence_quality == "high" or allow_degraded_sleep)
            )
        elif present is False:
            sleep_condition = False

        phone_condition = None
        if present is True and phone_use_result.ok:
            phone_condition = bool(phone.usage in {"call", "screen_use"})
        elif present is False:
            phone_condition = False

        stationary_condition = None
        if present is True and movement_result.ok:
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

        annotated = frame.copy()
        self._draw(
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
        )

        frame_log = {
            "frame": frame_idx,
            "time_seconds": now,
            "frame_status": "ok",
            "guard_track_id": selector.active_id,
            "guard_seen_now": guard_seen_now,
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

    def _get_pose(self, frame, guard_box, frame_idx) -> ModuleResult:
        load = self.registry.pose()
        if not load.ok:
            return self._propagate("pose", load)
        if frame_idx % self.pose_stride == 0:
            self.last_pose = safe_run(
                "pose",
                load.value.predict,
                frame,
                guard_box,
                health=self.health,
            )
            self.last_pose_frame = frame_idx
        if frame_idx - self.last_pose_frame <= self.pose_cache_max:
            return self.last_pose
        return self._unknown_recorded("pose", "pose_cache_expired")

    def _get_phone_detections(self, frame, guard_box, frame_idx) -> ModuleResult:
        load = self.registry.phone()
        if not load.ok:
            return self._propagate("phone_detection", load)
        if frame_idx % self.phone_stride == 0:
            self.last_phone_detections = safe_run(
                "phone_detection",
                load.value.predict,
                frame,
                guard_box,
                health=self.health,
            )
            self.last_phone_frame = frame_idx
        if frame_idx - self.last_phone_frame <= self.phone_cache_max:
            return self.last_phone_detections
        return self._unknown_recorded("phone_detection", "phone_cache_expired")

    def _get_eye_state(self, frame, guard_box, keypoints, frame_idx) -> ModuleResult:
        load = self.registry.eyes()
        if not load.ok:
            return self._propagate("eyes", load)
        if frame_idx % self.eye_stride == 0:
            self.last_eye = safe_run(
                "eyes",
                load.value.analyze,
                frame,
                guard_box,
                keypoints,
                int(frame_idx * 1000.0 / max(1.0, getattr(self, "video_fps", 30.0))),
                health=self.health,
            )
            self.last_eye_frame = frame_idx
        if frame_idx - self.last_eye_frame <= self.eye_cache_max:
            return self.last_eye
        return self._unknown_recorded("eyes", "eye_cache_expired")

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

    def _draw(
        self,
        frame,
        tracked,
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
    ):
        viz = self.cfg["visualization"]
        if viz.get("draw_zones", True):
            draw_polygon(frame, duty_zone, "duty zone")
            for name, polygon in patrol_zones:
                draw_polygon(frame, polygon, name)

        tracker_ids = getattr(tracked, "tracker_id", None) if tracked is not None else None
        if tracker_ids is not None:
            for idx, tid in enumerate(tracker_ids):
                if tid is None or int(tid) < 0:
                    continue
                box = tuple(float(v) for v in tracked.xyxy[idx])
                is_guard = guard is not None and int(tid) == guard.track_id
                label = f"GUARD ID {int(tid)}" if is_guard else f"person ID {int(tid)}"
                color = (0, 255, 0) if is_guard else (150, 150, 150)
                draw_box(frame, box, label, color=color, thickness=2 if is_guard else 1)

        pose_value = self.last_pose.value if self.last_pose.ok else None
        if guard is not None and pose_value is not None and viz.get("draw_pose", True):
            draw_pose(
                frame,
                pose_value.keypoints,
                self.cfg["pose_logic"].get("keypoint_confidence", 0.25),
            )
        if phone.bbox is not None and viz.get("draw_phone", True):
            draw_box(frame, phone.bbox, f"phone: {phone.usage}", color=(255, 0, 255), thickness=2)

        if viz.get("draw_panel", True):
            errors = [
                name for name, result in module_results.items()
                if result.status is ModuleStatus.ERROR
            ]
            unknowns = [
                name for name, result in module_results.items()
                if result.status is ModuleStatus.UNKNOWN
            ]
            lines = [
                f"Guard ID: {guard.track_id if guard else 'none'} | present={present}",
                f"Movement: {'stationary' if movement.stationary else 'moving/unknown'}",
                f"Patrol coverage: {self._fmt_ratio(movement.patrol_coverage_ratio)}",
                f"Posture: {posture.posture} | head-down={posture.head_down}",
                f"Phone: {phone.usage} | detected={phone.detected}",
                f"Eyes: {self._eye_text(eye)}",
                (
                    f"Sleep evidence: {sleep.candidate} | quality={sleep.evidence_quality} "
                    f"| score={sleep.score:.2f} | PERCLOS={self._fmt_ratio(sleep.perclos)}"
                ),
                f"Module errors: {', '.join(errors) if errors else 'none'}",
                f"Unknown modules: {', '.join(unknowns[:4]) if unknowns else 'none'}",
            ]
            alerts = [
                name for name in ("sleep", "phone", "stationary", "absence")
                if rules.is_triggered(name)
            ]
            draw_status_panel(frame, lines, alerts)

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
