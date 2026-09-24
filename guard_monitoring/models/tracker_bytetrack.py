from __future__ import annotations


class GuardByteTracker:
    def __init__(self, cfg: dict, frame_rate: float = 30.0):
        import math
        from trackers import ByteTrackTracker

        lost_buffer = int(cfg.get("lost_track_buffer", 60))
        if cfg.get("lost_track_seconds") is not None:
            # This package specifies the buffer in 30-FPS units, not input FPS.
            lost_buffer = int(math.ceil(float(cfg["lost_track_seconds"]) * 30.0))
        self.tracker = ByteTrackTracker(
            lost_track_buffer=lost_buffer,
            minimum_consecutive_frames=int(cfg.get("minimum_consecutive_frames", 2)),
            track_activation_threshold=float(cfg.get("track_activation_threshold", 0.45)),
            high_conf_det_threshold=float(cfg.get("high_conf_det_threshold", 0.35)),
            minimum_iou_threshold=float(cfg.get("minimum_iou_threshold", 0.10)),
            frame_rate=float(frame_rate),
        )

    def update(self, detections, timestamp: float):
        return self.tracker.update(detections, timestamp=timestamp)

    def reset(self):
        self.tracker.reset()
