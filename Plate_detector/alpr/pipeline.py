"""
Pipeline orchestration. DETECTION ONLY - this module does not read plate text.

Main thread (must stay real-time):
    capture -> vehicle detect+track -> plate detect on crops -> track bookkeeping -> draw -> show

Worker thread (may be slow, never blocks the display):
    finalized track -> enhance top-k crops -> save images / CSV -> post result back

OCR was deliberately removed: the job here is to produce clean plate IMAGES. Reading
them is a separate concern to be planned later. Note this also removed the ad-frame
filter, which used OCR to reject lettering-without-digits (dealer plaques such as
"SELENIA"); those can therefore reappear as false positives.

Only the worker touches disk; the main loop does two batched GPU calls per frame,
which is what keeps dense traffic at video framerate.
"""
import csv
import os
import queue
import re
import threading
import time
from datetime import datetime
from pathlib import Path

import cv2

from .capture import VideoStream
from .enhancer import PlateEnhancer
from .plate_detector import PlateDetector
from .tracker import TrackManager, TrackState
from .vehicle_detector import VehicleDetector
from .visualize import draw, fit_scale, resize_for_display, screen_size

from system_settings import get_display_prefs


def _make_writer(path, width, height, fps):
    """mp4v (OpenCV's always-available MPEG-4 Part 2 encoder) is not decodable by
    Chrome, Edge or Firefox - a browser <video> shows a black frame at 0:00
    duration even though the file itself is valid. Try the host app's H.264
    writer (same fix already applied to its other modules) and fall back to
    mp4v only if that's unavailable - this module stays usable standalone,
    copied to a host project that doesn't have guard_monitoring."""
    try:
        from guard_monitoring.io import make_writer
        return make_writer(path, width, height, fps)
    except Exception:  # noqa: BLE001 - any import/runtime issue, just fall back
        return cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))


def _write_latest_frame(path: Path, frame) -> None:
    """Write-then-rename so a concurrent MJPEG reader never opens a half-written
    JPEG - same fix already applied to the host app's other live-preview
    endpoints. Must still end in .jpg since cv2.imwrite picks its encoder from
    the file extension."""
    tmp_path = str(path.with_name(path.stem + ".tmp" + path.suffix))
    if not cv2.imwrite(tmp_path, frame):
        return
    for attempt in range(5):
        try:
            os.replace(tmp_path, str(path))
            return
        except PermissionError:
            if attempt == 4:
                return
            time.sleep(0.01)


class ResultWorker(threading.Thread):
    """Enhancement + saving, off the main thread."""

    _DEDUP_WINDOW_S = 120  # how long a finalized plate text stays eligible to be recognized as a duplicate

    _HEADER = ["timestamp", "track_id", "vehicle", "quality_score",
               "plate_conf", "enhanced_path", "raw_path"]

    def __init__(self, cfg, enhancer: PlateEnhancer):
        super().__init__(daemon=True)
        self.cfg = cfg
        self.enhancer = enhancer
        self.in_q: queue.Queue[TrackState | None] = queue.Queue()
        self.out_q: queue.Queue[tuple[int, str, str]] = queue.Queue()

        self.out_dir = Path(cfg.output.dir)
        self.plate_dir = self.out_dir / "plates"
        self.plate_dir.mkdir(parents=True, exist_ok=True)
        # tracks whose ONLY detection was below plate.conf land here instead of being
        # dropped, so raising the threshold can never lose a real plate silently
        self.lowconf_dir = self.out_dir / "plates_lowconf"
        self.csv_path = self.out_dir / cfg.output.csv

        # counters for the end-of-run diagnostic: a run where every finalized track
        # produces empty text is almost always a region-profile mismatch, not "no plates"
        self.n_finalized = 0
        self.n_lowconf = 0

        # loaded into memory (not append-only) so a better duplicate read can replace an
        # earlier row - see the dedup logic in _process()
        self._rows: list[list] = []
        if self.csv_path.exists():
            with open(self.csv_path, "r", newline="", encoding="utf-8") as f:
                rows = list(csv.reader(f))
            if rows and rows[0] == self._HEADER:
                self._rows = rows[1:]
        self._write_csv()

    def _write_csv(self):
        with open(self.csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(self._HEADER)
            w.writerows(self._rows)

    def submit(self, t: TrackState):
        self.in_q.put(t)

    def run(self):
        while True:
            t = self.in_q.get()
            if t is None:
                break
            try:
                self._process(t)
            except Exception as e:  # noqa: BLE001
                print(f"[worker] error on track {t.track_id}: {e}")
                self.out_q.put((t.track_id, "", "done"))
            finally:
                self.in_q.task_done()

    def _process(self, t: TrackState):
        ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        stem = f"{ts}_id{t.track_id}_{t.cls_name}"

        self.n_finalized += 1

        enhanced = [self.enhancer.enhance(d.crop) for d in t.best]
        best_enh = enhanced[0]

        # a track that never produced a confident detection is quarantined, not mixed in
        low = t.best[0].low_conf
        if low:
            self.n_lowconf += 1
        dest = self.lowconf_dir if low else self.plate_dir
        dest.mkdir(parents=True, exist_ok=True)

        enh_path = dest / f"{stem}_enhanced.png"
        cv2.imwrite(str(enh_path), best_enh)

        raw_path = ""
        if self.cfg.output.save_raw_plate:
            raw_path = dest / f"{stem}_raw.png"
            cv2.imwrite(str(raw_path), t.best[0].crop)
        if self.cfg.output.save_vehicle_crop and t.last_vehicle_crop is not None:
            cv2.imwrite(str(self.plate_dir / f"{stem}_vehicle.jpg"), t.last_vehicle_crop)

        self._rows.append([ts, str(t.track_id), t.cls_name, f"{t.best_score:.3f}",
                           f"{t.best[0].conf:.3f}", str(enh_path), str(raw_path)])
        self._write_csv()

        print(f"[result] #{t.track_id} {t.cls_name:<10} conf={t.best[0].conf:.2f} "
              f"score={t.best_score:.2f} -> {enh_path.name}")
        self.out_q.put((t.track_id, "", "lowconf" if low else "done"))

    @staticmethod
    def _parse_ts(ts: str) -> float:
        try:
            return datetime.strptime(ts, "%Y%m%d_%H%M%S_%f").timestamp()
        except ValueError:
            return 0.0


class ALPRPipeline:
    def __init__(self, cfg):
        self.cfg = cfg
        dev = cfg.device
        print(f"[init] device={dev} half={cfg.half}  (detection only - no OCR)")
        self.vehicles = VehicleDetector(cfg.vehicle, dev, cfg.half)
        self.plates = PlateDetector(cfg.plate, dev, cfg.half)
        self.tracks = TrackManager(cfg.tracking)
        self.enhancer = PlateEnhancer(cfg.enhance)

        # ---------- THE SWITCH ----------
        self.worker = ResultWorker(cfg, self.enhancer)
        self.worker.start()

        # Workspace-wide Display preferences (Settings page) - read once per
        # job, not per frame.
        display_prefs = get_display_prefs()
        self.show_detection_boxes = bool(display_prefs.get("show_detection_boxes", True))
        self.show_labels = bool(display_prefs.get("show_labels", True))
        self.show_confidence = bool(display_prefs.get("show_confidence", True))

    # ------------------------------------------------------------------ #
    def run(self, stop_event=None, progress_callback=None, latest_frame_path=None):
        """Blocking. `stop_event` (threading.Event) lets a host shut it down cleanly
        without a GUI keypress. Returns a summary dict instead of only writing files.

        `progress_callback`, if given, is called periodically with
        {"frames_processed": int, "total_frames": int|None} - total_frames is
        None for a live source with no fixed length. `latest_frame_path`, if
        given, gets the current annotated frame written to it continuously
        (atomic write-then-rename) so a host can offer a live preview while a
        long job is still processing."""
        cfg = self.cfg
        stream = VideoStream(cfg.source, cfg.drop_frames, cfg.reconnect).start()
        print(f"[capture] {cfg.source}  {stream.width}x{stream.height} @ {stream.fps:.1f}fps  "
              f"drop_frames={stream.drop}")

        writer = None
        out_path = None
        if cfg.output.save_video:
            # name per source so consecutive runs do not overwrite each other's review video
            src_stem = Path(str(cfg.source)).stem or "stream"
            out_path = Path(cfg.output.dir) / f"annotated_{src_stem}.mp4"
            writer = _make_writer(out_path, stream.width, stream.height,
                                   stream.fps if stream.fps > 1 else 25)

        latest_frame_path = Path(latest_frame_path) if latest_frame_path else None

        if cfg.display.show:
            # WINDOW_NORMAL + an explicit size avoids the AUTOSIZE+Windows-DPI-scaling bug
            # where the window renders only a cropped/zoomed fraction of the frame.
            cv2.namedWindow(cfg.display.window_name, cv2.WINDOW_NORMAL)
            # Fit BOTH limits. "auto" (or unset) = fill most of the actual screen. A hardcoded cap makes a
            # portrait source needlessly small on a big monitor; the 9:16 shape is inherent
            # to the footage and cannot be changed without cropping or distorting it.
            mw, mh = cfg.display.max_width, getattr(cfg.display, "max_height", None)
            if mw in (None, "auto", 0) or mh in (None, "auto", 0):
                sw, sh = screen_size()
                mw = sw - 80 if mw in (None, "auto", 0) else mw
                mh = int(sh * 0.90) if mh in (None, "auto", 0) else mh
            self._disp_wh = (mw, mh)
            r = fit_scale(stream.width, stream.height, mw, mh)
            disp_w, disp_h = max(1, int(stream.width * r)), max(1, int(stream.height * r))
            cv2.resizeWindow(cfg.display.window_name, disp_w, disp_h)
            print(f"[display] {stream.width}x{stream.height} -> window {disp_w}x{disp_h}")

        fps_t, fps_n, fps = time.time(), 0, 0.0
        frame_id = 0
        t_start = time.time()
        try:
            while True:
                if stop_event is not None and stop_event.is_set():
                    print("[exit] stop requested by host")
                    break
                fid, frame = stream.read()
                if fid is None:
                    break
                if frame is None:
                    continue
                frame_id += 1

                # ---- Stage 1: vehicles + tracking
                vehicles = self.vehicles.track(frame)

                # ---- Stage 2: plates on vehicle crops (only for tracks not yet finalized)
                todo = [v for v in vehicles
                        if not (self.tracks.tracks.get(v.track_id) and self.tracks.tracks[v.track_id].finalized)]
                if cfg.plate.every_n_frames > 1 and frame_id % cfg.plate.every_n_frames:
                    todo = []
                plate_map = {}
                if todo:
                    dets = self.plates.detect(frame, [v.xyxy for v in todo])
                    plate_map = {v.track_id: d for v, d in zip(todo, dets)}
                plates = [plate_map.get(v.track_id) for v in vehicles]

                # ---- Stage 3: track bookkeeping -> finalize
                for t in self.tracks.update(frame_id, vehicles, plates, frame):
                    self.worker.submit(t)

                # collect finished results from the worker
                while True:
                    try:
                        tid, text, status = self.worker.out_q.get_nowait()
                    except queue.Empty:
                        break
                    if tid in self.tracks.tracks:
                        self.tracks.tracks[tid].result_text = text
                        self.tracks.tracks[tid].result_status = status

                if frame_id % 300 == 0:
                    self.tracks.gc(frame_id)

                # ---- FPS
                fps_n += 1
                if time.time() - fps_t >= 1.0:
                    fps = fps_n / (time.time() - fps_t)
                    fps_t, fps_n = time.time(), 0

                # ---- Draw / show / save
                if cfg.display.show or writer is not None or latest_frame_path is not None:
                    annotated = draw(frame, self.tracks.active(frame_id), fps,
                                     self.worker.in_q.qsize(),
                                     show_boxes=self.show_detection_boxes,
                                     show_labels=self.show_labels,
                                     show_confidence=self.show_confidence)
                    if writer is not None:
                        writer.write(annotated)
                    if latest_frame_path is not None:
                        _write_latest_frame(latest_frame_path, annotated)
                    if cfg.display.show:
                        cv2.imshow(cfg.display.window_name, resize_for_display(annotated, *self._disp_wh))
                        key = cv2.waitKey(1) & 0xFF
                        if key in (27, ord("q")):
                            break

                if progress_callback is not None and frame_id % 10 == 0:
                    try:
                        progress_callback({
                            "frames_processed": frame_id,
                            "total_frames": stream.total_frames or None,
                        })
                    except Exception:  # noqa: BLE001 - progress reporting must never break inference
                        pass
        except KeyboardInterrupt:
            pass
        finally:
            stream.stop()
            if writer is not None:
                writer.release()
            if latest_frame_path is not None:
                Path(latest_frame_path).unlink(missing_ok=True)
            cv2.destroyAllWindows()
            # flush: finalize whatever tracks still have plates, then let the worker finish
            for t in self.tracks.update(frame_id + 10_000, [], [], None):
                self.worker.submit(t)
            elapsed = time.time() - t_start
            if elapsed > 0 and frame_id:
                print(f"[perf] {frame_id} frames in {elapsed:.1f}s = {frame_id / elapsed:.2f} avg FPS "
                      f"(vehicle imgsz={cfg.vehicle.imgsz}, plate imgsz={cfg.plate.imgsz})")
            print(f"[exit] waiting for {self.worker.in_q.qsize()} queued plates ...")
            self.worker.in_q.join()
            self.worker.submit(None)
            print(f"[exit] results in {Path(cfg.output.dir) / 'plates'}  |  {self.worker.csv_path}")

        return {
            "frames": frame_id,
            "elapsed_s": round(elapsed, 2),
            "fps": round(frame_id / elapsed, 2) if elapsed > 0 else 0.0,
            "finalized": self.worker.n_finalized,
            "lowconf": self.worker.n_lowconf,
            "output_dir": str(Path(cfg.output.dir)),
            "csv": str(self.worker.csv_path),
            "video": str(out_path) if writer is not None else None,
        }

