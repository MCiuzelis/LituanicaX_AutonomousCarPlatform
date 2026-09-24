"""Live apple detection on the car: IMX708 camera -> Hailo-8 -> boxes.

Runs on the Pi 5 (system python3 with python3-hailort + picamera2, no OpenCV needed).
Used standalone (benchmark below) and by dashboard.py (live overlay + telemetry).

Camera: same raw sensor mode as the training recordings (1536x864 @ up to 120 fps, via the
dashboard's 1280x720 setting). The ISP produces the 1280x720 main stream *and* a 640-wide
lores copy for free; lores is letterboxed to 640x640 (grey 114 bars, exactly like Ultralytics
training) and sent to the Hailo through the async API. NMS runs on-chip.

    python3 apple_detector.py                                            # Pi5/models/apple_yolov8s.hef
    python3 apple_detector.py --hef /usr/share/hailo-models/yolov8s_h8.hef --class-id 47  # stock COCO 'apple'
    python3 apple_detector.py --save 5                                   # also save annotated JPEGs

Library use:
    det = AppleDetector("apple_yolov8s.hef", lores_size=(640, 360))
    boxes = det.detect(rgb_lores)   # -> [(x1, y1, x2, y2, score, class_id)], coords 0..1 of the frame
"""
import argparse
import json
import os
import statistics
import threading
import time
from collections import deque
from contextlib import ExitStack

import numpy as np
from hailo_platform import FormatType, HailoSchedulingAlgorithm, VDevice

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # Pi5/

NET = 640                         # model input (square)
CAM_W, CAM_H = 1280, 720          # camera output the car works in (standalone benchmark)


class AppleDetector:
    """Async HailoRT pipeline: up to `depth` frames in flight so PCIe transfer, the chip and the
    Python side overlap (a blocking call per frame caps out around 96 FPS on this Pi).

    submit() and collect() may run on different threads: a capture thread submits frames while a
    result thread collects them, so boxes are available as soon as the chip is done with a frame
    instead of when the next camera frame arrives."""

    def __init__(self, hef_path, score_thr=0.35, class_ids=None, depth=4, lores_size=(640, 360)):
        self.hef_path = hef_path
        self.score_thr = score_thr
        self.class_ids = class_ids
        self.lo_w, self.lo_h = lores_size
        assert self.lo_w == NET and self.lo_h <= NET, f"lores must be {NET} wide and <= {NET} high"
        self.pad_top = (NET - self.lo_h) // 2
        params = VDevice.create_params()
        params.scheduling_algorithm = HailoSchedulingAlgorithm.ROUND_ROBIN
        self.vdevice = VDevice(params)
        self.model = self.vdevice.create_infer_model(hef_path)
        self.model.set_batch_size(1)
        self.model.input().set_format_type(FormatType.UINT8)
        self.model.output().set_format_type(FormatType.FLOAT32)
        h, w, _ = self.model.input().shape
        assert (h, w) == (NET, NET), f"expected {NET}x{NET} input, HEF has {w}x{h}"
        self._stack = ExitStack()
        self.cfg = self._stack.enter_context(self.model.configure())
        self._slots = []
        for _ in range(depth):
            canvas = np.full((NET, NET, 3), 114, np.uint8)
            b = self.cfg.create_bindings(
                output_buffers={self.model.output().name: np.empty(self.model.output().shape, np.float32)})
            b.input().set_buffer(canvas)
            self._slots.append((canvas, b))
        self._next = 0
        self._pending = deque()  # (job, bindings, user_tag, submit_time)
        self._free = threading.Semaphore(depth)   # slots not in flight
        self._cond = threading.Condition()
        self.last_infer_s = 0.0  # submit -> result of the last collected frame
        self.num_classes = None
        self.temp_error = None
        self._phys = None

    # -- async interface ---------------------------------------------------
    def submit(self, rgb_lores, tag=None):
        """Queue a lores RGB frame. Blocks while `depth` frames are in flight (single-threaded
        callers must collect() before submitting more than `depth`)."""
        self._free.acquire()
        canvas, b = self._slots[self._next]
        self._next = (self._next + 1) % len(self._slots)
        canvas[self.pad_top:self.pad_top + self.lo_h] = rgb_lores
        self.cfg.wait_for_async_ready(timeout_ms=1000)
        with self._cond:
            self._pending.append((self.cfg.run_async([b]), b, tag, time.perf_counter()))
            self._cond.notify()

    def collect(self, timeout=None):
        """Wait for the oldest in-flight frame -> (boxes, tag); None if nothing arrives in time."""
        with self._cond:
            if not self._cond.wait_for(lambda: self._pending, timeout):
                return None
            job, b, tag, t0 = self._pending.popleft()
        try:
            job.wait(1000)
            self.last_infer_s = time.perf_counter() - t0
            return self._decode(b.output().get_buffer()), tag
        finally:
            self._free.release()

    def set_lores_size(self, w, h):
        """Switch input frame size (e.g. after a camera mode change). Nothing may be in flight."""
        assert not self._pending, "drain in-flight frames first"
        assert w == NET and h <= NET, f"lores must be {NET} wide and <= {NET} high"
        self.lo_w, self.lo_h, self.pad_top = w, h, (NET - h) // 2
        for canvas, _ in self._slots:
            canvas.fill(114)   # old image rows must not survive in the new letterbox bars

    @property
    def in_flight(self):
        return len(self._pending)

    # -- blocking convenience ------------------------------------------------
    def detect(self, rgb_lores):
        self.submit(rgb_lores)
        return self.collect()[0]

    def _decode(self, raw):
        """HAILO_NMS_BY_CLASS -> [(x1, y1, x2, y2, score, class_id)] normalised to the camera frame.
        `raw` is a per-class list of (n, 5) arrays [ymin, xmin, ymax, xmax, score] in letterboxed
        640x640 space (0..1)."""
        self.num_classes = len(raw)
        out = []
        for cid, dets in enumerate(raw):
            if self.class_ids is not None and cid not in self.class_ids:
                continue
            for y1, x1, y2, x2, s in np.asarray(dets).reshape(-1, 5):
                if s < self.score_thr:
                    continue
                out.append((float(np.clip(x1, 0, 1)),
                            float(np.clip((y1 * NET - self.pad_top) / self.lo_h, 0, 1)),
                            float(np.clip(x2, 0, 1)),
                            float(np.clip((y2 * NET - self.pad_top) / self.lo_h, 0, 1)),
                            float(s), cid))
        return out

    def chip_temp_c(self):
        try:
            if self._phys is None:   # keep the handle: a one-liner lets it be released mid-call
                self._phys = self.vdevice.get_physical_devices()
            t = self._phys[0].control.get_chip_temperature()
            return round(max(t.ts0_temperature, t.ts1_temperature), 1)
        except Exception as e:  # noqa: BLE001 - optional telemetry
            self.temp_error = f"{type(e).__name__}: {str(e).splitlines()[0][:120]}"
            return None

    def close(self):
        while self._pending:
            self.collect(0)
        self._stack.close()
        self._phys = None
        self.vdevice.release()


def open_camera(fps):
    """Same raw mode the dashboard picks for 1280x720 (1536x864, 2x2 binned, max 120 fps)."""
    from picamera2 import Picamera2
    cam = Picamera2()
    # picamera2 "BGR888" = bytes in R,G,B order, i.e. an RGB numpy array (what the model expects)
    cfg = cam.create_video_configuration(
        main={"size": (CAM_W, CAM_H), "format": "BGR888"},
        lores={"size": (NET, NET * CAM_H // CAM_W), "format": "BGR888"},
        raw={"size": (1536, 864)},
        buffer_count=6,
    )
    cam.configure(cfg)
    dur = int(1e6 / fps)
    cam.set_controls({"FrameDurationLimits": (dur, dur)})
    cam.start()
    time.sleep(1.0)  # AE/AWB settle
    return cam


def draw(rgb_main, boxes, path):
    from PIL import Image, ImageDraw
    im = Image.fromarray(rgb_main)
    W, H = im.size
    d = ImageDraw.Draw(im)
    for x1, y1, x2, y2, s, _ in boxes:
        d.rectangle([x1 * W, y1 * H, x2 * W, y2 * H], outline=(0, 255, 0), width=3)
        d.text((x1 * W + 2, max(0, y1 * H - 12)), f"{s:.2f}", fill=(0, 255, 0))
    im.save(path, quality=85)


def pct(xs, p):
    return sorted(xs)[min(len(xs) - 1, int(len(xs) * p))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hef", default=os.path.join(ROOT, "models", "apple_yolov8s.hef"))
    ap.add_argument("--seconds", type=float, default=10.0)
    ap.add_argument("--fps", type=float, default=120.0, help="camera frame rate (sensor max 120)")
    ap.add_argument("--conf", type=float, default=0.35)
    ap.add_argument("--class-id", type=int, nargs="*", default=None,
                    help="keep only these classes (47 = apple for the stock COCO model)")
    ap.add_argument("--depth", type=int, default=4, help="max frames in flight on the Hailo")
    ap.add_argument("--save", type=int, default=0, help="save N annotated frames")
    ap.add_argument("--out", default=os.path.join(ROOT, "results", "vision"))
    args = ap.parse_args()

    det = AppleDetector(args.hef, args.conf, set(args.class_id) if args.class_id else None, args.depth,
                        (NET, NET * CAM_H // CAM_W))
    cam = open_camera(args.fps)
    try:
        # 1) Hailo alone, on a real camera frame, pipeline kept full
        lo = cam.capture_array("lores")
        for _ in range(20):
            det.detect(lo)
        n, t0 = 0, time.perf_counter()
        while time.perf_counter() - t0 < 3.0:
            if det.in_flight == args.depth:
                det.collect()
                n += 1
            det.submit(lo)
        hailo_fps = n / (time.perf_counter() - t0)
        while det.in_flight:
            det.collect()

        # 2) live loop: grab frame N and submit it, then decode frame N-1 while N is on the chip
        os.makedirs(args.out, exist_ok=True)
        save_every = max(1, int(args.seconds * args.fps / max(1, args.save))) if args.save else 0
        t_wait, t_lat, n_boxes, saved = [], [], [], []
        frames, t_start = 0, time.perf_counter()

        def finish():
            nonlocal frames
            boxes, (t_cap, main) = det.collect()
            t_lat.append(time.perf_counter() - t_cap)
            n_boxes.append(len(boxes))
            if main is not None:
                p = os.path.join(args.out, f"live_{len(saved):02d}.jpg")
                draw(main, boxes, p)
                saved.append(p)
            frames += 1

        while time.perf_counter() - t_start < args.seconds:
            a = time.perf_counter()
            req = cam.capture_request()
            lo = req.make_array("lores")
            t_cap = time.perf_counter()
            main = None
            if save_every and (frames + det.in_flight) % save_every == 0 and len(saved) < args.save:
                main = req.make_array("main")
            req.release()
            t_wait.append(t_cap - a)
            det.submit(lo, (t_cap, main))
            while det.in_flight > 1:   # keep one frame on the chip while we wait for the next
                finish()
        while det.in_flight:
            finish()
        e2e_fps = frames / (time.perf_counter() - t_start)
        temp = det.chip_temp_c()
    finally:
        cam.stop()
        det.close()

    ms = lambda xs: f"median {statistics.median(xs) * 1e3:.2f} ms, p99 {pct(xs, 0.99) * 1e3:.2f} ms"
    report = {
        "hef": args.hef,
        "hailo_only_fps": round(hailo_fps, 1),
        "camera_fps_requested": args.fps,
        "end_to_end_fps": round(e2e_fps, 1),
        "frames": frames,
        "frame_wait": ms(t_wait),
        "latency_frame_to_boxes": ms(t_lat),
        "boxes_per_frame": f"mean {statistics.mean(n_boxes):.1f}, max {max(n_boxes)}",
        "hailo_chip_temp_c": temp,
        "saved": saved,
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
