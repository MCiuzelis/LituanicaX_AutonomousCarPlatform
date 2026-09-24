#!/usr/bin/env python3
"""Camera pipeline sweep: which resolution / fps / encoder combinations the dashboard can
actually stream. For every mode it measures raw capture fps and then the fps each MJPEG
encoder delivers (the Pi 5 has no hardware video encoder, so this is CPU bound).

    python3 tests/camera_sweep.py                 # default matrix, ~3 min
    python3 tests/camera_sweep.py --seconds 2 --modes 1280x720@100,640x480@120
    python3 tests/camera_sweep.py --encoders jpeg-yuv,libav

Stop the dashboard first. Results -> results/camera/pipeline_sweep_<ts>.json (+ a table on stdout).
"""
import argparse
import io
import json
import os
import sys
import threading
import time
from datetime import datetime

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # rc_car/
sys.path.insert(0, ROOT)
from hardware import Camera  # noqa: E402

DEFAULT_MODES = ["640x360@120", "640x480@120", "800x600@120", "1024x576@120", "1280x720@120",
                 "1536x864@120", "1280x720@100", "1536x864@100", "1920x1080@56", "2304x1296@56",
                 "2304x1296@30"]
ENCODERS = ["jpeg-rgb", "jpeg-yuv", "jpeg-yuv-q50", "libav"]


class CountingOutput(io.BufferedIOBase):
    """Encoder sink that only counts frames and bytes."""

    def __init__(self):
        self.frames = self.bytes = 0
        self.t_first = self.t_last = None

    def write(self, buf):
        t = time.monotonic()
        if self.t_first is None:
            self.t_first = t
        self.t_last = t
        self.frames += 1
        self.bytes += len(buf)

    @property
    def fps(self):
        if self.frames < 2:
            return 0.0
        return (self.frames - 1) / (self.t_last - self.t_first)


class CpuMeter(threading.Thread):
    """Average overall CPU utilisation (%) while running, from /proc/stat."""

    def __init__(self):
        super().__init__(daemon=True)
        self.util = None
        self._stop = threading.Event()

    @staticmethod
    def _read():
        with open("/proc/stat") as f:
            v = list(map(int, f.readline().split()[1:]))
        return sum(v), v[3] + v[4]

    def run(self):
        t0, i0 = self._read()
        self._stop.wait()
        t1, i1 = self._read()
        self.util = round(100.0 * (1 - (i1 - i0) / max(1, t1 - t0)), 1)

    def stop(self):
        self._stop.set()
        self.join()
        return self.util


def make_encoder(name):
    from picamera2.encoders import JpegEncoder, LibavMjpegEncoder
    if name.startswith("jpeg"):
        q = 50 if name.endswith("q50") else 70
        return JpegEncoder(q=q)
    if name == "libav":
        return LibavMjpegEncoder()
    raise ValueError(name)


def pixel_format_for(name):
    return "RGB888" if name == "jpeg-rgb" else "YUV420"


def measure(cam, w, h, fps, enc_name, seconds):
    from picamera2.outputs import FileOutput
    cam.pixel_format = pixel_format_for(enc_name)
    mode = cam.set_mode(w, h, fps, settle_s=0.3)
    row = {"mode": "%dx%d@%g" % (w, h, fps), "sensor": "%dx%d" % mode.sensor.size,
           "fps_set": mode.fps, "encoder": enc_name, "format": cam.pixel_format}
    # 1) raw capture rate (no encoder)
    row["capture_fps"] = cam.measure_fps(min(1.5, seconds))
    # 2) encoder rate
    out = CountingOutput()
    meter = CpuMeter()
    meter.start()
    cam.cam.start_encoder(make_encoder(enc_name), FileOutput(out))
    time.sleep(seconds)
    cam.cam.stop_encoder()
    row["cpu_pct"] = meter.stop()
    row["encoded_fps"] = round(out.fps, 1)
    row["kb_per_frame"] = round(out.bytes / max(1, out.frames) / 1024, 1)
    row["mbit_s"] = round(out.bytes * 8 / 1e6 / max(1e-6, (out.t_last or 0) - (out.t_first or 0)), 1) if out.frames > 1 else 0
    row["ok"] = row["encoded_fps"] >= 0.95 * mode.fps
    return row


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--modes", default=",".join(DEFAULT_MODES), help="comma list of WxH@fps")
    ap.add_argument("--encoders", default=",".join(ENCODERS), help="comma list of %s" % ENCODERS)
    ap.add_argument("--seconds", type=float, default=3.0, help="encoder measurement time per combo")
    a = ap.parse_args()
    modes = []
    for m in a.modes.split(","):
        wh, fps = m.split("@")
        w, h = wh.split("x")
        modes.append((int(w), int(h), float(fps)))
    encoders = a.encoders.split(",")

    rows = []
    print("%-14s %-9s %-13s %-7s %8s %8s %6s %6s %7s  %s" % ("mode", "sensor", "encoder", "format", "capture", "encoded", "cpu%", "KB/f", "Mbit/s", ""))
    with Camera() as cam:
        cam.open()
        for w, h, fps in modes:
            for enc in encoders:
                try:
                    r = measure(cam, w, h, fps, enc, a.seconds)
                except Exception as e:  # noqa: BLE001 - record and carry on
                    r = {"mode": "%dx%d@%g" % (w, h, fps), "encoder": enc, "error": str(e).splitlines()[0][:80]}
                rows.append(r)
                if "error" in r:
                    print("%-14s %-9s %-13s ERROR %s" % (r["mode"], "", enc, r["error"]))
                else:
                    print("%-14s %-9s %-13s %-7s %8.1f %8.1f %6.0f %6.1f %7.1f  %s"
                          % (r["mode"], r["sensor"], enc, r["format"], r["capture_fps"], r["encoded_fps"],
                             r["cpu_pct"] or 0, r["kb_per_frame"], r["mbit_s"], "OK" if r["ok"] else "drops"))
    out = os.path.join(ROOT, "results", "camera", "pipeline_sweep_%s.json" % datetime.now().strftime("%Y%m%d_%H%M%S"))
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump({"seconds": a.seconds, "rows": rows}, f, indent=2)
    print("results ->", out)


if __name__ == "__main__":
    main()
