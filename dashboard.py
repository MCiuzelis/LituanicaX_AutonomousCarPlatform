#!/usr/bin/env python3
"""RC car dashboard: live camera, telemetry and remote control server.

    python3 ~/rc_car/dashboard.py            (or scripts/dashboard_restart.sh to run it detached)
    then open http://rp5:8080  (or http://<pi-ip>:8080)

Serves
  http://<pi>:8080/        the dashboard page (web/index.html): MJPEG camera, resolution
                           switching, mp4 recording, all telemetry, keyboard driving over a
                           WebSocket
  tcp  <pi>:8081           command/health link for car_client.py on a PC
  /ws/det, /api/vision     Hailo object detection on the camera (Vision class): boxes for the
                           page overlay, and on/off + model/threshold/class filter settings

Protocol on both the WebSocket and the tcp link (one JSON object per line):
  client -> pi   {"t": -1..1, "s": -1..1}   throttle (negative = brake) and steering, sent
                                            continuously (~20 Hz) while driving
  pi -> client   {"h":1, "v":batt, "temp":c, "rps":.., "us":pulse, "st":steer, "act":link,
                  "lk":{"ws":age_ms,"tcp":age_ms}, "stop":bool}   at HEALTH_HZ

Safety: throttle is capped at MAX_THROTTLE. If no command arrives on ANY link for DEADMAN_S
the motor goes to neutral and the steering to centre. The freshest command from any link wins.
"""
import asyncio
import io
import json
import logging
import os
import socket
import statistics
import sys
import threading
import time
from collections import deque
from fractions import Fraction

import numpy as np
from aiohttp import web
from picamera2.encoders import JpegEncoder, LibavH264Encoder
from picamera2.outputs import FileOutput, PyavOutput

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "vision"))
from hardware import Drive, Camera, BoardPower, BatteryMonitor, HailoBoard  # noqa: E402

HTTP_PORT, TCP_PORT = 8080, 8081
MAX_THROTTLE = 0.20          # fraction of full forward the server will ever apply
DEADMAN_S = 0.4              # stop if no command arrives on any link for this long
TELEMETRY_HZ = 4             # telemetry sampling + WebSocket push rate
HEALTH_HZ = 5                # compact health line rate on the tcp link
# (width, height, fps). Sensor modes: 1536x864@120, 2304x1296@56, 4608x2592@14; the ISP scales
# any smaller output from the chosen one. tests/camera_sweep.py showed the software MJPEG
# encoder keeps up with all of these (1536x864@120 at ~16 % CPU on YUV420).
CAMERA_MODES = [(640, 360, 120), (640, 480, 120), (800, 600, 120), (1024, 576, 120),
                (1280, 720, 120), (1280, 720, 100), (1280, 720, 60), (1536, 864, 120),
                (1920, 1080, 56), (2304, 1296, 56), (2304, 1296, 30)]
DEFAULT_MODE = (1280, 720, 60)
WEB_DIR = os.path.join(ROOT, "web")
REC_DIR = os.path.join(ROOT, "recordings")
MODEL_DIR = os.path.join(ROOT, "models")
STOCK_HEF = "/usr/share/hailo-models/yolov8s_h8.hef"   # COCO yolov8s: same arch/input as ours
DET_PUSH_HZ = 30             # max rate boxes are pushed to the page overlay
COCO_NAMES = (
    "person bicycle car motorcycle airplane bus train truck boat traffic_light fire_hydrant "
    "stop_sign parking_meter bench bird cat dog horse sheep cow elephant bear zebra giraffe "
    "backpack umbrella handbag tie suitcase frisbee skis snowboard sports_ball kite baseball_bat "
    "baseball_glove skateboard surfboard tennis_racket bottle wine_glass cup fork knife spoon bowl "
    "banana apple sandwich orange broccoli carrot hot_dog pizza donut cake chair couch potted_plant "
    "bed dining_table toilet tv laptop mouse remote keyboard cell_phone microwave oven toaster sink "
    "refrigerator book clock vase scissors teddy_bear hair_drier toothbrush").split()

log = logging.getLogger("dashboard")


# ---------------------------------------------------------------- control arbiter
class Control:
    """Applies the freshest command from any link; a watchdog stops the car when all go quiet."""

    def __init__(self, drive):
        self.drive = drive
        self.lock = threading.Lock()
        self.last = {}                  # link name -> monotonic time of its last command
        self.throttle = self.steer = 0.0
        self.active = None              # link that sent the command currently applied
        self.stopped = True             # True once the dead-man (or nobody) has stopped the car
        self._run = True
        threading.Thread(target=self._watchdog, daemon=True).start()

    def command(self, link, throttle, steer):
        t = max(-1.0, min(MAX_THROTTLE, float(throttle)))
        s = max(-1.0, min(1.0, float(steer)))
        with self.lock:
            self.last[link] = time.monotonic()
            self.throttle, self.steer, self.active, self.stopped = t, s, link, False
            if t >= 0:
                self.drive.forward(t)
            else:
                self.drive.brake(-t)
            self.drive.steer(s)

    def link_ages_ms(self):
        now = time.monotonic()
        return {k: int((now - v) * 1000) for k, v in self.last.items()}

    def status(self):
        return {"throttle": self.throttle, "steer": self.steer, "active": self.active,
                "stopped": self.stopped, "max_throttle": MAX_THROTTLE, "links_ms": self.link_ages_ms()}

    def _watchdog(self):
        while self._run:
            time.sleep(0.05)
            with self.lock:
                now = time.monotonic()
                fresh = [k for k, v in self.last.items() if now - v < DEADMAN_S]
                if not self.stopped and not fresh:
                    log.warning("dead-man: no command for %.0f ms on any link -> stop", DEADMAN_S * 1000)
                    self.drive.stop()
                    self.drive.steer(0)
                    self.throttle = self.steer = 0.0
                    self.active, self.stopped = None, True

    def close(self):
        self._run = False


# ---------------------------------------------------------------- camera stream
class StreamingOutput(io.BufferedIOBase):
    """Receives JPEG frames from the picamera2 encoder; readers wait() for a new one."""

    def __init__(self):
        self.frame, self.count = None, 0
        self.cond = threading.Condition()
        self.times = deque(maxlen=30)

    def write(self, buf):
        with self.cond:
            self.frame, self.count = buf, self.count + 1
            self.times.append(time.monotonic())
            self.cond.notify_all()

    def wait(self, seen, timeout=1.0):
        """Block until a frame newer than `seen` arrives; returns (frame, count)."""
        with self.cond:
            self.cond.wait_for(lambda: self.count != seen, timeout)
            return self.frame, self.count

    @property
    def fps(self):
        if len(self.times) < 2:
            return 0.0
        return (len(self.times) - 1) / max(1e-6, self.times[-1] - self.times[0])


class CameraStream:
    """Camera -> MJPEG stream for the page, plus optional H.264 mp4 recording to the SD card."""

    def __init__(self, mode=DEFAULT_MODE):
        self.cam = Camera(pixel_format="YUV420")   # cheaper to JPEG/H.264-encode than RGB888
        self.out = StreamingOutput()
        self.lock = threading.Lock()
        self.mode = None           # (width, height, fps) currently streaming
        self.lores_size = None     # (640, h) RGB copy of every frame for the detector
        self.rec = None            # (encoder, path, start_time) while recording
        os.makedirs(REC_DIR, exist_ok=True)
        self.set_mode(*mode)

    def set_mode(self, w, h, fps):
        with self.lock:
            if self.rec:
                raise RuntimeError("stop the recording before switching resolution")
            if self.mode is not None:
                self.cam.cam.stop_encoder()
            # detector input: 640 wide, same aspect as main (ISP-scaled, costs no CPU)
            lo = (640, min(640, int(h) * 640 // int(w)) // 2 * 2)
            m = self.cam.set_mode(int(w), int(h), float(fps), settle_s=0.0, lores=(*lo, "BGR888"))
            self.lores_size = lo
            self.cam.cam.start_encoder(JpegEncoder(q=70), FileOutput(self.out))
            self.mode = (m.width, m.height, m.fps)
            log.info("camera %dx%d @ %.1f fps", *self.mode)
            return self.mode

    def start_recording(self):
        with self.lock:
            if self.rec:
                return self.rec[1]
            w, h, fps = self.mode
            path = os.path.join(REC_DIR, time.strftime("rec_%Y%m%d_%H%M%S") + "_%dx%d.mp4" % (w, h))
            enc = LibavH264Encoder(bitrate=int(2e6 * (w * h) / (1280 * 720)))   # ~2 Mbit/s at 720p
            enc.framerate = Fraction(fps).limit_denominator(1000)                # PyAV wants a rational
            self.cam.cam.start_encoder(enc, PyavOutput(path))
            self.rec = (enc, path, time.monotonic())
            log.info("recording -> %s", path)
            return path

    def stop_recording(self):
        with self.lock:
            if not self.rec:
                return None
            enc, path, _ = self.rec
            self.cam.cam.stop_encoder(enc)
            self.rec = None
            os.sync()
            log.info("recording saved %s (%.1f MB)", path, os.path.getsize(path) / 1e6)
            return path

    @staticmethod
    def recordings():
        out = []
        for n in sorted(os.listdir(REC_DIR), reverse=True):
            if n.endswith(".mp4"):
                st = os.stat(os.path.join(REC_DIR, n))
                out.append({"name": n, "size_mb": round(st.st_size / 1e6, 1), "mtime": st.st_mtime})
        return out

    def status(self):
        rec = None
        if self.rec:
            rec = {"file": os.path.basename(self.rec[1]), "seconds": round(time.monotonic() - self.rec[2], 1)}
        fs = os.statvfs(REC_DIR)
        m = self.cam.mode
        return {"mode": self.mode, "stream_fps": round(self.out.fps, 1), "modes": CAMERA_MODES,
                "sensor": m and "%dx%d" % m.sensor.size,
                "recording": rec, "disk_free_gb": round(fs.f_bavail * fs.f_frsize / 1e9, 2)}

    def close(self):
        self.stop_recording()
        with self.lock:
            try:
                self.cam.cam.stop_encoder()
            except Exception:  # noqa: BLE001 - best effort on shutdown
                pass
            self.cam.close()


# ---------------------------------------------------------------- vision (Hailo detector)
def _pct(xs, p):
    if not xs:
        return None
    s = sorted(xs)
    return round(s[min(len(s) - 1, int(len(s) * p))], 2)


class Vision:
    """Runs the Hailo detector on the camera's lores stream while switched on.

    A capture thread submits every lores frame to the Hailo (async, several frames may be in
    flight); a result thread collects them in order and publishes boxes the moment the chip is
    done. Delay is measured from the frame's SensorTimestamp (start of sensor readout,
    CLOCK_MONOTONIC) to the moment its boxes are available here.
    """

    def __init__(self, camera):
        self.camera = camera
        self.cond = threading.Condition()
        self.latest = {"seq": 0, "boxes": []}
        default = os.path.join(MODEL_DIR, "apple_yolov8s.hef")
        self.cfg = {"on": False, "hef": default if os.path.exists(default) else STOCK_HEF,
                    "conf": 0.35, "filter": "apple"}
        self.det, self.names, self.error = None, [], None
        self.chip_temp, self._next_temp = None, 0.0
        self.frames = self.dropped = 0
        self._last_ts = None
        n = 240
        self.t_done, self.lat_total, self.lat_sensor = deque(maxlen=n), deque(maxlen=n), deque(maxlen=n)
        self.lat_hailo, self.n_boxes, self.max_score = deque(maxlen=n), deque(maxlen=n), deque(maxlen=n)
        self._det_lock = threading.Lock()   # guards self.det between the two threads
        self._run = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        self._results = threading.Thread(target=self._result_loop, daemon=True)
        self._results.start()

    # ---- control
    @staticmethod
    def models():
        out = sorted(os.path.join(MODEL_DIR, f) for f in os.listdir(MODEL_DIR) if f.endswith(".hef")) \
            if os.path.isdir(MODEL_DIR) else []
        return out + ([STOCK_HEF] if os.path.exists(STOCK_HEF) else [])

    def configure(self, **kw):
        for k in ("on", "hef", "conf", "filter"):
            if k in kw and kw[k] is not None:
                self.cfg[k] = kw[k]
        self.cfg["conf"] = max(0.05, min(0.95, float(self.cfg["conf"])))
        if kw.get("on"):
            self.error = None
        self._reset_stats()
        return self.status()

    def _reset_stats(self):
        for d in (self.t_done, self.lat_total, self.lat_sensor, self.lat_hailo, self.n_boxes, self.max_score):
            d.clear()
        self.frames = self.dropped = 0
        self._last_ts = None

    # ---- worker
    def _open(self):
        from apple_detector import AppleDetector   # needs python3-hailort; only loaded when used
        det = AppleDetector(self.cfg["hef"], lores_size=self.camera.lores_size)
        det.detect(np.full((det.lo_h, det.lo_w, 3), 114, np.uint8))   # learn class count
        self.names = COCO_NAMES if det.num_classes == 80 else ["apple"] if det.num_classes == 1 \
            else ["class%d" % i for i in range(det.num_classes)]
        log.info("vision: %s (%d classes) on %dx%d lores", self.cfg["hef"], det.num_classes, det.lo_w, det.lo_h)
        return det

    def _close(self):
        with self._det_lock:
            det, self.det = self.det, None
        if det is not None:
            try:
                det.close()
            except Exception:  # noqa: BLE001 - best effort
                pass
            with self.cond:
                self.latest = {"seq": self.latest["seq"] + 1, "boxes": []}
                self.cond.notify_all()

    def _apply_cfg(self):
        d = self.det
        d.score_thr = self.cfg["conf"]
        d.class_ids = {COCO_NAMES.index("apple")} if (self.cfg["filter"] == "apple" and len(self.names) == 80) else None

    def _result_loop(self):
        while self._run:
            with self._det_lock:
                r = self.det.collect(timeout=0.05) if self.det is not None else None
                infer_s = self.det.last_infer_s if r is not None else 0.0
            if r is None:
                time.sleep(0.005)
                continue
            try:
                self._publish(*r, infer_s)
            except Exception as e:  # noqa: BLE001 - never kill the result thread
                log.exception("vision publish: %s", e)

    def _publish(self, boxes, tag, infer_s):
        ts_sensor, ts_py = tag
        now = time.monotonic_ns()
        self.t_done.append(time.monotonic())
        self.lat_total.append((now - ts_sensor) / 1e6)
        self.lat_sensor.append((ts_py - ts_sensor) / 1e6)
        self.lat_hailo.append(infer_s * 1e3)
        self.n_boxes.append(len(boxes))
        self.max_score.append(max((b[4] for b in boxes), default=0.0))
        self.frames += 1
        with self.cond:
            self.latest = {"seq": self.latest["seq"] + 1, "lat_ms": round((now - ts_sensor) / 1e6, 1),
                           "boxes": [[round(b[0], 4), round(b[1], 4), round(b[2], 4), round(b[3], 4),
                                      round(b[4], 3), self.names[b[5]] if b[5] < len(self.names) else b[5]]
                                     for b in boxes]}
            self.cond.notify_all()

    def _loop(self):
        while self._run:
            if not self.cfg["on"]:
                self._close()
                time.sleep(0.1)
                continue
            try:
                if self.det is None or self.det.hef_path != self.cfg["hef"]:
                    self._close()
                    det = self._open()          # warm-up runs before the result thread sees it
                    with self._det_lock:
                        self.det = det
                    self._reset_stats()
                self._apply_cfg()
                with self.camera.lock:                     # a mode switch waits for this frame
                    req = self.camera.cam.cam.capture_request()
                try:
                    ts = req.get_metadata().get("SensorTimestamp")
                    lo = req.make_array("lores")
                finally:
                    req.release()
                if lo.shape[:2] != (self.det.lo_h, self.det.lo_w):   # camera mode was switched
                    while self.det.in_flight:                          # result thread drains them
                        time.sleep(0.002)
                    self.det.set_lores_size(lo.shape[1], lo.shape[0])
                    self._reset_stats()
                fps = self.camera.mode[2] if self.camera.mode else 0
                if self._last_ts and fps and (ts - self._last_ts) / 1e9 > 1.5 / fps:
                    self.dropped += round((ts - self._last_ts) / 1e9 * fps) - 1
                self._last_ts = ts
                self.det.submit(lo, (ts, time.monotonic_ns()))   # result thread publishes it
                if time.monotonic() > self._next_temp:
                    self.chip_temp, self._next_temp = self.det.chip_temp_c(), time.monotonic() + 2.0
            except Exception as e:  # noqa: BLE001 - report on the page, keep the dashboard up
                log.exception("vision: %s", e)
                self.error = str(e).splitlines()[0][:200]
                self.cfg["on"] = False
                self._close()

    def wait(self, seen, timeout=1.0):
        with self.cond:
            self.cond.wait_for(lambda: self.latest["seq"] != seen, timeout)
            return self.latest

    def status(self):
        t = list(self.t_done)
        recent = [x for x in t if x > time.monotonic() - 1.0]
        fps = (len(recent) - 1) / (recent[-1] - recent[0]) if len(recent) > 2 else 0.0
        tot, sen, hai = list(self.lat_total), list(self.lat_sensor), list(self.lat_hailo)
        other = [a - b - c for a, b, c in zip(tot, sen, hai)]
        seen = self.frames + self.dropped
        return {
            "on": self.cfg["on"] and self.det is not None, "requested": self.cfg["on"],
            "hef": os.path.basename(self.cfg["hef"]), "hef_path": self.cfg["hef"], "models": self.models(),
            "conf": self.cfg["conf"], "filter": self.cfg["filter"], "classes": len(self.names),
            "lores": self.camera.lores_size, "fps": round(fps, 1), "frames": self.frames,
            "dropped_pct": round(100.0 * self.dropped / seen, 1) if seen else 0.0,
            "lat_ms": {"p50": _pct(tot, 0.5), "p95": _pct(tot, 0.95), "max": _pct(tot, 1.0)},
            "sensor_ms": _pct(sen, 0.5), "hailo_ms": _pct(hai, 0.5), "other_ms": _pct(other, 0.5),
            "boxes": {"now": self.n_boxes[-1] if self.n_boxes else 0,
                      "mean": round(statistics.mean(self.n_boxes), 1) if self.n_boxes else 0,
                      "max_score": round(self.max_score[-1], 2) if self.max_score else None},
            "chip_temp_c": self.chip_temp, "error": self.error,
        }

    def close(self):
        self._run = False
        self._thread.join(2.0)
        self._results.join(2.0)
        self._close()


# ---------------------------------------------------------------- telemetry
class Telemetry:
    """Samples every subsystem at TELEMETRY_HZ into .data (a JSON-able dict)."""

    def __init__(self, drive, camera, control, vision):
        self.drive, self.camera, self.control, self.vision = drive, camera, control, vision
        self.power = BoardPower()
        self.battery = BatteryMonitor()
        self.hailo = self._hailo_info()
        self.data = {}
        self._stat_prev = None      # previous /proc/stat counters for cpu utilisation
        self._run = True
        threading.Thread(target=self._loop, daemon=True).start()

    @staticmethod
    def _hailo_info():
        """Identify the Hailo once at start-up; it does not change while running."""
        try:
            h = HailoBoard()
            info = h.identify() if h.available() else {}
            return {"up": bool(info), "board": info.get("board_name"),
                    "fw": info.get("firmware_version", "").split(" ")[0]}
        except Exception as e:  # noqa: BLE001 - the dashboard must come up without the HAT
            return {"up": False, "error": str(e).splitlines()[0]}

    def _cpu_util(self):
        """Overall + per-core utilisation (%) since the previous call, from /proc/stat."""
        rows = {}
        with open("/proc/stat") as f:
            for ln in f:
                if ln.startswith("cpu"):
                    parts = ln.split()
                    vals = list(map(int, parts[1:]))
                    rows[parts[0]] = (sum(vals), vals[3] + vals[4])   # total, idle+iowait
        prev, self._stat_prev = self._stat_prev, rows
        if not prev:
            return {"total": None, "cores": []}

        def pct(k):
            dt = rows[k][0] - prev[k][0]
            return round(100.0 * (1 - (rows[k][1] - prev[k][1]) / dt), 1) if dt else 0.0
        return {"total": pct("cpu"), "cores": [pct(k) for k in sorted(rows) if k != "cpu"]}

    @staticmethod
    def _wifi():
        try:
            with open("/proc/net/wireless") as f:
                lines = f.readlines()[2:]
            for ln in lines:
                if ln.strip().startswith("wlan0"):
                    parts = ln.split()
                    return {"up": True, "link": float(parts[2]), "level_dbm": float(parts[3])}
        except OSError:
            pass
        return {"up": False}

    def _loop(self):
        while self._run:
            t0 = time.monotonic()
            try:
                ps = self.power.sample()
                self.data = {
                    "time": time.time(),
                    "cpu": {"temp_c": ps.temp_c, "arm_freq_mhz": ps.arm_freq_mhz,
                            "throttled_hex": ps.throttled_hex, "util": self._cpu_util()},
                    "power": {"total_w": ps.total_w, "ext5v_v": ps.ext5v_v},
                    "battery_v": round(self.battery.read(samples=8).battery_volts, 3),
                    "drive": self.drive.status(),
                    "camera": self.camera.status(),
                    "wifi": self._wifi(),
                    "hailo": self.hailo,
                    "control": self.control.status(),
                    "vision": self.vision.status(),
                }
            except Exception as e:  # noqa: BLE001 - keep sampling after a transient failure
                log.exception("telemetry: %s", e)
            time.sleep(max(0.0, 1.0 / TELEMETRY_HZ - (time.monotonic() - t0)))

    def health_line(self):
        """Compact health for the tcp link."""
        d = self.data.get("drive", {})
        h = {"h": 1, "v": self.data.get("battery_v"), "temp": self.data.get("cpu", {}).get("temp_c"),
             "rps": d.get("rps"), "us": d.get("pulse_us"), "st": d.get("steer"),
             "act": self.control.active, "stop": self.control.stopped, "lk": self.control.link_ages_ms()}
        return json.dumps(h, separators=(",", ":")) + "\n"

    def close(self):
        self._run = False


# ---------------------------------------------------------------- tcp command link
def serve_lines(conn, link, control, telem):
    """Blocking: read {"t","s"} lines from conn, write health at HEALTH_HZ, until it drops."""
    conn.settimeout(0.2)
    buf, next_health = b"", time.monotonic()
    try:
        while True:
            try:
                chunk = conn.recv(4096)
                if not chunk:
                    break
                buf += chunk
                while b"\n" in buf:
                    line, buf = buf.split(b"\n", 1)
                    if line.strip():
                        try:
                            m = json.loads(line)
                            control.command(link, m.get("t", 0.0), m.get("s", 0.0))
                        except (ValueError, TypeError):
                            pass
            except socket.timeout:
                pass
            if time.monotonic() >= next_health:
                next_health += 1.0 / HEALTH_HZ
                conn.sendall(telem.health_line().encode())
    except OSError as e:
        log.info("%s link closed: %s", link, e)
    finally:
        conn.close()


def tcp_server(control, telem):
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(("", TCP_PORT))
    srv.listen(2)
    log.info("tcp control link on :%d", TCP_PORT)
    while True:
        conn, addr = srv.accept()
        conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        log.info("tcp client %s", addr)
        threading.Thread(target=serve_lines, args=(conn, "tcp", control, telem), daemon=True).start()


# ---------------------------------------------------------------- http / websocket
async def index(request):
    return web.FileResponse(os.path.join(WEB_DIR, "index.html"))


async def api_telemetry(request):
    return web.json_response(request.app["telem"].data)


async def api_mode(request):
    m = await request.json()
    cam = request.app["camera"]
    try:
        mode = await asyncio.get_running_loop().run_in_executor(None, cam.set_mode, m["w"], m["h"], m["fps"])
    except RuntimeError as e:
        return web.json_response({"error": str(e)}, status=409)
    return web.json_response({"mode": mode})


async def api_record(request):
    on = (await request.json()).get("on", True)
    cam = request.app["camera"]
    path = await asyncio.get_running_loop().run_in_executor(
        None, cam.start_recording if on else cam.stop_recording)
    return web.json_response({"recording": on, "file": path and os.path.basename(path)})


async def api_recordings(request):
    return web.json_response(request.app["camera"].recordings())


async def api_recording_delete(request):
    name = os.path.basename(request.match_info["name"])
    path = os.path.join(REC_DIR, name)
    if os.path.isfile(path):
        os.remove(path)
    return web.json_response({"deleted": name})


async def stream(request):
    """MJPEG: one multipart chunk per camera frame."""
    out = request.app["camera"].out
    resp = web.StreamResponse(headers={"Content-Type": "multipart/x-mixed-replace; boundary=frame",
                                       "Cache-Control": "no-cache"})
    await resp.prepare(request)
    loop, seen = asyncio.get_running_loop(), -1
    try:
        while True:
            frame, seen = await loop.run_in_executor(None, out.wait, seen)
            if frame is None:
                continue
            await resp.write(b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: %d\r\n\r\n" % len(frame))
            await resp.write(frame + b"\r\n")
    except (ConnectionResetError, asyncio.CancelledError):
        pass
    return resp


async def api_vision(request):
    """{"on": bool, "hef": path, "conf": 0..1, "filter": "apple"|"all"} - any subset."""
    m = await request.json()
    vision = request.app["vision"]
    if m.get("hef") and m["hef"] not in vision.models():
        return web.json_response({"error": "unknown model %s" % m["hef"]}, status=400)
    return web.json_response(vision.configure(**m))


async def ws_detections(request):
    """Latest boxes for the page overlay, pushed as soon as they exist (max DET_PUSH_HZ):
    {"seq", "lat_ms", "boxes": [[x1, y1, x2, y2, score, label], ...]}, coords 0..1 of the frame."""
    ws = web.WebSocketResponse(heartbeat=5)
    await ws.prepare(request)
    vision, loop, seen = request.app["vision"], asyncio.get_running_loop(), -1
    try:
        while not ws.closed:
            t0 = loop.time()
            latest = await loop.run_in_executor(None, vision.wait, seen)
            if latest["seq"] != seen:
                seen = latest["seq"]
                await ws.send_json(latest)
            await asyncio.sleep(max(0.0, 1.0 / DET_PUSH_HZ - (loop.time() - t0)))
    except (ConnectionResetError, asyncio.CancelledError):
        pass
    return ws


async def websocket(request):
    """Telemetry push at TELEMETRY_HZ; incoming messages are drive commands."""
    ws = web.WebSocketResponse(heartbeat=5)
    await ws.prepare(request)
    control, telem = request.app["control"], request.app["telem"]

    async def push():
        while not ws.closed:
            await ws.send_json(telem.data)
            await asyncio.sleep(1.0 / TELEMETRY_HZ)
    task = asyncio.ensure_future(push())
    try:
        async for msg in ws:
            if msg.type == web.WSMsgType.TEXT:
                try:
                    m = json.loads(msg.data)
                    control.command("ws", m.get("t", 0.0), m.get("s", 0.0))
                except (ValueError, TypeError):
                    pass
    finally:
        task.cancel()
    return ws


async def serve(app):
    runner = web.AppRunner(app, access_log=None)
    await runner.setup()
    await web.TCPSite(runner, port=HTTP_PORT).start()
    log.info("http://0.0.0.0:%d", HTTP_PORT)
    await asyncio.Event().wait()


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    drive = Drive(max_throttle=MAX_THROTTLE)
    camera = CameraStream()
    control = Control(drive)
    vision = Vision(camera)
    telem = Telemetry(drive, camera, control, vision)
    threading.Thread(target=tcp_server, args=(control, telem), daemon=True).start()

    app = web.Application()
    app.update(drive=drive, camera=camera, control=control, telem=telem, vision=vision)
    app.add_routes([
        web.get("/", index),
        web.get("/stream.mjpg", stream),
        web.get("/ws", websocket),
        web.get("/ws/det", ws_detections),
        web.post("/api/vision", api_vision),
        web.get("/api/telemetry", api_telemetry),
        web.post("/api/mode", api_mode),
        web.post("/api/record", api_record),
        web.get("/api/recordings", api_recordings),
        web.delete("/api/recordings/{name}", api_recording_delete),
        web.static("/recordings", REC_DIR),
    ])
    try:
        asyncio.run(serve(app))
    except KeyboardInterrupt:
        pass
    finally:
        control.close()
        telem.close()
        vision.close()
        camera.close()
        drive.close()


if __name__ == "__main__":
    main()
