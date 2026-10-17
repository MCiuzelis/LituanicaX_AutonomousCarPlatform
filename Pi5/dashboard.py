#!/usr/bin/env python3
"""RC car dashboard: live camera, telemetry and remote control server.

    python3 ~/LituanicaX_InteligentRobotCompetition/Pi5/dashboard.py            (or scripts/dashboard_restart.sh to run it detached)
    then open http://rp5:8080  (or http://<pi-ip>:8080)

Serves
  http://<pi>:8080/        the dashboard page (web/index.html): camera preview, resolution
                           switching, training-data recording (H.264 + per-frame IMU / odometry,
                           recorder.py), all telemetry, keyboard driving
  /ws/video                camera preview for the page: each message = one JPEG of the detector's own
                           frame + the boxes found in THAT frame (so video and boxes are in sync);
                           the page acks every frame, so nothing queues up on a slow WiFi link (Preview)
  /ws/det, /api/vision     Hailo object detection on the camera (Vision class): boxes, and on/off +
                           model/threshold/class filter settings
  /ws/imu, /api/imu/*      BNO085 attitude / acceleration / yaw rate in the car frame (Imu class)
  /ws/map, /api/map        track map: apples of the last few seconds (rolling window) and the track
                           boundaries + the planner's suggestion (shadow mode); vision/map_worker.py,
                           its own process (MapProcess); recorded into every take

Driving does NOT go through this process: drive_server.py (started from here, its own process with a
realtime network thread) owns throttle, steering and the motor encoder and serves the page's command
WebSocket (:8082/ws/ctl) and car_client.py's tcp link (:8081). Nothing here (camera, Hailo, recorder,
map) can delay a command. This process reads the drive state from shared memory (DriveLink).
Safety (drive_server.py): throttle capped at 40 %, neutral + centre if no command for 0.4 s.
"""
import asyncio
import json
import logging
import math
import os
import queue
import signal
import socket
import statistics
import struct
import subprocess
import sys
import threading
import time
from collections import deque
from multiprocessing.connection import Connection

import numpy as np
from aiohttp import web

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "vision"))
from hardware import Camera, BoardPower, BatteryMonitor, HailoBoard, BNO085, CarFrame, HardwareError  # noqa: E402
from recorder import Recording, list_recordings, delete_recording  # noqa: E402
import drive_server  # noqa: E402

HTTP_PORT = 8080
TELEMETRY_HZ = 4             # telemetry sampling + WebSocket push rate
# (width, height, fps). Sensor modes: 1536x864@120, 2304x1296@56, 4608x2592@14; the ISP scales
# any smaller output from the chosen one. tests/camera_sweep.py showed the software MJPEG
# encoder keeps up with all of these (1536x864@120 at ~16 % CPU on YUV420).
CAMERA_MODES = [(640, 360, 120), (640, 480, 120), (800, 600, 120), (1024, 576, 120),
                (1280, 720, 120), (1280, 720, 100), (1280, 720, 60), (1536, 864, 120),
                (1920, 1080, 56), (2304, 1296, 56), (2304, 1296, 30)]
DEFAULT_MODE = (1536, 864, 120)   # full 1536x864 sensor mode: detection at 120 fps, recordings at 60
BUFFERS_SMALL, BUFFERS_LARGE = 16, 10   # camera buffers for modes up to 1536x864 / larger
PREVIEW_FPS = 30             # preview to the page: at most this rate (the page's acks may slow it further)
PREVIEW_SIZES = {"480": (480, 270), "640": (640, 360), "960": (960, 540)}   # scaled from the detector frame
PREVIEW_DEFAULT = "640"
PREVIEW_QUALITY = 70
# Recording takes every Nth camera frame, at most this rate (YOLO training needs no more; the Pi 5
# has no H.264 hardware: x264 at 1536x864@60 took ~1.9 cores and pulled detection down to ~70 fps,
# at 30 fps detection keeps 120 fps while recording; measured 2026-10-11)
REC_FPS_SMALL, REC_FPS_LARGE = 30, 30    # modes up to 1536x864 / larger
WEB_DIR = os.path.join(ROOT, "web")
REC_DIR = os.path.join(ROOT, "recordings")
MODEL_DIR = os.path.join(ROOT, "models")
STOCK_HEF = "/usr/share/hailo-models/yolov8s_h8.hef"   # COCO yolov8s, 640x640 input
DET_INPUT = (960, 544)       # input (w, h) of models/apple_yolov8s.hef; the lores stream follows the loaded model
DET_PUSH_HZ = 30             # max rate boxes are pushed to the page overlay
IMU_PUSH_HZ = 25             # attitude / g-g plot push rate (/ws/imu); the IMU itself runs at 100 Hz
MAP_PUSH_HZ = 10             # track map push rate (/ws/map); the map itself updates on every detection frame
VIDEO_CREDITS = 2            # preview frames in flight to one page before it must ack
AUTO_STATES = ("off", "driving", "braking")   # drive_server.AUTO_OFF / _DRIVING / _BRAKING
COCO_NAMES = (
    "person bicycle car motorcycle airplane bus train truck boat traffic_light fire_hydrant "
    "stop_sign parking_meter bench bird cat dog horse sheep cow elephant bear zebra giraffe "
    "backpack umbrella handbag tie suitcase frisbee skis snowboard sports_ball kite baseball_bat "
    "baseball_glove skateboard surfboard tennis_racket bottle wine_glass cup fork knife spoon bowl "
    "banana apple sandwich orange broccoli carrot hot_dog pizza donut cake chair couch potted_plant "
    "bed dining_table toilet tv laptop mouse remote keyboard cell_phone microwave oven toaster sink "
    "refrigerator book clock vase scissors teddy_bear hair_drier toothbrush").split()

log = logging.getLogger("dashboard")


# ---------------------------------------------------------------- drive (separate process)
class DriveLink:
    """drive_server.py as a child process + a read view of its shared-memory state.

    Looks like the old in-process Drive / Control to the rest of this file and to recorder.py:
    .turns .rps .rpm .steering.position .esc.current_us .status(), .throttle .active .stopped
    .link_ages_ms() .control_status(). command() goes to the server as link "auto" (any human
    command in the last 0.5 s wins over it); the page and car_client.py talk to the server directly."""

    class _Steering:
        def __init__(self, link):
            self._l = link

        @property
        def position(self):
            return self._l.read()["steer_pos"]

    class _Esc:
        def __init__(self, link):
            self._l = link

        @property
        def current_us(self):
            return self._l.read()["esc_us"]

    def __init__(self):
        self.steering, self.esc = self._Steering(self), self._Esc(self)
        self.proc, self.state, self.restarts = None, None, 0
        self.closing = False
        self.wifi_dbm = None           # set by Telemetry (4 Hz), recorded with every drive sample
        self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
        self._cache, self._cache_t = None, 0.0
        self.mm_per_turn = drive_server._mm_per_turn()
        self._start()
        threading.Thread(target=self._monitor, daemon=True).start()

    def _start(self):
        subprocess.run(["pkill", "-f", "[d]rive_server.py"], check=False)   # a stale one from a killed dashboard
        time.sleep(0.3)
        self.proc = subprocess.Popen([sys.executable, os.path.join(ROOT, "drive_server.py")], cwd=ROOT)
        t_end = time.monotonic() + 15.0
        while time.monotonic() < t_end:
            try:
                st = drive_server.DriveState()
                if st.read()["pid"] == self.proc.pid:
                    self.state = st
                    log.info("drive server pid %d (rt priority %s)", self.proc.pid, bool(st.read()["rt"]))
                    return
                st.close()
            except FileNotFoundError:
                pass
            if self.proc.poll() is not None:
                raise RuntimeError("drive server exited with %s (see the log)" % self.proc.returncode)
            time.sleep(0.1)
        raise RuntimeError("drive server did not come up")

    def _monitor(self):
        while True:
            time.sleep(1.0)
            if self.proc.poll() is not None and not self.closing:
                log.error("drive server exited (%s): restarting", self.proc.returncode)
                self.restarts += 1
                try:
                    if self.state is not None:
                        st, self.state = self.state, None
                        st.close()
                    self._start()
                except Exception as e:  # noqa: BLE001 - keep trying
                    log.error("drive server restart failed: %s", e)

    def read(self):
        """Server state (dict of drive_server.FIELDS), at most 1 ms old."""
        now = time.monotonic()
        if self._cache is None or now - self._cache_t > 0.001:
            st = self.state
            self._cache = st.read() if st is not None else dict.fromkeys(drive_server.FIELDS, 0.0)
            self._cache_t = now
        return self._cache

    def _send(self, msg):
        try:
            self.sock.sendto(json.dumps(msg).encode(), drive_server.SOCK)
        except OSError:
            pass

    def command(self, link, throttle, steer):
        self._send({"link": link, "t": throttle, "s": steer})

    def send_health(self, volts, temp):
        self._send({"health": {"v": volts, "temp": temp}})

    # ---- drive view (recorder.py, telemetry, track map)
    @property
    def turns(self):
        return self.read()["turns"]

    @property
    def rps(self):
        return self.read()["rps"]

    @property
    def rpm(self):
        return self.rps * 60.0

    def status(self):
        s = self.read()
        return {"pulse_us": s["esc_us"], "rps": round(s["rps"], 2), "rpm": round(s["rps"] * 60),
                "turns": round(s["turns"], 2), "steer": s["steer_pos"]}

    # ---- control view
    @property
    def throttle(self):
        return self.read()["throttle"]

    @property
    def active(self):
        a = int(self.read()["active"])
        return drive_server.LINKS[a - 1] if a else None

    @property
    def stopped(self):
        return bool(self.read()["stopped"])

    def link_ages_ms(self):
        s, now = self.read(), time.monotonic()
        return {k: int((now - s["last_" + k]) * 1000) for k in drive_server.LINKS if s["last_" + k]}

    def control_status(self):
        s = self.read()
        return {"throttle": s["throttle"], "steer": s["steer"], "active": self.active, "stopped": bool(s["stopped"]),
                "max_throttle": s["max_throttle"], "links_ms": self.link_ages_ms(),
                "gap_max_ms": {"ws": round(s["gap_ws_ms"]), "tcp": round(s["gap_tcp_ms"])},
                "late": {"ws": int(s["late_ws"]), "tcp": int(s["late_tcp"])}, "deadman": int(s["deadman"]),
                "commands": int(s["commands"]), "rt": bool(s["rt"]), "ws_clients": int(s["ws_clients"]),
                "server_pid": int(s["pid"]), "server_restarts": self.restarts, "port": drive_server.WS_PORT,
                "auto": {"armed": bool(s["armed"]), "state": AUTO_STATES[int(s["auto_state"])],
                         "v_target": round(s["auto_v"]), "ceiling": s["arm_max"], "arms": int(s["arms"]),
                         "plan_age_ms": int((time.monotonic() - s["plan_t"]) * 1000) if s["plan_t"] else None,
                         "plan_steer": round(s["plan_steer"], 2), "plan_v": round(s["plan_v"]),
                         "plan_mode": drive_server.PLAN_MODES[int(s["plan_mode"])],
                         "speed_mm_s": round(abs(s["rps"]) * self.mm_per_turn)}}

    def link_sample(self):
        """Extra _drive.csv columns (recorder.LINK_COLUMNS)."""
        s, now = self.read(), time.monotonic()
        newest = max(s["last_ws"], s["last_tcp"], s["last_auto"])
        a = int(s["active"])
        return (drive_server.LINKS[a - 1] if a else "", int((now - newest) * 1000) if newest else "",
                "" if self.wifi_dbm is None else int(self.wifi_dbm), int(s["auto_state"]),
                round(s["auto_v"]) if s["auto_state"] else "",
                round(s["auto_u"], 3), round(s["auto_vmeas"]), round(s["arm_max"], 3),
                drive_server.PLAN_MODES[int(s["plan_mode"])] if s["plan_t"] else "",
                round(s["plan_steer"], 3) if s["plan_t"] else "", round(s["plan_v"]) if s["plan_t"] else "",
                int((now - s["plan_t"]) * 1000) if s["plan_t"] else "")

    def close(self):
        self.closing = True
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(5.0)
            except subprocess.TimeoutExpired:
                self.proc.kill()
        if self.state is not None:
            self.state.close()


# ---------------------------------------------------------------- camera preview
class Preview:
    """JPEG preview of the detector's frames for the page, with each frame's own boxes.

    While detection runs, Vision hands over every result (frame + boxes); the newest is scaled to
    the requested PREVIEW_SIZES and encoded at <= PREVIEW_FPS. With detection off this captures the
    lores stream itself. JPEG encoding and scaling release the GIL. Readers: /ws/video (one message
    per frame, sent only when the page has acked the previous ones) and /stream.mjpg."""

    def __init__(self, camera):
        self.camera = camera
        self.lock = threading.Lock()
        self.frames = {}               # size key -> (seq, payload bytes, jpeg bytes)
        self.wanted = {PREVIEW_DEFAULT: 0.0}   # size key -> last time a reader asked for it
        self.subs = set()              # (loop, asyncio.Event) of waiting readers
        self._offer, self._offer_evt = None, threading.Event()
        self.seq = 0
        self.times = deque(maxlen=60)
        self.enc_ms = deque(maxlen=120)
        self._run = True
        threading.Thread(target=self._loop, daemon=True).start()

    def offer(self, lores, header):
        """Vision result thread: newest detection frame (RGB lores array) + its header. Cheap."""
        self._offer = (lores, header, time.monotonic())
        self._offer_evt.set()

    def want(self, key):
        self.wanted[key] = time.monotonic()

    def subscribe(self, loop):
        item = (loop, asyncio.Event())
        with self.lock:
            self.subs.add(item)
        return item

    def unsubscribe(self, item):
        with self.lock:
            self.subs.discard(item)

    def latest(self, key):
        return self.frames.get(key)

    def _capture(self):
        """Own lores frame while detection is off."""
        cam = self.camera
        with cam.lock:
            req = cam.cam.cam.capture_request()
        try:
            md = req.get_metadata()
            lo = req.make_array("lores")
        finally:
            req.release()
        return lo, {"boxes": [], "det": False, "ts": md.get("SensorTimestamp")}

    def _loop(self):
        import cv2
        import simplejpeg
        nxt = time.monotonic()
        while self._run:
            off = self._offer
            if off is not None and time.monotonic() - off[2] < 0.25:   # detection running: wait for its next frame
                self._offer_evt.wait(1.5 / PREVIEW_FPS)
            self._offer_evt.clear()
            now = time.monotonic()
            if now < nxt:
                time.sleep(nxt - now)
                continue                   # then take the newest frame that arrives
            try:
                off = self._offer
                if off is not None and now - off[2] < 0.25:
                    lo, hdr = off[0], dict(off[1])
                else:
                    lo, hdr = self._capture()
                nxt = now + 1.0 / PREVIEW_FPS
                keys = [k for k, t in self.wanted.items() if now - t < 3.0 or k == PREVIEW_DEFAULT]
                a = time.monotonic()
                self.seq += 1
                for k in keys:
                    w, h = PREVIEW_SIZES[k]
                    img = lo if lo.shape[1] <= w else cv2.resize(lo, (w, h), interpolation=cv2.INTER_LINEAR)
                    jpg = simplejpeg.encode_jpeg(np.ascontiguousarray(img), quality=PREVIEW_QUALITY, colorspace="RGB")
                    ts = hdr.get("ts")
                    hd = dict(hdr, seq=self.seq, w=img.shape[1], h=img.shape[0],
                              age_ms=round((time.monotonic_ns() - ts) / 1e6, 1) if ts else None)
                    hb = json.dumps(hd, separators=(",", ":")).encode()
                    self.frames[k] = (self.seq, struct.pack("<I", len(hb)) + hb + jpg, jpg)
                self.enc_ms.append((time.monotonic() - a) * 1000)
                self.times.append(time.monotonic())
                with self.lock:
                    subs = list(self.subs)
                for loop, ev in subs:
                    loop.call_soon_threadsafe(ev.set)
            except Exception as e:  # noqa: BLE001 - camera switching etc.: try again
                log.warning("preview: %s", e)
                time.sleep(0.2)

    @property
    def fps(self):
        recent = [x for x in list(self.times) if x > time.monotonic() - 2.0]
        return (len(recent) - 1) / (recent[-1] - recent[0]) if len(recent) > 2 else 0.0

    def status(self):
        return {"fps": round(self.fps, 1), "enc_ms": _pct(list(self.enc_ms), 0.5),
                "sizes": sorted(k for k, t in self.wanted.items() if time.monotonic() - t < 3.0),
                "readers": len(self.subs)}

    def close(self):
        self._run = False


class CameraStream:
    """Camera: full-resolution main stream (recordings: H.264 + per-frame IMU / odometry in
    recordings/<id>/, recorder.Recording) + the lores stream sized for the detector (Vision, and the
    page preview: Preview). Nothing JPEG-encodes the main stream any more."""

    def __init__(self, mode=DEFAULT_MODE):
        self.cam = Camera(pixel_format="YUV420")   # cheaper to H.264-encode than RGB888
        self.preview = None        # Preview (its frame rate is reported here)
        self.lock = threading.Lock()
        self.mode = None           # (width, height, fps) currently streaming
        self.lores_size = None     # RGB copy of every frame for the detector, scaled to fit det_input
        self.det_input = DET_INPUT
        self.rec = None            # recorder.Recording while recording
        self.trackmap = None       # MapProcess: its log goes into every take
        os.makedirs(REC_DIR, exist_ok=True)
        self.set_mode(*mode)

    def set_mode(self, w, h, fps):
        with self.lock:
            if self.rec:
                raise RuntimeError("stop the recording before switching resolution")
            # detector input: same aspect as main, as large as fits the model input (ISP-scaled, costs no
            # CPU); never larger than main, the detector letterboxes whatever is left
            nw, nh = self.det_input
            s = min(nw / int(w), nh / int(h), 1.0)
            lo = (min(nw, round(int(w) * s / 2) * 2), min(nh, round(int(h) * s / 2) * 2))
            # extra buffers absorb hiccups of the software H.264 encoder while recording (default 6)
            nbuf = BUFFERS_SMALL if int(w) * int(h) <= 1536 * 864 else BUFFERS_LARGE
            m = self.cam.set_mode(int(w), int(h), float(fps), settle_s=0.0, lores=(*lo, "BGR888"), buffer_count=nbuf)
            self.lores_size = lo
            self.mode = (m.width, m.height, m.fps)
            log.info("camera %dx%d @ %.1f fps", *self.mode)
            return self.mode

    def set_det_input(self, w, h):
        """Resize the lores stream for a model with a (w, h) input; restarts the camera if it changes."""
        if (w, h) != self.det_input:
            self.det_input = (w, h)
            if self.mode is not None:
                self.set_mode(*self.mode)

    def start_recording(self, imu=None, drive=None, control=None):
        """Start a take (IMU / drive / control are sampled into it); returns its id."""
        with self.lock:
            if self.rec:
                return self.rec.id
            w, h, fps = self.mode
            cap = REC_FPS_SMALL if w * h <= 1536 * 864 else REC_FPS_LARGE
            skip = max(1, math.ceil(fps / cap - 0.01))
            self.rec = Recording(REC_DIR, self.cam.cam, self.cam.mode, skip=skip, imu=imu, drive=drive, control=control)
            if self.trackmap is not None:
                self.trackmap.start_log(self.rec.dir, self.rec.id)    # fresh map: the take's world frame
            log.info("recording -> %s (%.0f Mbit/s)", self.rec.dir, self.rec.bitrate / 1e6)
            return self.rec.id

    def stop_recording(self):
        with self.lock:
            if not self.rec:
                return None
            rec, self.rec = self.rec, None
            if self.trackmap is not None:
                tm = self.trackmap.stop_log()
                if tm:
                    rec.meta["files"].update(tm["files"])
                    rec.meta["track_map"] = tm["meta"]
            s = rec.stop()
            log.info("recording saved %s: %d frames, %d dropped, %d without IMU, %.1f MB", rec.id, s["frames"],
                     s["dropped_frames"], s["frames_without_imu"], s["video_bytes"] / 1e6)
            return rec.id

    @staticmethod
    def recordings():
        return list_recordings(REC_DIR)

    def status(self):
        rec, r = None, self.rec
        if r:
            rec = r.status()
            rec["file"] = rec["id"]
            if rec["disk_low"] and not getattr(r, "stopping", False):
                log.warning("recording: less than 500 MB free, stopping")
                r.stopping = True
                threading.Thread(target=self.stop_recording, daemon=True).start()
        fs = os.statvfs(REC_DIR)
        m = self.cam.mode
        return {"mode": self.mode, "stream_fps": round(self.preview.fps, 1) if self.preview else None,
                "preview": self.preview.status() if self.preview else None, "modes": CAMERA_MODES,
                "sensor": m and "%dx%d" % m.sensor.size,
                "recording": rec, "disk_free_gb": round(fs.f_bavail * fs.f_frsize / 1e9, 2)}

    def close(self):
        self.stop_recording()
        with self.lock:
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
        self.on_boxes = None       # callback(t_mid_s, boxes, crop, seq, sensor_ts_ns): the track map
        self.preview = None        # Preview: gets every result's frame + boxes (video and boxes in sync)
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
        from apple_detector import AppleDetector, hef_input_size   # needs python3-hailort; loaded when used
        self.camera.set_det_input(*hef_input_size(self.cfg["hef"]))
        det = AppleDetector(self.cfg["hef"], lores_size=self.camera.lores_size)
        det.detect(np.full((det.lo_h, det.lo_w, 3), 114, np.uint8))   # learn class count
        self.names = COCO_NAMES if det.num_classes == 80 else ["apple"] if det.num_classes == 1 \
            else ["class%d" % i for i in range(det.num_classes)]
        log.info("vision: %s (%d classes, %dx%d input) on %dx%d lores", self.cfg["hef"], det.num_classes,
                 det.net_w, det.net_h, det.lo_w, det.lo_h)
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
        ts_sensor, ts_py, t_mid, crop, lo = tag
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
            latest = self.latest
            self.cond.notify_all()
        if self.preview is not None:
            self.preview.offer(lo, {"boxes": latest["boxes"], "det": True, "ts": ts_sensor, "lat_ms": latest["lat_ms"],
                                    "det_seq": latest["seq"]})
        if self.on_boxes is not None and crop:
            self.on_boxes(t_mid, boxes, crop, self.latest["seq"], ts_sensor)

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
                    md = req.get_metadata()
                    ts = md.get("SensorTimestamp")
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
                # exposure middle of the middle row (as recorder.py's t_mid): when the boxes "happened"
                m = self.camera.cam.mode
                readout_ns = 1e9 / m.sensor.max_fps if m else 0.0
                t_mid = (ts - md.get("ExposureTime", 0) * 500 + readout_ns / 2) * 1e-9
                self.det.submit(lo, (ts, time.monotonic_ns(), t_mid, tuple(md.get("ScalerCrop", ())), lo))
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


# ---------------------------------------------------------------- IMU
class Imu:
    """BNO085 (SENSORS at RATE_HZ, no magnetometer) reported in the car frame.

    Opened in the background and retried, so the dashboard comes up without it. Attitude comes
    from game_rv (gyro + accel, no magnetometer: the motor and battery leads disturb it), so yaw
    is relative: it is zeroed at start-up and by zero_yaw(). The magnetometer is never enabled
    (nor rv, which fuses it): it is useless here and its reports cut the hub's rate for the rest. The board -> car rotation and the
    level reference live in config/imu.json (hardware.CarFrame); level() re-captures it.
    """

    SENSORS = ("game_rv", "accel", "gyro", "linear_accel", "gravity")
    RATE_HZ = 200
    RETRY_S = 5.0
    GAP_S = 0.05                 # a game_rv gap above 50 ms counts as a dropout
    WARMUP_S = 2.0

    def __init__(self):
        self.imu, self.error = None, None
        self.frame = CarFrame.load()
        self.yaw0 = 0.0
        self.cal = None              # dynamic calibration enables as read back from the hub
        self.gaps = {"count": 0, "max_ms": 0.0, "last": None, "prev_t": None, "since": None,
                     "armed_at": float("inf")}
        self._run = True
        threading.Thread(target=self._open, daemon=True).start()

    def _on_sample(self, r):
        """Every game_rv sample (driver's draining thread): track gaps in the data itself."""
        g = self.gaps
        if g["prev_t"] is not None and r.timestamp > g["armed_at"]:
            dt = r.timestamp - g["prev_t"]
            if dt > self.GAP_S:
                g["count"] += 1
                g["last"] = time.time()
            g["max_ms"] = max(g["max_ms"], dt * 1000)
        g["prev_t"] = r.timestamp

    def _open(self):
        while self._run and self.imu is None:
            imu = None
            try:
                imu = BNO085()
                imu.enable(*self.SENSORS, rate_hz=self.RATE_HZ)
                # gyro self-calibration is off at hub boot; the driver re-applies this after resets.
                # accel learning stays OFF: it needs still poses in several orientations, which a
                # driving car never gives, and the hub auto-saves what it learns to flash, so it
                # could overwrite the calibration from tools/imu_cal.py
                if not imu.set_calibration(accel=False, gyro=True, mag=False):
                    log.warning("imu: enabling dynamic calibration failed")
                self.cal = imu.calibration()
                # fusion outputs come at ~12 Hz for the first ~0.75 s after enabling (hub start-up),
                # so dropouts are counted from WARMUP_S on; later gaps (hub resets etc.) are real
                self.gaps["armed_at"] = time.monotonic() + self.WARMUP_S
                self.gaps["since"] = time.time() + self.WARMUP_S
                imu.on("game_rv", self._on_sample)
                self.imu, self.error = imu, None
                log.info("imu: BNO085 fw %s, %s, rt priority %s", (imu.product or [{}])[0].get("version"),
                         "levelled %s" % self.frame.level_time if self.frame.level_gravity else "NOT levelled",
                         imu.rt_priority)
                self.zero_yaw()
            except Exception as e:  # noqa: BLE001 - keep the dashboard up, retry
                self.error = str(e).splitlines()[0][:200]
                log.warning("imu: %s (retry in %.0f s)", self.error, self.RETRY_S)
                if imu is not None:
                    imu.close()
                time.sleep(self.RETRY_S)

    def sample(self):
        """Car-frame snapshot, or None while the IMU is down. Angles deg, rates deg/s, accel m/s^2.
        roll + = right side down, pitch + = nose up, yaw + = turned left; acc = (forward, left, up)
        without gravity; rate = (roll, pitch, yaw) rate, yaw + = turning left."""
        imu = self.imu
        if imu is None:
            return None
        r = imu.read("game_rv")
        if r is None:
            return None
        q = r.values[:4]
        att = self.frame.attitude(q)
        la = self.frame.to_car(imu.linear_acceleration) or (0.0, 0.0, 0.0)
        w = self.frame.to_car(imu.gyro) or (0.0, 0.0, 0.0)
        yaw = (att[2] - self.yaw0 + 180.0) % 360.0 - 180.0
        return {"roll": round(att[0], 2), "pitch": round(att[1], 2), "yaw": round(yaw, 1),
                "tilt": round(self.frame.tilt(q), 2),
                "acc": [round(x, 3) for x in la],
                "rate": [round(math.degrees(x), 2) for x in w],
                "age_ms": round((time.monotonic() - r.timestamp) * 1000, 1)}   # how old the attitude is

    def zero_yaw(self):
        att = self.imu and self.frame.attitude(self.imu.game_quaternion)
        if att:
            self.yaw0 = att[2]

    def level(self):
        """Current pose becomes roll = pitch = 0 (car still, on level ground); saved to config."""
        if self.imu is None:
            raise RuntimeError("IMU not running: %s" % self.error)
        try:
            g = CarFrame.capture_gravity(self.imu)
        except HardwareError as e:
            raise RuntimeError(str(e)) from e
        frame = CarFrame(self.frame.forward, self.frame.up, path=self.frame.path)
        frame.set_level(g)
        frame.save()
        self.frame = frame
        self.zero_yaw()
        log.info("imu: levelled, mount tilt %.2f deg, gravity %s", frame.mount_tilt_deg(), frame.level_gravity)
        return frame.status()

    def status(self):
        imu = self.imu
        if imu is None:
            return {"up": False, "error": self.error, "frame": self.frame.status()}
        st = imu.stats()
        link = st["link"]
        acc = {}
        for n in ("game_rv", "accel", "gyro"):
            r = imu.read(n)
            acc[n] = r.accuracy if r else None
        a = imu.acceleration
        return {"up": imu.available(), "error": imu.last_error, "values": self.sample(),
                "sensors": {n: {"hz": v["rate_hz"], "lost": v["lost"]} for n, v in st["sensors"].items()},
                "accuracy": acc, "g": a and round(math.sqrt(sum(x * x for x in a)), 3),
                "hub_resets": sum(v for k, v in link.items() if k.startswith("resets_") and k != "resets_start"),
                "worker_restarts": link.get("worker_restarts", 0), "rt": link.get("rt_priority"),
                "int_p99_us": (link.get("int_latency_us") or {}).get("p99"),
                "gaps": {"count": self.gaps["count"], "max_ms": round(self.gaps["max_ms"], 1),
                         "last": self.gaps["last"], "since": self.gaps["since"]},
                "self_cal": self.cal and [k for k in ("accel", "gyro") if self.cal.get(k)],
                "fw": (imu.product or [{}])[0].get("version"), "frame": self.frame.status()}

    def close(self):
        self._run = False
        if self.imu is not None:
            self.imu.close()


# ---------------------------------------------------------------- track map (separate process)
class MapProcess:
    """vision/map_worker.py (track map + planner) as a child process.

    A feeder thread forwards, every PERIOD_S, the raw game_rv samples (board-frame quaternions; the
    worker converts them to car attitude), the encoder turns and the detection frames
    (Vision.on_boxes) in one message. The worker answers at 10 Hz with the finished JSON snapshot,
    which /ws/map forwards as is. Recording: start_log() resets the map and logs into the take,
    stop_log() returns the take's map files + metadata."""

    PERIOD_S = 0.01

    def __init__(self, imu, drive):
        self.imu, self.drive = imu, drive
        self.lock = threading.Lock()              # serialises sends on the pipe
        self._dets = deque(maxlen=2000)
        self.sub = None
        self.snap_json, self.st = json.dumps({"up": False}), {"up": False}
        self.snap_at = 0.0
        self.replies = queue.Queue()
        self.restarts, self.error = 0, None
        self.conn = self.proc = None
        self._start()
        self._run = True
        threading.Thread(target=self._feeder, daemon=True).start()

    def _start(self):
        a, b = socket.socketpair()
        self.conn = Connection(a.detach())
        fd = b.detach()
        self.proc = subprocess.Popen([sys.executable, "-m", "vision.map_worker", str(fd)], cwd=ROOT, pass_fds=(fd,))
        os.close(fd)
        threading.Thread(target=self._reader, args=(self.conn,), daemon=True).start()
        log.info("track map worker pid %d", self.proc.pid)

    def _reader(self, conn):
        while True:
            try:
                msg = conn.recv()
            except (EOFError, OSError):
                return
            if msg[0] == "snap":
                self.snap_json, self.st, self.snap_at = msg[1], msg[2], time.monotonic()
            elif msg[0] == "stopped":
                self.replies.put(msg[1])

    def _send(self, msg):
        with self.lock:
            self.conn.send(msg)

    def on_boxes(self, t_mid, boxes, crop, seq, sensor_ts):
        """Vision result thread: keep it cheap."""
        self._dets.append((t_mid, boxes, crop, seq, sensor_ts))

    def _feeder(self):
        while self._run:
            t0 = time.monotonic()
            try:
                dev = self.imu.imu
                if dev is not None and self.sub is None:
                    self.sub = dev.subscribe("game_rv", maxlen=20000)
                imu = [(r.timestamp, *r.values[:4]) for r in self.sub.drain()] if self.sub is not None else []
                dets = []
                while self._dets:
                    dets.append(self._dets.popleft())
                self._send(("data", imu, [(t0, self.drive.turns)], dets))
                self.error = None
            except (OSError, EOFError, ValueError) as e:
                if self._run and self.proc.poll() is not None and not self.drive.closing:
                    log.error("track map worker exited (%s): restarting", self.proc.returncode)
                    self.restarts += 1
                    self._start()
                elif self._run:
                    self.error = str(e)
                    time.sleep(0.1)
            except Exception as e:  # noqa: BLE001 - keep feeding
                if self.error is None:
                    log.exception("track map feeder: %s", e)
                self.error = "%s: %s" % (type(e).__name__, str(e)[:160])
            time.sleep(max(0.0, self.PERIOD_S - (time.monotonic() - t0)))

    def reload_frame(self):
        self._send(("frame",))

    def reset(self):
        self._send(("reset",))

    def set_planner(self, **params):
        self._send(("planner", params))

    def start_log(self, directory, rid):
        self._send(("start_log", directory, rid))

    def stop_log(self):
        """Ends the take's map log -> {"files", "meta"} for the take's _meta.json (None if none)."""
        while not self.replies.empty():
            self.replies.get_nowait()
        try:
            self._send(("stop_log",))
            return self.replies.get(timeout=20.0)
        except (queue.Empty, OSError) as e:
            log.error("track map: no log from the worker (%s)", e)
            return None

    def status(self):
        st = dict(self.st)
        st.update(age_ms=int((time.monotonic() - self.snap_at) * 1000) if self.snap_at else None,
                  worker_restarts=self.restarts, error=self.error or st.get("error"))
        return st

    def close(self):
        self._run = False
        try:
            self._send(("quit",))
            self.proc.wait(5.0)
        except Exception:  # noqa: BLE001
            self.proc.kill()


# ---------------------------------------------------------------- telemetry
class Telemetry:
    """Samples every subsystem at TELEMETRY_HZ into .data (a JSON-able dict)."""

    def __init__(self, drive, camera, vision, imu, trackmap):
        self.drive, self.camera, self.vision, self.imu = drive, camera, vision, imu
        self.trackmap = trackmap
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
                batt = round(self.battery.read(samples=8).battery_volts, 3)
                wifi = self._wifi()
                self.drive.wifi_dbm = wifi.get("level_dbm")
                self.drive.send_health(batt, ps.temp_c)       # for the drive server's health lines
                self.data = {
                    "time": time.time(),
                    "cpu": {"temp_c": ps.temp_c, "arm_freq_mhz": ps.arm_freq_mhz,
                            "throttled_hex": ps.throttled_hex, "util": self._cpu_util()},
                    "power": {"total_w": ps.total_w, "ext5v_v": ps.ext5v_v},
                    "battery_v": batt,
                    "drive": self.drive.status(),
                    "camera": self.camera.status(),
                    "wifi": wifi,
                    "hailo": self.hailo,
                    "control": self.drive.control_status(),
                    "vision": self.vision.status(),
                    "imu": self._imu_status(),
                    "trackmap": self.trackmap.status(),
                }
            except Exception as e:  # noqa: BLE001 - keep sampling after a transient failure
                log.exception("telemetry: %s", e)
            time.sleep(max(0.0, 1.0 / TELEMETRY_HZ - (time.monotonic() - t0)))

    def _imu_status(self):
        try:
            return self.imu.status()
        except Exception as e:  # noqa: BLE001 - an IMU hiccup must not blank the other telemetry
            return {"up": False, "error": str(e).splitlines()[0][:200], "frame": self.imu.frame.status()}

    def close(self):
        self._run = False


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
    app = request.app
    cam = app["camera"]
    start = lambda: cam.start_recording(imu=app["imu"], drive=app["drive"], control=app["control"])  # noqa: E731
    try:
        rid = await asyncio.get_running_loop().run_in_executor(None, start if on else cam.stop_recording)
    except Exception as e:  # noqa: BLE001 - show it on the page
        log.exception("record: %s", e)
        return web.json_response({"error": str(e)}, status=500)
    return web.json_response({"recording": on, "file": rid})


async def api_recordings(request):
    return web.json_response(request.app["camera"].recordings())


async def api_recording_delete(request):
    """Deletes a whole take (folder: video + IMU + metadata) or a legacy loose mp4."""
    name = os.path.basename(request.match_info["name"])
    cam = request.app["camera"]
    if cam.rec and cam.rec.id == name:
        return web.json_response({"error": "recording in progress"}, status=409)
    ok = await asyncio.get_running_loop().run_in_executor(None, delete_recording, REC_DIR, name)
    return web.json_response({"deleted": name if ok else None})


async def api_recording_download(request):
    """One take as an uncompressed tar (<id>.tar: the folder with video, csvs and meta), streamed."""
    name = os.path.basename(request.match_info["name"])
    if not os.path.isdir(os.path.join(REC_DIR, name)) or name.startswith("."):
        raise web.HTTPNotFound()
    resp = web.StreamResponse(headers={"Content-Type": "application/x-tar",
                                       "Content-Disposition": 'attachment; filename="%s.tar"' % name})
    await resp.prepare(request)
    proc = await asyncio.create_subprocess_exec("tar", "-C", REC_DIR, "-cf", "-", name, stdout=asyncio.subprocess.PIPE)
    try:
        while True:
            chunk = await proc.stdout.read(1 << 20)
            if not chunk:
                break
            await resp.write(chunk)
    finally:
        if proc.returncode is None:
            proc.kill()
        await proc.wait()
    return resp


async def stream(request):
    """MJPEG of the preview frames (no boxes; the page uses /ws/video). ?w=480|640|960"""
    pv, key = request.app["preview"], request.query.get("w", PREVIEW_DEFAULT)
    key = key if key in PREVIEW_SIZES else PREVIEW_DEFAULT
    resp = web.StreamResponse(headers={"Content-Type": "multipart/x-mixed-replace; boundary=frame",
                                       "Cache-Control": "no-cache"})
    await resp.prepare(request)
    sub, seen = pv.subscribe(asyncio.get_running_loop()), -1
    try:
        while True:
            pv.want(key)
            await sub[1].wait()
            sub[1].clear()
            f = pv.latest(key)
            if f is None or f[0] == seen:
                continue
            seen = f[0]
            await resp.write(b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: %d\r\n\r\n" % len(f[2]))
            await resp.write(f[2] + b"\r\n")
    except (ConnectionResetError, asyncio.CancelledError):
        pass
    finally:
        pv.unsubscribe(sub)
    return resp


async def ws_video(request):
    """Preview for the page: binary messages [u32 header length][header JSON][JPEG]; the header has
    the boxes of exactly this frame (+ seq, sensor ts, age). ?w=480|640|960. Flow control: at most
    VIDEO_CREDITS frames unacknowledged (the page sends "a" after drawing one), and always the newest
    frame: on a slow link the frame rate drops instead of the delay growing."""
    pv, key = request.app["preview"], request.query.get("w", PREVIEW_DEFAULT)
    key = key if key in PREVIEW_SIZES else PREVIEW_DEFAULT
    ws = web.WebSocketResponse(heartbeat=5)
    await ws.prepare(request)
    sub = pv.subscribe(asyncio.get_running_loop())
    state = {"credits": VIDEO_CREDITS}

    async def acks():
        async for msg in ws:
            if msg.type == web.WSMsgType.TEXT:
                state["credits"] = min(VIDEO_CREDITS, state["credits"] + 1)
                sub[1].set()
    reader = asyncio.ensure_future(acks())
    seen = -1
    try:
        while not ws.closed:
            pv.want(key)
            try:
                await asyncio.wait_for(sub[1].wait(), 1.0)
            except asyncio.TimeoutError:
                continue
            sub[1].clear()
            f = pv.latest(key)
            if f is None or f[0] == seen or state["credits"] <= 0:
                continue
            seen = f[0]
            state["credits"] -= 1
            await ws.send_bytes(f[1])
    except (ConnectionResetError, asyncio.CancelledError):
        pass
    finally:
        reader.cancel()
        pv.unsubscribe(sub)
    return ws


async def api_vision(request):
    """{"on": bool, "hef": path, "conf": 0..1, "filter": "apple"|"all"} - any subset."""
    m = await request.json()
    vision = request.app["vision"]
    if m.get("hef") and m["hef"] not in vision.models():
        return web.json_response({"error": "unknown model %s" % m["hef"]}, status=400)
    return web.json_response(vision.configure(**m))


async def _read_until_closed(ws):
    """Push-only sockets must still read: aiohttp handles the heartbeat's pongs inside receive(),
    and without a reader it drops the connection every heartbeat + heartbeat/2 (7.5 s)."""
    async for _ in ws:
        pass


async def ws_detections(request):
    """Latest boxes for the page overlay, pushed as soon as they exist (max DET_PUSH_HZ):
    {"seq", "lat_ms", "boxes": [[x1, y1, x2, y2, score, label], ...]}, coords 0..1 of the frame."""
    ws = web.WebSocketResponse(heartbeat=5)
    await ws.prepare(request)
    vision, loop, seen = request.app["vision"], asyncio.get_running_loop(), -1
    reader = asyncio.ensure_future(_read_until_closed(ws))
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
    finally:
        reader.cancel()
    return ws


async def api_imu_level(request):
    """Current pose becomes level (roll = pitch = 0), saved to config/imu.json; 409 if moving."""
    try:
        frame = await asyncio.get_running_loop().run_in_executor(None, request.app["imu"].level)
    except RuntimeError as e:
        return web.json_response({"error": str(e)}, status=409)
    request.app["trackmap"].reload_frame()
    return web.json_response({"frame": frame})


async def api_imu_zero_yaw(request):
    request.app["imu"].zero_yaw()
    return web.json_response({"ok": True})


async def ws_imu(request):
    """Car-frame IMU snapshot (Imu.sample) at IMU_PUSH_HZ for the attitude / g-g display.
    sample() is a locked read of the driver's latest values (well under 1 ms), so it runs on the
    event loop: the shared executor is busy with MJPEG / detection waits of up to 1 s each."""
    ws = web.WebSocketResponse(heartbeat=5)
    await ws.prepare(request)
    imu, loop = request.app["imu"], asyncio.get_running_loop()
    reader = asyncio.ensure_future(_read_until_closed(ws))
    try:
        while not ws.closed:
            t0 = loop.time()
            s = imu.sample()
            if s is not None:
                await ws.send_json(s)
            await asyncio.sleep(max(0.0, 1.0 / IMU_PUSH_HZ - (loop.time() - t0)))
    except (ConnectionResetError, asyncio.CancelledError):
        pass
    finally:
        reader.cancel()
    return ws


async def api_map(request):
    """{"reset": true} clears the map (refused while recording: the take's map would be cut)."""
    m = await request.json()
    svc = request.app["trackmap"]
    if m.get("reset"):
        if request.app["camera"].rec:
            return web.json_response({"error": "recording: the map resets when a take starts"}, status=409)
        svc.reset()
    return web.json_response(svc.status())


def _drive_tuning_save(over):
    path = drive_server.DRIVE_CONFIG
    try:
        with open(path) as f:
            cfg = json.load(f)
    except (OSError, ValueError):
        cfg = {}
    cfg.update(over)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(cfg, f, indent=2)
    os.replace(tmp, path)


async def api_planner(request):
    """Autonomy tuning. GET -> {"steer_gain", "planner": {field: {value, min, max, step, label, help}},
    "drive": {...}}; POST {field: value, ...}: planner fields (vision/planner.py TUNABLES) go to the running
    planner + config/planner.json, speed-control fields (drive_server.DRIVE_TUNABLES) to the drive server +
    config/drive.json; values are clipped to their ranges, both kept across restarts."""
    from vision.planner import TUNABLES, PlannerParams, save_config
    if request.method == "POST":
        m = await request.json()
        if not isinstance(m, dict):
            return web.json_response({"error": "{field: value}"}, status=400)
        pl_over, dr_over = {}, {}
        for k, v in m.items():
            try:
                v = float(v)
            except (TypeError, ValueError):
                return web.json_response({"error": "%s: number" % k}, status=400)
            if not math.isfinite(v):
                return web.json_response({"error": "%s: number" % k}, status=400)
            if k in TUNABLES:
                lo, hi = TUNABLES[k][0], TUNABLES[k][1]
                pl_over[k] = max(lo, min(hi, v))
            elif k in drive_server.DRIVE_TUNABLES:
                lo, hi = drive_server.DRIVE_TUNABLES[k][1], drive_server.DRIVE_TUNABLES[k][2]
                dr_over[k] = max(lo, min(hi, v))
            else:
                return web.json_response({"error": "unknown field %s" % k}, status=400)
        if pl_over:
            save_config(pl_over)
            request.app["trackmap"].set_planner(**pl_over)
        if dr_over:
            _drive_tuning_save(dr_over)
            try:
                s = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
                s.sendto(json.dumps({"cfg": dr_over}).encode(), drive_server.SOCK)
                s.close()
            except OSError as e:
                log.warning("drive tuning not sent: %s", e)
        log.info("autonomy tuning: %s", ", ".join("%s %g" % kv for kv in sorted({**pl_over, **dr_over}.items())))
    pp, p0 = PlannerParams.load(), PlannerParams()
    dt = drive_server.load_drive_tuning()
    planner = {k: {"value": getattr(pp, k), "default": getattr(p0, k), "min": v[0], "max": v[1], "step": v[2],
                   "label": v[3], "help": v[4]}
               for k, v in TUNABLES.items()}
    drive = {k: {"value": dt[k], "default": v[0], "min": v[1], "max": v[2], "step": v[3], "label": v[4], "help": v[5]}
             for k, v in drive_server.DRIVE_TUNABLES.items()}
    return web.json_response({"steer_gain": pp.steer_gain, "planner": planner, "drive": drive})


async def ws_map(request):
    """Track map snapshots as the worker serialised them (vision/map_worker.py), each one once."""
    ws = web.WebSocketResponse(heartbeat=5)
    await ws.prepare(request)
    svc = request.app["trackmap"]
    reader = asyncio.ensure_future(_read_until_closed(ws))
    last = None
    try:
        while not ws.closed:
            js = svc.snap_json
            if js is not last:
                last = js
                await ws.send_str(js)
            await asyncio.sleep(1.0 / (2 * MAP_PUSH_HZ))
    except (ConnectionResetError, asyncio.CancelledError):
        pass
    finally:
        reader.cancel()
    return ws


async def websocket(request):
    """Telemetry push at TELEMETRY_HZ. (Drive commands go to drive_server.py, :8082/ws/ctl.)"""
    ws = web.WebSocketResponse(heartbeat=5)
    await ws.prepare(request)
    telem = request.app["telem"]
    reader = asyncio.ensure_future(_read_until_closed(ws))
    try:
        while not ws.closed:
            await ws.send_json(telem.data)
            await asyncio.sleep(1.0 / TELEMETRY_HZ)
    except (ConnectionResetError, asyncio.CancelledError):
        pass
    finally:
        reader.cancel()
    return ws


async def serve(app):
    runner = web.AppRunner(app, access_log=None)
    await runner.setup()
    await web.TCPSite(runner, port=HTTP_PORT).start()
    log.info("http://0.0.0.0:%d", HTTP_PORT)
    await asyncio.Event().wait()


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    drive = DriveLink()                      # drive_server.py: throttle / steering / encoder + command links
    camera = CameraStream()
    preview = Preview(camera)
    camera.preview = preview
    vision = Vision(camera)
    vision.preview = preview
    imu = Imu()
    trackmap = MapProcess(imu, drive)
    vision.on_boxes = trackmap.on_boxes
    camera.trackmap = trackmap
    telem = Telemetry(drive, camera, vision, imu, trackmap)

    app = web.Application()
    app.update(drive=drive, camera=camera, control=drive, telem=telem, vision=vision, imu=imu, trackmap=trackmap,
               preview=preview)
    app.add_routes([
        web.get("/", index),
        web.get("/stream.mjpg", stream),
        web.get("/ws/video", ws_video),
        web.get("/ws", websocket),
        web.get("/ws/det", ws_detections),
        web.post("/api/vision", api_vision),
        web.get("/ws/imu", ws_imu),
        web.post("/api/imu/level", api_imu_level),
        web.post("/api/imu/zero_yaw", api_imu_zero_yaw),
        web.get("/ws/map", ws_map),
        web.post("/api/map", api_map),
        web.get("/api/planner", api_planner),
        web.post("/api/planner", api_planner),
        web.get("/api/telemetry", api_telemetry),
        web.post("/api/mode", api_mode),
        web.post("/api/record", api_record),
        web.get("/api/recordings", api_recordings),
        web.delete("/api/recordings/{name}", api_recording_delete),
        web.get("/api/recordings/{name}/download", api_recording_download),
        web.static("/recordings", REC_DIR),
    ])
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))   # pkill / dashboard stop: finalise a take first
    try:
        asyncio.run(serve(app))
    except (KeyboardInterrupt, SystemExit):
        pass
    finally:
        drive.closing = True                 # no child restarts while shutting down
        telem.close()
        preview.close()
        camera.stop_recording()  # a running take is finalised first (its map log included)
        trackmap.close()
        vision.close()
        imu.close()
        camera.close()
        drive.close()


if __name__ == "__main__":
    main()
