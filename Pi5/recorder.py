"""Training-data recordings: H.264 video + the IMU / odometry state of every frame, in one folder.

    rec = Recording(REC_DIR, picam, mode, imu=imu, drive=drive, control=control)   # starts
    rec.status()                                                                     # live counters
    rec.stop()

Everything of one take lives in recordings/<id>/ and every file in it carries the id, so a file that
gets copied out alone still says where it came from; deleting the folder deletes the take:

    <id>.mp4           H.264 (fragmented mp4: playable up to the last second after a power cut).
                       Frame i of the video is row i of <id>_frames.csv; pts (us) = pts_us.
                       Container tags: title = id, comment = camera / stream / crop summary.
    <id>_frames.csv    one row per video frame: sensor timestamp, exposure, gains, and the IMU and
                       odometry interpolated to the middle of that frame's exposure: car attitude
                       (roll / pitch / yaw + quaternion), body rates, linear acceleration, the
                       camera's orientation in the world (quaternion + pitch / roll / heading in
                       CameraModel conventions), encoder turns / speed, steering, ESC pulse
    <id>_imu.csv       every IMU sample (game_rv, gyro, linear_accel, accel, gravity at ~200 Hz),
                       raw board frame, same clock: re-interpolate it any way you like
    <id>_drive.csv     encoder turns / speed, steering, ESC pulse, commanded throttle at DRIVE_HZ
    <id>_meta.json     what / where / how: camera (sensor, port, mount pose, intrinsics, the full
                       config/camera.json), this stream's ScalerCrop + K + distortion, IMU mount
                       (config/imu.json), column descriptions, git commit, and a summary at stop

<id> = YYYYmmdd_HHMMSS_<camera>_<w>x<h>_<fps>fps, e.g. 20261010_143000_cam0_1280x720_60fps.

Timing: every timestamp is CLOCK_MONOTONIC in ns (libcamera SensorTimestamp = start of readout of
the first row; BNO085 samples are hub-timestamped onto the same clock). The frame's reference time
t_mid_ns = sensor_ts - exposure / 2 + readout / 2 is the middle of the exposure of the middle row
(rolling shutter; readout estimated as the sensor mode's minimum frame time).

Nothing is drawn into the pixels: the video stays clean for training. The magnetometer is never
enabled (see dashboard.Imu).
"""
import bisect
import csv
import json
import math
import os
import queue
import shutil
import socket
import subprocess
import threading
import time
from collections import deque
from fractions import Fraction

import numpy as np
from picamera2.encoders import LibavH264Encoder
from picamera2.outputs import PyavOutput

from hardware.imu_frame import quat_to_matrix
from vision.camera_model import CameraModel, CONFIG as CAMERA_CONFIG

CAMERA_ID = "cam0"              # CSI port; the only camera on the car (front, IMX708 wide)
CAMERA_NAME = "front"
IMU_SENSORS = ("game_rv", "gyro", "linear_accel", "accel", "gravity")
DRIVE_HZ = 100
BITS_PER_PIXEL = 0.2            # H.264 bitrate = bpp * w * h * fps: ~11 Mbit/s at 720p60
BITRATE_MIN, BITRATE_MAX = 6e6, 40e6
WRITE_PERIOD_S = 0.1            # writer thread: drain, interpolate, write
IMU_KEEP_S = 4.0                # interpolation buffer depth
FRAME_WAIT_S = 0.5              # a frame waits at most this long for the IMU sample after it
MIN_FREE_BYTES = 500e6          # stop recording below this much free space
FORMAT_VERSION = 1

FRAME_COLUMNS = [
    ("frame", "index of the frame in the video (0-based)"),
    ("pts_us", "video presentation time, us from the first frame"),
    ("sensor_ts_ns", "libcamera SensorTimestamp, CLOCK_MONOTONIC ns (start of readout)"),
    ("t_mid_ns", "reference time of the frame: middle of the exposure of the middle row"),
    ("dropped_before", "recorded frames missing just before this one (pipeline could not keep up)"),
    ("exposure_us", "ExposureTime"),
    ("frame_duration_us", "FrameDuration"),
    ("analogue_gain", "AnalogueGain"),
    ("digital_gain", "DigitalGain"),
    ("colour_temp_k", "ColourTemperature"),
    ("lux", "Lux estimate"),
    ("lens_position", "LensPosition (dioptres; calibration valid for 1.0)"),
    ("imu_gap_ms", "time between the two game_rv samples t_mid_ns was interpolated between "
                   "(empty = no IMU / extrapolated)"),
    ("roll_deg", "car roll, + = right side down"),
    ("pitch_deg", "car pitch, + = nose up"),
    ("yaw_deg", "car yaw, + = turned left, 0 = heading at the first frame"),
    ("car_qw", "car -> world quaternion (w, x, y, z); world: X = car heading at the first frame, "
               "Y left, Z up"),
    ("car_qx", ""), ("car_qy", ""), ("car_qz", ""),
    ("gyro_x_dps", "body rate about car x (forward), right-hand rule, deg/s"),
    ("gyro_y_dps", "body rate about car y (left): + = nose DOWN"),
    ("gyro_z_dps", "body rate about car z (up): + = turning left"),
    ("acc_fwd", "linear acceleration (gravity removed), car frame, m/s^2"),
    ("acc_left", ""), ("acc_up", ""),
    ("cam_qw", "camera -> world quaternion (w, x, y, z), camera axes as OpenCV (x right, y down, "
               "z along the optical axis); camera mount from config/camera.json"),
    ("cam_qx", ""), ("cam_qy", ""), ("cam_qz", ""),
    ("cam_pitch_deg", "optical axis below horizontal (+ = looking down): CameraModel pitch_deg"),
    ("cam_roll_deg", "camera roll about the optical axis: CameraModel roll_deg (+ = left side down)"),
    ("cam_heading_deg", "optical axis heading in the world, + = left"),
    ("odo_turns", "motor encoder turns since the dashboard started (+ = forward)"),
    ("motor_rps", "motor shaft revolutions / s (median filtered)"),
    ("steer", "steering command -1 (left) .. +1 (right)"),
    ("esc_us", "ESC pulse width (1500 neutral, < 1500 forward on this car)"),
]


# ---------------------------------------------------------------- small math
def _quat_from_matrix(m):
    """Rotation matrix -> (w, x, y, z), w >= 0."""
    m = np.asarray(m)
    t = m[0, 0] + m[1, 1] + m[2, 2]
    if t > 0:
        s = math.sqrt(t + 1.0) * 2
        q = (0.25 * s, (m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s)
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2
        q = ((m[2, 1] - m[1, 2]) / s, 0.25 * s, (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s)
    elif m[1, 1] > m[2, 2]:
        s = math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2
        q = ((m[0, 2] - m[2, 0]) / s, (m[0, 1] + m[1, 0]) / s, 0.25 * s, (m[1, 2] + m[2, 1]) / s)
    else:
        s = math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2
        q = ((m[1, 0] - m[0, 1]) / s, (m[0, 2] + m[2, 0]) / s, (m[1, 2] + m[2, 1]) / s, 0.25 * s)
    return q if q[0] >= 0 else tuple(-x for x in q)


def _slerp(q0, q1, f):
    """Quaternions in the driver's (i, j, k, real) order."""
    d = sum(a * b for a, b in zip(q0, q1))
    if d < 0:
        q1, d = [-x for x in q1], -d
    if d > 0.9995:
        q = [a + f * (b - a) for a, b in zip(q0, q1)]
    else:
        th = math.acos(d)
        s0, s1 = math.sin((1 - f) * th) / math.sin(th), math.sin(f * th) / math.sin(th)
        q = [s0 * a + s1 * b for a, b in zip(q0, q1)]
    n = math.sqrt(sum(x * x for x in q))
    return [x / n for x in q]


def _rz(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


# CameraModel: world_to_cam = Rz(roll) @ Rx(pitch) @ B for a camera heading along +X
_B = np.array([[0.0, -1.0, 0.0], [0.0, 0.0, -1.0], [1.0, 0.0, 0.0]])


def _rx(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


def camera_angles(c2w):
    """camera -> world rotation -> (pitch_deg, roll_deg, heading_deg) in CameraModel conventions:
    world_to_cam = Rz(roll) @ Rx(pitch) @ B @ Rz_world(-heading)."""
    d = c2w[:, 2]                                         # optical axis in the world
    heading = math.atan2(d[1], d[0])
    pitch = math.asin(max(-1.0, min(1.0, -d[2])))
    m = c2w.T @ _rz(heading) @ _B.T @ _rx(pitch).T        # = Rz(roll)
    return math.degrees(pitch), math.degrees(math.atan2(m[1, 0], m[0, 0])), math.degrees(heading)


def camera_mount(pitch_deg, roll_deg):
    """camera -> car rotation for the static mount (car level, heading +X)."""
    return (_rz(math.radians(roll_deg)) @ _rx(math.radians(pitch_deg)) @ _B).T


# ---------------------------------------------------------------- encoder / output hooks
class _FrameLogEncoder(LibavH264Encoder):
    """Hands the metadata of every frame it actually encodes to on_frame(metadata, pts_us)."""

    def __init__(self, on_frame, **kw):
        super().__init__(**kw)
        self._on_frame = on_frame

    def _encode(self, stream, request):
        md = request.get_metadata()
        super()._encode(stream, request)
        try:
            self._on_frame(md, int(md["SensorTimestamp"] / 1000) - self.firsttimestamp)
        except Exception:  # noqa: BLE001 - the video must keep going whatever the logging does
            pass


class _TaggedOutput(PyavOutput):
    """PyavOutput that writes container tags (title / comment) into the mp4 and muxes on its own
    thread. Stock PyavOutput writes from the camera thread, and SD card write stalls (ext4 commits
    every ~5 s) then held up frame delivery long enough to drop frames."""

    def __init__(self, path, tags, **kw):
        super().__init__(path, **kw)
        self._tags = tags
        self._q = queue.Queue(maxsize=5000)     # encoded packets (~20-100 kB each)
        self._thread = None
        self.max_queue = 0

    def start(self):
        super().start()
        self._container.metadata.update(self._tags)
        self._thread = threading.Thread(target=self._mux_loop, daemon=True)
        self._thread.start()

    def outputframe(self, frame, keyframe=True, timestamp=None, packet=None, audio=False):
        self._q.put((frame, keyframe, timestamp, packet, audio))
        self.max_queue = max(self.max_queue, self._q.qsize())

    def _mux_loop(self):
        while True:
            item = self._q.get()
            if item is None:
                return
            super().outputframe(*item)

    def stop(self):
        if self._thread is not None:
            self._q.put(None)
            self._thread.join()
            self._thread = None
        super().stop()


# ---------------------------------------------------------------- recording
class _Series:
    """Time-sorted samples of one stream (t seconds monotonic, values tuple) for interpolation."""

    def __init__(self):
        self.t, self.v = [], []

    def add(self, t, v):
        if self.t and t < self.t[-1]:                 # IRQ jitter can reorder by < 1 ms
            i = bisect.bisect(self.t, t)
            self.t.insert(i, t)
            self.v.insert(i, v)
        else:
            self.t.append(t)
            self.v.append(v)

    def trim(self, before):
        i = bisect.bisect_left(self.t, before)
        if i > 1000:
            del self.t[:i], self.v[:i]

    @property
    def last_t(self):
        return self.t[-1] if self.t else None

    def at(self, t, quat=False):
        """(values at t, gap between the bracketing samples in s), or (None, None). Past the ends
        the nearest sample is returned with gap None."""
        if not self.t:
            return None, None
        i = bisect.bisect(self.t, t)
        if i == 0:
            return self.v[0], None
        if i == len(self.t):
            return self.v[-1], None
        t0, t1 = self.t[i - 1], self.t[i]
        f = (t - t0) / (t1 - t0) if t1 > t0 else 0.0
        a, b = self.v[i - 1], self.v[i]
        v = _slerp(a, b, f) if quat else [x + f * (y - x) for x, y in zip(a, b)]
        return v, t1 - t0


def _git_describe(root):
    try:
        rev = subprocess.run(["git", "-C", root, "rev-parse", "--short", "HEAD"], capture_output=True,
                             text=True, timeout=2).stdout.strip()
        dirty = subprocess.run(["git", "-C", root, "status", "--porcelain", "--untracked-files=no"],
                               capture_output=True, text=True, timeout=2).stdout.strip()
        return rev + ("-dirty" if dirty else "") if rev else None
    except Exception:  # noqa: BLE001
        return None


def _write_json(path, d):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(d, f, indent=1)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def _fmt(x, d):
    return "" if x is None else ("%.*f" % (d, x))


class Recording:
    """One take: starts in __init__, stop() finalises it. Thread-safe status()."""

    def __init__(self, rec_dir, picam, mode, skip=1, imu=None, drive=None, control=None):
        """skip: record every skip-th camera frame (the camera may run faster for detection)."""
        self.picam, self.mode, self.imu_src, self.drive, self.control = picam, mode, imu, drive, control
        self.skip = skip
        w, h, fps = mode.width, mode.height, mode.fps / skip
        self.fps = fps
        self.id = "%s_%s_%dx%d_%gfps" % (time.strftime("%Y%m%d_%H%M%S"), CAMERA_ID, w, h, round(fps, 2))
        self.dir = os.path.join(rec_dir, self.id)
        os.makedirs(self.dir)
        self.path = lambda suffix: os.path.join(self.dir, self.id + suffix)
        self.video = self.path(".mp4")
        self.t_start, self.wall_start = time.monotonic(), time.time()
        self.readout_ns = int(1e9 / mode.sensor.max_fps)
        self.frame_ns = int(1e9 / mode.fps)      # camera frame period

        self.cam_model = CameraModel.load()
        self.mount = camera_mount(self.cam_model.pitch_deg, self.cam_model.roll_deg)
        self.frame = imu.frame if imu is not None else None     # CarFrame at the start of the take
        self.yaw0 = None                                         # car yaw at the first frame

        self._frames = deque()                 # (metadata, pts_us) from the camera thread
        self._drive_q = deque()
        self.series = {n: _Series() for n in IMU_SENSORS}
        self.drive_series = _Series()
        self.sub = None
        self.n_frames = self.n_dropped = self.n_imu = self.n_drive = self.n_frames_no_imu = 0
        self.imu_overflow = 0
        self.imu_span = {}                     # sensor -> [count, first t, last t]
        self.crop = None
        self.first_ts = self.last_ts = None
        self.disk_low = False
        self.error = None
        self._stop = threading.Event()
        self._final = threading.Event()

        self.files = {k: open(self.path(s), "w", newline="") for k, s in
                      (("frames", "_frames.csv"), ("imu", "_imu.csv"), ("drive", "_drive.csv"))}
        self.csv = {k: csv.writer(f) for k, f in self.files.items()}
        self.csv["frames"].writerow([c for c, _ in FRAME_COLUMNS])
        self.csv["imu"].writerow(["t_ns", "sensor", "accuracy", "seq", "v0", "v1", "v2", "v3"])
        self.csv["drive"].writerow(["t_ns", "odo_turns", "motor_rps", "steer", "esc_us", "throttle_cmd"])
        self.meta = self._meta()
        _write_json(self.path("_meta.json"), self.meta)
        self._subscribe()

        self._threads = [threading.Thread(target=self._writer, daemon=True)]
        if drive is not None:
            self._threads.append(threading.Thread(target=self._drive_sampler, daemon=True))
        for t in self._threads:
            t.start()

        bitrate = int(min(BITRATE_MAX, max(BITRATE_MIN, BITS_PER_PIXEL * w * h * fps)))
        self.bitrate = bitrate
        self.encoder = _FrameLogEncoder(self._on_frame, bitrate=bitrate, iperiod=max(1, round(fps)))
        self.encoder.framerate = Fraction(fps).limit_denominator(1000)     # PyAV wants a rational
        self.encoder.frame_skip_count = skip
        tags = {"title": self.id, "comment": self._tag_comment()}
        self.output = _TaggedOutput(self.video, tags, options={"movflags": "frag_keyframe+empty_moov+default_base_moof"})
        picam.start_encoder(self.encoder, self.output)

    # ---- description ---------------------------------------------------------------------------
    def _meta(self):
        m, cm = self.mode, self.cam_model
        try:
            with open(CAMERA_CONFIG) as f:
                cam_cfg = json.load(f)
        except OSError:
            cam_cfg = None
        frame = self.frame
        return {
            "format_version": FORMAT_VERSION,
            "id": self.id,
            "started": time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime(self.wall_start)),
            "started_monotonic_ns": int(self.t_start * 1e9),
            "host": socket.gethostname(),
            "code": _git_describe(os.path.dirname(os.path.abspath(__file__))),
            "files": {"video": self.id + ".mp4", "frames": self.id + "_frames.csv", "imu": self.id + "_imu.csv",
                      "drive": self.id + "_drive.csv", "meta": self.id + "_meta.json"},
            "camera": {
                "id": CAMERA_ID, "name": CAMERA_NAME, "port": "CSI CAM0", "sensor": "Sony IMX708 (Arducam 12MP, wide lens)",
                "sensor_size": [4608, 2592],
                "mount": {"height_mm": cm.height_mm, "pitch_deg": cm.pitch_deg, "roll_deg": cm.roll_deg,
                          "frame": "car level on the floor; origin on the floor under the camera, X forward, Y left, "
                                   "Z up; pitch + = down, roll + = left side down"},
                "intrinsics_sensor_px": {"fx": cm.fx, "fy": cm.fy, "cx": cm.cx, "cy": cm.cy, "dist": list(cm.dist)},
                "calibration": cam_cfg,
            },
            "stream": {
                "width": m.width, "height": m.height, "fps": round(self.fps, 3), "camera_fps": m.fps,
                "frame_skip": self.skip,
                "raw_mode": [m.sensor.width, m.sensor.height], "raw_mode_max_fps": m.sensor.max_fps,
                "scaler_crop": None, "K": None, "dist": list(cm.dist),
                "codec": "H.264 (libx264 ultrafast, yuv420p), fragmented mp4", "bitrate": None,
                "keyframe_interval": max(1, round(self.fps)),
                "how_to_use": "pixels of this video map to sensor pixels through scaler_crop: "
                              "CameraModel.to_sensor(x, y, scaler_crop, (width, height)); K / dist are "
                              "OpenCV intrinsics for this stream directly",
            },
            "imu": {
                "device": "BNO085 (game_rv fusion: gyro + accel, no magnetometer)",
                "sensors": list(IMU_SENSORS),
                "rate_hz_requested": getattr(self.imu_src, "RATE_HZ", None),
                "mount": frame and {"forward": frame.forward, "up": frame.up, "level_gravity": frame.level_gravity,
                                    "level_time": frame.level_time, "car_from_board": [list(r) for r in frame.R]},
                "imu_csv": "raw board frame: game_rv v0..v3 = quaternion (i, j, k, real) board -> world (Z up, "
                           "arbitrary heading); gyro rad/s; accel / linear_accel / gravity m/s^2. "
                           "car vector = car_from_board @ board vector",
                "world": "X = car heading at the first frame (yaw 0), Y left, Z up; yaw drifts slowly "
                         "(no magnetometer)",
            },
            "timing": {
                "clock": "CLOCK_MONOTONIC (Python time.monotonic), ns",
                "sensor_ts": "libcamera SensorTimestamp: start of readout of the first row",
                "t_mid": "sensor_ts - exposure/2 + readout/2 (middle of the middle row's exposure)",
                "readout_ns_estimate": self.readout_ns,
            },
            "frames_columns": dict(FRAME_COLUMNS),
            "summary": None,
        }

    def _tag_comment(self):
        m = self.mode
        return ("%s camera %s (%s) %dx%d@%g raw %dx%d; per-frame IMU + calibration in %s_frames.csv / _meta.json"
                % (self.id, CAMERA_ID, CAMERA_NAME, m.width, m.height, round(self.fps, 2), m.sensor.width,
                   m.sensor.height, self.id))

    # ---- sources -------------------------------------------------------------------------------
    def _subscribe(self):
        dev = getattr(self.imu_src, "imu", None)
        if dev is not None and self.sub is None:
            self.sub = dev.subscribe(*IMU_SENSORS, maxlen=200000)
            for n in IMU_SENSORS:               # the latest sample before the subscription: the
                r = dev.read(n)                 # first frame can interpolate instead of extrapolate
                if r is not None:
                    self.sub._push(r)

    def _on_frame(self, md, pts_us):
        """Camera thread: keep it cheap."""
        self._frames.append((md, pts_us))

    def _drive_sampler(self):
        d, c, period = self.drive, self.control, 1.0 / DRIVE_HZ
        nxt = time.monotonic()
        while not self._stop.is_set():
            self._drive_q.append((time.monotonic_ns(), d.turns, d.rps, d.steering.position, d.esc.current_us,
                                  c.throttle if c is not None else None))
            nxt += period
            time.sleep(max(0.0, nxt - time.monotonic()))

    # ---- writer --------------------------------------------------------------------------------
    def _writer(self):
        # no periodic fsync: it also forces the mp4's dirty pages out, and the camera thread (which
        # writes the mp4) stalled long enough to drop frames. The kernel writes back within ~5 s.
        next_disk = 0.0
        while True:
            final = self._final.is_set()
            try:
                self._drain(final)
            except Exception as e:  # noqa: BLE001 - report, keep the video going
                self.error = "%s: %s" % (type(e).__name__, e)
            for f in self.files.values():
                f.flush()
            now = time.monotonic()
            if now > next_disk:
                st = os.statvfs(self.dir)
                self.disk_low = st.f_bavail * st.f_frsize < MIN_FREE_BYTES
                next_disk = now + 1.0
            if final:
                return
            self._final.wait(WRITE_PERIOD_S)

    def _drain(self, final):
        self._subscribe()                       # the IMU may come up after the recording started
        w = self.csv["imu"]
        if self.sub is not None:
            for r in self.sub.drain():
                w.writerow([int(r.timestamp * 1e9), r.sensor, r.accuracy, r.seq] + ["%.6g" % x for x in r.values])
                self.series[r.sensor].add(r.timestamp, r.values)
                sp = self.imu_span.setdefault(r.sensor, [0, r.timestamp, r.timestamp])
                sp[0] += 1
                sp[2] = r.timestamp
                self.n_imu += 1
            self.imu_overflow = self.sub.overflow
        w = self.csv["drive"]
        while self._drive_q:
            t, turns, rps, st, us, thr = self._drive_q.popleft()
            w.writerow([t, "%.4f" % turns, "%.3f" % rps, _fmt(st, 3), _fmt(us, 1), _fmt(thr, 3)])
            self.drive_series.add(t * 1e-9, (turns, rps, st if st is not None else float("nan"), us or float("nan")))
            self.n_drive += 1

        imu_last = self.series["game_rv"].last_t
        now = time.monotonic()
        while self._frames:
            md, pts = self._frames[0]
            t_mid = self._t_mid(md)
            if not final and imu_last is not None and t_mid > imu_last and now - t_mid < FRAME_WAIT_S:
                break                            # wait for the IMU sample after this frame
            self._frames.popleft()
            self._write_frame(md, pts, t_mid)

        keep = now - IMU_KEEP_S
        for s in self.series.values():
            s.trim(keep)
        self.drive_series.trim(keep)

    def _t_mid(self, md):
        return (md["SensorTimestamp"] - md.get("ExposureTime", 0) * 500 + self.readout_ns / 2) * 1e-9

    def _write_frame(self, md, pts, t_mid):
        ts = md["SensorTimestamp"]
        if self.crop is None:
            self.crop = tuple(md.get("ScalerCrop", ()))
        dropped = 0
        if self.last_ts is not None:
            dur = md.get("FrameDuration") * 1000 if md.get("FrameDuration") else self.frame_ns
            dropped = max(0, round((ts - self.last_ts) / (dur * self.skip)) - 1)   # in recorded frames
        else:
            self.first_ts = ts
        self.last_ts = ts
        self.n_dropped += dropped

        row = [self.n_frames, pts, ts, int(t_mid * 1e9), dropped, md.get("ExposureTime"), md.get("FrameDuration"),
               _fmt(md.get("AnalogueGain"), 3), _fmt(md.get("DigitalGain"), 3), md.get("ColourTemperature"),
               _fmt(md.get("Lux"), 1), _fmt(md.get("LensPosition"), 3)]
        row += self._imu_columns(t_mid)
        dv, _ = self.drive_series.at(t_mid)
        row += ["%.4f" % dv[0], "%.3f" % dv[1], _fmt(dv[2], 3), _fmt(dv[3], 1)] if dv else [""] * 4
        self.csv["frames"].writerow(row)
        self.n_frames += 1

    def _imu_columns(self, t):
        n_imu_cols = 3 + 4 + 3 + 3 + 4 + 3
        q, gap = self.series["game_rv"].at(t, quat=True)
        if q is None or self.frame is None:
            self.n_frames_no_imu += 1
            return [""] * (1 + n_imu_cols)
        s = self.series["game_rv"]
        if gap is None and min(abs(t - s.t[0]), abs(t - s.t[-1])) > 0.02:   # nearest sample > 20 ms away
            self.n_frames_no_imu += 1
        R = np.asarray(self.frame.R)                                  # car <- board
        c2w = np.asarray(quat_to_matrix(q)) @ R.T                     # car -> world (arbitrary heading)
        if self.yaw0 is None:
            self.yaw0 = math.atan2(c2w[1, 0], c2w[0, 0])
        c2w = _rz(-self.yaw0) @ c2w
        roll = math.degrees(math.atan2(c2w[2, 1], c2w[2, 2]))
        pitch = math.degrees(math.asin(max(-1.0, min(1.0, c2w[2, 0]))))
        yaw = math.degrees(math.atan2(c2w[1, 0], c2w[0, 0]))
        gyro, _ = self.series["gyro"].at(t)
        la, _ = self.series["linear_accel"].at(t)
        g = (R @ gyro[:3]) * (180 / math.pi) if gyro is not None else None
        a = R @ la[:3] if la is not None else None
        cam = c2w @ self.mount                                        # camera -> world
        cp, cr, ch = camera_angles(cam)
        out = [_fmt(gap and gap * 1000, 1), "%.3f" % roll, "%.3f" % pitch, "%.3f" % yaw]
        out += ["%.6f" % x for x in _quat_from_matrix(c2w)]
        out += ["%.3f" % x for x in g] if g is not None else [""] * 3
        out += ["%.4f" % x for x in a] if a is not None else [""] * 3
        out += ["%.6f" % x for x in _quat_from_matrix(cam)]
        out += ["%.3f" % cp, "%.3f" % cr, "%.3f" % ch]
        return out

    # ---- control -------------------------------------------------------------------------------
    def status(self):
        return {"id": self.id, "seconds": round(time.monotonic() - self.t_start, 1), "frames": self.n_frames,
                "dropped": self.n_dropped, "imu_samples": self.n_imu, "imu": self.sub is not None,
                "frames_no_imu": self.n_frames_no_imu, "error": self.error, "disk_low": self.disk_low}

    def stop(self):
        self.picam.stop_encoder(self.encoder)      # flushes x264 and closes the mp4
        self._stop.set()
        time.sleep(0.15)                           # the IMU samples after the last frame
        self._final.set()
        for t in self._threads:
            t.join(5.0)
        if self.sub is not None:
            self.sub.close()
        for f in self.files.values():
            f.flush()
            os.fsync(f.fileno())
            f.close()
        dur = time.monotonic() - self.t_start
        hz = {n: round((c - 1) / (t1 - t0), 1) if c > 2 and t1 > t0 else 0.0
              for n, (c, t0, t1) in self.imu_span.items()}
        self.meta["stream"].update(
            scaler_crop=list(self.crop) if self.crop else None, bitrate=self.bitrate,
            K=self.cam_model.K(self.crop, (self.mode.width, self.mode.height)).round(4).tolist() if self.crop else None)
        self.meta["summary"] = {
            "stopped": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "seconds": round(dur, 2), "frames": self.n_frames,
            "fps_measured": round((self.n_frames - 1) / ((self.last_ts - self.first_ts) * 1e-9), 2)
            if self.n_frames > 1 and self.last_ts > self.first_ts else None, "dropped_frames": self.n_dropped,
            "frames_without_imu": self.n_frames_no_imu, "imu_samples": self.n_imu, "imu_rate_hz": hz,
            "imu_buffer_overflow": self.imu_overflow, "drive_samples": self.n_drive,
            "mux_queue_max": self.output.max_queue,
            "video_bytes": os.path.getsize(self.video) if os.path.exists(self.video) else 0,
            "error": self.error, "stopped_for_low_disk": self.disk_low,
        }
        _write_json(self.path("_meta.json"), self.meta)
        os.sync()
        return self.meta["summary"]


# ---------------------------------------------------------------- listing / deleting
def list_recordings(rec_dir):
    """Newest first: folders (with their meta summary) and legacy loose mp4 files."""
    out = []
    for n in sorted(os.listdir(rec_dir), reverse=True):
        p = os.path.join(rec_dir, n)
        if os.path.isdir(p):
            size = sum(os.path.getsize(os.path.join(p, f)) for f in os.listdir(p))
            meta = None
            try:
                with open(os.path.join(p, n + "_meta.json")) as f:
                    meta = json.load(f)
            except (OSError, ValueError):
                pass
            s = (meta or {}).get("summary") or {}
            out.append({"name": n, "video": "%s/%s.mp4" % (n, n), "size_mb": round(size / 1e6, 1),
                        "mtime": os.path.getmtime(p), "seconds": s.get("seconds"), "frames": s.get("frames"),
                        "dropped": s.get("dropped_frames"), "no_imu": s.get("frames_without_imu"),
                        "imu_hz": (s.get("imu_rate_hz") or {}).get("game_rv"), "complete": bool(s),
                        "files": sorted(os.listdir(p))})
        elif n.endswith(".mp4"):
            st = os.stat(p)
            out.append({"name": n, "video": n, "size_mb": round(st.st_size / 1e6, 1), "mtime": st.st_mtime,
                        "legacy": True})
    return sorted(out, key=lambda r: r["mtime"], reverse=True)


def delete_recording(rec_dir, name):
    name = os.path.basename(name)
    p = os.path.join(rec_dir, name)
    if not name or name.startswith("."):
        return False
    if os.path.isdir(p):
        shutil.rmtree(p)
    elif os.path.isfile(p):
        os.remove(p)
    else:
        return False
    os.sync()
    return True
