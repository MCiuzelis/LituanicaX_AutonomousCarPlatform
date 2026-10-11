"""Track map: where the apples (track boundaries) are, from many frames instead of one.

numpy only; the dashboard runs it live (dashboard.TrackMapService) and tools/replay_map.py runs the
same code over a recorded take. Lengths mm, angles deg unless named *_rad, times CLOCK_MONOTONIC s.

    IMU game_rv (~200 Hz) --+
                            +--> Odometry: rear-axle dead reckoning, encoder distance along the IMU
    encoder turns (200 Hz) -+    heading; pose history of the last HISTORY_S (the "short-term memory")

    detections (up to 120 Hz, arrive ~15-30 ms after the exposure), per frame:
      1. camera pose AT THE FRAME'S EXPOSURE TIME: car roll / pitch from the IMU (interpolated),
         high-passed so only suspension motion moves the camera (a sloped driveway does not: the
         apples lie on the same slope), plus the suspension model (camera height vs pitch / roll,
         config/suspension.json from tools/suspension_cal.py) and the estimated pitch bias
      2. each box's bottom-centre -> floor point (CameraModel lens + that pose); noise grows with
         range (radial much more than sideways)
      3. placed in the world with the odometry pose at the exposure time (pose history, not "now")
      4. associated with known apples (Mahalanobis gate, greedy nearest first)
      5. pose correction from the matches with confirmed apples: a small Kalman-style update of
         (x, y, heading, camera pitch bias). It is applied as a correction of the odometry frame,
         so everything driven since the exposure moves with it ("adjust the path since then")
      6. apples refined (2D Kalman), unmatched boxes start tentative apples; apples that should be
         visible (in the image, 0.45-2.2 m) but are not detected lose credibility and are dropped
    tick (10 Hz): apples passed by the car get a side (left / right of the path); confirmed apples
      near the car are chained into a left and a right boundary -> polylines + centre line + track
      width, in the car frame (what a driver needs) and in the world (the map)

World frame: origin = camera floor point at reset, X = car heading at reset, Y left. The pose is the
camera's floor point (CameraModel origin), heading = car heading.
"""
import bisect
import csv
import json
import math
import os
import queue
import threading
import time
from collections import deque
from dataclasses import asdict, dataclass

import numpy as np

from vision.camera_model import CameraModel

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # Pi5/
ODO_CONFIG = os.path.join(ROOT, "config", "odometry.json")
SUSP_CONFIG = os.path.join(ROOT, "config", "suspension.json")
PARAMS_CONFIG = os.path.join(ROOT, "config", "track_map.json")   # optional: {"attitude": "static", ...}
OUT_DIR = os.path.join(ROOT, "results", "trackmap")

DEFAULT_ODO = {
    "mm_per_turn": 23.25,                    # car distance per motor (encoder) turn
    "cam_ahead_of_rear_axle_mm": 150.0,      # camera floor point ahead of the rear axle
    "source": "default: Sim guess (37 mm wheel radius, 10 motor turns per wheel turn); "
              "fit with tools/replay_map.py fit-scale",
}
DEFAULT_SUSP = {
    "h_per_pitch_mm_deg": 0.0,               # camera height change per deg of car pitch (nose up +)
    "h_per_roll_mm_deg": 0.0,                # ... per deg of car roll (right side down +)
    "h_per_abs_roll_mm_deg": 0.0,            # ... per deg of roll either way (a leaning car sits lower)
    "pitch_gain": 1.0,                       # camera pitch change per car pitch change (rigid: 1)
    "pitch_per_roll": 0.0,                   # camera pitch change per car roll (IMU / camera axes not aligned)
    "roll_gain": 1.0,
    "source": "default (not calibrated): tools/suspension_cal.py",
}

# raw columns of <id>_det.csv (one row per box; a frame without boxes has one row with box = -1)
DET_RAW_COLUMNS = ["seq", "sensor_ts_ns", "t_mid_ns", "n_boxes", "crop_x", "crop_y", "crop_w", "crop_h",
                   "box", "x1", "y1", "x2", "y2", "score", "cls"]
# ... followed by what the map made of it
DET_MAP_COLUMNS = ["status", "obs_x_mm", "obs_y_mm", "sigma_r_mm", "sigma_t_mm", "world_x_mm", "world_y_mm",
                   "landmark", "lm_state", "cam_pitch_deg", "cam_roll_deg", "cam_h_mm", "pitch_bias_deg",
                   "pose_x_mm", "pose_y_mm", "pose_yaw_deg", "matched_confirmed", "corr_x_mm", "corr_y_mm",
                   "corr_yaw_deg"]
POSE_COLUMNS = ["t_ns", "x_mm", "y_mm", "yaw_deg", "odo_x_mm", "odo_y_mm", "odo_yaw_deg", "speed_mm_s",
                "car_roll_deg", "car_pitch_deg", "d_roll_deg", "d_pitch_deg", "cam_pitch_deg", "cam_roll_deg",
                "cam_h_mm", "pitch_bias_deg", "sigma_xy_mm", "sigma_yaw_deg"]
LM_COLUMNS = ["t_ns", "id", "state", "side", "x_mm", "y_mm", "sigma_mm", "hits", "misses", "first_ns", "last_ns",
              "car_x_mm", "car_y_mm"]
STATE_NAME = {0: "tentative", 1: "confirmed", 2: "archived"}
SIDE_NAME = {1: "L", -1: "R", 0: ""}


@dataclass
class Params:
    history_s: float = 10.0            # pose / IMU history kept for late detections
    attitude: str = "highpass"         # "highpass": suspension motion only; "absolute": vs config/imu.json level;
                                       # "static": ignore IMU roll / pitch (comparison / uncalibrated IMU)
    level_tau_s: float = 2.0           # high-pass time constant (slower changes = ground slope)
    max_speed_mm_s: float = 8000.0     # encoder steps faster than this are glitches
    # observation noise: floor point from a box bottom
    sigma_base_mm: float = 10.0
    sigma_px_rad: float = 0.0025       # box-edge jitter as an angle (~1.5 px of the 960x540 lores)
    sigma_pitch_rad: float = 0.006     # uncorrected camera pitch (suspension, paving) ~0.35 deg
    sigma_az_rad: float = 0.004        # sideways angular noise
    min_range_mm: float = 300.0
    max_use_range_mm: float = 3500.0   # boxes beyond: not used
    max_init_range_mm: float = 3000.0  # new apples only start within this range
    edge_margin: float = 0.004         # boxes touching the frame edge (cut apples) are not used
    # association / lifecycle
    gate_chi2: float = 9.21            # 99 % for 2 dof
    gate_max_mm: float = 400.0
    new_min_sep_mm: float = 70.0       # no new apple this close to a known one
    confirm_hits: int = 5
    tentative_timeout_s: float = 0.4
    tentative_max_misses: int = 4
    confirmed_max_misses: int = 40     # consecutive frames in view without a detection -> gone
    miss_range_mm: tuple = (450.0, 2200.0)
    # rolling window: the map is local and short-lived, so it works on any track from the first
    # metre and never piles up laps (odometry drifts ~0.3-0.5 m per lap; old laps would be ghosts)
    window_s: float = 4.0              # apples not seen for this long are forgotten
    window_behind_mm: float = 1000.0   # ... and so are apples this far behind the camera
    trail_s: float = 10.0              # path kept for display
    lm_process_mm: float = 1.0         # random walk per update (apples get nudged)
    # pose filter
    correct: bool = True
    estimate_pitch: bool = False       # camera pitch bias from the apples (experimental: it also soaks up a
                                       # wrong odometry scale; validate on a take before switching it on)
    min_matches: int = 2
    # the encoder is on the MOTOR shaft: gearbox + open differential to the wheels, so an unloaded inner wheel
    # spinning in a corner and gear lash on throttle <-> brake both reach the odometry (the apple fit gave
    # 33-41 mm/turn while driving vs 55 pushed by hand, 2026-10-11): trust it less than a wheel encoder,
    # lean on the apples (replays: turn direction agrees 79-92 % vs 79-89 % at 0.03 / 0.02)
    odo_scale_sigma: float = 0.10      # distance error fraction
    odo_lat_sigma: float = 0.05        # sideways error per distance
    heading_rel_sigma: float = 0.01    # heading error per heading change
    heading_drift_deg_s: float = 0.02  # game_rv drift
    pitch_bias_sigma_deg: float = 1.0  # prior / steady-state spread of the bias
    pitch_bias_tau_s: float = 5.0      # bias decays to 0 without evidence
    pitch_bias_max_deg: float = 4.0
    # boundaries
    bound_ahead_mm: float = 4000.0
    bound_behind_mm: float = 600.0
    bound_max_gap_mm: float = 1400.0
    bound_max_turn_deg: float = 55.0
    default_width_mm: float = 1000.0


def load_params(path=PARAMS_CONFIG):
    """Params with overrides from config/track_map.json (unknown keys are an error)."""
    p = Params()
    try:
        with open(path) as f:
            over = json.load(f)
    except (OSError, ValueError):
        return p
    for k, v in over.items():
        if k.startswith("_"):
            continue
        if not hasattr(p, k):
            raise ValueError("%s: unknown track map parameter %r" % (path, k))
        setattr(p, k, tuple(v) if isinstance(getattr(p, k), tuple) else v)
    return p


def load_json(path, default):
    try:
        with open(path) as f:
            d = dict(default)
            d.update(json.load(f))
            return d
    except (OSError, ValueError):
        return dict(default)


def write_json(path, d):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(d, f, indent=1)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def take_files(take):
    """recordings/<id> (or its meta path) -> dict of the take's file paths."""
    take = take.rstrip("/")
    if take.endswith("_meta.json"):
        take = os.path.dirname(take)
    rid = os.path.basename(take)
    p = lambda s: os.path.join(take, rid + s)   # noqa: E731
    return {"id": rid, "dir": take, "meta": p("_meta.json"), "video": p(".mp4"), "frames": p("_frames.csv"),
            "imu": p("_imu.csv"), "drive": p("_drive.csv"), "det": p("_det.csv")}


# ---------------------------------------------------------------- rotations
_B = np.array([[0.0, -1.0, 0.0], [0.0, 0.0, -1.0], [1.0, 0.0, 0.0]])   # level camera looking along +X


def _rx(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


def _ry(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def _rz(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def cam_to_floor(pitch0_deg, roll0_deg, d_roll_deg=0.0, d_pitch_deg=0.0):
    """camera (OpenCV axes) -> camera floor frame (X car heading, Y left, Z up): the static mount
    (CameraModel pitch / roll) on a body rolled / pitched by d_roll (right side down +) /
    d_pitch (nose up +) relative to the ground."""
    mount = (_rz(math.radians(roll0_deg)) @ _rx(math.radians(pitch0_deg)) @ _B).T
    return _ry(-math.radians(d_pitch_deg)) @ _rx(math.radians(d_roll_deg)) @ mount


def camera_angles(c2w):
    """camera -> floor rotation -> (pitch, roll, heading) deg in CameraModel conventions."""
    d = c2w[:, 2]
    heading = math.atan2(d[1], d[0])
    pitch = math.asin(max(-1.0, min(1.0, -d[2])))
    m = c2w.T @ _rz(heading) @ _B.T @ _rx(pitch).T
    return math.degrees(pitch), math.degrees(math.atan2(m[1, 0], m[0, 0])), math.degrees(heading)


def _rot2(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s], [s, c]])


def _wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


_J = np.array([[0.0, -1.0], [1.0, 0.0]])


# ---------------------------------------------------------------- time series
class _Buf:
    """Time-sorted samples (t, tuple) with linear interpolation; ends clamp."""

    def __init__(self):
        self.t, self.v = [], []

    def add(self, t, v):
        if self.t and t < self.t[-1]:
            i = bisect.bisect(self.t, t)
            self.t.insert(i, t)
            self.v.insert(i, v)
        else:
            self.t.append(t)
            self.v.append(v)

    def trim(self, before):
        i = bisect.bisect_left(self.t, before)
        if i > 500 and i < len(self.t) - 2:
            del self.t[:i], self.v[:i]

    def at(self, t):
        n = len(self.t)
        if not n:
            return None
        i = bisect.bisect(self.t, t)
        if i == 0:
            return self.v[0]
        if i == n:
            return self.v[-1]
        t0, t1 = self.t[i - 1], self.t[i]
        f = (t - t0) / (t1 - t0) if t1 > t0 else 0.0
        return tuple(a + f * (b - a) for a, b in zip(self.v[i - 1], self.v[i]))

    @property
    def last_t(self):
        return self.t[-1] if self.t else None


# ---------------------------------------------------------------- landmarks
class Landmark:
    __slots__ = ("id", "x", "P", "hits", "misses", "first_t", "last_t", "state", "side", "score")

    def __init__(self, lid, x, P, t, score):
        self.id, self.x, self.P = lid, np.asarray(x, float), np.asarray(P, float)
        self.hits, self.misses, self.first_t, self.last_t = 1, 0, t, t
        self.state, self.side, self.score = 0, 0, score

    @property
    def sigma(self):
        return math.sqrt(max(np.linalg.eigvalsh(self.P)))


def _inv2(S):
    """Inverse of (..., 2, 2) symmetric matrices."""
    a, b, d = S[..., 0, 0], S[..., 0, 1], S[..., 1, 1]
    det = a * d - b * b
    out = np.empty_like(S)
    out[..., 0, 0], out[..., 1, 1] = d / det, a / det
    out[..., 0, 1] = out[..., 1, 0] = -b / det
    return out


# ---------------------------------------------------------------- the mapper
class TrackMapper:
    def __init__(self, cam=None, params=None, odo=None, susp=None):
        self.cam = cam or CameraModel.load()
        self.p = params or load_params()
        self.odo_cfg = odo or load_json(ODO_CONFIG, DEFAULT_ODO)
        self.susp = susp or load_json(SUSP_CONFIG, DEFAULT_SUSP)
        self.mm_per_turn = float(self.odo_cfg["mm_per_turn"])
        self.cam_ahead = float(self.odo_cfg["cam_ahead_of_rear_axle_mm"])
        self.log = None
        self.on_frame = None            # callback(frame result) after every detection frame (tools)
        self.map_id = 0
        self.reset()

    # ---- state -------------------------------------------------------------------------------
    def reset(self):
        self.map_id += 1
        self.imu = _Buf()               # (roll, pitch, yaw_unwrapped_rad, roll_lp, pitch_lp)
        self._imu_prev = None           # (t, yaw_raw_rad, yaw_unwrapped, roll_lp, pitch_lp)
        self._turns = []                # pending (t, turns), integrated once the IMU covers t
        self.odo = _Buf()               # (rear_x, rear_y, heading_rad) odometry frame
        self._odo_last = None           # (t, turns, rx, ry, th)
        self.yaw0 = None
        self.C = np.zeros(3)            # odometry -> world correction (x, y, angle)
        self.Pp = np.diag([1.0, 1.0, 1e-6])   # pose covariance (world x, y mm^2; heading rad^2)
        self.pitch_bias, self.pb_var = 0.0, self.p.pitch_bias_sigma_deg ** 2
        self._pb_t = None
        self.lms, self._next_id = [], 1
        self._dets = []                 # queued detection frames (t_mid order)
        self.trail = deque()            # (t, world x, world y) of the camera floor point, last trail_s, >= 30 mm apart
        self.width = self.p.default_width_mm
        self.bounds = None
        self.local_apples = np.zeros((0, 2))
        self.last_frame = None          # last processed detection frame (for the UI)
        self._last_pose_log = -1.0
        self.stats = {"frames": 0, "boxes": 0, "used": 0, "matched": 0, "corrected": 0, "created": 0,
                      "confirmed": 0, "dropped": 0, "forgotten": 0, "enc_glitches": 0, "no_odometry": 0,
                      "resid_mm": None}
        self._resid = []
        self.match_d = deque(maxlen=200000)   # (observation range, distance to its apple before the update)

    # ---- inputs ------------------------------------------------------------------------------
    def add_imu(self, t, roll, pitch, yaw):
        """Car attitude (CarFrame.attitude: deg, roll right-down +, pitch nose-up +, yaw left +)."""
        yr = math.radians(yaw)
        prev = self._imu_prev
        if prev is None:
            yu, rl, pl = yr, roll, pitch
        elif t < prev[0]:                       # rare reorder: keep the filters, store the sample
            self.imu.add(t, (roll, pitch, prev[2] + _wrap(yr - prev[1]), prev[3], prev[4]))
            return
        else:
            yu = prev[2] + _wrap(yr - prev[1])
            a = 1.0 - math.exp(-(t - prev[0]) / self.p.level_tau_s)
            rl, pl = prev[3] + a * (roll - prev[3]), prev[4] + a * (pitch - prev[4])
        self._imu_prev = (t, yr, yu, rl, pl)
        self.imu.add(t, (roll, pitch, yu, rl, pl))

    def add_turns(self, t, turns):
        self._turns.append((t, float(turns)))

    def add_detections(self, t_mid, boxes, crop, seq=None, sensor_ts=None):
        """boxes: [(x1, y1, x2, y2, score, cls)] normalised 0..1 of the frame showing `crop`
        (ScalerCrop, sensor px). t_mid: exposure time of the frame (s)."""
        self._dets.append({"t": t_mid, "boxes": list(boxes), "crop": tuple(crop), "seq": seq, "sensor_ts": sensor_ts})

    # ---- processing --------------------------------------------------------------------------
    def step(self):
        """Integrate odometry as far as the IMU covers it, then every queued detection frame the
        odometry has reached. Returns the number of frames processed."""
        imu_t = self.imu.last_t
        n = 0
        if imu_t is not None:
            i = 0
            while i < len(self._turns) and self._turns[i][0] <= imu_t:
                self._integrate(*self._turns[i])
                i += 1
            del self._turns[:i]
        odo_t = self.odo.last_t
        while self._dets:
            f = self._dets[0]
            if odo_t is None or f["t"] > odo_t:
                newest = self._dets[-1]["t"]
                if newest - f["t"] > 1.0:       # odometry is not coming (IMU down): give up on it
                    self._dets.pop(0)
                    self.stats["no_odometry"] += 1
                    self._log_frame(f, None)
                    continue
                break
            self._dets.pop(0)
            if f["t"] < self.odo.t[0]:
                self.stats["no_odometry"] += 1
                self._log_frame(f, None)
                continue
            self._process(f)
            n += 1
        if odo_t is not None:
            keep = odo_t - self.p.history_s
            self.odo.trim(keep)
            self.imu.trim(keep)
        if len(self._turns) > 2000:             # no IMU: do not grow forever
            del self._turns[:-200]
        return n

    def _integrate(self, t, turns):
        att = self.imu.at(t)
        if self._odo_last is None:
            self.yaw0 = att[2]
            self._odo_last = (t, turns, -self.cam_ahead, 0.0, 0.0)    # camera floor point at the origin
            self.odo.add(t, (-self.cam_ahead, 0.0, 0.0))
            return
        t0, turns0, rx, ry, th0 = self._odo_last
        dt = t - t0
        if dt <= 0:
            return
        ds = abs(turns - turns0) * self.mm_per_turn     # the ESC cannot reverse: |turns| is distance
        if ds > self.p.max_speed_mm_s * dt + 5.0:
            ds = 0.0
            self.stats["enc_glitches"] += 1
        th = att[2] - self.yaw0
        thm = 0.5 * (th + th0)
        rx += ds * math.cos(thm)
        ry += ds * math.sin(thm)
        # pose uncertainty grows with distance / heading change / time
        p = self.p
        c, s = math.cos(thm + self.C[2]), math.sin(thm + self.C[2])
        a2, l2 = (p.odo_scale_sigma * ds) ** 2, (p.odo_lat_sigma * ds) ** 2
        self.Pp[0, 0] += a2 * c * c + l2 * s * s
        self.Pp[1, 1] += a2 * s * s + l2 * c * c
        self.Pp[0, 1] += (a2 - l2) * c * s
        self.Pp[1, 0] = self.Pp[0, 1]
        self.Pp[2, 2] += (p.heading_rel_sigma * (th - th0)) ** 2 + (math.radians(p.heading_drift_deg_s)) ** 2 * dt
        self._odo_last = (t, turns, rx, ry, th)
        self.odo.add(t, (rx, ry, th))
        if self.log is not None and t - self._last_pose_log >= 0.02:
            self._last_pose_log = t
            self._log_pose(t)

    # ---- poses -------------------------------------------------------------------------------
    def odo_pose(self, t=None):
        """Odometry-only pose (camera floor point x, y, heading rad) at t (default: latest)."""
        v = self.odo.at(self.odo.last_t if t is None else t)
        if v is None:
            return None
        rx, ry, th = v
        return rx + self.cam_ahead * math.cos(th), ry + self.cam_ahead * math.sin(th), th

    def pose(self, t=None):
        """World pose (camera floor point x, y mm, heading rad) at t (default: latest)."""
        o = self.odo_pose(t)
        if o is None:
            return None
        cx, cy, ca = self.C
        c, s = math.cos(ca), math.sin(ca)
        return c * o[0] - s * o[1] + cx, s * o[0] + c * o[1] + cy, o[2] + ca

    def camera_at(self, t):
        """Camera state at t: dict(R cam->floor, h, pitch / roll used, car attitude)."""
        a = self.imu.at(t)
        if a is None:
            return None
        roll, pitch, _, rl, pl = a
        if self.p.attitude == "highpass":
            dr, dp = roll - rl, pitch - pl
        elif self.p.attitude == "absolute":
            dr, dp = roll, pitch
        else:                                   # "static": camera fixed to the ground (IMU attitude unused)
            dr = dp = 0.0
        sp = self.susp
        h = (self.cam.height_mm + sp["h_per_pitch_mm_deg"] * dp + sp["h_per_roll_mm_deg"] * dr
             + sp["h_per_abs_roll_mm_deg"] * abs(dr))
        dp, dr = sp["pitch_gain"] * dp + sp["pitch_per_roll"] * dr, sp["roll_gain"] * dr   # camera motion
        R = cam_to_floor(self.cam.pitch_deg, self.cam.roll_deg, dr, dp - self.pitch_bias)
        return {"R": R, "h": h, "d_roll": dr, "d_pitch": dp, "roll": roll, "pitch": pitch}

    # ---- geometry ----------------------------------------------------------------------------
    def floor_points(self, u, v, R, h):
        """Sensor pixels -> floor points (N, 2) in the camera floor frame (NaN: above the horizon)."""
        x, y = self.cam.undistort(u, v)
        d = np.stack([x, y, np.ones_like(x)], -1) @ R.T
        with np.errstate(divide="ignore", invalid="ignore"):
            t = -h / d[:, 2]
        t = np.where(t > 0, t, np.nan)
        return d[:, :2] * t[:, None]

    def project(self, pts, R, h, crop):
        """Floor points (N, 2) in the camera floor frame -> normalised frame coords (N, 2), NaN behind."""
        pts = np.asarray(pts, float).reshape(-1, 2)
        p = np.c_[pts, np.full(len(pts), -h)]
        c = p @ R                                   # floor -> camera: R.T @ p
        with np.errstate(divide="ignore", invalid="ignore"):
            x = np.where(c[:, 2] > 1.0, c[:, 0] / c[:, 2], np.nan)
            y = np.where(c[:, 2] > 1.0, c[:, 1] / c[:, 2], np.nan)
        u, v = self.cam.distort(x, y)
        nx, ny = self.cam.from_sensor(u, v, crop)
        return np.stack([nx, ny], -1)

    def _obs_cov(self, xy, h):
        d = np.hypot(xy[:, 0], xy[:, 1])
        p = self.p
        sa = math.hypot(p.sigma_px_rad, p.sigma_pitch_rad)
        sr = np.hypot(p.sigma_base_mm, (d * d + h * h) / h * sa)
        st = np.hypot(p.sigma_base_mm, d * p.sigma_az_rad)
        u = xy / np.maximum(d, 1.0)[:, None]
        uu = u[:, :, None] * u[:, None, :]
        S = sr[:, None, None] ** 2 * uu + st[:, None, None] ** 2 * (np.eye(2) - uu)
        return S, sr, st

    # ---- one detection frame -----------------------------------------------------------------
    def _process(self, f):
        p, t = self.p, f["t"]
        st = self.stats
        st["frames"] += 1
        st["boxes"] += len(f["boxes"])
        self._decay_pitch_bias(t)
        cam = self.camera_at(t)
        boxes = f["boxes"]
        status = ["ok"] * len(boxes)
        m = p.edge_margin
        use = []
        for i, b in enumerate(boxes):
            if b[3] >= 1 - m or b[0] <= m or b[2] >= 1 - m:
                status[i] = "edge"
            else:
                use.append(i)
        crop = f["crop"]
        if use:
            bx = np.array([boxes[i][:4] for i in use], float)
            u, v = self.cam.to_sensor((bx[:, 0] + bx[:, 2]) / 2, bx[:, 3], crop)
        else:
            u = v = np.zeros(0)

        def observe():
            c = self.camera_at(t)
            xy = self.floor_points(u, v, c["R"], c["h"]) if len(use) else np.zeros((0, 2))
            return c, xy

        cam, xy = observe()
        rng = np.hypot(xy[:, 0], xy[:, 1]) if len(xy) else np.zeros(0)
        ok = ~np.isnan(rng) & (rng >= p.min_range_mm) & (rng <= p.max_use_range_mm)
        for k, i in enumerate(use):
            if not ok[k]:
                status[i] = "range"
        sel = np.nonzero(ok)[0]
        idx = [use[k] for k in sel]                 # box index of each observation
        u, v = u[sel], v[sel]
        xy = xy[sel]
        st["used"] += len(idx)

        pose = self.pose(t)
        active = [lm for lm in self.lms if lm.state < 2]
        matches = self._associate(xy, cam["h"], pose, active)

        corr = np.zeros(3)
        conf = [(k, j) for k, j in matches if active[j].state == 1]
        if p.correct and len(conf) >= p.min_matches:
            corr = self._correct(xy, cam["h"], pose, active, conf)
            if corr is not None:
                st["corrected"] += 1
                cam, xy = observe()
                xy = xy if len(xy) else np.zeros((0, 2))
                pose = self.pose(t)
            else:
                corr = np.zeros(3)

        # update / create landmarks with the (corrected) observations
        S, sr, stt = self._obs_cov(xy, cam["h"]) if len(xy) else (np.zeros((0, 2, 2)), [], [])
        R2 = _rot2(pose[2])
        w = pose[:2] + xy @ R2.T if len(xy) else np.zeros((0, 2))
        Sw = R2 @ S @ R2.T
        matched_lm, lm_of = set(), {}
        for k, j in matches:
            lm = active[j]
            self.match_d.append((float(np.hypot(*xy[k])), float(np.hypot(*(w[k] - lm.x)))))
            Pk = lm.P
            K = Pk @ _inv2(Pk + Sw[k])
            lm.x = lm.x + K @ (w[k] - lm.x)
            lm.P = (np.eye(2) - K) @ Pk + np.eye(2) * p.lm_process_mm ** 2
            lm.hits += 1
            lm.misses = 0
            lm.last_t = t
            lm.score = max(lm.score, boxes[idx[k]][4])
            matched_lm.add(lm.id)
            lm_of[k] = lm
            if lm.state == 0 and lm.hits >= p.confirm_hits:
                lm.state = 1
                st["confirmed"] += 1
        st["matched"] += len(matches)
        matched_obs = {k for k, _ in matches}
        pos_cov = self.Pp[:2, :2]
        for k in range(len(xy)):
            if k in matched_obs or np.hypot(*xy[k]) > p.max_init_range_mm:
                continue
            if any(np.hypot(*(lm.x - w[k])) < p.new_min_sep_mm for lm in active):
                continue
            lm = Landmark(self._next_id, w[k], Sw[k] + pos_cov, t, boxes[idx[k]][4])
            self._next_id += 1
            self.lms.append(lm)
            active.append(lm)
            lm_of[k] = lm
            st["created"] += 1

        self._misses(active, matched_lm, pose, cam, crop, t)
        self._prune(t, pose)

        res = {"t": t, "seq": f["seq"], "sensor_ts": f["sensor_ts"], "boxes": boxes, "crop": crop,
               "status": status, "idx": idx, "xy": xy, "w": w, "sr": sr, "st": stt, "lm": lm_of,
               "cam": cam, "pose": pose, "n_conf": len(conf), "corr": corr}
        self.last_frame = res
        if self.log is not None:
            self._log_frame(f, res)
        if self.on_frame is not None:
            self.on_frame(res)
        return res

    def _associate(self, xy, h, pose, active):
        """Greedy nearest-first matching under a Mahalanobis gate -> [(obs k, landmark j)]."""
        if not len(xy) or not active:
            return []
        p = self.p
        S, _, _ = self._obs_cov(xy, h)
        R2 = _rot2(pose[2])
        w = pose[:2] + xy @ R2.T
        Sw = R2 @ S @ R2.T
        X = np.array([lm.x for lm in active])
        P = np.array([lm.P for lm in active])
        lever = (w - pose[:2]) @ _J.T                       # d w / d heading
        Spose = self.Pp[:2, :2] + self.Pp[2, 2] * lever[:, :, None] * lever[:, None, :]
        D = X[None, :, :] - w[:, None, :]                   # (N, M, 2)
        Ctot = Sw[:, None] + P[None, :] + Spose[:, None]
        d2 = np.einsum("nmi,nmij,nmj->nm", D, _inv2(Ctot), D)
        e2 = (D ** 2).sum(-1)
        cand = np.argwhere((d2 < p.gate_chi2) & (e2 < p.gate_max_mm ** 2))
        if not len(cand):
            return []
        order = np.argsort(d2[cand[:, 0], cand[:, 1]])
        used_k, used_j, out = set(), set(), []
        for k, j in cand[order]:
            if k in used_k or j in used_j:
                continue
            used_k.add(k)
            used_j.add(j)
            out.append((int(k), int(j)))
        return out

    def _correct(self, xy, h, pose, active, pairs):
        """Kalman-style update of (dx, dy, dheading, dpitch_bias) from matched confirmed apples;
        applies it to the odometry -> world correction. Returns the applied (dx, dy, dyaw_deg) or None."""
        p = self.p
        S, _, _ = self._obs_cov(xy, h)
        R2 = _rot2(pose[2])
        w = pose[:2] + xy @ R2.T
        Sw = R2 @ S @ R2.T
        npar = 4 if p.estimate_pitch else 3
        prior = np.zeros((npar, npar))
        prior[:3, :3] = self.Pp
        if npar == 4:
            prior[3, 3] = self.pb_var
        prior_inv = np.linalg.inv(prior + np.eye(npar) * 1e-9)

        def solve(pp):
            A, b = prior_inv.copy(), np.zeros(npar)
            for k, j in pp:
                lm = active[j]
                H = np.zeros((2, npar))
                H[:, :2] = np.eye(2)
                H[:, 2] = _J @ (w[k] - pose[:2])
                if npar == 4:                                # camera pitch down + -> point comes closer
                    d = math.hypot(*xy[k])
                    H[:, 3] = R2 @ (xy[k] / d) * (-(d * d + h * h) / h * math.pi / 180.0)
                Wk = _inv2(Sw[k] + lm.P)
                A += H.T @ Wk @ H
                b += H.T @ Wk @ (lm.x - w[k])
            cov = np.linalg.inv(A)
            return cov @ b, cov

        def chi2(pp, dlt):
            out = []
            for k, j in pp:
                lm = active[j]
                pred = w[k] + dlt[:2] + dlt[2] * (_J @ (w[k] - pose[:2]))
                if npar == 4:
                    d = math.hypot(*xy[k])
                    pred = pred + R2 @ (xy[k] / d) * (-(d * d + h * h) / h * math.pi / 180.0) * dlt[3]
                r = lm.x - pred
                out.append((float(r @ _inv2(Sw[k] + lm.P) @ r), float(np.hypot(*r))))
            return out

        dlt, cov = solve(pairs)
        c2 = chi2(pairs, dlt)
        keep = [pr for pr, (x2, _) in zip(pairs, c2) if x2 < p.gate_chi2]
        if len(keep) < p.min_matches:
            return None
        if len(keep) < len(pairs):
            dlt, cov = solve(keep)
            c2 = chi2(keep, dlt)
        self._resid.extend(e for _, e in c2)
        if len(self._resid) > 2000:
            del self._resid[:-1000]
        # apply: rotate about the camera point at the frame time, then shift
        px, py = pose[0], pose[1]
        cx, cy, ca = self.C
        dth = float(dlt[2])
        c, s = math.cos(dth), math.sin(dth)
        cx, cy = c * (cx - px) - s * (cy - py) + px + dlt[0], s * (cx - px) + c * (cy - py) + py + dlt[1]
        self.C = np.array([cx, cy, ca + dth])
        self.Pp = 0.5 * (cov[:3, :3] + cov[:3, :3].T)
        if npar == 4:
            self.pitch_bias = max(-p.pitch_bias_max_deg, min(p.pitch_bias_max_deg, self.pitch_bias + float(dlt[3])))
            self.pb_var = float(cov[3, 3])
        return np.array([dlt[0], dlt[1], math.degrees(dth)])

    def _decay_pitch_bias(self, t):
        """Ornstein-Uhlenbeck: the bias relaxes to 0 and its variance back to the prior."""
        if self._pb_t is not None and t > self._pb_t:
            a = math.exp(-(t - self._pb_t) / self.p.pitch_bias_tau_s)
            self.pitch_bias *= a
            self.pb_var = a * a * self.pb_var + (1 - a * a) * self.p.pitch_bias_sigma_deg ** 2
        self._pb_t = t

    def _misses(self, active, matched, pose, cam, crop, t):
        """Apples that should have been detected in this frame (in the image, in range) but were not."""
        cand = [lm for lm in active if lm.id not in matched]
        if not cand:
            return
        R2 = _rot2(pose[2])
        rel = (np.array([lm.x for lm in cand]) - pose[:2]) @ R2           # world -> camera floor frame
        d = np.hypot(rel[:, 0], rel[:, 1])
        uv = self.project(rel, cam["R"], cam["h"], crop)
        lo, hi = self.p.miss_range_mm
        inview = (d > lo) & (d < hi) & (uv[:, 0] > 0.03) & (uv[:, 0] < 0.97) & (uv[:, 1] > 0.03) & (uv[:, 1] < 0.97)
        for lm, iv in zip(cand, inview):
            if iv:
                lm.misses += 1

    def _prune(self, t, pose):
        """Drops tentative apples that did not repeat, confirmed ones that stopped being detected
        while in view, and everything outside the rolling window (not seen for window_s, or more
        than window_behind_mm behind the camera)."""
        p, keep = self.p, []
        fx, fy = math.cos(pose[2]), math.sin(pose[2])
        for lm in self.lms:
            if lm.state == 0:
                if lm.misses >= p.tentative_max_misses or t - lm.last_t > p.tentative_timeout_s:
                    continue
            else:
                if lm.misses >= p.confirmed_max_misses and t - lm.last_t > 0.3:
                    self.stats["dropped"] += 1
                    continue
                if (t - lm.last_t > p.window_s
                        or (lm.x[0] - pose[0]) * fx + (lm.x[1] - pose[1]) * fy < -p.window_behind_mm):
                    self.stats["forgotten"] += 1
                    continue
            keep.append(lm)
        self.lms = keep

    # ---- 10 Hz: sides, boundaries, logs -------------------------------------------------------
    def tick(self, t=None):
        """Side labels + boundaries (+ trail, landmark / track logs). t defaults to the odometry time."""
        t = self.odo.last_t if t is None else t
        pose = self.pose(t)
        if pose is None:
            return None
        if not self.trail or math.hypot(pose[0] - self.trail[-1][1], pose[1] - self.trail[-1][2]) > 30:
            self.trail.append((t, pose[0], pose[1]))
        while self.trail and t - self.trail[0][0] > self.p.trail_s:
            self.trail.popleft()
        R2 = _rot2(pose[2])
        conf = [lm for lm in self.lms if lm.state == 1]
        if conf:
            rel = (np.array([lm.x for lm in conf]) - pose[:2]) @ R2
            for lm, (x, y) in zip(conf, rel):
                if lm.side == 0 and -250 < x < 0 and 120 < abs(y) < 2000:   # the car is passing it
                    lm.side = 1 if y > 0 else -1
        else:
            rel = np.zeros((0, 2))
        self.local_apples = rel                  # confirmed apples in the car frame (vision/planner.py)
        self.bounds = self._boundaries(conf, rel, pose)
        if self.log is not None:
            self._log_tick(t, pose, R2)
        return self.bounds

    def _boundaries(self, lms, rel, pose):
        """Chain confirmed apples ahead into a left and a right boundary (car frame, mm)."""
        p = self.p
        pts = [(lm, r) for lm, r in zip(lms, rel)
               if -p.bound_behind_mm < r[0] < p.bound_ahead_mm and abs(r[1]) < 3000]
        half = self.width / 2
        chains = {1: [np.array([-p.bound_behind_mm, half])], -1: [np.array([-p.bound_behind_mm, -half])]}
        ids = {1: [], -1: []}
        dirs = {1: np.array([1.0, 0.0]), -1: np.array([1.0, 0.0])}
        free = list(range(len(pts)))
        cos_max = math.cos(math.radians(p.bound_max_turn_deg))
        done = {1: False, -1: False}
        while free and not (done[1] and done[-1]):
            # extend the chain that is further behind (by its last point's distance from the car)
            sides = [s for s in (1, -1) if not done[s]]
            s = min(sides, key=lambda q: np.hypot(*chains[q][-1]))
            o = -s
            cur, dr = chains[s][-1], dirs[s]
            ocur, odr = chains[o][-1], dirs[o]
            best, bcost = None, None
            for k in free:
                lm, r = pts[k]
                if lm.side == o:
                    continue
                v = r - cur
                dist = math.hypot(*v)
                if dist < 30 or dist > p.bound_max_gap_mm:
                    continue
                along = v @ dr
                if along <= 0 or along / dist < cos_max:
                    continue
                # must stay on its side of the other boundary's current line
                if s * float(_J @ odr @ (r - ocur)) <= 0:
                    continue
                lat = abs(dr[0] * v[1] - dr[1] * v[0])
                cost = dist + 2.0 * lat - (300.0 if lm.side == s else 0.0)
                if bcost is None or cost < bcost:
                    best, bcost = k, cost
            if best is None:
                done[s] = True
                continue
            free.remove(best)
            lm, r = pts[best]
            v = r - cur
            n = np.hypot(*v)
            if len(chains[s]) > 1:
                d = 0.5 * dr + 0.5 * v / n
                dirs[s] = d / np.hypot(*d)
            else:
                dirs[s] = v / n if (v / n) @ dr > cos_max else dr
            chains[s].append(r)
            ids[s].append(lm.id)
        left = [c.tolist() for c in chains[1][1:]]
        right = [c.tolist() for c in chains[-1][1:]]
        # centre line: from the longer side, midpoints to the nearest point of the other polyline
        center, widths = [], []
        a, b = (left, right) if len(left) >= len(right) else (right, left)
        if len(b) >= 1:
            B = np.array(b)
            for q in a:
                qn = np.array(q)
                cp = _nearest_on_polyline(qn, B)
                wd = np.hypot(*(cp - qn))
                if wd < 2500:
                    center.append(((qn + cp) / 2).tolist())
                    widths.append(wd)
        if len(widths) >= 2:
            self.width = 0.8 * self.width + 0.2 * float(np.median(widths))
        R2 = _rot2(pose[2])
        to_w = lambda pl: [(np.array(q) @ R2.T + pose[:2]).tolist() for q in pl]   # noqa: E731
        return {"left": left, "right": right, "center": center, "ids_left": ids[1], "ids_right": ids[-1],
                "width": self.width, "left_w": to_w(left), "right_w": to_w(right), "center_w": to_w(center)}

    # ---- outputs ------------------------------------------------------------------------------
    def snapshot(self, with_proj=True):
        """JSON-able state for the dashboard (world mm, rounded). Small by construction: the rolling
        window keeps a few dozen apples and the last trail_s of path."""
        t = self.odo.last_t
        pose = self.pose()
        if pose is None:
            return {"map_id": self.map_id, "up": False, "stats": self.stats}
        o = self.odo_pose()
        R = lambda v: int(round(v))   # noqa: E731
        lms = [[lm.id, R(lm.x[0]), R(lm.x[1]), lm.state, lm.side, R(lm.sigma), lm.hits] for lm in self.lms]
        f = self.last_frame
        det = []
        if f is not None and t is not None and t - f["t"] < 0.3:
            for k in range(len(f["w"])):
                det.append([R(f["w"][k][0]), R(f["w"][k][1]), 1 if k in f["lm"] and f["lm"][k].state == 1 else 0])
        b = self.bounds or {}
        rd = lambda pl: [[R(x), R(y)] for x, y in pl]   # noqa: E731
        out = {"map_id": self.map_id, "up": True, "t": t,
               "pose": [R(pose[0]), R(pose[1]), round(math.degrees(pose[2]), 2)],
               "odo": [R(o[0]), R(o[1]), round(math.degrees(o[2]), 2)],
               "corr": [R(self.C[0]), R(self.C[1]), round(math.degrees(self.C[2]), 2)],
               "sigma_xy": R(math.sqrt(max(np.linalg.eigvalsh(self.Pp[:2, :2])))),
               "pitch_bias": round(self.pitch_bias, 2), "lms": lms, "det": det,
               "left": rd(b.get("left_w", [])), "right": rd(b.get("right_w", [])), "center": rd(b.get("center_w", [])),
               "width": R(self.width), "trail": [[R(x), R(y)] for _, x, y in self.trail],
               "window_s": self.p.window_s, "stats": dict(self.stats, resid_mm=self.resid_mm())}
        if with_proj and f is not None and t is not None:
            out["proj"] = self.image_overlay(t, f["crop"])
        if f is not None:
            c = f["cam"]
            out["cam"] = {"h": round(c["h"], 1), "d_pitch": round(c["d_pitch"], 2), "d_roll": round(c["d_roll"], 2)}
        return out

    def image_overlay(self, t, crop):
        """Confirmed apples and boundaries projected into the camera frame (normalised 0..1)."""
        cam, pose = self.camera_at(t), self.pose(t)
        if cam is None or pose is None:
            return {}
        R2 = _rot2(pose[2])
        out = {"lms": [], "left": [], "right": []}
        near = [lm for lm in self.lms if lm.state == 1]
        if near:
            rel = (np.array([lm.x for lm in near]) - pose[:2]) @ R2
            ok = (rel[:, 0] > 250) & (np.hypot(rel[:, 0], rel[:, 1]) < 5000)
            if ok.any():
                uv = self.project(rel[ok], cam["R"], cam["h"], crop)
                for lm, (x, y) in zip([lm for lm, k in zip(near, ok) if k], uv):
                    if not np.isnan(x):
                        out["lms"].append([round(float(x), 4), round(float(y), 4), lm.id, lm.side])
        b = self.bounds or {}
        for key in ("left", "right"):
            pl = [q for q in b.get(key, []) if q[0] > 250]
            if len(pl) >= 2:
                dense = []
                for a, c in zip(pl[:-1], pl[1:]):
                    for f in np.linspace(0, 1, 6)[:-1]:
                        dense.append([a[0] + f * (c[0] - a[0]), a[1] + f * (c[1] - a[1])])
                dense.append(pl[-1])
                uv = self.project(np.array(dense), cam["R"], cam["h"], crop)
                out[key] = [[round(float(x), 4), round(float(y), 4)] for x, y in uv if not np.isnan(x)]
        return out

    def resid_mm(self):
        return round(float(np.median(self._resid[-500:])), 1) if self._resid else None

    def final_map(self):
        R = lambda v: round(float(v), 1)   # noqa: E731
        return {
            "map_id": self.map_id, "params": asdict(self.p), "odometry": self.odo_cfg, "suspension": self.susp,
            "camera": {"height_mm": self.cam.height_mm, "pitch_deg": self.cam.pitch_deg, "roll_deg": self.cam.roll_deg},
            "frame": "world: origin = camera floor point at reset, X = car heading at reset, Y left, mm",
            "window": "rolling: only the apples of the last window_s / window_behind_mm (the full path is in _pose.csv)",
            "landmarks": [{"id": lm.id, "x": R(lm.x[0]), "y": R(lm.x[1]), "sigma": R(lm.sigma), "state": STATE_NAME[lm.state],
                           "side": SIDE_NAME[lm.side], "hits": lm.hits, "first_t": lm.first_t, "last_t": lm.last_t}
                          for lm in self.lms if lm.state > 0],
            "trail": [[R(x), R(y)] for _, x, y in self.trail],
            "track_width_mm": R(self.width),
            "stats": dict(self.stats, resid_mm=self.resid_mm()),
        }

    # ---- recording ----------------------------------------------------------------------------
    def _log_frame(self, f, res):
        if self.log is None:
            return
        crop = list(f["crop"]) if f["crop"] else [""] * 4
        base = [f["seq"], f["sensor_ts"], int(f["t"] * 1e9), len(f["boxes"])] + crop
        rows = []
        if res is None:
            for i, b in enumerate(f["boxes"] or [None]):
                raw = [i, *("%.5f" % x for x in b[:4]), "%.4f" % b[4], b[5]] if b else [-1] + [""] * 6
                rows.append(base + raw + ["no_odometry"] + [""] * (len(DET_MAP_COLUMNS) - 1))
            self.log.put("det", rows)
            return
        cam, pose, corr = res["cam"], res["pose"], res["corr"]
        pitch, roll, _ = camera_angles(cam["R"])
        frame_cols = ["%.3f" % pitch, "%.3f" % roll, "%.1f" % cam["h"], "%.3f" % self.pitch_bias,
                      "%.1f" % pose[0], "%.1f" % pose[1], "%.3f" % math.degrees(pose[2]), res["n_conf"],
                      "%.1f" % corr[0], "%.1f" % corr[1], "%.4f" % corr[2]]
        k_of = {i: k for k, i in enumerate(res["idx"])}
        for i, b in enumerate(f["boxes"] or [None]):
            if b is None:
                rows.append(base + [-1] + [""] * 6 + ["none"] + [""] * 8 + frame_cols)
                continue
            raw = [i, *("%.5f" % x for x in b[:4]), "%.4f" % b[4], b[5]]
            k = k_of.get(i)
            if k is None:
                rows.append(base + raw + [res["status"][i]] + [""] * 8 + frame_cols)
                continue
            lm = res["lm"].get(k)
            rows.append(base + raw + ["used", "%.1f" % res["xy"][k][0], "%.1f" % res["xy"][k][1], "%.1f" % res["sr"][k],
                                      "%.1f" % res["st"][k], "%.1f" % res["w"][k][0], "%.1f" % res["w"][k][1],
                                      lm.id if lm else "", STATE_NAME[lm.state] if lm else ""] + frame_cols)
        self.log.put("det", rows)

    def _log_pose(self, t):
        pose, o, cam = self.pose(t), self.odo_pose(t), self.camera_at(t)
        prev = self.odo.at(t - 0.2)
        cur = self.odo.at(t)
        speed = math.hypot(cur[0] - prev[0], cur[1] - prev[1]) / 0.2 if prev and cur else 0.0
        pitch, roll, _ = camera_angles(cam["R"])
        self.log.put("pose", [[int(t * 1e9), "%.1f" % pose[0], "%.1f" % pose[1], "%.3f" % math.degrees(pose[2]),
                               "%.1f" % o[0], "%.1f" % o[1], "%.3f" % math.degrees(o[2]), "%.0f" % speed,
                               "%.3f" % cam["roll"], "%.3f" % cam["pitch"], "%.3f" % cam["d_roll"], "%.3f" % cam["d_pitch"],
                               "%.3f" % pitch, "%.3f" % roll, "%.1f" % cam["h"], "%.3f" % self.pitch_bias,
                               "%.1f" % math.sqrt(max(np.linalg.eigvalsh(self.Pp[:2, :2]))),
                               "%.3f" % math.degrees(math.sqrt(self.Pp[2, 2]))]])

    def _log_tick(self, t, pose, R2):
        tn = int(t * 1e9)
        rows = []
        for lm in self.lms:
            if lm.state == 2:
                continue
            rel = (lm.x - pose[:2]) @ R2
            rows.append([tn, lm.id, STATE_NAME[lm.state], SIDE_NAME[lm.side], "%.1f" % lm.x[0], "%.1f" % lm.x[1],
                         "%.1f" % lm.sigma, lm.hits, lm.misses, int(lm.first_t * 1e9), int(lm.last_t * 1e9),
                         "%.1f" % rel[0], "%.1f" % rel[1]])
        self.log.put("landmarks", rows)
        b = self.bounds or {}
        rd = lambda pl: [[round(x), round(y)] for x, y in pl]   # noqa: E731
        self.log.put("track", {"t_ns": tn, "pose": [round(pose[0], 1), round(pose[1], 1), round(math.degrees(pose[2]), 3)],
                               "width_mm": round(self.width), "left": rd(b.get("left", [])), "right": rd(b.get("right", [])),
                               "center": rd(b.get("center", [])), "ids_left": b.get("ids_left", []),
                               "ids_right": b.get("ids_right", [])})


def _nearest_on_polyline(q, P):
    if len(P) == 1:
        return P[0]
    a, b = P[:-1], P[1:]
    ab = b - a
    tt = np.clip(((q - a) * ab).sum(1) / np.maximum((ab * ab).sum(1), 1e-9), 0, 1)
    c = a + tt[:, None] * ab
    return c[np.argmin(((c - q) ** 2).sum(1))]


# ---------------------------------------------------------------- recording sink
class MapLog:
    """Writes the map's side of a take into <dir>/<id>_*.{csv,jsonl,json} on its own thread (the SD
    card stalls writes for seconds at times; the mapper must never wait for it).

        <id>_det.csv        every detection box (raw: DET_RAW_COLUMNS) + what the map made of it
                            (DET_MAP_COLUMNS); frames without boxes have one row with box = -1
        <id>_pose.csv       fused pose + odometry-only pose + camera attitude / height at 50 Hz
        <id>_landmarks.csv  every active apple (estimated position, sigma, state, side) at 10 Hz
        <id>_track.jsonl    left / right boundary + centre line in the car frame at 10 Hz
        <id>_map.json       the final map at stop (apples, path, parameters, statistics)
    """

    def __init__(self, directory, rid):
        self.dir, self.id = directory, rid
        p = lambda s: os.path.join(directory, rid + s)   # noqa: E731
        self.paths = {"det": p("_det.csv"), "pose": p("_pose.csv"), "landmarks": p("_landmarks.csv"),
                      "track": p("_track.jsonl"), "map": p("_map.json")}
        self.f = {k: open(self.paths[k], "w", newline="") for k in ("det", "pose", "landmarks", "track")}
        self.w = {k: csv.writer(self.f[k]) for k in ("det", "pose", "landmarks")}
        self.w["det"].writerow(DET_RAW_COLUMNS + DET_MAP_COLUMNS)
        self.w["pose"].writerow(POSE_COLUMNS)
        self.w["landmarks"].writerow(LM_COLUMNS)
        self.q = queue.Queue(maxsize=100000)
        self.dropped = 0
        self.rows = {"det": 0, "pose": 0, "landmarks": 0, "track": 0}
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def put(self, kind, payload):
        try:
            self.q.put_nowait((kind, payload))
        except queue.Full:
            self.dropped += 1

    def _run(self):
        last_flush = time.monotonic()
        while True:
            try:
                item = self.q.get(timeout=0.5)
            except queue.Empty:
                item = ()
            if item is None:
                break
            if item:
                kind, payload = item
                if kind == "track":
                    self.f["track"].write(json.dumps(payload, separators=(",", ":")) + "\n")
                    self.rows["track"] += 1
                else:
                    self.w[kind].writerows(payload)
                    self.rows[kind] += len(payload)
            if time.monotonic() - last_flush > 0.5:
                for f in self.f.values():
                    f.flush()
                last_flush = time.monotonic()

    def close(self, final_map=None):
        self.q.put(None)
        self._thread.join(10.0)
        for f in self.f.values():
            f.flush()
            os.fsync(f.fileno())
            f.close()
        files = {k: os.path.basename(v) for k, v in self.paths.items()}
        if final_map is not None:
            final_map = dict(final_map, log_rows=self.rows, log_dropped=self.dropped)
            write_json(self.paths["map"], final_map)
        else:
            files.pop("map")
        return files
