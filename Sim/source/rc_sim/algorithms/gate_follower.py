"""Baseline: map the track objects near the car, track the corridor centre line, pure-pursuit it.

1. Odometry: speed from the motor encoder, yaw rate from the IMU gyro -> dead-reckoned pose in a
   local 'odom' frame (no absolute position is ever needed).
2. Landmarks: every box -> object position (GroundProjector, size-based range) at the pose the frame
   was CAPTURED at (boxes arrive ~25 ms late), fused into a short-lived landmark memory with
   weights that fall with range (range noise grows with distance). The memory keeps objects that
   have left the 47 deg field of view in a tight turn.
3. Centre line: landmarks clearly left or right of the PREVIOUS centre line (not of the car's
   heading, which fails in hairpins) are labelled first; the labels then spread along each side to
   neighbours within the object spacing, so the object at the apex of a hairpin - which sits dead
   ahead of the old line - still joins the outer side. Centre points are the
   midpoints of left/right "gates" (pairs about a corridor width apart); where a stretch has only
   one side, that side's objects offset half a corridor inwards stand in. The new line is GROWN
   from the car through those points in short steps with a turn-rate limit, so a hairpin bends
   the line instead of folding it. Starts as a straight line ahead.
4. Pure pursuit on the centre line; speed limited by the line's curvature ahead (lateral
   acceleration budget) and by how far ahead it is known; PI on encoder speed -> throttle.

Parameters: DEFAULTS below, each overridable from run.py with --param name=value.
"""
from __future__ import annotations

import math
from collections import deque

import numpy as np

from ..interface import CarInfo, Command, SensorFrame
from .base import Algorithm
from .geometry import GroundProjector

DEFAULTS = dict(
    lookahead_m=0.35,        # pure-pursuit lookahead at standstill
    lookahead_per_mps=0.12,  # + this much per m/s
    half_width_m=0.50,       # corridor half width (object line to centre)
    side_min_m=0.20,         # landmarks this close to / this far from the old centre line
    side_max_m=0.90,         #   are not labelled directly (only by propagation)
    seed_range_m=1.2,        # direct labels only this far along the old line
    link_m=0.80,             # labels propagate to neighbours this close
    gate_min_m=0.70,         # left/right pairs this far apart form a gate
    gate_max_m=1.40,
    grow_step_m=0.15,        # centre-line growth step
    grow_radius_m=0.40,      # centre points within this of the next step pull the line
    grow_max_m=3.0,
    min_turn_radius_m=0.45,  # the grown line never bends tighter than this (tightest track radius)
    v_max=2.0,               # m/s
    v_min=0.6,
    a_lat_max=2.0,           # m/s^2 budget for the speed limit from curvature
    a_brake=2.0,             # m/s^2 assumed when planning to slow for a bend ahead
    latency_s=0.15,          # command -> wheels + detection delay, travelled before braking starts
    known_for_vmax_m=3.0,    # full speed only when this much centre line is known
    kp=0.10, ki=0.15,        # throttle per (m/s) error, per (m/s * s)
    ff_per_mps=0.075,        # feed-forward duty per m/s (1 / no-load speed at full duty)
    ff_offset=0.06,          # duty that overcomes rolling resistance
    brake_gain=0.5,          # brake per m/s over the target
    memory_s=3.0,            # forget landmarks not re-seen for this long
    merge_m=0.15,            # merge radius at 1 m, grows with range
    range_noise=0.05,        # 1 sigma range error as a fraction of range (size-based ranging)
    max_range_m=2.5,         # ignore boxes further than this (beyond it they are mostly other legs
                             #   of the track seen across the infield)
    width_check=0.25,        # reject boxes whose width and height ranges differ by more than this
    min_hits=4,              # a landmark must be seen this often before it is used (FP ghosts
                             #   live ~3 frames)
    object_radius_m=0.05,    # size and shape of the track objects (Track.blend: 10 cm spheres,
    object_height_m=0.0875,  #   8.75 cm tall)
    object_shape="ellipsoid",
)


def _polyline_frame(path: np.ndarray):
    """Cumulative arc length, unit tangents and left normals of a polyline [K, 2]."""
    d = np.diff(path, axis=0)
    seg = np.maximum(np.linalg.norm(d, axis=1), 1e-9)
    t = d / seg[:, None]
    return np.r_[0.0, np.cumsum(seg)], t, np.stack([-t[:, 1], t[:, 0]], 1)


def _resample(path: np.ndarray, step: float) -> np.ndarray:
    """Points every `step` metres along a polyline (keeps the first point)."""
    s0 = _polyline_frame(path)[0]
    if s0[-1] < step:
        return path
    q = np.arange(0.0, s0[-1] + 1e-9, step)
    return np.stack([np.interp(q, s0, path[:, 0]), np.interp(q, s0, path[:, 1])], 1)


def _project(path: np.ndarray, pts: np.ndarray):
    """(s, signed lateral + = left, segment index) of points onto a polyline, the last segment
    extended straight beyond its end."""
    s0, t, n = _polyline_frame(path)
    a = path[:-1]
    seg_len = np.diff(s0)
    along = np.clip(((pts[:, None, :] - a[None]) * t[None]).sum(-1), 0.0, None)          # [P, K]
    along = np.where(np.arange(len(a))[None] < len(a) - 1, np.minimum(along, seg_len[None]), along)
    foot = a[None] + along[..., None] * t[None]
    k = np.argmin(np.linalg.norm(pts[:, None, :] - foot, axis=-1), axis=1)
    i = np.arange(len(pts))
    lat = ((pts - foot[i, k]) * n[k]).sum(-1)
    return s0[k] + along[i, k], lat, k


class GateFollower(Algorithm):
    name = "gate_follower"

    def __init__(self, **params):
        p = dict(DEFAULTS)
        p.update({k: type(DEFAULTS[k])(v) if k in DEFAULTS else v for k, v in params.items()})
        super().__init__(**p)
        self.p = p

    def reset(self, car: CarInfo) -> None:
        super().reset(car)
        self.proj = GroundProjector(car.camera, self.p["object_radius_m"], self.p["object_height_m"], self.p["object_shape"])
        self.pose = np.zeros(3)                     # odom x, y, yaw
        self.poses = deque(maxlen=200)              # (t, x, y, yaw) for capture-time lookup
        self.t = None
        self.v = 0.0
        self.gyro_z = 0.0
        self.lm = np.zeros((0, 2))                  # landmarks, odom frame
        self.lm_w = np.zeros(0)                     # accumulated weight (1 / variance)
        self.lm_t = np.zeros(0)                     # last seen
        self.lm_hits = np.zeros(0, dtype=int)
        self.center = np.array([[0.0, 0.0], [3.0, 0.0]])   # odom frame, starts straight ahead
        self.integral = 0.0
        self._dbg = {}

    # ---- odometry -----------------------------------------------------------------------------
    def _odometry(self, f: SensorFrame) -> None:
        if f.encoder is not None:
            self.v = f.encoder.rps * self.car.wheel_m_per_motor_turn
        if self.t is None:
            self.t = f.t
            self.poses.append((f.t, *self.pose))
            return
        t_prev = self.t
        for s in f.imu:                             # integrate over the IMU samples
            if s.t > t_prev:
                self._integrate(s.t - t_prev, s.gyro[2])
                t_prev = s.t
                self.poses.append((s.t, *self.pose))
            self.gyro_z = s.gyro[2]
        if f.t > t_prev:
            self._integrate(f.t - t_prev, self.gyro_z)
        self.t = f.t
        self.poses.append((f.t, *self.pose))

    def _integrate(self, dt: float, wz: float) -> None:
        x, y, th = self.pose
        th_mid = th + 0.5 * wz * dt
        self.pose = np.array([x + self.v * dt * math.cos(th_mid), y + self.v * dt * math.sin(th_mid), th + wz * dt])

    def _pose_at(self, t: float) -> np.ndarray:
        return np.array(min(self.poses, key=lambda p: abs(p[0] - t))[1:])

    @staticmethod
    def _to_odom(pose, pts):
        c, s = math.cos(pose[2]), math.sin(pose[2])
        return np.stack([pose[0] + c * pts[:, 0] - s * pts[:, 1], pose[1] + s * pts[:, 0] + c * pts[:, 1]], 1)

    def _to_car(self, pts):
        x, y, th = self.pose
        c, s = math.cos(th), math.sin(th)
        d = np.asarray(pts, dtype=np.float64).reshape(-1, 2) - np.array([x, y])
        return np.stack([c * d[:, 0] + s * d[:, 1], -s * d[:, 0] + c * d[:, 1]], 1)

    # ---- landmarks ----------------------------------------------------------------------------
    def _update_landmarks(self, f: SensorFrame) -> None:
        p = self.p
        for df in f.detections:
            meas = [self.proj.object_position(d, p["max_range_m"], p["width_check"]) for d in df.detections]
            meas = np.array([q for q in meas if q is not None])
            if not len(meas):
                continue
            pose = self._pose_at(df.t_capture)
            rng = np.hypot(meas[:, 0] - self.car.camera.x_m, meas[:, 1])
            w = 1.0 / np.maximum(p["range_noise"] * rng, 0.02) ** 2
            for q, wq, r in zip(self._to_odom(pose, meas), w, rng):
                if len(self.lm):
                    d = np.linalg.norm(self.lm - q, axis=1)
                    i = int(np.argmin(d))
                    if d[i] < p["merge_m"] * max(1.0, r):
                        self.lm[i] = (self.lm_w[i] * self.lm[i] + wq * q) / (self.lm_w[i] + wq)
                        self.lm_w[i] = min(self.lm_w[i] + wq, 50.0 * wq)     # stay responsive
                        self.lm_t[i], self.lm_hits[i] = df.t_capture, self.lm_hits[i] + 1
                        continue
                self.lm = np.vstack([self.lm, q])
                self.lm_w, self.lm_t = np.append(self.lm_w, wq), np.append(self.lm_t, df.t_capture)
                self.lm_hits = np.append(self.lm_hits, 1)
        keep = (f.t - self.lm_t) < p["memory_s"]
        if len(self.lm):
            keep &= self._to_car(self.lm)[:, 0] > -0.8
        self.lm, self.lm_w, self.lm_t, self.lm_hits = self.lm[keep], self.lm_w[keep], self.lm_t[keep], self.lm_hits[keep]

    # ---- centre line --------------------------------------------------------------------------
    def _update_center(self):
        p = self.p
        car_xy = self.pose[:2]
        # the old line, cut to what lies ahead of the car (its end extends straight)
        ahead = self.center[self._to_car(self.center)[:, 0] > 0.1]
        heading = np.array([math.cos(self.pose[2]), math.sin(self.pose[2])])
        old = np.vstack([car_xy, ahead]) if len(ahead) else np.vstack([car_xy, car_xy + heading])
        pts = self.lm[self.lm_hits >= p["min_hits"]]
        left = right = cands = np.zeros((0, 2))
        if len(pts):
            s, lat, seg = _project(old, pts)
            label = np.where((np.abs(lat) > p["side_min_m"]) & (np.abs(lat) < p["side_max_m"])
                             & (s > -0.3) & (s < p["seed_range_m"]), np.sign(lat), 0.0)
            dist = np.linalg.norm(pts[:, None] - pts[None], axis=-1)
            np.fill_diagonal(dist, np.inf)
            for _ in range(len(pts)):                                   # spread labels along each side
                unl = np.nonzero(label == 0)[0]
                lab = np.nonzero(label != 0)[0]
                if not len(unl) or not len(lab):
                    break
                d = dist[np.ix_(unl, lab)]
                k = np.argmin(d, axis=1)
                take = d[np.arange(len(unl)), k] < p["link_m"]
                if not take.any():
                    break
                label[unl[take]] = label[lab[k[take]]]
            left, right = pts[label > 0], pts[label < 0]
            mids = np.zeros((0, 2))
            if len(left) and len(right):
                d = np.linalg.norm(left[:, None] - right[None], axis=-1)
                li, ri = np.nonzero((d > p["gate_min_m"]) & (d < p["gate_max_m"]))
                mids = 0.5 * (left[li] + right[ri])
            # one-sided stand-ins: an unpaired object, half a corridor towards the nearest object of
            # the other side, or along the old line's normal when the other side is not seen at all
            nrm = _polyline_frame(old)[2]
            offs = []
            for q, lb, sg in zip(pts, label, seg):
                if lb == 0 or (len(mids) and np.min(np.linalg.norm(mids - q, axis=1)) < 0.6):
                    continue
                other = right if lb > 0 else left
                if len(other):
                    o = other[np.argmin(np.linalg.norm(other - q, axis=1))]
                    if np.linalg.norm(o - q) < p["gate_max_m"] + 0.5:
                        offs.append(q + p["half_width_m"] * (o - q) / np.linalg.norm(o - q))
                        continue
                offs.append(q - lb * p["half_width_m"] * nrm[sg])
            cands = np.vstack([mids.reshape(-1, 2), np.array(offs).reshape(-1, 2)])
        self.center = self._grow(car_xy, old, cands) if len(cands) else old
        return left, right

    def _grow(self, start, old, cands):
        """Grow the centre line from the car through the candidate centre points."""
        p = self.p
        h = float(self.pose[2])                    # grow from the car, along its heading
        max_turn = p["grow_step_m"] / p["min_turn_radius_m"]
        path, q, misses = [start], start.copy(), 0
        for _ in range(int(p["grow_max_m"] / p["grow_step_m"])):
            look = q + 2.5 * p["grow_step_m"] * np.array([math.cos(h), math.sin(h)])
            d = np.linalg.norm(cands - look, axis=1)
            near = d < p["grow_radius_m"]
            fwd = ((cands - q) @ np.array([math.cos(h), math.sin(h)])) > 0.02
            near &= fwd
            if near.any():
                w = 1.0 / (d[near] + 0.05)
                tgt = (cands[near] * w[:, None]).sum(0) / w.sum()
                dh = math.atan2(tgt[1] - q[1], tgt[0] - q[0]) - h
                dh = (dh + math.pi) % (2 * math.pi) - math.pi
                h += max(-max_turn, min(max_turn, dh))
                misses = 0
            else:
                misses += 1
                if misses > 2:
                    break
            q = q + p["grow_step_m"] * np.array([math.cos(h), math.sin(h)])
            path.append(q.copy())
        path = np.array(path)
        return path if _polyline_frame(path)[0][-1] >= 0.45 else old

    # ---- control ------------------------------------------------------------------------------
    def step(self, f: SensorFrame) -> Command:
        p, car = self.p, self.car
        self._odometry(f)
        self._update_landmarks(f)
        left, right = self._update_center()
        path = _resample(self._to_car(self.center), 0.05)
        s_path = _polyline_frame(path)[0]
        known = s_path[-1]

        v = self.v
        ld = p["lookahead_m"] + p["lookahead_per_mps"] * max(v, 0.0)
        r = np.hypot(path[:, 0], path[:, 1])
        beyond = np.nonzero((r >= ld) & (path[:, 0] > 0.05))[0]         # first point ahead at >= ld
        target = path[beyond[0]] if len(beyond) else path[-1]
        dist = max(float(np.hypot(*target)), 0.2)
        kappa = 2.0 * math.sin(math.atan2(target[1], target[0])) / dist
        steer = max(-1.0, min(1.0, -math.atan(kappa * car.wheelbase_m) / car.max_steer_rad))   # + = right

        # speed profile: corner speed sqrt(a_lat / curvature) at every point of the line ahead,
        # reachable from now with a_brake over the distance to it; slower when little line is known
        kmax = abs(kappa)
        v_target = min(p["v_max"], math.sqrt(p["a_lat_max"] / max(kmax, 1e-3)))
        coarse = path[::3]                                                 # 0.15 m spacing
        if len(coarse) >= 3:
            a, b, c = coarse[:-2], coarse[1:-1], coarse[2:]
            ab, bc, ca = (np.linalg.norm(b - a, axis=1), np.linalg.norm(c - b, axis=1), np.linalg.norm(a - c, axis=1))
            cross = np.abs((b - a)[:, 0] * (c - a)[:, 1] - (b - a)[:, 1] * (c - a)[:, 0])
            curv = 2.0 * cross / np.maximum(ab * bc * ca, 1e-9)                # 1 / circumradius
            dist = s_path[::3][1:-1]
            v_corner = np.sqrt(p["a_lat_max"] / np.maximum(curv, 1e-3))
            brake_dist = np.maximum(dist - p["latency_s"] * max(v, 0.0), 0.0)
            v_target = min(v_target, float(np.min(np.sqrt(v_corner ** 2 + 2 * p["a_brake"] * brake_dist))))
            kmax = max(kmax, float(curv[dist < 1.5].max()) if (dist < 1.5).any() else 0.0)
        v_target = max(p["v_min"], min(v_target, p["v_min"] + (p["v_max"] - p["v_min"]) * known / p["known_for_vmax_m"]))

        err = v_target - v
        self.integral = max(-2.0, min(2.0, self.integral + err / car.control_hz))
        if err < -0.3:
            throttle = -min(1.0, p["brake_gain"] * (-err))
            self.integral *= 0.9
        else:
            throttle = p["ff_offset"] + p["ff_per_mps"] * v_target + p["kp"] * err + p["ki"] * self.integral
        throttle = max(-1.0, min(car.max_throttle, throttle))

        car_pts = lambda a: [tuple(q) for q in self._to_car(a)] if len(a) else []
        self._dbg = {"landmarks": car_pts(self.lm), "left": car_pts(left), "right": car_pts(right),
                     "path": [tuple(q) for q in path], "target": tuple(target),
                     "text": f"v {v:4.2f}/{v_target:4.2f} m/s  k {kmax:4.2f}  lm {len(self.lm)}  L{len(left)} R{len(right)}"}
        return Command(throttle, steer)

    def debug(self) -> dict:
        return self._dbg
