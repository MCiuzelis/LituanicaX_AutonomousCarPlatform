#!/usr/bin/env python3
"""Closed-loop test of the autonomy on synthetic apple tracks: the real TrackMapper + Planner (what
vision/map_worker.py runs) drive a simulated car. Replays of manual takes (tools/autonomy_eval.py) cannot
show what the planner's own steering leads to; this can (corner entries, latency, gains, speed).

    python3 tools/track_sim.py [--track loop|hairpin|bean|all] [--laps 2] [--vmax 1500] [--plot out.png]
                               [--set planner_field=value ...] [--map map_field=value ...] [--seed 1]

Simulated (2026-10-11 measurements):
  car       kinematic, rear axle; curvature = -1.45 1/m x steering (+ 0.07 1/m right drift at 0, which the
            planner's steer_trim cancels); steering acts 67 ms after the command, then a first-order 30 ms
            servo; grip limit 4 m/s^2 (beyond: understeer); speed follows the planner's target (+2 / -3 m/s^2),
            never above --vmax (the page's throttle ceiling)
  camera    config/camera.json mount and lens, 1536x864 crop (73 deg wide), 120 fps; every apple's box from
            the projection of its floor point (+ 1.5 px noise, a common 0.3 deg pitch wobble per frame);
            misses 3 % near, up to 35 % at 4 m; boxes reach the map 25 ms after the exposure
  odometry  IMU yaw (200 Hz) + encoder (mm_per_turn 2 % off) delivered 10 ms late, in 10 ms batches
  planner   20 Hz, input age = now - odometry time (as on the car); the plan acts at once in the drive
            server
Tracks: two apple rows (0.8 m apart, an apple every 0.35 m +- 4 cm) along a closed centre line with
rounded corners (radius 0.75 m): "loop" (CCW, like the 2026-10-11 track), "hairpin" (two 180 deg turns),
"bean" (a right-hand bend in a left-hand loop).
Result per track: progress (laps), mean speed, apples touched by the car body (220 x 340 mm around the
rear axle), worst |offset| of the camera from the centre line, time near the rows (|offset| > half width -
0.15 m); the run ends when the car leaves the track (|offset| > half width + 0.15 m).
"""
import argparse
import math
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # Pi5/
sys.path.insert(0, ROOT)
from vision.camera_model import CROP_1536                             # noqa: E402
from vision.planner import Planner, cast_like, plan_with_map           # noqa: E402
from vision.track_map import TrackMapper, cam_to_floor, load_params   # noqa: E402

WIDTH = 800.0
SPACING = 350.0
APPLE_R = 37.0
CAR_HALF_W, CAR_FRONT, CAR_BACK = 110.0, 250.0, 90.0    # body around the rear axle (mm)
FULL_LOCK = 1.45                                          # 1/m
DRIFT = -0.07                                             # 1/m at steering 0 (right)
STEER_DELAY, SERVO_TAU = 0.067, 0.03
A_LAT_MAX = 4000.0
DET_LAG, ODO_LAG, BATCH = 0.025, 0.010, 0.010
CAM_HZ, IMU_HZ = 120.0, 200.0


# ---------------------------------------------------------------- tracks
def _rounded(corners, radius, step=20.0):
    """Closed polygon (CCW or not) with arcs of `radius` at the corners -> dense centre line (N, 2)."""
    P = np.asarray(corners, float)
    n = len(P)
    out = []
    for i in range(n):
        a, b, c = P[i - 1], P[i], P[(i + 1) % n]
        u1, u2 = (b - a) / np.hypot(*(b - a)), (c - b) / np.hypot(*(c - b))
        turn = math.atan2(u1[0] * u2[1] - u1[1] * u2[0], u1 @ u2)
        t = radius * math.tan(abs(turn) / 2)
        p_in, p_out = b - u1 * t, b + u2 * t
        normal = np.array([-u1[1], u1[0]]) * (1 if turn > 0 else -1)
        ctr = p_in + normal * radius
        a0 = math.atan2(*(p_in - ctr)[::-1])
        k = max(2, int(abs(turn) * radius / step))
        for f in np.linspace(0, 1, k):
            ang = a0 + f * turn
            out.append(ctr + radius * np.array([math.cos(ang), math.sin(ang)]))
        # straight to the next corner's arc start is filled by the resampling below
    out = np.array(out)
    return _resample(np.vstack([out, out[:1]]), step)[:-1]


def _resample(P, step):
    seg = np.hypot(*np.diff(P, axis=0).T)
    cum = np.r_[0, np.cumsum(seg)]
    u = np.arange(0, cum[-1], step)
    return np.stack([np.interp(u, cum, P[:, 0]), np.interp(u, cum, P[:, 1])], 1)


TRACKS = {
    # CCW loop ~7 x 5 m with corners of 70-130 deg (shape of the 2026-10-11 driveway track)
    "loop": [(0, 0), (1600, -200), (3300, 3200), (3000, 4600), (-2500, 3300), (-2800, 1200)],
    # long and thin: two 180 deg turns
    "hairpin": [(0, 0), (5000, 0), (5000, 1700), (0, 1700)],
    # left-hand loop with a right-left chicane in the back straight
    "bean": [(0, 0), (4500, 0), (4500, 3500), (3000, 3500), (2300, 2500), (1300, 2500), (700, 3500), (-1000, 3500)],
    # CCW ~8 x 3.5 m, a tight ~110 deg end (shape of the 2026-10-15 driveway track, from the takes' maps)
    "drive": [(0, 0), (6200, 0), (5200, 3000), (2000, 3200), (-1000, 1500)],
}


def make_track(name, rng, radius=750.0, jitter=40.0):
    C = _rounded(TRACKS[name], radius)
    # driving direction: as listed (CCW for all three); rows: offset curves of the centre line
    t = np.roll(C, -1, axis=0) - np.roll(C, 1, axis=0)       # closed curve: central differences wrap
    t /= np.hypot(t[:, 0], t[:, 1])[:, None]
    nrm = np.stack([-t[:, 1], t[:, 0]], 1)
    apples = []
    for side in (1, -1):
        R = C + side * WIDTH / 2 * nrm
        # inner rows of tight corners fold back on themselves: drop points closer than half the width
        keep = []
        for i, q in enumerate(R):
            d = np.hypot(*(C - q).T).min()
            if d > WIDTH / 2 - 30:
                keep.append(i)
        R = R[keep]
        A = _resample(np.vstack([R, R[:1]]), SPACING)
        A = A + rng.normal(0, jitter / math.sqrt(2), A.shape)
        apples.append(A)
    gap = np.hypot(*(apples[0][:, None] - apples[1][None]).transpose(2, 0, 1)).min()
    assert gap > 0.75 * WIDTH, "track %s: rows only %.0f mm apart" % (name, gap)
    side = np.r_[np.ones(len(apples[0])), -np.ones(len(apples[1]))]      # true side: + left row, - right row
    return C, np.vstack(apples), side


# ---------------------------------------------------------------- simulation
class Sim:
    def __init__(self, track, args, rng):
        self.C, self.apples, self.side = make_track(track, rng, args.radius, args.jitter)
        self.args, self.rng = args, rng
        prm = load_params()
        prm.imu_lag_s = 0.0                       # the simulated IMU has no lag
        for kv in args.map:
            k, v = kv.split("=", 1)
            setattr(prm, k, cast_like(getattr(prm, k), v))
        self.m = TrackMapper(params=prm)
        self.pl = Planner()
        for kv in args.set:
            k, v = kv.split("=", 1)
            setattr(self.pl.p, k, cast_like(getattr(self.pl.p, k), v))
        self.cam = self.m.cam
        self.cam_ahead = self.m.cam_ahead
        self.mmpt_true = self.m.mm_per_turn * 1.02
        # start on the centre line, heading along it
        # (--start-offset: that far left of it, --start-yaw: turned that much left; --start-at: along the line)
        i0 = int(getattr(args, "start_at", 0.0) / 20.0) % len(self.C)
        d = self.C[(i0 + 1) % len(self.C)] - self.C[i0]
        th = math.atan2(d[1], d[0])
        off = getattr(args, "start_offset", 0.0)
        cx, cy = self.C[i0][0] - off * math.sin(th), self.C[i0][1] + off * math.cos(th)
        th += math.radians(getattr(args, "start_yaw", 0.0))
        self.x, self.y, self.th = cx - self.cam_ahead * math.cos(th), cy - self.cam_ahead * math.sin(th), th
        self.v = 0.0
        self.k = 0.0                              # actual curvature 1/mm
        self.steer_q = []                         # (t_apply, steer)
        self.steer_cmd = 0.0
        self.v_target = 0.0
        self.dist = 0.0
        self.log = []                             # planner steps: (t, mode, steer, v target, known, v, yaw rate, nL, nR, why)
        self.R0 = cam_to_floor(self.cam.pitch_deg, self.cam.roll_deg)

    def cam_pose(self):
        return self.x + self.cam_ahead * math.cos(self.th), self.y + self.cam_ahead * math.sin(self.th), self.th

    def boxes(self):
        cx, cy, th = self.cam_pose()
        c, s = math.cos(th), math.sin(th)
        rel = (self.apples - [cx, cy]) @ np.array([[c, -s], [s, c]])
        d = np.hypot(rel[:, 0], rel[:, 1])
        sel = np.nonzero((rel[:, 0] > 150) & (d < 5000))[0]
        if not len(sel):
            return []
        pitch_noise = self.rng.normal(0, 0.3)
        R = cam_to_floor(self.cam.pitch_deg, self.cam.roll_deg, 0.0, pitch_noise)
        h = self.cam.height_mm
        out = []
        P = rel[sel]
        lat = np.stack([-np.sin(np.arctan2(P[:, 1], P[:, 0])), np.cos(np.arctan2(P[:, 1], P[:, 0]))], 1)
        uv = self.m.project(P, R, h, CROP_1536)
        uvl = self.m.project(P + APPLE_R * lat, R, h, CROP_1536)
        uvr = self.m.project(P - APPLE_R * lat, R, h, CROP_1536)
        for k, i in enumerate(sel):
            if np.isnan(uv[k]).any() or np.isnan(uvl[k]).any() or np.isnan(uvr[k]).any():
                continue
            p_miss = 0.03 + 0.32 * max(0.0, (d[i] - 2000) / 2000)
            if self.rng.random() < p_miss:
                continue
            x1, x2 = sorted((uvl[k][0], uvr[k][0]))
            y2 = uv[k][1] + self.rng.normal(0, 1.5 / 540)
            xc = 0.5 * (x1 + x2) + self.rng.normal(0, 1.5 / 960)
            w = x2 - x1
            y1 = y2 - w * 16 / 9
            if xc < 0 or xc > 1 or y2 < 0 or y2 > 1:
                continue
            out.append((max(0.0, xc - w / 2), max(0.0, y1), min(1.0, xc + w / 2), min(1.0, y2), 0.9, 0))
        return out

    def inner_hits(self, hits):
        """How many of the touched apples were on the inside of a bend of the centre line."""
        C = self.C
        n = len(C)
        k = 0
        for j in hits:
            q = self.apples[j]
            i = int(np.argmin(np.hypot(C[:, 0] - q[0], C[:, 1] - q[1])))
            a, b, c = C[(i - 15) % n], C[i], C[(i + 15) % n]
            turn = (b - a)[0] * (c - b)[1] - (b - a)[1] * (c - b)[0]      # + left bend
            if abs(turn) > 1e3 and np.sign(turn) == self.side[j]:
                k += 1
        return k

    def body_touch(self):
        """Indices of the apples the car body (rectangle around the rear axle) touches now."""
        c, s = math.cos(self.th), math.sin(self.th)
        rel = (self.apples - [self.x, self.y]) @ np.array([[c, -s], [s, c]])
        dx = np.maximum(0, np.maximum(rel[:, 0] - CAR_FRONT, -CAR_BACK - rel[:, 0]))
        dy = np.maximum(0, np.abs(rel[:, 1]) - CAR_HALF_W)
        return np.nonzero(np.hypot(dx, dy) < APPLE_R)[0]

    def offset(self):
        cx, cy, _ = self.cam_pose()
        d = np.hypot(self.C[:, 0] - cx, self.C[:, 1] - cy)
        i = int(np.argmin(d))
        t = self.C[(i + 1) % len(self.C)] - self.C[i - 1]
        side = np.sign(t[0] * (cy - self.C[i][1]) - t[1] * (cx - self.C[i][0]))
        return i, float(d[i] * side)

    def run(self):
        a = self.args
        dt = 0.002
        t = 0.0
        next_cam, next_imu, next_plan, next_batch = 0.0, 0.0, 0.05, 0.0
        pending = []                              # (t_deliver, kind, payload)
        n = len(self.C)
        i0, _ = self.offset()
        prog, last_i = 0, i0
        path, worst, off_t, touched, hits = [], 0.0, 0.0, None, {}
        speeds = []
        while t < a.max_s:
            # sensors
            if t >= next_imu:
                next_imu += 1 / IMU_HZ
                pending.append((t + a.odo_lag, "imu", (t, 0.0, 0.0, math.degrees(self.th))))
                pending.append((t + a.odo_lag, "turns", (t, self.dist / self.mmpt_true)))
            if t >= next_cam:
                next_cam += 1 / CAM_HZ
                pending.append((t + a.det_lag, "det", (t, self.boxes())))
            # worker: batches of what has arrived, map step, 20 Hz plan
            if t >= next_batch:
                next_batch += BATCH
                due = [e for e in pending if e[0] <= t]
                pending = [e for e in pending if e[0] > t]
                for _, kind, x in due:
                    if kind == "imu":
                        self.m.add_imu(*x)
                    elif kind == "turns":
                        self.m.add_turns(*x)
                    else:
                        self.m.add_detections(x[0], x[1], CROP_1536)
                self.m.step()
            if t >= next_plan:
                next_plan += 1 / a.plan_hz
                b = self.m.tick()
                to = self.m.odo.last_t
                if b is not None and to is not None:
                    o, prev = self.m.odo_pose(to), self.m.odo_pose(to - 0.2)
                    v = math.hypot(o[0] - prev[0], o[1] - prev[1]) / 0.2 if o and prev else 0.0
                    p = plan_with_map(self.pl, self.m, v, to, age_s=t - to)
                    self.steer_q.append((t + a.steer_delay, p.get("steer") or 0.0))
                    self.v_target = min(a.vmax, p.get("speed_mm_s") or 0.0)
                    self.log.append((t, p["mode"], p.get("steer"), p.get("speed_mm_s"), p["known_mm"], v,
                                     p.get("yaw_rate", 0.0), len(p["left_idx"]), len(p["right_idx"]), p.get("why", "")))
            while self.steer_q and self.steer_q[0][0] <= t:
                self.steer_cmd = self.steer_q.pop(0)[1]
            # car
            k_cmd = (-FULL_LOCK * self.steer_cmd + DRIFT) / 1000.0
            self.k += (k_cmd - self.k) * min(1.0, dt / a.servo_tau)
            k_eff = self.k
            if self.v > 1 and abs(k_eff) * self.v ** 2 > a.grip:
                k_eff = math.copysign(a.grip / self.v ** 2, k_eff)
            dv = self.v_target - self.v
            self.v += max(-a.dec * dt, min(a.acc * dt, dv * dt / 0.25 if abs(dv) > 1 else dv))
            self.v = max(0.0, self.v)
            ds = self.v * dt
            self.th += k_eff * ds
            self.x += ds * math.cos(self.th)
            self.y += ds * math.sin(self.th)
            self.dist += ds
            t += dt
            # scoring
            if int(t / dt) % 10 == 0:
                i, off = self.offset()
                step = (i - last_i) % n
                if step < n // 2:
                    prog += step
                last_i = i
                path.append((self.x, self.y, t, off, self.v))
                speeds.append(self.v)
                worst = max(worst, abs(off))
                if abs(off) > WIDTH / 2 - 150:
                    off_t += 10 * dt
                for j in self.body_touch():
                    hits.setdefault(int(j), (t, self.x, self.y))
                if abs(off) > WIDTH / 2 + 150:      # left the track
                    touched = (t, self.x, self.y)
                    break
                if prog >= a.laps * n:
                    break
        laps = prog / n
        return {"laps": laps, "t": t, "v_mean": float(np.mean(speeds)) if speeds else 0.0, "worst_off": worst,
                "off_s": off_t, "touched": touched, "hits": hits, "path": np.array(path)}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--track", default="all")
    ap.add_argument("--laps", type=float, default=2.0)
    ap.add_argument("--vmax", type=float, default=1500.0)
    ap.add_argument("--odo-lag", type=float, default=ODO_LAG, help="IMU + encoder delivery delay (s)")
    ap.add_argument("--det-lag", type=float, default=DET_LAG, help="exposure -> boxes at the map (s)")
    ap.add_argument("--steer-delay", type=float, default=STEER_DELAY, help="command -> servo starts moving (s)")
    ap.add_argument("--servo-tau", type=float, default=SERVO_TAU, help="servo + chassis first-order time constant (s)")
    ap.add_argument("--grip", type=float, default=A_LAT_MAX, help="lateral grip mm/s^2 (beyond: understeer)")
    ap.add_argument("--acc", type=float, default=2000.0, help="car acceleration limit mm/s^2")
    ap.add_argument("--dec", type=float, default=3000.0, help="car braking limit mm/s^2 (2026-10-16 autonomous takes: "
                    "~1200-2000 measured at brake 0.4-0.8)")
    ap.add_argument("--max-s", type=float, default=90.0)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--plan-hz", type=float, default=50.0, help="planner rate (vision/map_worker.py PLAN_HZ)")
    ap.add_argument("--radius", type=float, default=750.0, help="corner radius of the centre line (mm)")
    ap.add_argument("--jitter", type=float, default=40.0, help="apple position noise (mm)")
    ap.add_argument("--start-offset", type=float, default=0.0, help="start this far left of the centre line (mm)")
    ap.add_argument("--start-yaw", type=float, default=0.0, help="start turned this much left of the track (deg)")
    ap.add_argument("--start-at", type=float, default=0.0, help="start this far along the centre line (mm)")
    ap.add_argument("--plot")
    ap.add_argument("--set", nargs="*", default=[])
    ap.add_argument("--map", nargs="*", default=[])
    a = ap.parse_args()
    names = list(TRACKS) if a.track == "all" else a.track.split(",")
    res = {}
    for name in names:
        rng = np.random.default_rng(a.seed)
        sim = Sim(name, a, rng)
        r = sim.run()
        res[name] = (sim, r)
        inner = sim.inner_hits(r["hits"])
        print("%-8s laps %.2f in %.1f s, mean %.2f m/s, apples touched %2d (inside of a turn %d), worst offset %3.0f mm, near the rows %.1f s%s" % (
            name, r["laps"], r["t"], r["v_mean"] / 1000, len(r["hits"]), inner, r["worst_off"], r["off_s"],
            ", LEFT THE TRACK at %.1f s" % r["touched"][0] if r["touched"] else ""), flush=True)
    if a.plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axs = plt.subplots(1, len(res), figsize=(7 * len(res), 6), squeeze=False)
        for ax, (name, (sim, r)) in zip(axs[0], res.items()):
            ax.plot(sim.apples[:, 0], sim.apples[:, 1], "o", color="tab:red", ms=3)
            ax.plot(sim.C[:, 0], sim.C[:, 1], ":", color="0.6")
            P = r["path"]
            if len(P):
                sc = ax.scatter(P[:, 0], P[:, 1], c=P[:, 4], s=2, cmap="viridis", vmin=0, vmax=a.vmax)
            if r["touched"]:
                ax.plot(r["touched"][1], r["touched"][2], "kx", ms=14, mew=3)
            for j in r["hits"]:
                ax.plot(*sim.apples[j], "o", mfc="none", mec="k", ms=9, mew=2)
            ax.set_aspect("equal")
            ax.set_title("%s: %.2f laps, %.2f m/s, %d apples touched%s" % (name, r["laps"], r["v_mean"] / 1000, len(r["hits"]),
                                                                          ", LEFT THE TRACK" if r["touched"] else ""))
        plt.colorbar(sc, ax=axs[0, -1], label="mm/s")
        plt.tight_layout()
        plt.savefig(a.plot, dpi=70)


if __name__ == "__main__":
    main()
