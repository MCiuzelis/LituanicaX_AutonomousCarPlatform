"""Path follower for the apple track: the track map (car frame) -> steering + speed.

numpy + scipy (Delaunay). tools/autonomy_eval.py replays recorded takes through TrackMapper +
Planner and compares what it would have commanded with what the car really did; the car runs the
same code.

    plan = Planner().update(mapper.bounds, speed_mm_s, t, apples=mapper.local_apples)  # after tick()
    plan["steer"], plan["speed_mm_s"], plan["mode"]

Path, best first:
  "gates"   Delaunay triangulation of the apples around the car; every triangle edge about one track
            width long is a gate (one apple left, one right); a beam search walks gate midpoints
            through neighbouring triangles from the gate the car is in, scoring constant width,
            smooth turns and length, and never using an apple on both sides. This is what makes
            sharp corners work: looking into a corner the outer row lies straight ahead, and a
            boundary chained "straight on" (the map's own left / right boundaries) runs into it.
  "center" / "left" / "right"   the map's chained boundaries: centre line, or one side shifted by
            half the track width (when the gate search finds nothing)
  "lost"    nothing usable: hold the last steering briefly, then straight, and ask for a stop. Steering: pure pursuit from the rear axle to the path point `lookahead` away, so
curvature = 2 y / L^2, mapped to the servo with the measured full-lock curvature. Speed: limited by
the lateral acceleration the curve needs and by how far ahead the path is known.

Car frame (TrackMapper.bounds): origin = the camera's floor point, X forward, Y left, mm.
Steering command: -1 full left .. +1 full right (hardware.Steering).
"""
import math
from dataclasses import dataclass

import numpy as np


@dataclass
class PlannerParams:
    # measured 2026-10-11 (5 takes, ~1.2 m/s): full lock = 1.45 1/m (radius 0.69 m) both ways,
    # servo + chassis respond ~67 ms after the command; nearly linear in between
    curv_full_lock_per_m: float = 1.45
    actuation_lag_s: float = 0.07
    cam_ahead_of_rear_axle_mm: float = 176.5      # config/odometry.json
    lookahead_mm: float = 650.0                   # pure-pursuit distance from the rear axle ...
    lookahead_s: float = 0.25                     # ... + this many seconds of travel
    min_path_mm: float = 500.0                    # a path shorter than this (ahead of the camera) is not followed
    # speed
    max_speed_mm_s: float = 1500.0
    min_speed_mm_s: float = 500.0
    max_lat_acc_mm_s2: float = 2500.0             # ~0.25 g: v = sqrt(a / curvature)
    stop_within_known_s: float = 0.8              # never drive faster than the known path allows in this time
    lost_hold_s: float = 0.3                      # path lost: keep the last steering this long, then stop
    # gate search
    gates: bool = True
    gate_min_mm: float = 450.0                    # a gate is an apple pair this far apart ...
    gate_max_mm: float = 1500.0                   # ... at most
    gate_area: tuple = (-900.0, 4000.0, 3000.0)   # apples used: x from, x to, |y| up to (car frame)
    beam: int = 30
    max_steps: int = 14
    max_turn_deg: float = 80.0                    # between consecutive midpoint steps
    w_width: float = 3.0                          # cost weights
    w_turn: float = 1.0
    w_long_step: float = 1.0
    reward_per_m: float = 2.5


class Planner:
    def __init__(self, params=None):
        self.p = params or PlannerParams()
        self.last_steer = 0.0
        self.lost_since = None
        self.last = None

    # ---- gate search -----------------------------------------------------------------------------
    def gate_path(self, apples, width):
        """apples (N, 2) car frame -> (midpoints (M, 2), left apples, right apples) or None."""
        p = self.p
        A = np.asarray(apples, float).reshape(-1, 2)
        x0, x1, ym = p.gate_area
        A = A[(A[:, 0] > x0) & (A[:, 0] < x1) & (np.abs(A[:, 1]) < ym)]
        if len(A) < 3:
            return None
        try:
            from scipy.spatial import Delaunay
            tri = Delaunay(A)
        except Exception:  # noqa: BLE001 - collinear / too few points: no triangulation
            return None
        gate_of, gates = {}, []
        for s in tri.simplices:
            for i, j in ((s[0], s[1]), (s[1], s[2]), (s[0], s[2])):
                e = (min(i, j), max(i, j))
                if e in gate_of:
                    continue
                d = float(np.hypot(*(A[e[0]] - A[e[1]])))
                gate_of[e] = len(gates) if p.gate_min_mm <= d <= p.gate_max_mm else -1
                if gate_of[e] >= 0:
                    gates.append((e, d, (A[e[0]] + A[e[1]]) / 2))
        if not gates:
            return None
        nbr = [set() for _ in gates]
        for s in tri.simplices:
            g = [gate_of[(min(i, j), max(i, j))] for i, j in ((s[0], s[1]), (s[1], s[2]), (s[0], s[2]))]
            g = [k for k in g if k >= 0]
            for a in g:
                nbr[a].update(b for b in g if b != a)
        wref = float(width or 800.0)
        cmax = math.cos(math.radians(p.max_turn_deg))

        def step(state, k):
            cost, path, hd, left, right, length = state
            (i, j), w, m = gates[k]
            d = m - (path[-1][1] if path else np.zeros(2))
            n = float(np.hypot(*d))
            if n < 60.0:
                return None
            u = d / n
            c = float(u @ hd)
            if c < cmax:
                return None
            # which apple of the gate is on the left of the direction of travel
            a, b = (i, j) if (u[0] * (A[i] - m)[1] - u[1] * (A[i] - m)[0]) > 0 else (j, i)
            if a in right or b in left:
                return None
            turn = math.degrees(math.acos(max(-1.0, min(1.0, c))))
            dc = (p.w_width * ((w - wref) / wref) ** 2 + p.w_turn * (turn / 45.0) ** 2
                  + p.w_long_step * max(0.0, (n - 1000.0) / 500.0) ** 2)
            return (cost + dc, path + [(k, m)], u, left | {a}, right | {b}, length + n)

        # start: gates the car is passing now / about to pass (beside or just ahead, around y = 0)
        root = (0.0, [], np.array([1.0, 0.0]), frozenset(), frozenset(), 0.0)
        beam = []
        for k, ((i, j), w, m) in enumerate(gates):
            if -700 < m[0] < 1200 and abs(m[1]) < 600 and A[i][1] * A[j][1] < 0 or \
                    (-700 < m[0] < 1200 and abs(m[1]) < 300):
                s = step(root, k)
                if s is not None:
                    beam.append(s)
        if not beam:
            return None
        best = min(beam, key=lambda s: s[0] - p.reward_per_m * s[5] / 1000.0)
        for _ in range(p.max_steps):
            nxt = []
            for s in beam:
                used = {k for k, _ in s[1]}
                for k in nbr[s[1][-1][0]]:
                    if k not in used:
                        ns = step(s, k)
                        if ns is not None:
                            nxt.append(ns)
            if not nxt:
                break
            nxt.sort(key=lambda s: s[0] - p.reward_per_m * s[5] / 1000.0)
            beam = nxt[:p.beam]
            if beam[0][0] - p.reward_per_m * beam[0][5] / 1000.0 < best[0] - p.reward_per_m * best[5] / 1000.0:
                best = beam[0]
        mids = np.array([m for _, m in best[1]])
        return mids, A[sorted(best[3])], A[sorted(best[4])]

    # ---- path ------------------------------------------------------------------------------------
    @staticmethod
    def _offset(pl, d):
        """Polyline (N, 2) shifted sideways by d (left +) along its normals."""
        P = np.asarray(pl, float)
        if len(P) < 2:
            return P
        t = np.gradient(P, axis=0)
        t /= np.maximum(np.hypot(t[:, 0], t[:, 1]), 1e-9)[:, None]
        return P + d * np.stack([-t[:, 1], t[:, 0]], 1)

    def path(self, b, apples=None):
        """Target path (N, 2) in the car frame -> (path, kind, left apples, right apples)."""
        width = float((b or {}).get("width") or 800.0)
        if self.p.gates and apples is not None:
            g = self.gate_path(apples, width)
            if g is not None and len(g[0]) and g[0][:, 0].max() >= self.p.min_path_mm:
                return g[0], "gates", g[1], g[2]
        if not b:
            return None, "none", None, None
        left, right, center = (np.asarray(b.get(k) or np.zeros((0, 2)), float) for k in ("left", "right", "center"))
        half = 0.5 * width
        reach = lambda pl: float(pl[:, 0].max()) if len(pl) else -1e9   # noqa: E731
        cands = []
        if len(center) >= 2:
            cands.append((reach(center), center, "center"))
        if len(left) >= 2:
            cands.append((reach(left), self._offset(left, -half), "left"))
        if len(right) >= 2:
            cands.append((reach(right), self._offset(right, half), "right"))
        if not cands:
            return None, "none", left, right
        # the centre line if it reaches about as far as the best single boundary, else that boundary
        best = max(c[0] for c in cands)
        for r, pl, kind in cands:
            if kind == "center" and r >= best - 400:
                return pl, kind, left, right
        r, pl, kind = max(cands, key=lambda c: c[0])
        return pl, kind, left, right

    # ---- control ---------------------------------------------------------------------------------
    def update(self, bounds, speed_mm_s, t=None, apples=None):
        """bounds: TrackMapper.bounds; apples: TrackMapper.local_apples (car frame, mm)."""
        p = self.p
        pl, kind, left, right = self.path(bounds, apples)
        ahead = float(pl[:, 0].max()) if pl is not None and len(pl) else 0.0
        out = {"mode": kind, "path": pl, "known_mm": ahead, "left": left, "right": right}
        if pl is None or ahead < p.min_path_mm:
            if self.lost_since is None:
                self.lost_since = t if t is not None else 0.0
            hold = t is not None and t - self.lost_since < p.lost_hold_s
            out.update(mode="lost", steer=self.last_steer if hold else 0.0, curvature=None, target=None,
                       speed_mm_s=p.min_speed_mm_s if hold else 0.0)
            self.last = out
            return out
        self.lost_since = None
        v = max(0.0, float(speed_mm_s))
        # where the rear axle will be when this command takes effect: straight ahead by v * lag
        # (good enough at these latencies; the arc correction is < 1 cm)
        rear = np.array([-p.cam_ahead_of_rear_axle_mm + v * p.actuation_lag_s, 0.0])
        L = p.lookahead_mm + v * p.lookahead_s
        tgt = self._lookahead_point(pl, rear, L)
        dx, dy = tgt - rear
        d2 = dx * dx + dy * dy
        k = 2.0 * dy / max(d2, 1.0) * 1000.0              # 1/m, left +
        steer = float(np.clip(-k / p.curv_full_lock_per_m, -1.0, 1.0))
        # speed: lateral acceleration in the curve, and enough known path to stop in
        kk = max(abs(k), self._path_curvature(pl) * 1000.0)
        v_curve = math.sqrt(p.max_lat_acc_mm_s2 / max(kk / 1000.0, 1e-6))
        v_known = ahead / p.stop_within_known_s
        vt = max(p.min_speed_mm_s, min(p.max_speed_mm_s, v_curve, v_known))
        self.last_steer = steer
        out.update(steer=steer, curvature=k, target=tgt.tolist(), speed_mm_s=vt, lookahead_mm=L)
        self.last = out
        return out

    @staticmethod
    def _lookahead_point(pl, origin, L):
        """First point along the polyline at distance L from origin (interpolated); the path's
        nearest-to-L point if it never gets that far, its first point if it starts beyond L."""
        d = np.hypot(pl[:, 0] - origin[0], pl[:, 1] - origin[1])
        if d[0] >= L:
            return pl[0]
        for i in range(1, len(pl)):
            if d[i] >= L:
                f = (L - d[i - 1]) / max(d[i] - d[i - 1], 1e-9)
                return pl[i - 1] + f * (pl[i] - pl[i - 1])
        return pl[-1]

    @staticmethod
    def _path_curvature(pl):
        """Max curvature (1/mm) along the path from consecutive triples, a little smoothed."""
        if len(pl) < 3:
            return 0.0
        a, b, c = pl[:-2], pl[1:-1], pl[2:]
        ab, bc, ca = (np.hypot(*(b - a).T), np.hypot(*(c - b).T), np.hypot(*(a - c).T))
        cross = np.abs((b - a)[:, 0] * (c - a)[:, 1] - (b - a)[:, 1] * (c - a)[:, 0])
        k = 2 * cross / np.maximum(ab * bc * ca, 1e-6)
        return float(np.median(k)) if len(k) < 3 else float(np.percentile(k, 75))
