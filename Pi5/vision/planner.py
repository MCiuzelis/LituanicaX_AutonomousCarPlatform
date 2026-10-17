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
            smooth turns and length, never using an apple on both sides, and respecting each apple's
            side memory (TrackMapper votes: earlier plans + passed apples). Sharp corners work because
            the outer row seen straight ahead pairs with the apex apple, not with the inner row.
  "row"     no gates (one row in view, e.g. deep in a sharp corner): the visible row is chained from
            the apple nearest the car; its side comes from the votes, else from geometry (the car is
            right of a left boundary: apples on the left that bend towards the centre are a left
            boundary turning right). Path = that row shifted half a track width towards the track.
  "lost"    nothing usable: hold the last steering briefly, speed 0 (the drive server brakes).
Steering: pure pursuit from the rear axle to the path point `lookahead` away (curvature = 2 y / L^2),
times steer_gain, mapped to the servo with the measured full-lock curvature, plus a centre trim.
Speed: limited by the lateral acceleration the curve needs, by how far ahead the path is known and by
a confidence (row path, few gates, path jumping between updates).

plan_with_map(planner, mapper, v, t) is what the car runs (map_worker) and the replay tools use: it
passes the map's side memory in and votes the plan's left / right apples back.

Car frame (TrackMapper.bounds): origin = the camera's floor point, X forward, Y left, mm.
Steering command: -1 full left .. +1 full right (hardware.Steering).
"""
import json
import math
import os
from dataclasses import dataclass, fields

import numpy as np

try:                                   # imported here, not at the first gate path: ~0.5 s on the Pi, the moment
    from scipy.optimize import lsq_linear   # the car would start
except ImportError:                    # no scipy: racing_line() falls back to the gate midpoints
    lsq_linear = None

# live-tunable from the dashboard ("autonomy tuning", saved in config/planner.json): PlannerParams field ->
# (min, max, step, label, help). 2026-10-16 speed defaults are less timid: the one-row / memory / few-gate
# confidence and the steering-jump penalty no longer halve the speed, corners 2.0 m/s^2 (the driver: 2-2.6)
TUNABLES = {
    "steer_gain": (0.3, 2.5, 0.05, "steer gain", "servo command = gain x commanded curvature / full lock (1 = as measured)"),
    "ff_dist_mm": (300.0, 1500.0, 50.0, "steer calmness mm",
                   "sideways / heading errors are corrected over this much travel: larger = calmer, later; smaller = tighter"),
    "ff_time_s": (0.0, 1.2, 0.05, "steer calmness s",
                  "at speed: errors are corrected over at least this many seconds of travel (more = calmer; too little = "
                  "the car weaves at 2-3 Hz above ~1.5 m/s)"),
    "ff_preview_s": (0.0, 0.3, 0.01, "steer preview s", "the path's curvature this far ahead is steered now: more = turns in earlier"),
    "racing_margin_mm": (150.0, 400.0, 10.0, "apex margin mm",
                         "the racing line keeps this far from the gate apples (car half width 110 + apple 37 + safety)"),
    "max_speed_mm_s": (500.0, 4000.0, 50.0, "top speed mm/s", "never faster (also capped by the page's throttle level)"),
    "max_lat_acc_mm_s2": (800.0, 5000.0, 100.0, "corner grip mm/s\u00b2",
                          "lateral acceleration allowed in bends: higher = faster corners (the driver: 2800-3700)"),
    "brake_mm_s2": (500.0, 4000.0, 100.0, "brake for bends mm/s\u00b2", "planned deceleration before a bend: lower = slows down earlier"),
    "known_end_frac": (0.0, 1.5, 0.05, "blind corner speed",
                       "x full-lock corner speed the car must be able to slow to by the end of the path it can see "
                       "(0 = able to stop there: cautious; 1 = a full-lock corner right after it)"),
    "min_speed_mm_s": (200.0, 1200.0, 50.0, "min speed mm/s", "never slower while a path is known"),
    "conf_one_side": (0.2, 1.0, 0.05, "one-row speed", "top speed x this when only one row is followed"),
    "conf_memory": (0.2, 1.0, 0.05, "memory speed", "top speed x this when driving on the remembered path"),
}

CONFIG_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config", "planner.json")


@dataclass
class PlannerParams:
    # measured 2026-10-11 (5 takes, ~1.2 m/s): full lock = 1.45 1/m (radius 0.69 m) both ways,
    # servo + chassis respond ~67 ms after the command; nearly linear in between
    curv_full_lock_per_m: float = 1.45
    steer_gain: float = 1.0                       # servo command = gain * commanded curvature / full lock; > 1 steers harder.
                                                  # 1.0 with control "ff" (it asks 0.98-1.04 of the driven corner curvature);
                                                  # pure pursuit on the midline needed 1.5-1.6 (it asked ~0.6-0.9, late).
                                                  # The dashboard slider overrides (config/planner.json)
    steer_trim: float = -0.05                     # added to every command: at 0 the car curved right (-0.07 1/m,
                                                  # autonomous takes 2026-10-11 13:31-13:36)
    actuation_lag_s: float = 0.08                 # from sending the plan to the chassis turning: servo + chassis ~67 ms +
                                                  # drive server (50 Hz) ~10 ms; the input's age is added per plan
    cam_ahead_of_rear_axle_mm: float = 176.5      # config/odometry.json
    control: str = "ff"                           # "ff": path curvature feed-forward + error feedback; "pp": pure pursuit
    ff_dist_mm: float = 600.0                     # ff: errors decay (critically damped) over this much travel (track_sim:
                                                  # 350 / 450 / 600 / 800 mm -> 37 / 29 / 22 / 18 apples touched in 24 laps, 1 / 0 / 0 / 1 off)
    ff_time_s: float = 0.5                        # ff: ... and over at least this many seconds of travel (= ff_dist_mm at
                                                  # 1.2 m/s, 900 mm at 1.8 m/s)
    ff_preview_s: float = 0.05                    # ff: the path curvature this far ahead of the predicted car (servo)
    lookahead_mm: float = 650.0                   # pure-pursuit distance from the rear axle ...
    lookahead_s: float = 0.25                     # ... + this many seconds of travel
    min_path_mm: float = 500.0                    # a path reaching less far than this from the camera is not followed
    smooth_sigma_mm: float = 250.0                # the gate-midpoint polyline is smoothed along its length (Gaussian)
    smooth_step_mm: float = 40.0                  # ... after resampling at this spacing; 0 = no smoothing
    # racing line (gates): instead of the gate midpoints, the smoothest line through the gates that keeps
    # racing_margin_mm from every gate apple (car half width 110 + apple 37 + ~70 safety), starting at the
    # car's position and heading. A 0.75 m corner (midline: ~92 % of full lock, the car ran wide into the
    # outer row in tools/track_sim.py) becomes a wide-entry / inside-apex line of about twice the radius
    racing: bool = True
    racing_margin_mm: float = 220.0
    path_from_car: bool = True                    # every path starts along the car's heading (see _from_car)
    racing_tail_mm: float = 250.0                 # the heading anchor: a fixed point this far behind the camera
    racing_reg: float = 0.02                      # pull towards the midline (relative to the curvature term)
    # speed (the drive server turns it into throttle, never above the page's manual throttle level,
    # and brakes when the car is faster than this)
    max_speed_mm_s: float = 2500.0                # (was 1500: the cap on 41-75 % of the 2026-10-16 15:2x plans; the page's
                                                  # throttle level is the real limit)
    min_speed_mm_s: float = 600.0                 # crawl while any path is known (stopping would freeze the view)
    max_lat_acc_mm_s2: float = 2800.0             # full lock (1.45 1/m) -> 1.39 m/s, 0.7 1/m -> 2.0 m/s. The driver's tight
                                                  # corners 2026-10-11: 2.8-3.7 m/s^2 at 1.4-1.6 m/s; autonomy at 2000 + the old
                                                  # known-path rule: 1.1-1.6 m/s^2 at 0.86-1.08 m/s
    brake_mm_s2: float = 1500.0                   # planned deceleration towards the curve speed of a bend ahead
    # beyond the known path the track may turn at full lock: the car must be able to brake to that corner's speed
    # (x known_end_frac; 0 = to a stop) by the end of the known path. Before (2026-10-16 15:2x): "never faster than
    # the known path / 1 s" - the path end stays put while the car approaches it, so the target fell to 0.6-0.8 m/s
    # before every corner (0.6-0.8 m known) and the car braked although the curve allowed 1.2-3.5 m/s
    known_end_frac: float = 1.0
    lost_hold_s: float = 0.3                      # path lost: keep the last steering this long; speed 0 at once ...
    # ... unless there was a path a moment ago: "crawl" on at min_speed_mm_s with the last steering for at most
    # lost_crawl_mm / lost_crawl_s. A stopped car sees the same nothing forever (2026-10-15: stopped deep in
    # corners with "gate path short" and only went on when the map had forgotten enough apples, 2-3 s later)
    lost_crawl_mm: float = 1000.0
    lost_crawl_s: float = 2.0
    # path continuation: a path that ends where the remembered one (memory_s) went on is continued along it, so a
    # corner seen on the way in is still followed when its apples have left the view (73 deg): the car turned
    # only when the outer row was all it saw
    continue_path: bool = True
    continue_join_mm: float = 250.0               # the new path's end must be this close to the remembered path
    row_side_hold_s: float = 0.8                  # "row": keep the side of the previous row path this long unless the
                                                  # side memory is firm (150022: sides flipped 6x in 1 s, off the track)
    # confidence (0..1) scales the top speed: one-sided path, few gates, a path that jumps between updates
    conf_one_side: float = 0.8                    # "row" path (one row in view)
    conf_memory: float = 0.8                      # "memory": the last gate path (nothing usable in view)
    memory_s: float = 1.5                         # ... at most this old
    memory_prefer_s: float = 0.6                  # a gate path this fresh beats a one-row guess
    # weak gate paths (<= weak_max_apples apples on a side) that disagree with the fresh remembered path by more than
    # weak_gap_mm (within weak_cmp_mm along the path) are followed only once they have held for weak_confirm_s
    weak_hold: bool = True
    weak_max_apples: int = 1
    weak_gap_mm: float = 250.0
    weak_cmp_mm: float = 1200.0
    weak_confirm_s: float = 0.15
    # boundary memory (BoundaryMemory, 2026-10-15): every apple a gate path put on a side (and every apple the car
    # passed) is remembered at its map position. A new / re-created apple within bmem_match_mm takes that side
    # (a row seen with both sides a second ago keeps its side when it is all that is in view), and remembered
    # boundary apples the map no longer has (out of view, dropped, duplicated) are planned with again as
    # "virtual" apples: the corner seen from the straight is still there deep in the corner
    boundary_memory: bool = True
    bmem_keep_s: float = 8.0                      # remembered this long after it was last confirmed ...
    bmem_behind_mm: float = 1500.0                # ... and until this far behind the camera
    bmem_merge_mm: float = 200.0                  # observations this close update one remembered apple
    bmem_match_mm: float = 250.0                  # a map apple this close to a remembered one takes its side
    bmem_virtual: bool = True                     # plan with remembered apples the map no longer has
    conf_few_gates: float = 0.85                   # fewer than 4 gate midpoints
    conf_jump_steer: float = 1.0                  # steering change per 50 ms between plans that halves the confidence
    # gate search
    gates: bool = True
    gate_min_mm: float = 450.0                    # a gate is an apple pair this far apart ...
    gate_max_mm: float = 1500.0                   # ... at most
    gate_area: tuple = (-900.0, 4000.0, 3000.0)   # apples used: x from, x to, |y| up to (car frame)
    beam: int = 30
    max_steps: int = 14
    car_side_tol_mm: float = 100.0                # the car may be this far on the wrong side of a first-gate row
    gate_cross_min_deg: float = 35.0              # the path crosses every gate at least this steeply (not along it)
    max_turn_deg: float = 80.0                    # between consecutive midpoint steps
    w_width: float = 3.0                          # cost weights
    w_turn: float = 1.0
    w_long_step: float = 1.0
    reward_per_m: float = 2.5
    # side memory (TrackMapper.local_votes: + left / - right)
    vote_lock: float = 4.0                        # an apple this sure of its side is never used on the other
    w_vote: float = 1.5                           # cost per gate apple against a weaker side memory
    # rows: apples closer than this are one row (2026-10-11 manual laps: along a row p50 ~350 mm, p90 ~545 mm;
    # nearest apple across the track p5 ~530-580 mm, p50 ~780 mm). Two apples of one row never make a gate
    # (two apples apart they look like a perfect 0.8 m gate: the path then crossed the outer row in corners),
    # and a row's side memory (mostly apples the car has passed) reaches the new apples ahead of it.
    row_link_frac: float = 0.85                   # x track width ...
    row_link_max_mm: float = 680.0                # ... at most (a chain link: a missed apple is bridged)
    row_near_mm: float = 600.0                    # one-row fallback: rows whose nearest apple is further than this ...
    row_far_penalty: float = 2.0                  # ... lose this much score per mm
    row_dup_mm: float = 180.0                     # apples this close are one apple mapped twice
    row_bend_deg: float = 60.0                    # a row bends at most this much at an apple
    row_majority: float = 3.0                     # a row with memory both ways takes the side holding this many times more
    same_row_mixed: bool = True                   # no gate inside a row even when its memory is mixed (rows are chains)
    row_side_min: float = 6.0                    # a row with this much net side memory has that side (a passed apple: 6)
    # one-row fallback
    row_min_apples: int = 2

    @classmethod
    def load(cls, path=CONFIG_PATH):
        """Defaults overridden by config/planner.json (any field above, e.g. {\"steer_gain\": 1.3})."""
        try:
            with open(path) as f:
                over = json.load(f)
        except (OSError, ValueError):
            over = {}
        names = {fl.name for fl in fields(cls)}
        return cls(**{k: v for k, v in over.items() if k in names})


def save_config(over, path=CONFIG_PATH):
    """Merge {field: value} into config/planner.json (the dashboard's steer gain slider)."""
    try:
        with open(path) as f:
            cfg = json.load(f)
    except (OSError, ValueError):
        cfg = {}
    cfg.update(over)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(cfg, f, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


class Planner:
    def __init__(self, params=None):
        self.p = params or PlannerParams.load()
        self.last_steer = 0.0
        self.lost_since = None
        self.last = None
        self.memory = None                        # (t, last gate path in map coordinates)
        self.last_t = None                        # time of the previous plan (rate-independent memory / confidence)
        self.why = ""
        self.now = None                           # time of the plan being made
        self.row_side = None                      # (side, t) of the last row path
        self.last_path_t = None                   # last time a path was followed (crawl)
        self.crawl_mm = 0.0
        self.crawl_t = None
        self.weak_cand = None                     # (t first seen, map path) of a weak gate path not yet followed

    # ---- gate search -----------------------------------------------------------------------------
    # ---- rows ------------------------------------------------------------------------------------
    def rows(self, apples, width, votes=None):
        """Apples -> (row id per apple, side memory per apple with its row's side added, mixed per apple).
        Rows: chains of apples (see below). A row whose net memory
        reaches row_side_min lends that side (vote_lock strength) to all its apples; a row with strong
        memory both ways (two rows bridged by a stray apple) lends nothing and does not block gates."""
        p = self.p
        A = np.asarray(apples, float).reshape(-1, 2)
        V = np.zeros(len(A)) if votes is None else np.asarray(votes, float).reshape(-1)
        n = len(A)
        comp = np.arange(n)
        self._chains, self._chain_of = [], {}
        if n < 2:
            return comp, V.copy(), np.zeros(n, bool)
        link = min(p.row_link_max_mm, p.row_link_frac * float(width or 800.0))

        def root(a):
            while comp[a] != a:
                comp[a] = comp[comp[a]]
                a = comp[a]
            return a

        def union(a, b):
            ra, rb = root(a), root(b)
            if ra != rb:
                comp[max(ra, rb)] = min(ra, rb)
        D = np.sqrt(((A[:, None, :] - A[None, :, :]) ** 2).sum(-1))
        # 1. near-duplicates (one apple mapped twice) are one node
        for a, b in zip(*np.nonzero(np.triu(D < p.row_dup_mm, 1))):
            union(a, b)
        node = np.array([root(a) for a in range(n)])
        nodes = np.unique(node)
        C = np.array([A[node == k].mean(0) for k in nodes])
        m = len(nodes)
        # 2. rows are chains: shortest links first, every node at most two links, the second one continuing
        # the first (bend <= row_bend_deg), no loops. Distance alone (union-find) joined the inner row's end to
        # the outer row where a corner wraps round (0.34-0.43 m in the map, 2026-10-15 150022): one "mixed"
        # row, no same-row protection, a gate inside the outer row and the car aimed at it
        comp = np.arange(m)
        if m >= 2:
            DC = np.sqrt(((C[:, None, :] - C[None, :, :]) ** 2).sum(-1))
            ii, jj = np.nonzero(np.triu(DC <= link, 1))
            order = np.argsort(DC[ii, jj])
            nb = [[] for _ in range(m)]
            cmin = math.cos(math.radians(180.0 - p.row_bend_deg))
            for e in order:
                a, b = int(ii[e]), int(jj[e])
                if len(nb[a]) >= 2 or len(nb[b]) >= 2 or root(a) == root(b):
                    continue
                ok = True
                for x, y in ((a, b), (b, a)):
                    if nb[x]:
                        u, w = C[nb[x][0]] - C[x], C[y] - C[x]
                        if float(u @ w) / (float(np.hypot(*u)) * float(np.hypot(*w)) + 1e-9) > cmin:
                            ok = False                     # the new link folds back on the existing one
                if ok:
                    nb[a].append(b)
                    nb[b].append(a)
                    union(a, b)
        comp_node = np.array([root(a) for a in range(m)])
        comp = comp_node[np.searchsorted(nodes, node)]
        # the rows as ordered chains (node centroids, apple indices per node) for row_path()
        members = [np.nonzero(node == k)[0] for k in nodes]
        chains = []
        if m >= 2:
            seen = np.zeros(m, bool)
            for st in [k for k in range(m) if len(nb[k]) <= 1] + list(range(m)):
                if seen[st]:
                    continue
                order, prv, cur = [], -1, st
                while cur >= 0 and not seen[cur]:
                    seen[cur] = True
                    order.append(cur)
                    nxt = [x for x in nb[cur] if x != prv and not seen[x]]
                    prv, cur = cur, (nxt[0] if nxt else -1)
                if len(order) >= 2:
                    chains.append((C[order], [members[k] for k in order]))
        self._chains = chains
        self._chain_of = {int(i): ci for ci, (_, groups) in enumerate(chains) for g in groups for i in g}
        Ve, mixed = V.copy(), np.zeros(n, bool)
        for c in np.unique(comp):
            idx = np.nonzero(comp == c)[0]
            if len(idx) < 2:
                continue
            pos, neg = float(V[idx][V[idx] > 0].sum()), float(-V[idx][V[idx] < 0].sum())
            if pos >= p.row_side_min and neg >= p.row_side_min and max(pos, neg) < p.row_majority * min(pos, neg):
                mixed[idx] = True
            elif pos - neg >= p.row_side_min:
                # the row's side, also over apples a stray gate voted the other way (2026-10-15 145744: one wrong
                # far apple made the right row "mixed" and a gate inside it was planned)
                Ve[idx] = np.maximum(V[idx], 0.0) + p.vote_lock
            elif neg - pos >= p.row_side_min:
                Ve[idx] = np.minimum(V[idx], 0.0) - p.vote_lock
        return comp, Ve, mixed

    def gate_path(self, apples, width, votes=None, rows=None):
        """apples (N, 2) car frame, votes (N,) side memory, rows: self.rows() of them -> (midpoints (M, 2),
        left indices, right indices; indices into `apples`) or None."""
        p = self.p
        A0 = np.asarray(apples, float).reshape(-1, 2)
        V0 = np.zeros(len(A0)) if votes is None else np.asarray(votes, float).reshape(-1)
        if rows is None:
            rows = self.rows(A0, width, V0)
        C0, V0, M0 = rows
        x0, x1, ym = p.gate_area
        keep = np.nonzero((A0[:, 0] > x0) & (A0[:, 0] < x1) & (np.abs(A0[:, 1]) < ym))[0]
        A, V, C, M = A0[keep], V0[keep], C0[keep], M0[keep]
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
                same_row = C[e[0]] == C[e[1]] and (p.same_row_mixed or not M[e[0]])
                gate_of[e] = len(gates) if p.gate_min_mm <= d <= p.gate_max_mm and not same_row else -1
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
        cross_max = math.cos(math.radians(p.gate_cross_min_deg))

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
            # the path must go through the gate, not along it (2026-10-16: started turned 45 deg towards the left
            # row, the first "gate" joined a near left apple to a far right apple lying straight ahead: crossed
            # lengthwise it made the left row "right" and the car drove into it)
            g = A[j] - A[i]
            if abs(float(u @ g)) / max(float(np.hypot(*g)), 1e-9) > cross_max:
                return None
            # which apple of the gate is on the left of the direction of travel
            a, b = (i, j) if (u[0] * (A[i] - m)[1] - u[1] * (A[i] - m)[0]) > 0 else (j, i)
            if a in right or b in left:
                return None
            if V[a] <= -p.vote_lock or V[b] >= p.vote_lock:      # memory says the other side
                return None
            turn = math.degrees(math.acos(max(-1.0, min(1.0, c))))
            dc = (p.w_width * ((w - wref) / wref) ** 2 + p.w_turn * (turn / 45.0) ** 2
                  + p.w_long_step * max(0.0, (n - 1000.0) / 500.0) ** 2
                  + p.w_vote * (max(0.0, -V[a]) + max(0.0, V[b])) / p.vote_lock)
            return (cost + dc, path + [(k, m, a, b)], u, left | {a}, right | {b}, length + n)

        # start: gates the car is passing now / about to pass (beside or just ahead, around y = 0). Both
        # apples may lie on one side of the car's axis (deep in a hairpin the whole track is to one side);
        # gates inside one row are already excluded
        root = (0.0, [], np.array([1.0, 0.0]), frozenset(), frozenset(), 0.0)
        beam = []
        for k, ((i, j), w, m) in enumerate(gates):
            if -700 < m[0] < 1200 and abs(m[1]) < 600:
                s = step(root, k)
                # the car is on the track: it is left of the first gate's right row and right of its left row,
                # looking the way the path goes (2026-10-16 sim: started turned 30-45 deg towards the left row, the
                # gap between the left row and the next straight's row looked like the track and the left row
                # became "right")
                if s is not None and self._car_inside(keep[s[1][0][2]], keep[s[1][0][3]], s[2]):
                    beam.append(s)
        if not beam:
            return None
        best = min(beam, key=lambda s: s[0] - p.reward_per_m * s[5] / 1000.0)
        for _ in range(p.max_steps):
            nxt = []
            for s in beam:
                used = {e[0] for e in s[1]}
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
        mids = np.array([e[1] for e in best[1]])
        pairs = keep[np.array([[e[2], e[3]] for e in best[1]], int)]       # (left, right) apple of each gate
        return mids, keep[sorted(best[3])], keep[sorted(best[4])], pairs

    def _car_inside(self, left, right, u):
        """Indices (into the apples rows() saw) of the first gate's left / right apple, u: the way the path goes
        through it -> is the car on the track side of both apples' rows (near the car)?"""
        chain_of = getattr(self, "_chain_of", None)
        if chain_of is None:
            return True
        for idx, want in ((left, -1.0), (right, 1.0)):          # want: car left (+1) / right (-1) of the row
            c = chain_of.get(int(idx), -1)
            if c < 0:
                continue
            P = self._chains[c][0]
            i = int(np.argmin(np.hypot(P[:, 0], P[:, 1])))
            t = P[min(i + 1, len(P) - 1)] - P[max(i - 1, 0)]
            nt = float(np.hypot(*t))
            if nt < 1e-6:
                continue
            t = t / nt
            if float(t @ u) < 0:
                t = -t
            cross = float(t[0] * (0.0 - P[i][1]) - t[1] * (0.0 - P[i][0]))   # + car left of the row
            if want * cross < -self.p.car_side_tol_mm:
                return False
        return True

    def racing_line(self, apples, pairs):
        """Gate pairs (M, 2) of (left, right) apple indices along the path -> one point per gate (M', 2):
        each gate's point slides along its gate (keeping racing_margin_mm from both apples) so that the
        polyline car -> points has the least squared curvature (bounded linear least squares). Gates at /
        behind the camera are left out (the car is already there)."""
        p = self.p
        A = np.asarray(apples, float).reshape(-1, 2)
        Lp, Rp = A[pairs[:, 0]], A[pairs[:, 1]]
        M = (Lp + Rp) / 2
        k0 = _first_ahead(M)
        Lp, Rp, M = Lp[k0:], Rp[k0:], M[k0:]
        n = len(M)
        if n < 2:
            return M
        d = Rp - Lp
        w = np.hypot(d[:, 0], d[:, 1])
        U = d / np.maximum(w, 1e-9)[:, None]                    # across the gate, left -> right
        lim = np.maximum(0.0, w / 2 - p.racing_margin_mm)
        X0 = np.vstack([[[-p.racing_tail_mm, 0.0], [0.0, 0.0]], M])
        N = n + 2
        seg = np.hypot(*np.diff(X0, axis=0).T)
        h = np.maximum(0.5 * (seg[:-1] + seg[1:]), 50.0)        # spacing around each inner point
        rows = N - 2
        D = np.zeros((rows, N))
        r = np.arange(rows)
        # second difference scaled to curvature for uneven spacing: 2 / (h1 h2 (h1 + h2)) weights
        h1, h2 = seg[:-1], seg[1:]
        D[r, r] = 2.0 / (h1 * (h1 + h2))
        D[r, r + 1] = -2.0 / (h1 * h2)
        D[r, r + 2] = 2.0 / (h2 * (h1 + h2))
        D *= np.sqrt(h)[:, None] * 1000.0                       # integrate k^2 along the length (1/m units)
        b = -(D @ X0)                                           # (rows, 2)
        Am = np.zeros((2 * rows, n))
        for k in range(n):
            Am[:rows, k] = D[:, k + 2] * U[k, 0]
            Am[rows:, k] = D[:, k + 2] * U[k, 1]
        bv = np.r_[b[:, 0], b[:, 1]]
        reg = math.sqrt(p.racing_reg) * float(np.linalg.norm(Am) / max(1.0, math.sqrt(n))) / max(1.0, float(np.max(lim)))
        Am = np.vstack([Am, reg * np.eye(n)])
        bv = np.r_[bv, np.zeros(n)]
        if lsq_linear is None:
            return M
        try:
            s = lsq_linear(Am, bv, bounds=(-lim - 1e-6, lim + 1e-6), method="bvls").x
        except Exception:  # noqa: BLE001 - degenerate: midpoints
            return M
        return M + s[:, None] * U

    # ---- path ------------------------------------------------------------------------------------
    def _from_car(self, pl):
        """The path starts at the car, along its heading: a point racing_tail_mm behind the camera, the camera,
        then the path's points ahead of the camera. Before (2026-10-16), the path went straight from the camera
        to its first point, often 20-30 deg to the side: the heading-error term alone asked for full lock and
        the car turned in early and hit the apex apples (143705: k_ff ~0, psi -20..-33 deg, steer -0.7..-1)."""
        P = np.asarray(pl, float).reshape(-1, 2)
        if not len(P):
            return P
        P = P[min(_first_ahead(P), len(P) - 1):]
        if not self.p.path_from_car:                  # the old way, for comparisons
            return np.vstack([[[0.0, 0.0]], P])
        return np.vstack([[[-self.p.racing_tail_mm, 0.0], [0.0, 0.0]], P])

    @staticmethod
    def _offset(pl, d):
        """Polyline (N, 2) shifted sideways by d (left +) along its normals."""
        P = np.asarray(pl, float)
        if len(P) < 2:
            return P
        t = np.gradient(P, axis=0)
        t /= np.maximum(np.hypot(t[:, 0], t[:, 1]), 1e-9)[:, None]
        return P + d * np.stack([-t[:, 1], t[:, 0]], 1)

    def row_path(self, apples, width, votes=None, rows=None):
        """Fallback when the gates give no usable path (one row in view, or the other row ends beside the car):
        every row (the chains of rows()) is a candidate. Its side: the rows' side memory if firm, else geometry -
        the car is on the track, so a row the car is right of (looking along the row the way the car goes) is
        a left boundary (2026-10-15: "first apple close on one side, the next ones further and further across"
        is the boundary on that side bending across the view). Path = the row shifted half a track width
        towards the track, from the row's apple nearest the car on. The row reaching furthest wins (a firmly
        known side counts 1 m extra; the side of the previous row path 0.5 m while row_side_hold_s).
        -> (path, left indices, right indices) or None."""
        p = self.p
        A = np.asarray(apples, float).reshape(-1, 2)
        V = np.zeros(len(A)) if votes is None else np.asarray(votes, float).reshape(-1)
        if rows is not None:
            V = rows[1]                               # with the rows' sides
        x0, x1, ym = p.gate_area
        w2 = 0.5 * float(width or 800.0)

        def side_of(Pr, i):                           # +1: car right of the row (left boundary)
            d = Pr[min(i + 1, len(Pr) - 1)] - Pr[max(i - 1, 0)]
            return 1.0 if (d[0] * (0.0 - Pr[i][1]) - d[1] * (0.0 - Pr[i][0])) < 0 else -1.0
        rs = self.row_side
        hold = rs is not None and self.now is not None and self.now - rs[1] < p.row_side_hold_s
        best = None
        for P, groups in getattr(self, "_chains", []):
            flat = np.concatenate(groups)
            if len(flat) < p.row_min_apples:
                continue
            if not np.any((A[flat, 0] > -400.0) & (A[flat, 0] < x1) & (np.abs(A[flat, 1]) < ym)):
                continue
            vs = float(np.sum(V[flat]))
            firm = abs(vs) >= p.row_side_min
            i0 = int(np.argmin(np.hypot(P[:, 0], P[:, 1])))
            for Pr, i in ((P, i0), (P[::-1], len(P) - 1 - i0)):
                tang = Pr[min(i + 1, len(Pr) - 1)] - Pr[max(i - 1, 0)]
                fwd = float(tang[0]) / max(float(np.hypot(*tang)), 1e-9)
                if fwd < -0.2:                        # this way along the row the car would drive backwards
                    continue
                sd = side_of(Pr, i)
                if firm and sd * vs < 0:              # the memory says this row is the other side: wrong way round
                    continue
                path = self._offset(Pr, -sd * w2)[i:]
                if len(path) < 2:
                    continue
                # a boundary to follow is near the car: a far row (2026-10-16 sim: 2 apples 2.2 m away) reached
                # further but was the next straight's
                dnear = float(np.hypot(*P[i0]))
                score = (_reach(path) + (1000.0 if firm else 0.0) + (500.0 if hold and sd == rs[0] else 0.0) + 200.0 * fwd
                         - p.row_far_penalty * max(0.0, dnear - p.row_near_mm))
                if best is None or score > best[0]:
                    best = (score, path, flat, sd)
        if best is None:
            return None
        _, path, flat, sd = best
        if self.now is not None:
            self.row_side = (sd, self.now)
        return path, (flat if sd > 0 else flat[:0]), (flat if sd < 0 else flat[:0])

    def path(self, b, apples=None, votes=None):
        """Target path (N, 2) in the car frame -> (path, kind, left indices, right indices) (indices
        into `apples`, ordered along the path)."""
        width = float((b or {}).get("width") or 800.0)
        self.why = ""
        if apples is None or not len(apples):
            self.why = "no confirmed apples"
            return None, "none", None, None
        A = np.asarray(apples, float).reshape(-1, 2)
        rows = self.rows(A, width, votes)
        if self.p.gates:
            g = self.gate_path(A, width, votes, rows)
            if g is not None and len(g[0]) and _reach(g[0]) >= self.p.min_path_mm:
                pl = g[0]
                if self.p.racing:
                    rl = self.racing_line(A, g[3])
                    if len(rl) >= 1:
                        pl = rl
                pl = pl[min(_first_ahead(pl), len(pl) - 1):]
                return pl, "gates", _order_along(A, g[1], g[0]), _order_along(A, g[2], g[0])
            self.why = "no gate path" if g is None else "gate path short"
        r = self.row_path(A, width, votes, rows)
        if r is not None:
            return r[0], "row", r[1], r[2]
        self.why += ", no row (%d apples)" % len(A)
        return None, "none", None, None

    # ---- control ---------------------------------------------------------------------------------
    def update(self, bounds, speed_mm_s, t=None, apples=None, votes=None, yaw_rate=0.0, age_s=0.0, pose=None):
        """bounds: TrackMapper.bounds (track width); apples / votes: TrackMapper.local_apples / local_votes;
        yaw_rate rad/s (left +) and age_s (how old the map / odometry is now) for the latency prediction;
        pose: the car's map pose (x, y, heading rad) for the path memory.
        out["left"] / ["right"]: the apples used on each side (car frame, ordered along the path),
        out["left_idx"] / ["right_idx"]: their rows in `apples`."""
        p = self.p
        self.now = t
        pl, kind, li, ri = self.path(bounds, apples, votes)
        if pl is not None:
            pl = self._from_car(pl)
            pl = smooth_path(pl, p.smooth_sigma_mm, p.smooth_step_mm)
        ahead = _reach(pl) if pl is not None and len(pl) else 0.0
        extended = False
        if (p.continue_path and pl is not None and len(pl) >= 2 and pose is not None and t is not None
                and self.memory is not None and t - self.memory[0] < p.memory_s):
            # the remembered path (car frame now) goes on beyond where this one ends: continue along it
            c0, s0 = math.cos(pose[2]), math.sin(pose[2])
            M = (self.memory[1] - [pose[0], pose[1]]) @ np.array([[c0, -s0], [s0, c0]])
            d = np.hypot(M[:, 0] - pl[-1][0], M[:, 1] - pl[-1][1])
            j = int(np.argmin(d))
            if d[j] < p.continue_join_mm and j < len(M) - 2 and _reach(M[j + 1:]) > ahead + 100.0:
                pl = smooth_path(np.vstack([pl, M[j + 1:]]), p.smooth_sigma_mm, p.smooth_step_mm)
                ahead = _reach(pl)
                extended = True
        weak = kind == "gates" and min(len(li if li is not None else ()), len(ri if ri is not None else ())) <= p.weak_max_apples
        if (p.weak_hold and weak and pl is not None and len(pl) >= 2 and pose is not None and t is not None
                and self.memory is not None and t - self.memory[0] < p.memory_prefer_s):
            # a gate path resting on one apple a side flipped between two candidate gates (2026-10-17: target
            # +-300-400 mm sideways every 2-4 plans, full lock each way, 16x the flips of 2+ gate paths): if it
            # disagrees with the fresh remembered path it must stay the same for weak_confirm_s, the remembered
            # path is followed meanwhile (a corner really revealed passes after that; a flip-flop never does)
            c0, s0 = math.cos(pose[2]), math.sin(pose[2])
            to_car, to_map = np.array([[c0, -s0], [s0, c0]]), np.array([[c0, s0], [-s0, c0]])
            M = (self.memory[1] - [pose[0], pose[1]]) @ to_car
            if _path_gap(pl, M, p.weak_cmp_mm) > p.weak_gap_mm:
                wc = self.weak_cand
                if wc is not None and _path_gap(pl, (wc[1] - [pose[0], pose[1]]) @ to_car, p.weak_cmp_mm) <= p.weak_gap_mm:
                    confirmed = t - wc[0] >= p.weak_confirm_s
                else:
                    self.weak_cand = (t, pl @ to_map + [pose[0], pose[1]])
                    confirmed = False
                if not confirmed:
                    M = M[int(np.argmin(np.hypot(M[:, 0], M[:, 1]))):]
                    if len(M) >= 2 and _reach(M) >= p.min_path_mm:
                        pl = smooth_path(self._from_car(M), p.smooth_sigma_mm, p.smooth_step_mm)
                        kind, ahead = "memory", _reach(pl)
                        self.why = "one-gate path disagrees, held"
            else:
                self.weak_cand = None
        elif kind == "gates":
            self.weak_cand = None
        if pose is not None and t is not None:
            if pl is not None and ahead >= p.min_path_mm and kind == "gates":
                c0, s0 = math.cos(pose[2]), math.sin(pose[2])
                # continued along the memory: it keeps the memory's age (what lies beyond the view expires
                # memory_s after it was last seen), else the path is new
                t_mem = self.memory[0] if extended else t
                self.memory = (t_mem, pl @ np.array([[c0, s0], [-s0, c0]]) + [pose[0], pose[1]])
            elif self.memory is not None and (
                    ((pl is None or ahead < p.min_path_mm) and t - self.memory[0] < p.memory_s)
                    or (kind == "row" and t - self.memory[0] < p.memory_prefer_s)):
                # nothing usable in view (deep in a hairpin, a blurred frame): the last gate path, from where
                # the car is now on; stopping here would freeze the view
                c0, s0 = math.cos(pose[2]), math.sin(pose[2])
                M = (self.memory[1] - [pose[0], pose[1]]) @ np.array([[c0, -s0], [s0, c0]])
                d = np.hypot(M[:, 0], M[:, 1])
                M = M[int(np.argmin(d)):]
                if len(M) >= 2 and _reach(M) >= p.min_path_mm:
                    M = smooth_path(self._from_car(M), p.smooth_sigma_mm, p.smooth_step_mm)
                    pl, kind, ahead = M, "memory", _reach(M)
                    self.why = ""
        A = np.asarray(apples if apples is not None else np.zeros((0, 2)), float).reshape(-1, 2)
        li = np.zeros(0, int) if li is None else np.asarray(li, int)
        ri = np.zeros(0, int) if ri is None else np.asarray(ri, int)
        out = {"mode": kind, "path": pl, "known_mm": ahead, "left": A[li], "right": A[ri],
               "left_idx": li, "right_idx": ri}
        if pl is None or ahead < p.min_path_mm:
            why = getattr(self, "why", "") or "path reaches only %.0f mm" % ahead
            if self.lost_since is None:
                self.lost_since = t if t is not None else 0.0
                self.crawl_mm, self.crawl_t = 0.0, t
            elif t is not None and self.crawl_t is not None and t > self.crawl_t:
                self.crawl_mm += max(0.0, float(speed_mm_s)) * (t - self.crawl_t)
                self.crawl_t = t
            crawl = (t is not None and self.last_path_t is not None
                     and self.lost_since - self.last_path_t < p.lost_crawl_s
                     and t - self.lost_since < p.lost_crawl_s and self.crawl_mm < p.lost_crawl_mm)
            if crawl:
                # keep going slowly the way the last path went: the view changes, the path comes back
                out.update(mode="crawl", steer=self.last_steer, curvature=None, target=None,
                           speed_mm_s=p.min_speed_mm_s, confidence=0.3, why=why + ", crawling")
                self.last = out
                return out
            hold = t is not None and t - self.lost_since < p.lost_hold_s
            out.update(mode="lost", steer=self.last_steer if hold else 0.0, curvature=None, target=None,
                       speed_mm_s=0.0, confidence=0.0, why=why)
            self.last = out
            return out
        self.lost_since = None
        self.last_path_t = t
        v = max(0.0, float(speed_mm_s))
        # where the rear axle will be, and where it will point, when this command takes effect: along the
        # arc the car is driving now (speed, yaw rate) for the input's age + the actuation lag. The heading
        # matters: at 135 deg/s the car turns ~18 deg in 0.14 s, ~0.3 m sideways on a 1 m lookahead
        dt = p.actuation_lag_s + max(0.0, float(age_s))
        phi = float(yaw_rate) * dt
        s_arc = v * dt
        if abs(phi) > 1e-4:
            r_arc = s_arc / phi
            adv = np.array([r_arc * math.sin(phi), r_arc * (1.0 - math.cos(phi))])
        else:
            adv = np.array([s_arc, 0.0])
        rear = np.array([-p.cam_ahead_of_rear_axle_mm, 0.0]) + adv
        c, s = math.cos(phi), math.sin(phi)
        L = p.lookahead_mm + v * p.lookahead_s
        tgt = self._lookahead_point(pl, rear, L)
        fb = None
        if p.control == "ff" and len(pl) >= 2:
            # path in the predicted rear-axle frame; nearest point; feed-forward curvature a little further
            # on (servo response) + feedback on the sideways and heading error (critically damped over
            # ff_dist_mm of travel: k = k_path - e / D^2 - 2 psi / D, speed independent)
            Q = (pl - rear) @ np.array([[c, -s], [s, c]])
            kp = _signed_curvature(Q)
            seg = np.hypot(*np.diff(Q, axis=0).T)
            cum = np.r_[0.0, np.cumsum(seg)]
            i = int(np.argmin(np.hypot(Q[:, 0], Q[:, 1])))
            j = min(len(Q) - 2, i)
            T = Q[j + 1] - Q[j]
            T = T / max(float(np.hypot(*T)), 1e-9)
            Nl = np.array([-T[1], T[0]])
            e = float(-(Q[i] @ Nl))                                   # car left of the path +
            psi = -math.atan2(T[1], T[0])                             # car heading - path heading
            s_ff = cum[i] + v * p.ff_preview_s
            kff = float(np.interp(s_ff, cum, kp))
            # at speed the errors are corrected over a fixed time, not distance: the yaw-rate prediction is a damping term
            # of strength ~ v x latency / D on a yaw rate that is ~75 ms old, and with D fixed it made the car weave at
            # 2-2.5 Hz from ~1.5 m/s (2026-10-17: steering +-0.6..1.0, yaw +-100 deg/s, the swing outwards at some
            # corner entries; track_sim with the real delays: weave 0.22 -> ~0.1)
            D = max(p.ff_dist_mm, v * p.ff_time_s)
            k = (kff - e / (D * D) - 2.0 * psi / D) * 1000.0         # 1/m, left +
            fb = {"e_mm": round(e), "psi_deg": round(math.degrees(psi), 1), "k_ff": round(kff * 1000.0, 2)}
        else:                                                         # pure pursuit
            rx, ry = tgt - rear
            dx, dy = c * rx + s * ry, -s * rx + c * ry                # in the predicted car frame
            k = 2.0 * dy / max(dx * dx + dy * dy, 1.0) * 1000.0      # 1/m, left +
        steer = float(np.clip(-p.steer_gain * k / p.curv_full_lock_per_m + p.steer_trim, -1.0, 1.0))
        # speed: now within the lateral acceleration of the commanded curvature, and able to brake down to
        # every point's curve speed before reaching it (after the latency); able to take a full-lock corner
        # right after the known path; confidence. Curvatures count up to full lock only: the error feedback may
        # ask for more than the car can steer, and going slower does not make it turn tighter
        k_cap = p.curv_full_lock_per_m / 1000.0
        v_lock = math.sqrt(p.max_lat_acc_mm_s2 / k_cap)
        v_curve = math.sqrt(p.max_lat_acc_mm_s2 / min(max(abs(k) / 1000.0, 1e-6), k_cap))
        kp_abs = np.abs(_signed_curvature(pl))
        if len(kp_abs):
            sp = np.r_[0.0, np.cumsum(np.hypot(*np.diff(pl, axis=0).T))]
            v_at = np.sqrt(p.max_lat_acc_mm_s2 / np.clip(kp_abs, 1e-6, k_cap))
            reach = np.maximum(0.0, sp - v * dt)
            v_curve = min(v_curve, float(np.min(np.sqrt(v_at ** 2 + 2.0 * p.brake_mm_s2 * reach))))
        v_known = math.sqrt((p.known_end_frac * v_lock) ** 2 + 2.0 * p.brake_mm_s2 * max(0.0, ahead - v * dt))
        conf = 1.0
        if kind == "row":
            conf *= p.conf_one_side
        if kind == "memory":
            conf *= p.conf_memory
        if kind == "gates" and len(pl) < 4:
            conf *= p.conf_few_gates
        # steering jump between plans, per 50 ms (the plan rate may be 20 or 50 Hz)
        dt_plan = (t - self.last_t) if (t is not None and self.last_t is not None and t > self.last_t) else 0.05
        conf *= 0.5 ** (abs(steer - self.last_steer) * 0.05 / max(dt_plan, 0.01) / p.conf_jump_steer)
        v_conf = p.max_speed_mm_s * conf
        vt = max(p.min_speed_mm_s, min(p.max_speed_mm_s, v_curve, v_known, v_conf))
        self.last_steer = steer
        self.last_t = t
        out.update(steer=steer, curvature=k, target=tgt.tolist(), speed_mm_s=vt, lookahead_mm=L, tracking=fb,
                   confidence=round(conf, 2), limits={"curve": round(v_curve), "known": round(v_known),
                                                      "conf": round(v_conf)})
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


def _first_ahead(P, x_min=150.0):
    """Index of the first point of a path that is ahead of the camera (the leading gates beside / behind
    it are already reached; later points may come back, e.g. around a hairpin); len(P) if none."""
    idx = np.nonzero(np.asarray(P)[:, 0] > x_min)[0]
    return int(idx[0]) if len(idx) else len(P)


def _signed_curvature(P, span=3):
    """Signed curvature (1/mm, left +) at every point of a densely resampled polyline, from the points
    `span` before / after (~120 mm at the 40 mm resampling)."""
    n = len(P)
    if n < 3:
        return np.zeros(n)
    i = np.arange(n)
    a, b, c = P[np.maximum(i - span, 0)], P, P[np.minimum(i + span, n - 1)]
    ab, bc, ca = np.hypot(*(b - a).T), np.hypot(*(c - b).T), np.hypot(*(a - c).T)
    cross = (b - a)[:, 0] * (c - a)[:, 1] - (b - a)[:, 1] * (c - a)[:, 0]
    k = 2 * cross / np.maximum(ab * bc * ca, 1e-6)
    k[(ab < 1e-6) | (bc < 1e-6)] = 0.0
    if n > 2 * span:
        k[:span], k[-span:] = k[span], k[-span - 1]
    return k


def _order_along(A, idx, path):
    """Indices idx of points A, ordered by where along the path polyline they lie."""
    idx = np.asarray(idx, int)
    if len(idx) < 2 or len(path) < 2:
        return idx
    seg = np.diff(path, axis=0)
    cum = np.r_[0.0, np.cumsum(np.hypot(seg[:, 0], seg[:, 1]))]
    pos = []
    for i in idx:
        q = A[i]
        tt = np.clip(((q - path[:-1]) * seg).sum(1) / np.maximum((seg * seg).sum(1), 1e-9), 0, 1)
        c = path[:-1] + tt[:, None] * seg
        k = int(np.argmin(((c - q) ** 2).sum(1)))
        pos.append(cum[k] + tt[k] * np.hypot(*seg[k]))
    return idx[np.argsort(pos)]


YAW_RATE_WINDOW_S = 0.06


class BoundaryMemory:
    """Apples with a known side in map (world) coordinates: P (N, 2), S (N,) side strength in vote units
    (+ left / - right, |S| <= CAP), T (N,) last confirmed. Independent of the map's landmark ids, which change
    when an apple is dropped and found again."""

    CAP = 8.0

    def __init__(self):
        self.P, self.S, self.T = np.zeros((0, 2)), np.zeros(0), np.zeros(0)

    def __len__(self):
        return len(self.S)

    def add(self, W, sides, t, weight, merge_mm, set_to=None):
        """World points W (n, 2) seen on `sides` (+1 / -1): strengthen the remembered apple within merge_mm (its
        position follows the map's), else remember a new one. set_to: strength to set (a passed apple)."""
        for q, sd in zip(np.asarray(W, float).reshape(-1, 2), sides):
            if len(self.S):
                d = np.hypot(self.P[:, 0] - q[0], self.P[:, 1] - q[1])
                j = int(np.argmin(d))
                if d[j] < merge_mm:
                    self.P[j] = 0.7 * self.P[j] + 0.3 * q
                    if set_to is not None:
                        self.S[j] = sd * set_to
                    else:
                        self.S[j] = max(-self.CAP, min(self.CAP, self.S[j] + sd * weight))
                    self.T[j] = t
                    continue
            self.P = np.vstack([self.P, q])
            self.S = np.r_[self.S, sd * (set_to if set_to is not None else weight)]
            self.T = np.r_[self.T, t]

    def prune(self, t, pose, keep_s, behind_mm):
        if not len(self.S):
            return
        c, s = math.cos(pose[2]), math.sin(pose[2])
        x = (self.P[:, 0] - pose[0]) * c + (self.P[:, 1] - pose[1]) * s
        k = (t - self.T < keep_s) & (x > -behind_mm) & (self.S != 0)
        self.P, self.S, self.T = self.P[k], self.S[k], self.T[k]

    def lookup(self, W, radius):
        """-> (side strength of the remembered apple nearest each point within radius (0: none), mask of the
        remembered apples within radius of some point)."""
        W = np.asarray(W, float).reshape(-1, 2)
        out, used = np.zeros(len(W)), np.zeros(len(self.S), bool)
        if not len(self.S) or not len(W):
            return out, used
        D = np.hypot(W[:, None, 0] - self.P[None, :, 0], W[:, None, 1] - self.P[None, :, 1])
        j = D.argmin(1)
        ok = D[np.arange(len(W)), j] < radius
        out[ok] = self.S[j[ok]]
        used |= (D < radius).any(0)
        return out, used


def plan_with_map(planner, mapper, speed_mm_s, t, age_s=0.0):
    """One planner step on a TrackMapper after its tick() (t = its odometry time): side memory in, the
    plan's sides voted back. age_s: how much older t is than now (live: the map worker's input age)."""
    p = planner.p
    b = mapper.bounds
    o, prev = mapper.odo_pose(t), mapper.odo_pose(t - YAW_RATE_WINDOW_S)
    yaw_rate = (o[2] - prev[2]) / YAW_RATE_WINDOW_S if o and prev else 0.0
    pose = mapper.pose(t)
    A = np.asarray(mapper.local_apples, float).reshape(-1, 2)
    V = np.asarray(mapper.local_votes, float).reshape(-1)
    ids = mapper.local_ids
    n_real = len(A)
    mem = getattr(planner, "bmem", None)
    if mem is None:
        mem = planner.bmem = BoundaryMemory()
    R = None
    n_virtual = 0
    if p.boundary_memory and pose is not None:
        mem.prune(t, pose, p.bmem_keep_s, p.bmem_behind_mm)
        c, s = math.cos(pose[2]), math.sin(pose[2])
        R = np.array([[c, -s], [s, c]])                          # car frame -> map
        W = A @ R.T + [pose[0], pose[1]]
        ms, matched = mem.lookup(W, p.bmem_match_mm)
        V = V + ms
        if p.bmem_virtual and len(mem):
            sel = ~matched & (np.abs(mem.S) >= p.vote_lock)
            if sel.any():
                Q = (mem.P[sel] - [pose[0], pose[1]]) @ R              # map -> car frame
                x0, x1, ym = p.gate_area
                ok = (Q[:, 0] > x0) & (Q[:, 0] < x1) & (np.abs(Q[:, 1]) < ym)
                if ok.any():
                    A = np.vstack([A, Q[ok]])
                    V = np.r_[V, mem.S[sel][ok]]
                    n_virtual = int(ok.sum())
    out = planner.update(b, speed_mm_s, t, apples=A, votes=V, yaw_rate=yaw_rate, age_s=age_s, pose=pose)
    out["yaw_rate"] = yaw_rate
    out["n_virtual"] = n_virtual
    vl = [i for i in out["left_idx"] if i >= n_real]
    vr = [i for i in out["right_idx"] if i >= n_real]
    # left / right as the page draws them include the remembered (virtual) apples; the indices are the map's
    out["left_idx"] = np.asarray([i for i in out["left_idx"] if i < n_real], int)
    out["right_idx"] = np.asarray([i for i in out["right_idx"] if i < n_real], int)
    out["virtual_used"] = len(vl) + len(vr)
    # only a gate path (both sides seen) votes: a one-row guess that is wrong would otherwise become the
    # memory of the whole row (rows lend their side to new apples)
    if out["mode"] == "gates" and ids:
        # 20 votes per second of planning whatever the plan rate (memory builds up in the same time)
        prev = getattr(planner, "_vote_t", None)
        weight = min(2.0, max(0.0, (t - prev) * 20.0)) if prev is not None and t > prev else 1.0
        planner._vote_t = t
        mapper.vote_sides([ids[i] for i in out["left_idx"]], [ids[i] for i in out["right_idx"]], weight)
        if R is not None:
            li, ri = out["left_idx"], out["right_idx"]
            idx = np.r_[li, ri].astype(int)
            if len(idx):
                mem.add(A[idx] @ R.T + [pose[0], pose[1]], [1] * len(li) + [-1] * len(ri), t, weight, p.bmem_merge_mm)
    if R is not None and n_real:
        # apples the car drove past: their side is a fact
        side_of = {lm.id: lm.side for lm in mapper.lms if lm.side}
        k = [i for i, lid in enumerate(ids) if lid in side_of]
        if k:
            mem.add(A[k] @ R.T + [pose[0], pose[1]], [side_of[ids[i]] for i in k], t, 0.0, p.bmem_merge_mm,
                    set_to=BoundaryMemory.CAP)
    return out


def _path_gap(P, M, upto_mm):
    """Largest distance from the points of path P (car frame, ahead of the camera, the first upto_mm of it) to path
    M, where M reaches that far; 0 if nothing to compare."""
    P, M = np.asarray(P, float).reshape(-1, 2), np.asarray(M, float).reshape(-1, 2)
    P = P[P[:, 0] > 150.0]
    if len(P) < 2 or len(M) < 2:
        return 0.0
    cum = np.r_[0.0, np.cumsum(np.hypot(*np.diff(P, axis=0).T))]
    P = P[(cum <= upto_mm) & (np.hypot(P[:, 0], P[:, 1]) < _reach(M) - 100.0)]
    if not len(P):
        return 0.0
    return float(np.max(np.min(np.hypot(P[:, None, 0] - M[None, :, 0], P[:, None, 1] - M[None, :, 1]), 1)))


def _reach(pl):
    """How far the path is known: its farthest point from the camera (not its forward x: in a sharp
    corner the path runs sideways and its x stays small)."""
    return float(np.max(np.hypot(pl[:, 0], pl[:, 1])))


def smooth_path(pl, sigma_mm, step_mm):
    """Gate midpoints (one per apple pair) make a kinked polyline; the car should follow a curve.
    Resample at step_mm by arc length, then a Gaussian (sigma_mm along the path) on x and y. The ends
    are padded by extending the end segments straight, so the curve keeps its start and direction."""
    P = np.asarray(pl, float).reshape(-1, 2)
    if len(P) < 3 or not step_mm or sigma_mm <= 0:
        return P
    seg = np.hypot(*np.diff(P, axis=0).T)
    keep = np.r_[True, seg > 1e-6]
    P = P[keep]
    if len(P) < 3:
        return P
    cum = np.r_[0.0, np.cumsum(np.hypot(*np.diff(P, axis=0).T))]
    n = max(3, int(cum[-1] / step_mm) + 1)
    u = np.linspace(0.0, cum[-1], n)
    R = np.stack([np.interp(u, cum, P[:, 0]), np.interp(u, cum, P[:, 1])], 1)
    h = max(1, int(round(sigma_mm / step_mm)))
    w = np.exp(-0.5 * (np.arange(-3 * h, 3 * h + 1) / h) ** 2)
    w /= w.sum()
    pad = 3 * h
    d0, d1 = R[1] - R[0], R[-1] - R[-2]
    head = R[0] - d0 * np.arange(pad, 0, -1)[:, None]
    tail = R[-1] + d1 * np.arange(1, pad + 1)[:, None]
    E = np.vstack([head, R, tail])
    return np.stack([np.convolve(E[:, 0], w, "valid"), np.convolve(E[:, 1], w, "valid")], 1)


def cast_like(current, text):
    """A command-line value ("1.5", "false", "-900,4000,3000") as the type of a parameter's current value."""
    if isinstance(current, bool):
        return str(text).strip().lower() in ("1", "true", "yes", "on")
    if isinstance(current, tuple):
        return tuple(float(x) for x in str(text).split(","))
    return type(current)(text)
