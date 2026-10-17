#!/usr/bin/env python3
"""How often does the planner put an apple on the wrong side, and how often has it no path? Replays takes
through TrackMapper + Planner (as the car runs them) and scores every plan against the side each apple was
really on: the side the car passed it on (TrackMapper sets Landmark.side when the camera goes by it,
-250 < x < 0, |y| 120..2000 mm), known only at the end of the replay.

    python3 tools/side_eval.py recordings/<id> [...] [--set planner_field=v ...] [--map map_field=v ...]

Per take, over the ticks with the car moving (> 200 mm/s): modes (gates / row / memory / crawl / lost),
"wrong side": apples the plan used on the side opposite to the one they were passed on (of all used apples
that were later passed), "wrong-side ticks": plans with at least one such apple ahead of the car within 1.5 m.
Autonomous takes are open loop here (the car drove the old planner's path), but the sides are facts.
"""
import argparse
import math
import os
import sys
from collections import Counter

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from tools.replay_map import load_take                                   # noqa: E402
from vision.planner import Planner, cast_like, plan_with_map              # noqa: E402
from vision.track_map import TrackMapper, load_params                     # noqa: E402

TICK_S = 0.05


def evaluate(take, a):
    f, meta, events = load_take(take)
    p = load_params()
    for kv in a.map:
        k, v = kv.split("=", 1)
        setattr(p, k, cast_like(getattr(p, k), v))
    m, pl = TrackMapper(params=p), Planner()
    for kv in a.set:
        k, v = kv.split("=", 1)
        setattr(pl.p, k, cast_like(getattr(pl.p, k), v))
    passed = {}                                                    # landmark id -> side it was passed on
    ticks = []
    nt = None
    for t, _, kind, x in events:
        if kind == "imu":
            m.add_imu(t, *x)
        elif kind == "turns":
            m.add_turns(t, x)
        else:
            m.add_detections(x["t"], x["boxes"], x["crop"], x["seq"], x["sensor_ts"])
        m.step()
        if nt is None:
            nt = t
        while t >= nt:
            b = m.tick(nt)
            for lm in m.lms:
                if lm.side and lm.id not in passed:
                    passed[lm.id] = lm.side
            if b is not None:
                o, pr = m.odo_pose(nt), m.odo_pose(nt - 0.2)
                v = math.hypot(o[0] - pr[0], o[1] - pr[1]) / 0.2 if o and pr else 0.0
                out = plan_with_map(pl, m, v, nt)
                ids = m.local_ids
                A = m.local_apples
                near = lambda i: 0 < A[i][0] < 1500                 # noqa: E731
                used = [(ids[i], 1, near(i)) for i in out["left_idx"]] + [(ids[i], -1, near(i)) for i in out["right_idx"]]
                ticks.append((nt, v, out["mode"], used))
            nt += TICK_S
    rows = [tk for tk in ticks if tk[1] > 200]
    modes = Counter(tk[2] for tk in rows)
    n_used = n_wrong = 0
    wrong_ticks = 0
    for _, _, _, used in rows:
        bad = False
        for lid, sd, nr in used:
            if lid in passed:
                n_used += 1
                if passed[lid] != sd:
                    n_wrong += 1
                    bad = bad or nr
        wrong_ticks += bad
    n = max(1, len(rows))
    if a.times:
        t0 = ticks[0][0]
        bad_t = [tk[0] - t0 for tk in rows if any(lid in passed and passed[lid] != sd and nr for lid, sd, nr in tk[3])]
        segs, cur = [], []
        for x in bad_t:
            if cur and x - cur[-1] > 0.3:
                segs.append(cur)
                cur = []
            cur.append(x)
        if cur:
            segs.append(cur)
        modes_at = {round(tk[0] - t0, 2): tk[2] for tk in rows}
        print("  wrong-side moments (s from the first tick): " + " ".join(
            "%.1f-%.1f(%s)" % (sg[0], sg[-1], modes_at.get(round(sg[0], 2), "?")) for sg in segs))
    print("%s: %d moving ticks | %s | wrong side %.1f %% of %d used apples, wrong-side ticks %.1f %%" % (
        f["id"][:15], len(rows), " ".join("%s %.0f%%" % (k, 100 * c / n) for k, c in modes.most_common()),
        100 * n_wrong / max(1, n_used), n_used, 100 * wrong_ticks / n), flush=True)
    return len(rows), modes, n_used, n_wrong, wrong_ticks


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("takes", nargs="+")
    ap.add_argument("--set", nargs="*", default=[])
    ap.add_argument("--map", nargs="*", default=[])
    ap.add_argument("--times", action="store_true", help="list the moments with a wrong-side apple ahead")
    a = ap.parse_args()
    for take in a.takes:
        evaluate(take, a)


if __name__ == "__main__":
    main()
