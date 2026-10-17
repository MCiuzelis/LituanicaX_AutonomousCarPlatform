"""Path flips of the planner on recorded takes (open loop): the take's detections + odometry replayed through
TrackMapper + Planner at the live plan rate; a flip = the lookahead target jumps > 400 mm sideways between two
consecutive plans while the steering jumps > 0.8 (2026-10-17: the "turns to the outside at a corner entry"
moments were these, mostly on gate paths resting on one apple a side). Counted only while the car was autonomous
in the take (its _drive.csv), split by plan mode.

    cd Pi5 && uv run --project ../YoloTraining --with scipy python tools/pc_run.py tools/analysis/flip_eval.py \\
        recordings/<id> [...] [--set planner_field=value ...]
"""
import argparse
import csv
import math
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from tools.replay_map import load_take                    # noqa: E402
from vision.planner import Planner, cast_like, plan_with_map           # noqa: E402
from vision.track_map import TrackMapper                   # noqa: E402

PLAN_S = 0.02                                              # vision/map_worker.py PLAN_HZ 50


def auto_spans(take):
    tid = os.path.basename(take.rstrip("/\\"))
    path = os.path.join(take, tid + "_drive.csv")
    spans, start, last = [], None, None
    with open(path) as f:
        for r in csv.DictReader(f):
            t = int(r["t_ns"]) / 1e9
            on = (r.get("auto_state") or "0") not in ("0", "0.0", "")
            if on and start is None:
                start = t
            if not on and start is not None:
                spans.append((start, t))
                start = None
            last = t
    if start is not None:
        spans.append((start, last))
    return spans


def run(take, sets):
    f, meta, events = load_take(take)
    spans = auto_spans(take)
    m, pl = TrackMapper(), Planner()
    for kv in sets:
        k, v = kv.split("=", 1)
        setattr(pl.p, k, cast_like(getattr(pl.p, k), v))
    plans = []
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
            o = m.odo_pose(nt)
            if b is not None and o:
                pr = m.odo_pose(nt - 0.2)
                v = math.hypot(o[0] - pr[0], o[1] - pr[1]) / 0.2 if pr else 0
                p = plan_with_map(pl, m, v, nt)
                if any(a <= nt <= b2 for a, b2 in spans) and p.get("target") is not None:
                    plans.append((nt, p["mode"], p["target"][1], p["steer"]))
            nt += PLAN_S
    out = {}
    for (t0, m0, y0, s0), (t1, m1, y1, s1) in zip(plans, plans[1:]):
        if t1 - t0 > 0.1:
            continue
        d = out.setdefault(m1, [0, 0])
        d[0] += 1
        if abs(y1 - y0) > 400 and abs(s1 - s0) > 0.8:
            d[1] += 1
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("takes", nargs="+")
    ap.add_argument("--set", nargs="*", default=[])
    a = ap.parse_args()
    tot = {}
    for take in a.takes:
        r = run(take, a.set)
        print("%s  %s" % (os.path.basename(take.rstrip("/\\"))[:15],
                          "  ".join("%s %d/%d" % (k, v[1], v[0]) for k, v in sorted(r.items()))), flush=True)
        for k, v in r.items():
            t = tot.setdefault(k, [0, 0])
            t[0] += v[0]
            t[1] += v[1]
    n, fl = sum(v[0] for v in tot.values()), sum(v[1] for v in tot.values())
    print("TOTAL plans %d, flips %d (%.2f %%): %s" % (n, fl, 100.0 * fl / max(n, 1),
                                                    ", ".join("%s %d/%d" % (k, v[1], v[0]) for k, v in sorted(tot.items()))))


if __name__ == "__main__":
    main()
