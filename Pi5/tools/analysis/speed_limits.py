"""Why is the car slow? Replays a take through the map + planner and, for the moments the car drove itself
(drive.csv auto_state == 1), tells which speed limit set the planner's target (curve / known path / confidence
/ the 0.4 m/s floor) next to the speed and throttle the car really had.

    python tools/analysis/speed_limits.py recordings/<id> [...] [--set planner_field=v ...]
"""
import argparse
import csv
import math
import os
import sys
from collections import Counter

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from tools.replay_map import load_take                    # noqa: E402
from vision.planner import Planner, cast_like, plan_with_map   # noqa: E402
from vision.track_map import TrackMapper                  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("takes", nargs="+")
    ap.add_argument("--set", nargs="*", default=[])
    a = ap.parse_args()
    for take in a.takes:
        f, meta, events = load_take(take)
        dr = list(csv.DictReader(open(f["drive"])))
        dt_ = np.array([int(r["t_ns"]) for r in dr]) * 1e-9
        auto = np.array([int(float(r["auto_state"] or 0)) for r in dr])
        thr = np.array([float(r["throttle_cmd"]) for r in dr])
        m, pl = TrackMapper(), Planner()
        for kv in a.set:
            k, v = kv.split("=", 1)
            setattr(pl.p, k, cast_like(getattr(pl.p, k), v))
        rows = []
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
                if b is not None:
                    o, pr = m.odo_pose(nt), m.odo_pose(nt - 0.2)
                    v = math.hypot(o[0] - pr[0], o[1] - pr[1]) / 0.2 if o and pr else 0.0
                    out = plan_with_map(pl, m, v, nt)
                    j = min(len(dt_) - 1, int(np.searchsorted(dt_, nt)))
                    if auto[j] == 1:
                        lim = out.get("limits") or {}
                        vt = out.get("speed_mm_s") or 0.0
                        which = "lost" if out["mode"] in ("lost", "crawl") else (
                            min(lim, key=lim.get) if lim and min(lim.values()) <= vt + 1 else "floor")
                        rows.append((nt, v, vt, which, out["mode"], thr[j], lim, out.get("confidence")))
                nt += 0.05
        n = len(rows)
        if not n:
            print("%s: no autonomous driving" % f["id"][:15])
            continue
        vs = np.array([r[1] for r in rows])
        vt = np.array([r[2] for r in rows])
        th = np.array([r[5] for r in rows])
        c = Counter(r[3] for r in rows)
        print("%s: %d autonomous ticks | speed p50 %.0f, target p50 %.0f mm/s | target < 700 mm/s %.0f %% of the time | "
              "throttle p50 %.2f, <= 0 %.0f %%" % (f["id"][:15], n, np.median(vs), np.median(vt), 100 * np.mean(vt < 700),
                                                 np.median(th), 100 * np.mean(th <= 0)))
        print("   target set by: " + ", ".join("%s %.0f %%" % (k, 100 * v / n) for k, v in c.most_common()))
        for key in ("curve", "known", "conf"):
            vals = [r[6][key] for r in rows if r[6] and key in r[6]]
            if vals:
                print("   %-5s limit p10 %5.0f p50 %5.0f" % (key, np.percentile(vals, 10), np.percentile(vals, 50)))
        conf = [r[7] for r in rows if r[7] is not None]
        if conf:
            print("   confidence p10 %.2f p50 %.2f" % (np.percentile(conf, 10), np.percentile(conf, 50)))


if __name__ == "__main__":
    main()
