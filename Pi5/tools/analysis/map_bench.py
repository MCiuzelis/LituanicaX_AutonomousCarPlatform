"""CPU benchmark of the map worker's work: a take segment through TrackMapper + Planner as fast as possible,
the way vision/map_worker.py schedules it (10 ms batches, a plan every 1 / --plan-hz). Prints CPU ms per
second of data (1000 = one core flat out), per detection frame and per plan.

    python3 tools/analysis/map_bench.py recordings/<id> [--from 10] [--to 30] [--plan-hz 20] [--repeat 3]
"""
import argparse
import math
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from tools.replay_map import load_take                    # noqa: E402
from vision.planner import Planner, plan_with_map          # noqa: E402
from vision.track_map import TrackMapper, load_params     # noqa: E402


def run(events, t0, t1, plan_hz):
    m, pl = TrackMapper(params=load_params()), Planner()
    T0 = events[0][0]
    step_s = plan_s = 0.0
    frames = plans = 0
    nb = nt = None
    for t, _, kind, x in events:
        rel = t - T0
        if rel > t1:
            break
        if kind == "imu":
            m.add_imu(t, *x)
        elif kind == "turns":
            m.add_turns(t, x)
        else:
            m.add_detections(x["t"], x["boxes"], x["crop"], x["seq"], x["sensor_ts"])
        if nb is None:
            nb = nt = t
        if t < nb:
            continue
        nb += 0.01
        timed = rel >= t0
        a = time.perf_counter()
        n = m.step()
        if timed:
            step_s += time.perf_counter() - a
            frames += n
        if t >= nt:
            nt += 1.0 / plan_hz
            a = time.perf_counter()
            b = m.tick()
            to = m.odo.last_t
            if b is not None and to is not None:
                o, pr = m.odo_pose(to), m.odo_pose(to - 0.2)
                v = math.hypot(o[0] - pr[0], o[1] - pr[1]) / 0.2 if o and pr else 0.0
                plan_with_map(pl, m, v, to)
            if timed:
                plan_s += time.perf_counter() - a
                plans += 1
    return step_s, plan_s, frames, plans


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("take")
    ap.add_argument("--from", dest="t0", type=float, default=10.0)
    ap.add_argument("--to", dest="t1", type=float, default=30.0)
    ap.add_argument("--plan-hz", type=float, default=20.0)
    ap.add_argument("--repeat", type=int, default=3)
    a = ap.parse_args()
    _, _, events = load_take(a.take)
    best = None
    for _ in range(a.repeat):
        r = run(events, a.t0, a.t1, a.plan_hz)
        if best is None or r[0] + r[1] < best[0] + best[1]:
            best = r
    step_s, plan_s, frames, plans = best
    dur = a.t1 - a.t0
    print("%s %.0f-%.0f s, plans %.0f Hz: CPU %.0f ms per s of data (frames %.0f + plans %.0f) | %.2f ms / frame (%d), "
          "%.2f ms / plan (%d)" % (os.path.basename(a.take.rstrip("/"))[:15], a.t0, a.t1, a.plan_hz,
                                    1000 * (step_s + plan_s) / dur, 1000 * step_s / dur, 1000 * plan_s / dur,
                                    1000 * step_s / max(frames, 1), frames, 1000 * plan_s / max(plans, 1), plans))


if __name__ == "__main__":
    main()
