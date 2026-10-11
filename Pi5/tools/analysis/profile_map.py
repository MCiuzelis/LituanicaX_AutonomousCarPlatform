"""Profile vision/track_map.py on a take: cost of step() per detection frame and of the 10 Hz tick() +
snapshot() vs. the number of apples, and the share of a core that would be at 120 fps. Run it ON THE PI
(the PC is several times faster). 2026-10-11 (old unbounded map): 2.3-5.5 ms/frame.

    python3 tools/analysis/profile_map.py recordings/<id>
"""
import sys, time, os
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from tools.replay_map import load_take
from vision.track_map import TrackMapper

f, meta, events = load_take(sys.argv[1])
m = TrackMapper()
rows = []          # (t, n_lms, step_ms, frames_processed)
ticks = []         # (t, n_lms, tick_ms, snap_ms, snap_bytes)
import json
next_tick = None
for t, _, kind, x in events:
    if kind == "imu":
        m.add_imu(t, *x)
    elif kind == "turns":
        m.add_turns(t, x)
    else:
        m.add_detections(x["t"], x["boxes"], x["crop"], x["seq"], x["sensor_ts"])
    a = time.perf_counter()
    n = m.step()
    b = time.perf_counter()
    if n:
        rows.append((t, len(m.lms), (b - a) * 1e3 / n))
    if next_tick is None:
        next_tick = t
    if t >= next_tick:
        next_tick += 0.1
        a = time.perf_counter()
        m.tick()
        b = time.perf_counter()
        s = m.snapshot(trail_from=len(m.trail))
        js = json.dumps(s)
        c = time.perf_counter()
        ticks.append((t, len(m.lms), (b - a) * 1e3, (c - b) * 1e3, len(js)))
rows, ticks = np.array(rows), np.array(ticks)
t0 = rows[0, 0]
print("per detection frame step(): p50 %.2f ms  p90 %.2f  max %.1f" % (*np.percentile(rows[:, 2], [50, 90]), rows[:, 2].max()))
print("%5s %6s %9s %9s %9s %9s %8s" % ("t", "lms", "step_ms", "tick_ms", "snap_ms", "snapKB", "cpu@120"))
for a in np.arange(0, rows[-1, 0] - t0, 5):
    r = rows[(rows[:, 0] - t0 >= a) & (rows[:, 0] - t0 < a + 5)]
    k = ticks[(ticks[:, 0] - t0 >= a) & (ticks[:, 0] - t0 < a + 5)]
    if not len(r):
        continue
    # CPU share of one core if detection ran at 120 fps + 10 Hz tick/snapshot
    share = (r[:, 2].mean() * 120 + (k[:, 2].mean() + k[:, 3].mean()) * 10) / 10 if len(k) else 0
    print("%5.0f %6.0f %9.2f %9.1f %9.1f %9.1f %7.0f%%" % (a, r[:, 1].max(), r[:, 2].mean(), k[:, 2].mean(), k[:, 3].mean(), k[:, 4].mean() / 1024, share))
