"""Car-frame views at chosen moments of a replayed take: apples (blue left / orange right / grey unknown,
hollow = tentative), the map's chained boundaries, the planner's path (green dotted) and the apples it used
as gates (squares), and the path really driven next (black, odometry). Used to find the corner failure of the
chained boundaries. Times = seconds from the first odometry tick (same as autonomy_eval --csv "t").

    python tools/analysis/car_frame_views.py recordings/<id> out.png 4.0,9.3,17.2
(on Windows through tools/pc_run.py, with scipy: uv run --project YoloTraining --with scipy python Pi5/tools/pc_run.py ...)
"""
import math, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from tools.replay_map import load_take
from vision.track_map import TrackMapper
from vision.planner import Planner

take, out = sys.argv[1], sys.argv[2]
want = [float(x) for x in sys.argv[3].split(",")]
f, meta, events = load_take(take)
m, pl = TrackMapper(), Planner()
snaps, odo = {}, []
nt = None
t_first = None
for t, _, kind, x in events:
    if kind == "imu": m.add_imu(t, *x)
    elif kind == "turns": m.add_turns(t, x)
    else: m.add_detections(x["t"], x["boxes"], x["crop"], x["seq"], x["sensor_ts"])
    m.step()
    if nt is None: nt = t
    while t >= nt:
        b = m.tick(nt)
        o = m.odo_pose(nt)
        if t_first is None and o: t_first = nt
        if o: odo.append((nt, *o))
        if b is not None:
            pr = m.odo_pose(nt - 0.1)
            v = math.hypot(o[0] - pr[0], o[1] - pr[1]) / 0.1 if pr else 0
            p = pl.update(b, v, nt, apples=m.local_apples)
            rel_t = nt - t_first
            for w in want:
                if abs(rel_t - w) < 0.05 and w not in snaps:
                    pose = m.pose(nt); R = np.array([[math.cos(pose[2]), -math.sin(pose[2])], [math.sin(pose[2]), math.cos(pose[2])]])
                    lms = [((lm.x - pose[:2]) @ R, lm.state, lm.side) for lm in m.lms]
                    snaps[w] = (nt, b, p, lms)
        nt += 0.1
odo = np.array(odo)
fig, axs = plt.subplots(1, len(snaps), figsize=(5 * len(snaps), 6))
for ax, (w, (nt, b, p, lms)) in zip(np.atleast_1d(axs), sorted(snaps.items())):
    i = int(np.searchsorted(odo[:, 0], nt))
    c, s = math.cos(odo[i, 3]), math.sin(odo[i, 3])
    F = (odo[i:i + 40, 1:3] - odo[i, 1:3]) @ np.array([[c, -s], [s, c]])
    col = {1: "tab:blue", -1: "tab:orange", 0: "0.5"}
    for xy, st, sd in lms:
        ax.plot(-xy[1], xy[0], "o", color=col[sd], mfc=col[sd] if st == 1 else "none", ms=7)
    for k, cc in (("left", "tab:blue"), ("right", "tab:orange"), ("center", "tab:green")):
        P = np.array(b.get(k) or np.zeros((0, 2)))
        if len(P): ax.plot(-P[:, 1], P[:, 0], "-", color=cc, lw=2)
    for k2, cc in (("left", "tab:blue"), ("right", "tab:orange")):
        if p.get(k2) is not None and len(p[k2]): ax.plot(-np.asarray(p[k2])[:, 1], np.asarray(p[k2])[:, 0], "s", mfc="none", mec=cc, ms=13, mew=2)
    if p.get("path") is not None:
        ax.plot(-p["path"][:, 1], p["path"][:, 0], ":", color="g", lw=2)
    ax.plot(-F[:, 1], F[:, 0], "k-", lw=2.5, label="driven (next 4 s)")
    ax.plot([0], [0], "k^", ms=10)
    ax.set_xlim(-2000, 2000); ax.set_ylim(-1000, 3500); ax.set_aspect("equal"); ax.grid(alpha=.3)
    ax.set_title("t=%.1f s mode=%s steer=%s" % (w, p["mode"], None if p.get("steer") is None else round(p["steer"], 2)))
plt.tight_layout(); plt.savefig(out, dpi=70)
