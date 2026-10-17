"""Planner moments, one row each: the video frame (live detections drawn in) next to a large car-frame
view of what the planner saw and did: confirmed apples coloured by their side memory (blue left /
orange right / grey none, number = votes), tentative apples hollow, the apples the plan used as left /
right (squares), the planned path (green), the pure-pursuit target (green x), the path really driven over
the next 2.5 s (black, odometry) and the camera's field of view. Times = seconds from the first odometry
tick (autonomy_eval --csv "t").

    python tools/analysis/plan_views.py recordings/<id> out.png 4.0,4.3,4.6 [--video path.mp4] [--set k=v ...]
(PC: uv run --project YoloTraining --with scipy --with matplotlib python Pi5/tools/pc_run.py ...; the video
defaults to YoloTraining/data/videos_<date>/<id>.mp4)
"""
import argparse
import csv
import glob
import math
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from tools.replay_map import load_take                    # noqa: E402
from vision.planner import Planner, cast_like, plan_with_map           # noqa: E402
from vision.track_map import TrackMapper                   # noqa: E402

TICK_S = 0.05                                              # the car plans at 20 Hz


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("take")
    ap.add_argument("out")
    ap.add_argument("times")
    ap.add_argument("--video")
    ap.add_argument("--set", nargs="*", default=[])
    ap.add_argument("--hfov-deg", type=float, default=50.0)
    a = ap.parse_args()
    want = [float(x) for x in a.times.split(",")]
    f, meta, events = load_take(a.take)
    boxes_of = {x["sensor_ts"]: x["boxes"] for _, _, kind, x in events if kind == "det"}
    m, pl = TrackMapper(), Planner()
    for kv in a.set:
        k, v = kv.split("=", 1)
        setattr(pl.p, k, cast_like(getattr(pl.p, k), v))
    snaps, odo, passed = {}, [], {}
    nt = t_first = None
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
            for lm in m.lms:                                   # the side each apple was passed on (the truth)
                if lm.side:
                    passed.setdefault(lm.id, lm.side)
            o = m.odo_pose(nt)
            if t_first is None and o:
                t_first = nt
            if o:
                odo.append((nt, *o))
            if b is not None:
                pr = m.odo_pose(nt - 0.2)
                v = math.hypot(o[0] - pr[0], o[1] - pr[1]) / 0.2 if pr else 0
                p = plan_with_map(pl, m, v, nt)
                for w in want:
                    if abs(nt - t_first - w) < TICK_S / 2 + 1e-6 and w not in snaps:
                        pose = m.pose(nt)
                        c, s = math.cos(pose[2]), math.sin(pose[2])
                        R = np.array([[c, -s], [s, c]])
                        lms = [((lm.x - pose[:2]) @ R, lm.state, lm.votes, lm.side, lm.id) for lm in m.lms]
                        lf = m.last_frame
                        snaps[w] = dict(t=nt, p=p, lms=lms, v=v, sensor_ts=lf["sensor_ts"] if lf else None,
                                        boxes=lf["boxes"] if lf else [])
            nt += TICK_S
    odo = np.array(odo)
    # video frames by sensor timestamp
    vid = a.video
    if vid is None:
        rid = f["id"]
        c = glob.glob(os.path.join(ROOT, "..", "YoloTraining", "data", "videos*", rid + ".mp4"))
        vid = c[0] if c else None
    fr = list(csv.DictReader(open(f["frames"])))
    fts = np.array([int(r["t_mid_ns"]) for r in fr]) * 1e-9
    fsts = [int(r["sensor_ts_ns"]) for r in fr]
    steer = np.array([float(r["steer"]) for r in fr])
    gz = np.array([float(r["gyro_z_dps"]) for r in fr])
    cap = None
    if vid:
        import cv2
        cap = cv2.VideoCapture(vid)
    n = len(snaps)
    fig, axs = plt.subplots(n, 2, figsize=(16, 6.2 * n), squeeze=False, gridspec_kw={"width_ratios": [1.45, 1]})
    for row, (w, sn) in zip(axs, sorted(snaps.items())):
        nt, p = sn["t"], sn["p"]
        j = int(np.clip(np.searchsorted(fts, nt), 0, len(fts) - 1))
        axv, ax = row
        if cap is not None:
            cap.set(1, j)
            ok, img = cap.read()
            if ok:
                axv.imshow(img[:, :, ::-1])
                H, W = img.shape[:2]
                for bx in boxes_of.get(fsts[j], []):          # the boxes detected in this very frame
                    axv.add_patch(plt.Rectangle((bx[0] * W, bx[1] * H), (bx[2] - bx[0]) * W, (bx[3] - bx[1]) * H,
                                                fill=False, ec="yellow", lw=1))
        axv.set_axis_off()
        axv.set_title("video frame %d (t=%.2f s)  driver steer %+.0f  yaw rate %+.0f deg/s" % (j, w, steer[j], gz[j]))
        i = int(np.searchsorted(odo[:, 0], nt))
        c, s = math.cos(odo[i, 3]), math.sin(odo[i, 3])
        F = (odo[i:i + int(2.5 / TICK_S), 1:3] - odo[i, 1:3]) @ np.array([[c, -s], [s, c]])
        for xy, st, vt, sd, lid in sn["lms"]:
            vv = vt + 6 * sd
            col = "tab:blue" if vv >= 2 else "tab:orange" if vv <= -2 else "0.5"
            ax.plot(-xy[1], xy[0], "o", color=col, mfc=col if st == 1 else "none", ms=8)
            if st == 1 and abs(xy[1]) < 2400 and -900 < xy[0] < 4000:
                truth = {1: " L", -1: " R"}.get(passed.get(lid), "")
                ax.text(-xy[1] + 50, xy[0] + 40, "%d%s" % (vv, truth), fontsize=7, color=col)
        for k2, cc in (("left", "tab:blue"), ("right", "tab:orange")):
            P = np.asarray(p.get(k2) if p.get(k2) is not None else np.zeros((0, 2))).reshape(-1, 2)
            if len(P):
                ax.plot(-P[:, 1], P[:, 0], "s-", mfc="none", color=cc, ms=15, mew=2, lw=1)
        if p.get("path") is not None:
            ax.plot(-p["path"][:, 1], p["path"][:, 0], "-", color="tab:green", lw=2.5)
        if p.get("target") is not None:
            ax.plot(-p["target"][1], p["target"][0], "x", color="tab:green", ms=14, mew=3)
        ax.plot(-F[:, 1], F[:, 0], "k-", lw=2, label="driven next 2.5 s")
        h = math.radians(a.hfov_deg / 2)
        for sg in (-1, 1):
            ax.plot([0, -sg * 4000 * math.sin(h)], [0, 4000 * math.cos(h)], "-", color="0.8", lw=1)
        ax.plot([0], [0], "k^", ms=10)
        ax.set_xlim(-2400, 2400)
        ax.set_ylim(-900, 4000)
        ax.set_aspect("equal")
        ax.grid(alpha=.3)
        st = p.get("steer")
        ax.set_title("%s  steer %s  v %.0f  known %.0f  %s" % (p["mode"], "-" if st is None else "%+.2f" % st,
                                                               sn["v"], p["known_mm"], p.get("why", "")), fontsize=10)
    plt.tight_layout()
    plt.savefig(a.out, dpi=60)


if __name__ == "__main__":
    main()
