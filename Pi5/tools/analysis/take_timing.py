"""Per-take timing health: detection rate over the take (from _det.csv), drive.csv sampler regularity,
map size (landmarks per tick), in 5 s bins. Found the 2026-10-11 detector collapse (120 -> 20-40 fps as the
map grew inside the dashboard process).

    python tools/analysis/take_timing.py <dir with flat take files> [glob, default 2026*]
(flat = <id>_frames.csv, <id>_det.csv ... in one folder, like YoloTraining/data/videos_20261011)
"""
import csv, glob, json, os, sys
import numpy as np

D = sys.argv[1]
for meta_p in sorted(glob.glob(os.path.join(D, (sys.argv[2] if len(sys.argv) > 2 else "2026*") + "_meta.json"))):
    rid = os.path.basename(meta_p)[:-10]
    p = lambda s: os.path.join(D, rid + s)
    meta = json.load(open(meta_p))
    t0 = meta["started_monotonic_ns"] * 1e-9
    fr = list(csv.DictReader(open(p("_frames.csv"))))
    ts = np.array([int(r["sensor_ts_ns"]) for r in fr]) * 1e-9
    t0 = ts[0]
    dur = ts[-1] - t0
    # detection frames
    det_ts = {}
    for r in csv.DictReader(open(p("_det.csv"))):
        k = int(r["sensor_ts_ns"])
        det_ts.setdefault(k, [int(r["seq"]), 0])
        if int(r["box"]) >= 0:
            det_ts[k][1] += 1
    dts = np.array(sorted(det_ts)) * 1e-9
    seqs = np.array([det_ts[k][0] for k in sorted(det_ts)])
    nbox = np.array([det_ts[k][1] for k in sorted(det_ts)])
    # drive sampler
    dr = np.array([[int(r["t_ns"]) * 1e-9, float(r["motor_rps"]), float(r["throttle_cmd"] or 0), float(r["steer"])]
                   for r in csv.DictReader(open(p("_drive.csv")))])
    ddt = np.diff(dr[:, 0]) * 1000
    # landmarks per tick
    lm_t = {}
    for r in csv.DictReader(open(p("_landmarks.csv"))):
        lm_t[int(r["t_ns"])] = lm_t.get(int(r["t_ns"]), 0) + 1
    mp = json.load(open(p("_map.json"))) if os.path.exists(p("_map.json")) else {}
    print("=" * 100)
    print(rid, "dur %.1f s, video frames %d, det frames %d (%.1f/s), seq span %d (%.1f/s published)"
          % (dur, len(fr), len(dts), len(dts) / dur, seqs.max() - seqs.min() + 1 if len(seqs) else 0,
             (seqs.max() - seqs.min() + 1) / dur if len(seqs) else 0))
    g = np.diff(dts) * 1000
    if len(g):
        print("  det interval ms: p50 %.1f p90 %.1f p99 %.1f max %.0f; gaps>50ms %d, >200ms %d" %
              (*np.percentile(g, [50, 90, 99]), g.max(), (g > 50).sum(), (g > 200).sum()))
        sg = np.diff(seqs)
        print("  seq steps: ", {int(k): int(v) for k, v in zip(*np.unique(sg, return_counts=True)) if v > 2})
    print("  drive sampler ms: p50 %.1f p99 %.1f max %.0f, >50ms: %d" % (*np.percentile(ddt, [50, 99]), ddt.max(), (ddt > 50).sum()))
    # per 5-s bins
    print("  %6s %7s %7s %7s %7s %6s %6s" % ("t", "det/s", "boxes", "rps", "thr", "lms", "drvmax"))
    for a in np.arange(0, dur, 5):
        m = (dts - t0 >= a) & (dts - t0 < a + 5)
        md = (dr[:, 0] - t0 >= a) & (dr[:, 0] - t0 < a + 5)
        lms = [v for k, v in lm_t.items() if a <= k * 1e-9 - t0 < a + 5]
        dd = np.diff(dr[md, 0]) * 1000
        print("  %6.0f %7.1f %7.1f %7.1f %7.2f %6s %6.0f" % (a, m.sum() / 5, nbox[m].mean() if m.any() else 0,
              dr[md, 1].mean() if md.any() else 0, dr[md, 2].max() if md.any() else 0,
              max(lms) if lms else "-", dd.max() if len(dd) else 0))
    if mp:
        st = {}
        for l in mp["landmarks"]:
            st[l["state"]] = st.get(l["state"], 0) + 1
        print("  final map:", st, "trail pts", len(mp["trail"]), "stats", mp["stats"])
