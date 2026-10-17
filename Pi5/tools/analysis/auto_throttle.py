"""Autonomy throttle / brake behaviour of takes from their _drive.csv (link / auto_state / auto_v_target
columns, takes since 2026-10-11 afternoon): time driving vs braking, throttle distribution, how often
the throttle switches, target vs measured speed.

    python tools/analysis/auto_throttle.py <dir with flat take files> [glob, default 2026*]
"""
import csv
import glob
import json
import os
import sys

import numpy as np

D = sys.argv[1]
MM = 55.074
for meta_p in sorted(glob.glob(os.path.join(D, (sys.argv[2] if len(sys.argv) > 2 else "2026*") + "_meta.json"))):
    rid = os.path.basename(meta_p)[:-10]
    rows = list(csv.DictReader(open(os.path.join(D, rid + "_drive.csv"))))
    if not rows or "auto_state" not in rows[0]:
        continue
    t = np.array([int(r["t_ns"]) for r in rows]) * 1e-9
    t -= t[0]
    st = np.array([int(r["auto_state"] or 0) for r in rows])
    thr = np.array([float(r["throttle_cmd"] or 0) for r in rows])
    v = np.abs([float(r["motor_rps"]) for r in rows]) * MM
    vt = np.array([float(r["auto_v_target"]) if r["auto_v_target"] else np.nan for r in rows])
    dt = np.r_[np.diff(t), 0.01]
    drv, brk = st == 1, st == 2
    if not (drv | brk).any():
        print(rid[9:15], "no autonomy")
        continue
    tp = thr[drv]
    ceil = tp.max() if len(tp) else 0
    switches = int(np.sum(np.abs(np.diff(np.sign(np.round(tp, 3)))) > 0)) if len(tp) else 0
    # driving episodes: how they ended
    ep, start = [], None
    for i in range(len(st)):
        if st[i] == 1 and start is None:
            start = i
        if start is not None and st[i] != 1:
            ep.append((t[start], t[i] - t[start], v[i]))
            start = None
    print("%s %4.0f s | auto driving %4.1f s, braking %4.1f s | throttle while driving: at ceiling %.0f%%, 0 %.0f%%, "
          "brake %.0f%% | switches %.1f/s | speed p50 %.2f (target p50 %.2f) m/s | %d episodes, median %.1f s" % (
              rid[9:15], t[-1], dt[drv].sum(), dt[brk].sum(), 100 * np.mean(tp >= ceil - 1e-6) if len(tp) else 0,
              100 * np.mean(tp == 0) if len(tp) else 0, 100 * np.mean(tp < 0) if len(tp) else 0,
              switches / max(dt[drv].sum(), 1e-6), np.median(v[drv]) / 1000 if drv.any() else 0,
              np.nanmedian(vt[drv]) / 1000 if drv.any() else 0, len(ep), np.median([e[1] for e in ep]) if ep else 0))
