"""Steering command -> curvature (gyro yaw rate / encoder speed) with the servo + chassis lag (best lag by
correlation). 2026-10-11: full lock = 1.45 1/m (0.69 m radius) at ~1.2 m/s, lag ~67 ms. NB keyboard driving is
bang-bang (-1/0/+1), so intermediate steer values are rare.

    python tools/analysis/steer_model.py <dir with flat take files> Pi5/config/odometry.json [glob, default 2026*]
"""
import csv, glob, os, sys, json
import numpy as np

D = sys.argv[1]
MM = json.load(open(sys.argv[2]))["mm_per_turn"]
S, K, V = [], [], []
lags = []
for p in sorted(glob.glob(os.path.join(D, (sys.argv[3] if len(sys.argv) > 3 else "2026*") + "_frames.csv"))):
    r = list(csv.DictReader(open(p)))
    t = np.array([int(x["t_mid_ns"]) for x in r]) * 1e-9
    v = np.abs(np.array([float(x["motor_rps"]) for x in r])) * MM          # mm/s
    w = np.radians(np.array([float(x["gyro_z_dps"]) for x in r]))          # yaw rate, left +
    st = np.array([float(x["steer"]) for x in r])                           # -1 left .. +1 right
    ok = v > 400
    k = np.where(ok, w / np.maximum(v, 1), np.nan) * 1000                   # 1/m, left +
    best = None
    for lag in range(0, 30):                                                # frames at 60 fps
        a, b = st[:len(st) - lag], k[lag:]
        m = ~np.isnan(b)
        c = np.corrcoef(a[m], b[m])[0, 1]
        if best is None or c < best[1]:
            best = (lag, c)
    lags.append(best)
    lag = best[0]
    a, b = st[:len(st) - lag], k[lag:]
    m = ~np.isnan(b)
    S.append(a[m]); K.append(b[m]); V.append(v[lag:][m])
    print(os.path.basename(p)[:15], "best lag %d frames (%.0f ms) corr %.3f" % (lag, lag / 60 * 1000, best[1]))
S, K, V = np.concatenate(S), np.concatenate(K), np.concatenate(V)
print("speed while moving: p50 %.0f mm/s p90 %.0f max %.0f" % (*np.percentile(V, [50, 90]), V.max()))
for lo, hi in [(-1.01, -0.9), (-0.9, -0.5), (-0.5, -0.1), (-0.1, 0.1), (0.1, 0.5), (0.5, 0.9), (0.9, 1.01)]:
    m = (S >= lo) & (S < hi)
    if m.sum() > 20:
        print("steer [%5.2f,%5.2f): n %5d  curvature p50 %+.2f 1/m  (radius %.2f m)  p10 %+.2f p90 %+.2f" %
              (lo, hi, m.sum(), np.median(K[m]), 1 / abs(np.median(K[m])) if abs(np.median(K[m])) > 0.05 else np.inf,
               *np.percentile(K[m], [10, 90])))
print("values of steer used:", np.unique(np.round(S, 2))[:20])
