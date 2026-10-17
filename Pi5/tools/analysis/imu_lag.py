"""How late is each IMU stream? Cross-correlates, on a recorded take:
  1. the body rate differentiated from the game_rv quaternions vs the gyro reports (both board frame):
     the fusion's own delay relative to the gyro
  2. the camera's yaw rate (bearing change of the same apple between consecutive detection frames, 120 fps,
     median over the apples matched by the track map) vs the gyro / game_rv yaw rate: the absolute delay of
     each stream against the frames' t_mid (exposure middle)
A positive lag = that stream describes the car this long before its timestamp (it arrives late).

    python tools/analysis/imu_lag.py recordings/<id> [recordings/<id2> ...]
(PC: uv run --project YoloTraining --with scipy python Pi5/tools/pc_run.py Pi5/tools/analysis/imu_lag.py ...)
"""
import csv
import json
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from tools.replay_map import load_take                    # noqa: E402
from vision.track_map import TrackMapper, load_params     # noqa: E402


def qmul(a, b):
    """Quaternions (..., 4) as (i, j, k, real)."""
    x1, y1, z1, w1 = np.moveaxis(a, -1, 0)
    x2, y2, z2, w2 = np.moveaxis(b, -1, 0)
    return np.stack([w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2, w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                     w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2, w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2], -1)


def best_lag(t_ref, x_ref, t_sig, x_sig, lags):
    """Lag (s) of signal vs reference maximising correlation: corr(x_ref(t), x_sig(t + lag))."""
    out = []
    for L in lags:
        y = np.interp(t_ref + L, t_sig, x_sig)
        out.append(np.corrcoef(x_ref, y)[0, 1])
    out = np.array(out)
    i = int(np.argmax(out))
    if 0 < i < len(lags) - 1:                                   # parabolic refinement
        a, b, c = out[i - 1], out[i], out[i + 1]
        d = 0.5 * (a - c) / (a - 2 * b + c) if (a - 2 * b + c) != 0 else 0.0
        return lags[i] + d * (lags[1] - lags[0]), b
    return lags[i], out[i]


def main():
    lags = np.arange(-0.06, 0.0601, 0.001)
    for take in sys.argv[1:]:
        rid = os.path.basename(take.rstrip("/"))
        imu = {"game_rv": [], "gyro": []}
        for r in csv.DictReader(open(os.path.join(take, rid + "_imu.csv"))):
            if r["sensor"] in imu:
                imu[r["sensor"]].append([int(r["t_ns"]) * 1e-9] + [float(r["v%d" % k]) for k in range(4 if r["sensor"] == "game_rv" else 3)])
        g = np.array(sorted(imu["game_rv"]))
        w = np.array(sorted(imu["gyro"]))
        # 1. body rate from game_rv: dq = q_k^-1 q_k+1 -> 2 * vector part / dt (board frame)
        q = g[:, 1:5]
        qc = q * [-1, -1, -1, 1]
        dq = qmul(qc[:-1], q[1:])
        dq *= np.sign(dq[:, 3:4])
        dt = np.diff(g[:, 0])
        ok = dt > 1e-4
        tr = 0.5 * (g[:-1, 0] + g[1:, 0])[ok]
        rate = (2 * dq[:, :3] / dt[:, None])[ok]
        # the board axis that carries the yaw: the gyro axis with the most variance
        ax = int(np.argmax(w[:, 1:4].var(0)))
        sel = np.abs(w[:, 1 + ax]) > 0.0
        L1, c1 = best_lag(w[sel, 0], w[sel, 1 + ax], tr, rate[:, ax], lags)
        print("%s: yaw axis = board %s; game_rv vs gyro: game_rv is %+.1f ms later (r %.3f); rates %.0f / %.0f Hz" % (
            rid[:15], "xyz"[ax], 1000 * L1, c1, len(g) / (g[-1, 0] - g[0, 0]), len(w) / (w[-1, 0] - w[0, 0])))
        # 2. camera yaw rate from apple bearings (map replay without the IMU lag correction)
        f, meta, events = load_take(take)
        p = load_params()
        p.imu_lag_s = 0.0
        m = TrackMapper(params=p)
        last = {}                                                # landmark id -> (t, bearing)
        vis = []                                                 # (t mid, bearing rate)

        def on_frame(res):
            cur = {}
            for k, lm in res["lm"].items():
                x, y = res["xy"][k]
                if x > 400:
                    cur[lm.id] = (res["t"], math.atan2(y, x))
            rates = []
            for lid, (t, b) in cur.items():
                if lid in last and 0 < t - last[lid][0] < 0.02:
                    rates.append((b - last[lid][1]) / (t - last[lid][0]))
            if len(rates) >= 2:
                vis.append((res["t"] - 0.5 * (1 / 120), -float(np.median(rates))))
            last.clear()
            last.update(cur)
        m.on_frame = on_frame
        for t, _, kind, x in events:
            if kind == "imu":
                m.add_imu(t, *x)
            elif kind == "turns":
                m.add_turns(t, x)
            else:
                m.add_detections(x["t"], x["boxes"], x["crop"], x["seq"], x["sensor_ts"])
            m.step()
        V = np.array(vis)
        # visual yaw rate is noisy: low-pass both the same way (3-frame median then compare)
        sign = np.sign(np.corrcoef(V[:, 1], np.interp(V[:, 0], w[:, 0], w[:, 1 + ax]))[0, 1])
        L2, c2 = best_lag(V[:, 0], V[:, 1], w[:, 0], sign * w[:, 1 + ax], lags)
        L3, c3 = best_lag(V[:, 0], V[:, 1], tr, sign * rate[:, ax], lags)
        print("   camera (%d frame pairs) vs gyro: gyro is %+.1f ms later (r %.3f); vs game_rv: %+.1f ms later (r %.3f)" % (
            len(V), 1000 * L2, c2, 1000 * L3, c3))


if __name__ == "__main__":
    main()
