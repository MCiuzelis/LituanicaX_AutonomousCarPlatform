#!/usr/bin/env python3
"""Which BNO085 output reports a rotation first? Records raw_gyro (the sensor's own samples, no hub
processing), gyro (calibrated), game_rv (fusion, what the track map uses) and girv (gyro-integrated
rotation vector: CEVA's low-latency output for head tracking) together while you turn the car by hand,
then cross-correlates each stream's yaw rate with raw_gyro's.

    sudo systemctl stop ... / stop the dashboard first (it owns the IMU), then:
    python3 tools/imu_latency_test.py [--seconds 25] [--rate 200]

Turn the car left / right about its vertical axis by hand, briskly (1-3 swings a second), for the
whole recording. Background: replays of 2026-10-11 put game_rv ~21 ms behind the camera's frames
and the gyro report ~17 ms (tools/analysis/imu_lag.py); the map corrects it with imu_lag_s.
Writes results/imu_latency/<time>.csv (every sample) next to the printed lags.
"""
import argparse
import csv
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from hardware import BNO085                                           # noqa: E402
from tools.analysis.imu_lag import best_lag, qmul                     # noqa: E402

STREAMS = ("raw_gyro", "gyro", "game_rv", "girv")


def quat_rate(t, q):
    """Body rate (board frame, rad/s) between consecutive quaternions (i, j, k, real)."""
    qc = q * [-1, -1, -1, 1]
    dq = qmul(qc[:-1], q[1:])
    dq *= np.sign(dq[:, 3:4])
    dt = np.diff(t)
    ok = dt > 1e-4
    return (0.5 * (t[:-1] + t[1:]))[ok], (2 * dq[:, :3] / dt[:, None])[ok]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seconds", type=float, default=25.0)
    ap.add_argument("--rate", type=float, default=200.0)
    a = ap.parse_args()
    rows = []
    with BNO085() as imu:
        imu.enable(*STREAMS, rate_hz=a.rate)
        sub = imu.subscribe(*STREAMS)
        time.sleep(1.0)
        sub.drain()
        print("go: turn the car left / right by hand for %.0f s" % a.seconds, flush=True)
        t_end = time.monotonic() + a.seconds
        while time.monotonic() < t_end:
            for r in sub.drain():
                rows.append((r.sensor, r.timestamp, list(r.values)))
            time.sleep(0.02)
        print("done", flush=True)
    out_dir = os.path.join(ROOT, "results", "imu_latency")
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, time.strftime("%Y%m%d_%H%M%S") + ".csv")
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["sensor", "t", "values"])
        for s, t, v in rows:
            w.writerow([s, "%.6f" % t, " ".join("%.6g" % x for x in v)])
    data = {s: (np.array([t for n, t, _ in rows if n == s]), np.array([v for n, _, v in rows if n == s], float))
            for s in STREAMS}
    rates = {}
    for s in ("raw_gyro", "gyro"):
        rates[s] = (data[s][0], data[s][1][:, :3])
    rates["game_rv"] = quat_rate(data["game_rv"][0], data["game_rv"][1][:, :4])
    rates["girv (quaternion)"] = quat_rate(data["girv"][0], data["girv"][1][:, :4])
    rates["girv (its rate)"] = (data["girv"][0], data["girv"][1][:, 4:7])
    t_ref, X = rates["gyro"]
    ax = int(np.argmax(X.var(0)))                              # the axis that was turned
    lags = np.arange(-0.04, 0.0401, 0.0005)
    print("%s (%d samples): yaw = board %s axis" % (path, len(rows), "xyz"[ax]))
    for name, (t, R) in rates.items():
        if len(t) < 50:
            print("  %-18s no data" % name)
            continue
        # every stream against raw_gyro (raw units: only the correlation matters, sign fixed by it)
        tr, Xr = rates["raw_gyro"]
        sign = np.sign(np.corrcoef(np.interp(tr, t, R[:, ax]), Xr[:, ax])[0, 1]) or 1.0
        L, c = best_lag(tr, Xr[:, ax], t, sign * R[:, ax], lags)
        print("  %-18s %5.0f Hz, %+5.1f ms behind raw_gyro (r %.3f)" % (name, len(t) / (t[-1] - t[0]), 1000 * L, c))


if __name__ == "__main__":
    main()
