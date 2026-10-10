#!/usr/bin/env python3
"""BNO085 accelerometer multi-pose test. Read-only: turns the hub's dynamic calibration off (RAM only,
undone by the next reset / dashboard start) and never saves anything to the hub's flash.

    python3 ~/LituanicaX_InteligentRobotCompetition/Pi5/tools/imu_accel_check.py [seconds]

While it runs, rest the car (still!) in different orientations for ~3 s each: wheels down, roof
down, nose up, nose down, left side down, right side down, plus a few in-between angles. The board
may be tilted in the car: no pose has to line up with a board axis. Still poses at least 25 deg
apart are captured automatically (up to 10).

A healthy accelerometer reads |a| = 9.81 +- 0.2 m/s^2 in every pose. With 6+ well spread poses a
per-axis offset + scale is fitted (axis-aligned ellipsoid): reading = scale * true + offset.
"""
import math
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from hardware import BNO085  # noqa: E402

G = 9.80665
DUR = float(sys.argv[1]) if len(sys.argv) > 1 else 150.0
STILL_S, MAX_W, MIN_SEP_DEG, MAX_POSES = 1.5, 0.03, 25.0, 10


def fit(P):
    """Axis-aligned ellipsoid sum c_i (a_i - b_i)^2 = k through the poses -> offset b, scale s."""
    A = np.hstack([P ** 2, P])
    c, *_ = np.linalg.lstsq(A, np.ones(len(P)), rcond=None)
    q, l = c[:3], c[3:]
    if np.any(q <= 0):
        return None
    b = -l / (2 * q)
    k = 1 + np.sum(q * b ** 2)
    return b, np.sqrt(k / q) / G


def main():
    poses = []
    with BNO085() as imu:
        imu.set_calibration(accel=False, gyro=False, mag=False)
        imu.enable("accel", "gyro", rate_hz=100)
        time.sleep(1.5)
        print("go: rest the car still in different orientations, ~3 s each (%.0f s)" % DUR, flush=True)
        buf, t_end = [], time.monotonic() + DUR
        while time.monotonic() < t_end and len(poses) < MAX_POSES:
            r = imu.wait("accel", timeout=0.5)
            w = imu.gyro
            if r is None or w is None:
                continue
            if math.sqrt(sum(x * x for x in w)) > MAX_W:
                buf = []
                continue
            buf.append(r.values[:3])
            if len(buf) < STILL_S * 100:
                continue
            m = np.mean(buf, axis=0)
            buf = []
            u = m / np.linalg.norm(m)
            if all(math.degrees(math.acos(np.clip(u @ p / np.linalg.norm(p), -1, 1))) > MIN_SEP_DEG for p in poses):
                poses.append(m)
                print("  pose %d: accel %s  |a| %.2f%s" % (len(poses), np.round(m, 2), np.linalg.norm(m),
                      "" if abs(np.linalg.norm(m) - G) < 0.3 else "  <-- BAD"), flush=True)
    if not poses:
        print("no still poses captured")
        return
    P = np.array(poses)
    n = np.linalg.norm(P, axis=1)
    print("\n%d poses, |a| %.2f .. %.2f m/s^2 (healthy: all 9.6 .. 10.0)" % (len(P), n.min(), n.max()))
    if len(P) < 6:
        print("need 6+ spread poses for the per-axis fit")
        return
    f = fit(P)
    if f is None:
        print("fit failed: poses not spread enough (include nose up / down and both sides)")
        return
    b, s = f
    resid = np.linalg.norm((P - b) / s, axis=1) - G
    print("per axis: reading = scale * true + offset   (fit residual rms %.2f m/s^2)" % np.sqrt(np.mean(resid ** 2)))
    for i, a in enumerate("xyz"):
        ok = abs(s[i] - 1) < 0.03 and abs(b[i]) < 0.5
        print("  %s: scale %.3f  offset %+.2f m/s^2  %s" % (a, s[i], b[i], "ok" if ok else "FAULTY"))


if __name__ == "__main__":
    main()
