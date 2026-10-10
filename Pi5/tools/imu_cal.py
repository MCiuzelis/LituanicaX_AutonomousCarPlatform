#!/usr/bin/env python3
"""BNO085 on-hub calibration (CEVA procedure: accel by still poses, gyro by staying still),
verified before anything is written to the hub's flash.

    python3 ~/LituanicaX_InteligentRobotCompetition/Pi5/tools/imu_cal.py            # calibrate + verify, never saves
    python3 ~/LituanicaX_InteligentRobotCompetition/Pi5/tools/imu_cal.py --save     # ... and saves the result (DCD) to flash if verify passes

After a successful --save run tools/imu_accel_check.py (offsets ~0) and tools/imu_level.py.
Stop the dashboard first (it owns the IMU). Run it in an interactive ssh session to see the
live status line.

Phase 1 CALIBRATE: rest the car still in 6+ different orientations, 4 s each (wheels down, roof
  down, nose up, nose down, left side down, right side down); wait for ">>> pose N captured"
  before moving. Ends when accel accuracy reaches 3 (high) with 6+ poses, or on timeout.
Phase 2 GYRO: 15 s countdown to put the car on the floor, then it must be wheels down and still
  for 10 s uninterrupted (any movement restarts the 10 s).
Phase 3 VERIFY: 10 s countdown to pick it up, then 4-6 new still poses (include roof down and
  nose up); each must read |a| = 9.81 +- 0.25 m/s^2.
All timings: see --help.
Magnetometer: not calibrated (never used on this car; the motor disturbs it).
"""
import argparse
import math
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from hardware import BNO085  # noqa: E402

G = 9.80665
MAX_W, MIN_SEP_DEG, TOL = 0.03, 25.0, 0.25
UP_COS = 0.90          # wheels down: board -x within ~25 deg of up (mount ~12 deg + uncalibrated accel ~9 deg)


def norm(v):
    return math.sqrt(sum(x * x for x in v))


class Poses:
    def __init__(self, still_s):
        self.buf, self.poses, self.still_s = [], [], still_s

    def feed(self, a, w):
        """Returns a newly captured still pose (mean accel) or None."""
        if norm(w) > MAX_W:
            self.buf = []
            return None
        self.buf.append(a)
        if len(self.buf) < self.still_s * 100:
            return None
        m = np.mean(self.buf, axis=0)
        self.buf = []
        u = m / np.linalg.norm(m)
        if all(math.degrees(math.acos(np.clip(u @ p / np.linalg.norm(p), -1, 1))) > MIN_SEP_DEG for p in self.poses):
            self.poses.append(m)
            return m
        return None


def countdown(seconds, msg):
    for s in range(int(seconds), 0, -1):
        print("\r  %s: %2d s " % (msg, s), end="", flush=True)
        time.sleep(1.0)
    print("\r  %s: go      " % msg, flush=True)


def hold_wheels_down(imu, hold_s, timeout):
    """Wait until the car is wheels down and still, then for hold_s of uninterrupted stillness.
    Any movement restarts the hold. True when held, False on timeout."""
    t_end, since, last_print = time.monotonic() + timeout, None, 0.0
    while time.monotonic() < t_end:
        r = imu.wait("accel", timeout=0.5)
        w = imu.gyro
        if r is None or w is None:
            continue
        a = r.values[:3]
        upright = -a[0] / max(norm(a), 1e-6) > UP_COS
        still = norm(w) < MAX_W
        now = time.monotonic()
        if upright and still:
            since = since or now
            if now - since >= hold_s:
                print("\r  gyro: held still %.0f s                                        " % hold_s, flush=True)
                return True
        else:
            since = None
        if now - last_print > 0.5:
            last_print = now
            state = ("still %4.1f / %.0f s" % (now - since, hold_s)) if since else \
                    ("not wheels down" if not upright else "moving - keep it still")
            print("\r  gyro: %-28s gyro %.3f rad/s   %3.0f s left   " % (state, norm(w), t_end - now),
                  end="", flush=True)
    print(flush=True)
    return False


def run_phase(imu, name, seconds, done, still_s):
    p, t_end, last_print, acc = Poses(still_s), time.monotonic() + seconds, 0.0, 0
    print("\n== %s (%d s max, %.0f s still per pose)" % (name, seconds, still_s), flush=True)
    while time.monotonic() < t_end:
        r = imu.wait("accel", timeout=0.5)
        w = imu.gyro
        if r is None or w is None:
            continue
        acc = r.accuracy
        m = p.feed(r.values[:3], w)
        if m is not None:
            n = np.linalg.norm(m)
            print("\n  >>> pose %d captured: accel %s  |a| %.2f%s  -> move to the next orientation"
                  % (len(p.poses), np.round(m, 2), n, "" if abs(n - G) < TOL else "  (off)"), flush=True)
        now = time.monotonic()
        if now - last_print > 0.5:
            last_print = now
            state = "moving" if norm(w) > MAX_W else "still %.1f/%.0f s" % (len(p.buf) / 100, still_s)
            print("\r  accel accuracy %d/3  |a| %5.2f  %-16s  poses %d  %3.0f s left   "
                  % (acc, norm(r.values[:3]), state, len(p.poses), t_end - now), end="", flush=True)
        if done(p, acc):
            break
    print(flush=True)
    return p.poses, acc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--save", action="store_true", help="save the calibration to flash if verify passes")
    ap.add_argument("--still-seconds", type=float, default=4.0, help="stillness needed per pose")
    ap.add_argument("--min-poses", type=int, default=6, help="calibration poses before the accel phase may end")
    ap.add_argument("--place-seconds", type=float, default=15.0, help="time to put the car down before the gyro step")
    ap.add_argument("--gyro-seconds", type=float, default=10.0, help="uninterrupted wheels-down stillness for the gyro")
    ap.add_argument("--pickup-seconds", type=float, default=10.0, help="pause before the verify phase")
    ap.add_argument("--cal-seconds", type=float, default=300)
    ap.add_argument("--verify-seconds", type=float, default=150)
    a = ap.parse_args()
    with BNO085() as imu:
        imu.enable("accel", "gyro", "game_rv", rate_hz=100)
        imu.set_periodic_calibration_save(False)    # the hub auto-saves DCD by default: only --save writes
        if not imu.set_calibration(accel=True, gyro=True, mag=False):
            sys.exit("could not enable dynamic calibration")
        print("dynamic calibration:", imu.calibration())
        time.sleep(1.0)
        cal, acc = run_phase(imu, "CALIBRATE: %d+ still orientations (all 6 sides)" % a.min_poses,
                             a.cal_seconds, lambda p, acc: acc == 3 and len(p.poses) >= a.min_poses,
                             a.still_seconds)
        if acc != 3:
            print("  accel accuracy only %d after %d poses: the hub did not converge" % (acc, len(cal)))

        print("\n== GYRO: put the car on the floor, wheels down, and let go", flush=True)
        countdown(a.place_seconds, "place the car")
        if not hold_wheels_down(imu, a.gyro_seconds, timeout=120):
            print("  gyro step timed out (car not wheels down and still)")

        print("\n== VERIFY next: pick the car up", flush=True)
        countdown(a.pickup_seconds, "verify starts in")
        ver, _ = run_phase(imu, "VERIFY: 4+ NEW still orientations (include roof down and nose up)",
                           a.verify_seconds, lambda p, acc: len(p.poses) >= 6, a.still_seconds)
        imu.set_calibration(accel=False, gyro=False, mag=False)   # stop learning; RAM only
        n = [np.linalg.norm(m) for m in ver]
        ok = len(ver) >= 4 and all(abs(x - G) < TOL for x in n)
        print("\nverify: %d poses, |a| %s -> %s" % (len(ver), [round(x, 2) for x in n], "PASS" if ok else "FAIL"))
        if a.save and ok:
            print("saving calibration (DCD) to flash:", "ok" if imu.save_calibration() else "FAILED")
        elif a.save:
            print("not saved (verify failed). The hub keeps its previous flash calibration after a reset.")
        else:
            print("not saved (dry run). Re-run with --save to keep it.")


if __name__ == "__main__":
    main()
