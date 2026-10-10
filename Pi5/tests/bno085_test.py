#!/usr/bin/env python3
"""BNO085 functional test through hardware.BNO085 (~1 min). Keep the board still.

    python3 tests/bno085_test.py [--seconds 2] [--no-save]

Checks: worker boots with SCHED_FIFO, product id, FRS metadata of every sensor, every sensor
alone at its max rate (rate + no lost samples), enable_default() at 100 Hz with plausible
values at rest, event sensors accepted, commands (calibration, tare / clear tare), wake-up
from idle, hub reset recovery. Results -> results/bno085_test_<ts>.json.
For load / fault / soak testing see tests/bno085_endurance.py.
"""
import argparse
import json
import math
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # Pi5/
sys.path.insert(0, ROOT)
from hardware import HardwareError  # noqa: E402
from hardware.bno085 import BNO085, SENSORS, MAX_RATE_HZ, TARE_GAME_RV, _PERIODIC  # noqa: E402

CHECKS = []
G = 9.80665


def check(name, ok, detail="", warn=False):
    st = "PASS" if ok else ("WARN" if warn else "FAIL")
    CHECKS.append({"check": name, "status": st, "detail": detail})
    print("  [%s] %-36s %s" % (st, name, detail), flush=True)
    return ok


def norm(v):
    return math.sqrt(sum(x * x for x in v))


def expected_hz(name):
    base = _PERIODIC[name][4]        # raw sensors run at their calibrated base sensor's rate
    return min(MAX_RATE_HZ[name], MAX_RATE_HZ[base]) if base else MAX_RATE_HZ[name]


def measure(imu, names, seconds):
    s0 = {n: imu.stats()["sensors"].get(n, {}) for n in names}
    time.sleep(seconds)
    s1 = imu.stats()["sensors"]
    return {n: {"hz": (s1[n]["samples"] - s0[n].get("samples", 0)) / seconds, "lost": s1[n]["lost"]}
            for n in names}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seconds", type=float, default=2.0, help="measurement time per sensor")
    ap.add_argument("--no-save", action="store_true")
    args = ap.parse_args()
    res = {"time": time.strftime("%Y-%m-%d %H:%M:%S")}

    print("== bring-up")
    t = time.monotonic()
    try:
        imu = BNO085()
    except HardwareError as e:
        check("worker boots", False, str(e))
        return 1
    try:
        check("worker boots", True, "%.0f ms" % ((time.monotonic() - t) * 1e3))
        check("SCHED_FIFO priority", imu.rt_priority, "" if imu.rt_priority else
              "not allowed: run scripts/setup_spi0_mode3.sh and log in again", warn=True)
        res["product"] = imu.product
        check("product id (4 firmware parts)", len(imu.product) == 4,
              ", ".join("%d v%s" % (p["part"], p["version"]) for p in imu.product))

        print("\n== FRS metadata")
        res["metadata"] = {}
        for n in MAX_RATE_HZ:
            m = imu.metadata(n)
            res["metadata"][n] = m and {k: m[k] for k in ("min_period_us", "max_rate_hz", "power_mA")}
            check("metadata %s" % n, bool(m and m["min_period_us"]),
                  "FRS min period %s us" % (m and m["min_period_us"]))

        print("\n== each sensor alone at max rate (%.0f s each)" % args.seconds)
        res["max_rate"] = {}
        for n in MAX_RATE_HZ:
            imu.disable()
            try:
                imu.enable(n)
            except HardwareError as e:
                check("%s max rate" % n, False, str(e))
                continue
            time.sleep(0.5)
            r = measure(imu, [n], args.seconds)[n]
            exp = expected_hz(n)
            res["max_rate"][n] = dict(r, expected_hz=exp)
            # arvr_* run a little below their nominal 400 Hz on this firmware
            check("%s max rate" % n, r["hz"] >= 0.9 * exp and r["lost"] == 0,
                  "%.0f Hz (max %.0f) lost %d" % (r["hz"], exp, r["lost"]))

        print("\n== enable_default() (recommended setup)")
        imu.disable()
        imu.enable_default()
        time.sleep(3.5)                              # the hub re-plans for ~3 s after a change
        r = measure(imu, BNO085.DEFAULT_SENSORS, 3.0)
        res["default"] = r
        check("all at ~100 Hz, nothing lost", all(v["hz"] >= 90 and v["lost"] == 0 for v in r.values()),
              " ".join("%s=%.0f" % (n, v["hz"]) for n, v in r.items()))
        a, g, gy = imu.acceleration, imu.gravity, imu.gyro
        check("|accel| ~ g", abs(norm(a) - G) < 0.6, "%.2f m/s^2" % norm(a))
        check("|gravity| ~ g", abs(norm(g) - G) < 0.6, "%.2f m/s^2" % norm(g))
        check("gyro ~ 0 at rest", norm(gy) < 0.05, "%.4f rad/s" % norm(gy))
        check("|quaternion| = 1", abs(norm(imu.quaternion) - 1) < 0.01, "euler %s" % [round(x, 1) for x in imu.euler])
        mag = norm(imu.magnetic)
        check("|mag| Earth-like", 20 < mag < 70, "%.0f uT (magnets / motors nearby raise it)" % mag, warn=True)

        print("\n== events")
        ev = [s for s in SENSORS if s not in MAX_RATE_HZ]
        imu.enable(*ev)
        time.sleep(1.0)
        st = imu.stats()["sensors"]
        check("event sensors accepted", all(st[n]["applied_hz"] for n in ev),
              " ".join("%s=%s" % (n, st[n]["applied_hz"]) for n in ev), warn=True)
        check("stability classifier reports", imu.stability is not None, str(imu.stability), warn=True)

        print("\n== commands")
        c = imu.calibration()
        check("calibration config read", c is not None and c["status"] == 0, str(c))
        check("calibration config write", imu.set_calibration(accel=False, gyro=True, mag=False))   # never accel / mag: see tools/imu_cal.py
        q0 = imu.game_quaternion
        imu.tare(TARE_GAME_RV)
        time.sleep(0.4)
        q1 = imu.game_quaternion
        imu.clear_tare()
        time.sleep(0.4)
        q2 = imu.game_quaternion
        check("tare -> identity", abs(abs(q1[3]) - 1) < 0.02, "real %.3f" % q1[3])
        check("clear tare -> original", abs(sum(x * y for x, y in zip(q0, q2))) > 0.98)

        print("\n== idle / reset")
        imu.disable()
        time.sleep(2.0)
        t = time.monotonic()
        imu.enable("game_rv")
        check("wake from idle", True, "first sample after %.0f ms" % ((time.monotonic() - t) * 1e3))
        t = time.monotonic()
        imu.reset()
        ok = imu.wait("game_rv", 3.0) is not None
        check("hub reset, config re-applied", ok, "data back after %.0f ms" % ((time.monotonic() - t) * 1e3))
        res["link"] = imu.stats()["link"]
        check("no driver errors", not res["link"].get("errors") and not imu.last_error,
              str({k: v for k, v in res["link"].items() if k.startswith(("errors", "resets", "timeout"))}))
    finally:
        imu.close()

    res["checks"] = CHECKS
    fails = [c for c in CHECKS if c["status"] == "FAIL"]
    print("\n%d checks, %d failed" % (len(CHECKS), len(fails)))
    if not args.no_save:
        os.makedirs(os.path.join(ROOT, "results"), exist_ok=True)
        path = os.path.join(ROOT, "results", "bno085_test_%s.json" % time.strftime("%Y%m%d_%H%M%S"))
        with open(path, "w") as f:
            json.dump(res, f, indent=1, default=str)
        os.sync()
        print("saved", path)
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
