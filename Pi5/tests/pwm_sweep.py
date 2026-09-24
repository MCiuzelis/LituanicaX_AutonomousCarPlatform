#!/usr/bin/env python3
"""Hardware PWM sweep test for RC servos on RP1 PWM0 (50 Hz).

Sweeps the pulse width min -> max -> min on the given pins, reads back the sysfs state at
every step, and estimates the real duty cycle by sampling the pad level with pinctrl so that
a wrong mux / dead channel is caught even without a scope.

SAFETY: GPIO13 is the ESC throttle on this car - a 1000..2000 us sweep there is full throttle
to full brake. It is refused unless --allow-esc-pin is given (motor/battery disconnected!).
Default is the steering servo on GPIO12 only, over its default 1000..2000 us range.

    python3 tests/pwm_sweep.py                       # 1000..2000 us, 100 us steps, 2 cycles
    python3 tests/pwm_sweep.py --min 500 --max 2500 --step 50 --dwell 0.15 --cycles 3
    python3 tests/pwm_sweep.py --pins 12 --mode angle --step 15      # angle sweep -90..90
    python3 tests/pwm_sweep.py --no-verify                           # fast, no pad sampling
    python3 tests/pwm_sweep.py --smooth 5                            # continuous 5 s ramps
"""
import argparse
import json
import os
import sys
import time
from datetime import datetime

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # Pi5/
sys.path.insert(0, ROOT)
from hardware import Servo, HardwareError, PWM_PINS  # noqa: E402
from hardware.esc import ESC_PWM_GPIO  # noqa: E402
from hardware.base import run_cmd  # noqa: E402

FRAME_US = 20_000.0  # 50 Hz


def pad_high_fraction(gpio, samples):
    """Fraction of pinctrl samples that see the pad high (~ duty cycle)."""
    hi = 0
    for _ in range(samples):
        _, out = run_cmd(["pinctrl", "get", str(gpio)])
        if "| hi" in out:
            hi += 1
    return hi / samples


def build_steps(a):
    if a.mode == "angle":
        vals, v = [], a.min_angle
        while v <= a.max_angle + 1e-9:
            vals.append(round(v, 2))
            v += a.step
    else:
        vals, v = [], a.min
        while v <= a.max + 1e-9:
            vals.append(round(v, 1))
            v += a.step
    return vals + vals[-2::-1]  # up then back down (no repeated end point)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pins", default="12", help="comma list of GPIOs (%s)" % sorted(PWM_PINS))
    ap.add_argument("--allow-esc-pin", action="store_true",
                    help="permit GPIO%d (ESC throttle) - only with the motor disconnected" % ESC_PWM_GPIO)
    ap.add_argument("--mode", choices=("pulse", "angle"), default="pulse")
    ap.add_argument("--min", type=float, default=1000.0, help="min pulse us (pulse mode)")
    ap.add_argument("--max", type=float, default=2000.0, help="max pulse us (pulse mode)")
    ap.add_argument("--min-angle", type=float, default=-90.0)
    ap.add_argument("--max-angle", type=float, default=90.0)
    ap.add_argument("--step", type=float, default=None, help="us or degrees per step (default 100 us / 15 deg)")
    ap.add_argument("--dwell", type=float, default=0.2, help="seconds to hold each step")
    ap.add_argument("--cycles", type=int, default=2)
    ap.add_argument("--smooth", type=float, default=0.0,
                    help="if >0: instead of steps, ramp continuously taking this many seconds per direction")
    ap.add_argument("--samples", type=int, default=200, help="pinctrl samples per step for duty verification")
    ap.add_argument("--tol", type=float, default=0.03, help="allowed |measured - expected| duty (fraction)")
    ap.add_argument("--no-verify", action="store_true")
    ap.add_argument("--out", default=None, help="results JSON (default results/pwm_sweep_<ts>.json)")
    a = ap.parse_args()
    if a.step is None:
        a.step = 15.0 if a.mode == "angle" else 100.0
    pins = [int(p) for p in a.pins.split(",") if p]
    if ESC_PWM_GPIO in pins and not a.allow_esc_pin:
        print("refusing GPIO%d: it drives the ESC throttle. Disconnect the motor and pass --allow-esc-pin."
              % ESC_PWM_GPIO)
        return 2

    print("PWM sweep: pins=%s mode=%s range=%s step=%s dwell=%.2fs cycles=%d verify=%s"
          % (pins, a.mode, (a.min_angle, a.max_angle) if a.mode == "angle" else (a.min, a.max),
             a.step, a.dwell, a.cycles, not a.no_verify))

    servos = {}
    results = {"started": datetime.now().isoformat(timespec="seconds"), "args": vars(a), "pins": {}, "steps": []}
    fails = 0
    try:
        for g in pins:
            s = Servo(g, min_us=a.min, max_us=a.max, min_angle=a.min_angle, max_angle=a.max_angle)
            servos[g] = s
            st = s.status()
            results["pins"][g] = st
            ok = st["sysfs_period"] == "20000000" and st["sysfs_enable"] == "1"
            print("  GPIO%d -> %s ch%d  period=%s ns enable=%s  %s"
                  % (g, st["chip"], st["channel"], st["sysfs_period"], st["sysfs_enable"], "OK" if ok else "BAD"))
            if not ok:
                fails += 1

        if a.smooth > 0:
            # continuous ramp: update as fast as sysfs allows
            for cyc in range(a.cycles):
                for direction in (1, -1):
                    t0 = time.monotonic()
                    while True:
                        f = min(1.0, (time.monotonic() - t0) / a.smooth)
                        frac = f if direction == 1 else 1.0 - f
                        for s in servos.values():
                            s.set_fraction(frac)
                        if f >= 1.0:
                            break
                        time.sleep(0.005)
                    print("  cycle %d ramp %s done (%.1f s)" % (cyc + 1, "up" if direction == 1 else "down", a.smooth))
        else:
            steps = build_steps(a)
            for cyc in range(a.cycles):
                print("--- cycle %d/%d ---" % (cyc + 1, a.cycles))
                for val in steps:
                    t_set = time.monotonic()
                    row = {"cycle": cyc + 1, "value": val, "pins": {}}
                    for g, s in servos.items():
                        us = s.set_angle(val) if a.mode == "angle" else s.set_pulse_us(val)
                        row["pins"][g] = {"pulse_us": us, "sysfs_duty": int(s.pwm._r("duty_cycle"))}
                    time.sleep(a.dwell)
                    line = "  %-7s" % (("%6.1f deg" % val) if a.mode == "angle" else ("%6.0f us" % val))
                    for g, s in servos.items():
                        r = row["pins"][g]
                        expect = r["pulse_us"] / FRAME_US
                        sysfs_ok = r["sysfs_duty"] == int(round(r["pulse_us"] * 1000))
                        line += "  GPIO%d: %4.0f us sysfs=%s" % (g, r["pulse_us"], "ok" if sysfs_ok else "MISMATCH")
                        if not sysfs_ok:
                            fails += 1
                        if not a.no_verify:
                            meas = pad_high_fraction(g, a.samples)
                            r["duty_expected"], r["duty_measured"] = round(expect, 4), round(meas, 4)
                            ok = abs(meas - expect) <= a.tol
                            r["ok"] = ok
                            if not ok:
                                fails += 1
                            line += " duty %.1f%% (exp %.1f%%) %s" % (meas * 100, expect * 100, "ok" if ok else "FAIL")
                    row["elapsed_s"] = round(time.monotonic() - t_set, 3)
                    results["steps"].append(row)
                    print(line)

        for s in servos.values():
            s.center()
        time.sleep(0.3)
    except HardwareError as e:
        print("HARDWARE ERROR:", e)
        fails += 1
    finally:
        for s in servos.values():
            s.close()

    results["failures"] = fails
    out = a.out or os.path.join(ROOT, "results",
                                "pwm_sweep_%s.json" % datetime.now().strftime("%Y%m%d_%H%M%S"))
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print("\n%s  (%d steps, %d failures) -> %s" % ("PASS" if fails == 0 else "FAIL", len(results["steps"]), fails, out))
    return 0 if fails == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
