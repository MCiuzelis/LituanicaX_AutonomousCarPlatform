#!/usr/bin/env python3
"""Motor test for the Hobbywing 1080 G2 + brushed motor, verified with the AS5600 encoder.

SAFETY: the car can drive away. This script always prints its plan and asks for an explicit
"yes" before any throttle is applied. Ctrl-C at any time -> throttle neutral immediately.
Put the car on a stand (wheels free) for the first runs. Switch the ESC on by hand first.

    python3 tools/drive_test.py                       # ramp forward throttle from 6 % until the
                                                      # encoder sees motion, hold 2 s, stop
    python3 tools/drive_test.py --us 1400 --seconds 3 # fixed raw pulse instead of a ramp
    python3 tools/drive_test.py --ramp-max 0.30 --ramp-step 0.01 --min-dps 30

The ramp is in forward-throttle fraction (0..1), mapped through the ESC calibration in
hardware/esc.py (reversed on this car: 1000 us = full forward). --us is a raw pulse: below
1500 us is forward, above is brake. The pin is left at neutral when the script exits.
"""
import argparse
import os
import signal
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # Pi5/
sys.path.insert(0, ROOT)
from hardware import HobbywingESC, AS5600, HardwareError  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--us", type=float, default=None, help="fixed raw throttle pulse (us); default = ramp")
    ap.add_argument("--ramp-start", type=float, default=0.06, help="forward throttle fraction to start at")
    ap.add_argument("--ramp-max", type=float, default=0.24, help="never exceed this throttle fraction")
    ap.add_argument("--ramp-step", type=float, default=0.01, help="throttle fraction per step")
    ap.add_argument("--step-seconds", type=float, default=0.3)
    ap.add_argument("--min-dps", type=float, default=30.0, help="encoder speed that counts as moving")
    ap.add_argument("--max-dps", type=float, default=3000.0, help="abort if the encoder exceeds this")
    ap.add_argument("--seconds", type=float, default=2.0, help="hold time once moving / for --us")
    ap.add_argument("--no-encoder", action="store_true")
    ap.add_argument("--yes", action="store_true", help="skip the confirmation prompt (NOT recommended)")
    a = ap.parse_args()

    enc = None
    if not a.no_encoder:
        try:
            enc = AS5600()
            enc.zero()
            print("encoder: magnet=%s agc=%d" % (enc.magnet(), enc.agc()))
        except HardwareError as e:
            print("encoder unavailable (%s) - continuing blind" % e)

    print("\nPLAN  (ESC must already be switched on)")
    print("  1. neutral 1500 us on GPIO13 for 0.5 s")
    if a.us is not None:
        print("  2. throttle %.0f us for %.1f s" % (a.us, a.seconds))
    else:
        print("  2. ramp forward %.0f%% -> max %.0f%% in %.1f%% steps every %.1f s until encoder > %.0f deg/s, then hold %.1f s"
              % (a.ramp_start * 100, a.ramp_max * 100, a.ramp_step * 100, a.step_seconds, a.min_dps, a.seconds))
    print("  3. neutral")
    print("  abort: Ctrl-C, or encoder > %.0f deg/s" % a.max_dps)
    print("\n*** THE MOTOR WILL SPIN. Wheels off the ground? ***")
    if not a.yes:
        if input("type yes to continue: ").strip().lower() != "yes":
            print("aborted, nothing was sent.")
            return 1

    esc = HobbywingESC()
    state = {"peak": 0.0, "moved_at_us": None}

    def stop(*_):
        esc.neutral()
        raise KeyboardInterrupt

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)

    def watch(seconds, label):
        """Hold the current throttle for `seconds`, sampling the encoder. Returns peak dps."""
        t0 = time.monotonic()
        peak = 0.0
        while time.monotonic() - t0 < seconds:
            if enc is not None:
                r = enc.read()
                v = abs(r.velocity_dps)
                peak = max(peak, v)
                state["peak"] = max(state["peak"], v)
                if v > a.max_dps:
                    esc.neutral()
                    raise RuntimeError("encoder %.0f deg/s exceeds limit %.0f - throttle cut" % (v, a.max_dps))
                print("\r  %s %5.0f us  encoder %+8.1f deg/s  turns %+7.2f   "
                      % (label, esc.current_us, r.velocity_dps, r.turns), end="", flush=True)
            time.sleep(0.01)
        return peak

    try:
        esc.neutral()
        time.sleep(0.5)
        if a.us is not None:
            esc.pulse_us(a.us)
            peak = watch(a.seconds, "fixed")
            print("\n  peak %.0f deg/s" % peak)
        else:
            frac = a.ramp_start
            while frac <= a.ramp_max + 1e-9:
                us = esc.throttle(frac)
                peak = watch(a.step_seconds, "ramp ")
                if enc is None or peak >= a.min_dps:
                    state["moved_at_us"] = us
                    break
                frac += a.ramp_step
            if state["moved_at_us"] is None:
                print("\n  no motion detected up to %.0f%% throttle (encoder peak %.0f deg/s)"
                      % (a.ramp_max * 100, state["peak"]))
            else:
                print("\n  motion at %.0f us -> holding %.1f s" % (state["moved_at_us"], a.seconds))
                watch(a.seconds, "hold ")
        esc.neutral()
        print("\nneutral. waiting for the motor to stop...")
        watch(1.5, "coast")
        print()
    except KeyboardInterrupt:
        esc.neutral()
        print("\ninterrupted -> neutral")
    except RuntimeError as e:
        print("\nABORT:", e)
    finally:
        esc.close()   # neutral, signal kept running
        if enc is not None:
            r = enc.read()
            print("summary: peak %.0f deg/s, total %+.2f turns, moved at %s us"
                  % (state["peak"], r.turns, state["moved_at_us"]))
            enc.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
