#!/usr/bin/env python3
"""Steering servo calibration: tune the center pulse and the amplitude, saved automatically.

Run it in a terminal (it needs a TTY):   ssh -t rp5 python3 ~/rc_car/tools/steering_cal.py

Keys
  a / d    center  -/+ 5 us          A / D    center  -/+ 25 us
  [ / ]    amplitude -/+ 5 us        { / }    amplitude -/+ 25 us
  l / r    move to full left / right (center -/+ amplitude) to check the end stops
  c        move to center
  i        toggle invert (swap left/right)
  x        reset to defaults (1500 us / 300 us)
  ?        help      q   quit (servo centred)

Every change is written to config/steering.json immediately (fsync'd). Steering.set(-1..+1)
and Drive.steer() read that file.
"""
import os
import select
import sys
import termios
import tty

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # rc_car/
sys.path.insert(0, ROOT)
from hardware.base import HardwareError  # noqa: E402
from hardware.steering import Steering, DEFAULTS, save_config, CONFIG_PATH  # noqa: E402

STEP, BIG = 5.0, 25.0


def getch(timeout=0.1):
    r, _, _ = select.select([sys.stdin], [], [], timeout)
    return sys.stdin.read(1) if r else None


def main():
    if not sys.stdin.isatty():
        print("needs a TTY: run with  ssh -t rp5 python3 ~/rc_car/tools/steering_cal.py")
        return 2
    try:
        st = Steering()
    except HardwareError as e:
        print("%s\nStop it first (the dashboard keeps re-centring the servo), e.g.:\n"
              "  pkill -f \"python3 dashboard.py\"\n"
              "and restart it afterwards with scripts/dashboard_restart.sh" % e)
        return 1
    print(__doc__)
    print("config:", CONFIG_PATH)
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    tty.setcbreak(fd)

    def show(msg=""):
        print("\rcenter=%4.0f us  amplitude=%3.0f us  invert=%s  -> pulse=%4.0f us  %s"
              % (st.center_us, st.amplitude_us, st.cfg["invert"], st.servo.pulse_us, msg) + " " * 6,
              end="", flush=True)

    def change(**kw):
        st.cfg.update(kw)
        st.cfg["amplitude_us"] = max(0.0, min(1000.0, st.cfg["amplitude_us"]))
        st.cfg["center_us"] = max(500.0, min(2500.0, st.cfg["center_us"]))
        save_config(st.cfg)
        st.set(st.position)  # re-apply the current position with the new numbers
        show("saved")

    show()
    try:
        while True:
            k = getch()
            if k is None:
                continue
            if k == "q":
                break
            elif k == "?":
                print(__doc__)
            elif k in "aAdD":
                d = (STEP if k in "ad" else BIG) * (-1 if k in "aA" else 1)
                change(center_us=st.center_us + d)
            elif k in "[]{}":
                d = (STEP if k in "[]" else BIG) * (-1 if k in "[{" else 1)
                change(amplitude_us=st.amplitude_us + d)
            elif k == "i":
                change(invert=not st.cfg["invert"])
            elif k == "x":
                change(**DEFAULTS)
            elif k == "l":
                st.set(-1.0); show("full left")
            elif k == "r":
                st.set(1.0); show("full right")
            elif k == "c":
                st.center(); show("center")
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)
        st.close()
        print("\nsaved:", st.cfg)


if __name__ == "__main__":
    sys.exit(main())
