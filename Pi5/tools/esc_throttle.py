#!/usr/bin/env python3
"""Send raw throttle pulses to the ESC from the keyboard (ESC calibration / bench checks).

Run it in a terminal (it needs a TTY):   ssh -t rp5 python3 ~/LituanicaX_InteligentRobotCompetition/Pi5/tools/esc_throttle.py

Keys
  n / f / b   neutral / full forward / full brake (1500 / 1000 / 2000 us - the throttle is
              reversed on this car, see hardware/esc.py)
  + / -       pulse +/- 25 us            0   stop sending pulses (no signal)
  ?           help                       q   quit (pulse back to neutral)

1080 G2 throttle-range calibration (ESC buttons are pressed by hand, the Pi only sends pulses):
  ESC off. Hold SET, switch the ESC on, release SET when the LED flashes. Then:
  n (neutral) -> press SET,  f (full forward) -> press SET,  b (full brake) -> press SET.

*** f / b / +/- WILL SPIN THE MOTOR once the ESC is calibrated - wheels off the ground. ***
"""
import os
import select
import sys
import termios
import tty

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # Pi5/
sys.path.insert(0, ROOT)
from hardware import HobbywingESC  # noqa: E402

STEP_US = 25


def getch(timeout=0.1):
    r, _, _ = select.select([sys.stdin], [], [], timeout)
    return sys.stdin.read(1) if r else None


def main():
    if not sys.stdin.isatty():
        print("needs a TTY: run with  ssh -t rp5 python3 ~/LituanicaX_InteligentRobotCompetition/Pi5/tools/esc_throttle.py")
        return 2
    esc = HobbywingESC()
    print(__doc__)
    print("ready. pulse=%.0f us" % esc.current_us)
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    tty.setcbreak(fd)

    def say(msg):
        print("\r" + msg + " " * 10)

    def current():
        return esc.current_us if esc.current_us is not None else esc.neutral_us

    try:
        while True:
            k = getch()
            if k is None:
                continue
            if k == "q":
                break
            elif k == "?":
                print(__doc__)
            elif k == "n":
                esc.neutral()
                say("neutral %.0f us" % esc.current_us)
            elif k == "f":
                say("FULL FORWARD %.0f us" % esc.pulse_us(esc.full_us))
            elif k == "b":
                say("FULL BRAKE %.0f us" % esc.pulse_us(esc.brake_us))
            elif k in "+-":
                say("pulse %.0f us" % esc.pulse_us(current() + (STEP_US if k == "+" else -STEP_US)))
            elif k == "0":
                esc.signal_off()
                say("signal OFF")
            else:
                say("unknown key %r (? for help)" % k)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)
        esc.close()
        print("\npulse at neutral, exiting.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
