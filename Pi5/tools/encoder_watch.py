#!/usr/bin/env python3
"""AS5600 encoder monitor: prints angle / cumulative turns / velocity while you turn the shaft.

    python3 tools/encoder_watch.py [--seconds 10] [--rate 100] [--zero]
"""
import argparse
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # Pi5/
sys.path.insert(0, ROOT)
from hardware import AS5600  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--seconds", type=float, default=10)
ap.add_argument("--rate", type=float, default=100, help="poll rate Hz")
ap.add_argument("--zero", action="store_true", help="take the current position as 0")
a = ap.parse_args()

e = AS5600()
if a.zero:
    e.zero()
r = e.read()
print("AS5600 on i2c-%d 0x%02x  magnet=%s agc=%d magnitude=%d  start angle=%.2f deg"
      % (e.bus_num, e.address, r.magnet, r.agc, r.magnitude, r.angle_deg))
t0 = time.monotonic()
n = 0
vmax = 0.0
last_print = 0.0
while time.monotonic() - t0 < a.seconds:
    r = e.read()
    n += 1
    vmax = max(vmax, abs(r.velocity_dps))
    if time.monotonic() - last_print >= 0.1:
        print("\rangle=%7.2f deg  turns=%+8.3f  vel=%+8.1f dps  magnet=%-6s agc=%2d   "
              % (r.angle_deg, r.turns, r.velocity_dps, r.magnet, r.agc), end="", flush=True)
        last_print = time.monotonic()
    time.sleep(1.0 / a.rate)
print("\ndone: %d reads (%.0f Hz), final turns=%+.3f, peak |velocity|=%.0f dps"
      % (n, n / a.seconds, r.turns, vmax))
e.close()
