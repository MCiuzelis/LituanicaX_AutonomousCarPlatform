#!/usr/bin/env python3
"""Odometry scale: push the car exactly 4 m straight, count motor encoder turns -> mm per turn.

The track map (vision/track_map.py) converts encoder turns into distance with config/odometry.json
mm_per_turn; every apple position along the track scales with it.

    ssh -t rp5 python3 ~/LituanicaX_InteligentRobotCompetition/Pi5/tools/odometry_cal.py

Car at the start mark, Enter; push it straight 4 m by hand (ESC off or at neutral: the motor just
turns with the wheels), Enter. Repeat as often as you like; q saves the median to
config/odometry.json. Works with the dashboard running (reads drive.turns from /api/telemetry) or
without it (reads the AS5600 directly).
"""
import json
import os
import statistics
import sys
import threading
import time
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # Pi5/
sys.path.insert(0, ROOT)
from vision.track_map import DEFAULT_ODO, ODO_CONFIG, load_json, write_json   # noqa: E402

DISTANCE_MM = 4000.0
URL = "http://localhost:8080/api/telemetry"


def _dashboard_turns():
    with urllib.request.urlopen(URL, timeout=1.0) as r:
        return float(json.load(r)["drive"]["turns"])


class Encoder:
    """Motor turns: from the running dashboard, else from the AS5600 itself."""

    def __init__(self):
        try:
            _dashboard_turns()
            self.enc = None
            print("reading the encoder through the running dashboard")
        except OSError:
            from hardware.encoder import AS5600
            self.enc = AS5600()
            self.enc.zero()
            self.turns = 0.0
            threading.Thread(target=self._spin, daemon=True).start()
            print("dashboard not running: reading the AS5600 directly")

    def _spin(self):
        while True:
            _, self.turns, _ = self.enc.read_fast()

    def read(self):
        """Turns with the car standing still (waits until the value has settled)."""
        get = _dashboard_turns if self.enc is None else (lambda: self.turns)
        prev = get()
        while True:
            time.sleep(0.5)             # dashboard telemetry refreshes at 4 Hz
            cur = get()
            if abs(cur - prev) < 0.02:
                return cur
            prev = cur


def main():
    enc = Encoder()
    runs = []
    while True:
        if input("\nrun %d: car at the start mark, Enter (q = save and quit) " % (len(runs) + 1)).strip().lower() == "q":
            break
        t0 = enc.read()
        input("push it straight %.0f m, stop, Enter " % (DISTANCE_MM / 1000))
        turns = abs(enc.read() - t0)
        if turns < 10:
            print("only %.2f turns: the encoder did not see it, run ignored" % turns)
            continue
        runs.append(DISTANCE_MM / turns)
        print("  %.2f motor turns -> %.3f mm/turn" % (turns, runs[-1]))
    if not runs:
        return
    k = statistics.median(runs)
    print("\nmm_per_turn %.3f (median of %d run%s, spread %.1f %%)"
          % (k, len(runs), "s" * (len(runs) > 1), (max(runs) - min(runs)) / k * 100))
    odo = load_json(ODO_CONFIG, DEFAULT_ODO)
    odo.update(mm_per_turn=round(k, 3), runs_mm_per_turn=[round(x, 3) for x in runs],
               source="tools/odometry_cal.py: pushed %.0f mm, %s" % (DISTANCE_MM, time.strftime("%Y-%m-%dT%H:%M:%S")))
    write_json(ODO_CONFIG, odo)
    print("written %s - restart the dashboard (scripts/dashboard_restart.sh) to use it" % ODO_CONFIG)


if __name__ == "__main__":
    main()
