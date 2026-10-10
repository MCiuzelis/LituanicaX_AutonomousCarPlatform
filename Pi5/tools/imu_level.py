#!/usr/bin/env python3
"""Capture the IMU level reference: the car's current pose becomes roll = pitch = 0.

    python3 ~/LituanicaX_InteligentRobotCompetition/Pi5/tools/imu_level.py [--show]

Put the car on level ground in its nominal pose and keep it still. Writes config/imu.json
(the dashboard's "set level" button does the same while it owns the IMU; stop it first).
--show only prints the current car-frame attitude with the saved reference.
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from hardware import BNO085, CarFrame  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--show", action="store_true", help="print attitude, do not change the reference")
    a = ap.parse_args()
    frame = CarFrame.load()
    with BNO085() as imu:
        imu.enable_default()
        time.sleep(1.0)
        if not a.show:
            frame.set_level(CarFrame.capture_gravity(imu))
            frame.save()
            print("saved %s: %s" % (frame.path, frame.status()))
        roll, pitch, _ = frame.attitude(imu.game_quaternion)
        print("roll %.2f deg (right down +)  pitch %.2f deg (nose up +)  accel car (fwd, left, up) %s"
              % (roll, pitch, tuple(round(x, 2) for x in frame.to_car(imu.acceleration))))


if __name__ == "__main__":
    main()
