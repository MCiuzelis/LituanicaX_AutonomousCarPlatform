"""Hold the steering at full lock so the road-wheel angle can be measured.

Why this exists
---------------
`max_steer_rad = 0.488` (28 deg) in the sim is the one drivetrain number that was
never checked against the car.  It matters more than it looks: the policy plans a
line, and the line it can plan is bounded by tan(delta)/wheelbase.  Recovering the
yaw rate from the depth-camera image flow in a logged run put the real car at
1.1-1.7 1/m of curvature at steer = 1.0, against the sim's 2.37 1/m — a minimum
turn radius 40-100 % larger than the one the policy learned to use.  That is
and the obvious suspect was SERVO_AMPLITUDE = 0.35 not reaching 28 deg.

An image-flow estimate is an estimate.  This script replaced it with a protractor, and
the protractor said the suspect was innocent (2026-08-02):

    200 us -> 16.0 deg    250 us -> 18.4 deg    300 us -> 23.5 deg    350 us -> 29.0 deg

A full-lock command at SERVO_AMPLITUDE = 0.35 is 350 us, so delta_max is 28-29 deg — the
geometry was right all along.  The car only PULLS 1.1-1.7 1/m out of a kinematic 2.90,
which makes the gap tyre slip, not linkage.  Re-run this after any change to the servo
horn, the linkage, or SERVO_AMPLITUDE; use the skid pad for the grip half of the story.

It drives the servo through the EXACT mapping `Drive._set_throttle`'s sibling
`Drive._steering_pulse_ns` uses — same SERVO_DIRECTION, SERVO_AMPLITUDE,
SERVO_CENTER_US, SERVO_OFFSET_US — so the angle you measure is the angle the
policy commands, not the angle some other pulse width happens to produce.

Usage
-----
    sudo Deployment/Python/HardwareTests/steering_angle_test.py

Put the car on a box with the front wheels clear of the ground, or on paper so you
can trace the wheel edges.  Measure the angle of the front wheels off straight
ahead at each hold.  What you want out of it:

    delta_max = (|angle at +1.0| + |angle at -1.0|) / 2

Feed that back either by raising SERVO_AMPLITUDE until delta_max reaches the sim's
28 deg, or by lowering the sim's max_steer_rad to match the car.  Note both
extremes separately as well: a large asymmetry means SERVO_OFFSET_US is wrong and
the car has a permanent steering bias the policy has to fight.
"""

from __future__ import annotations

import math
import os
import subprocess
import sys
import time
from pathlib import Path

# Import the deployment package rather than copying its constants: a test that
# drifts from the code it is calibrating is worse than no test.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lituanicax.config import (  # noqa: E402
    SERVO_AMPLITUDE,
    SERVO_CENTER_US,
    SERVO_DIRECTION,
    SERVO_GPIO,
    SERVO_MAX_US,
    SERVO_MIN_US,
    SERVO_OFFSET_US,
    SERVO_PWM_CHANNEL,
    SERVO_PWM_FREQ,
)
from lituanicax.modules.drive import Drive, SysfsHardwarePWM  # noqa: E402

#: Steering commands to visit, in order.  The +1 / -1 holds are the measurement;
#: the intermediate values are free (the servo is already powered and waiting) and
#: tell you whether the linkage is linear, which is what the sim assumes when it
#: maps the action straight onto an angle.
SWEEP = (1.0, -1.0, 0.5, -0.5, 0.0)

#: The sim's max_steer_rad, for the comparison printout.
SIM_MAX_STEER_DEG = math.degrees(0.488)
#: Wheelbase in metres. The real car measures 0.187 m; the sim's scaled MuSHR USD
#: is 0.224 m, which is itself part of the curvature mismatch.
REAL_WHEELBASE_M = 0.187
SIM_WHEELBASE_M = 0.224


def _prompt(message: str) -> None:
    """Wait for Enter, tolerating a closed stdin (e.g. run under nohup)."""
    try:
        input(message)
    except EOFError:
        print(f"{message}  [no stdin — holding 5 s]")
        time.sleep(5.0)


def main() -> int:
    print("Steering angle test")
    print("-" * 62)
    print(f"  SERVO_AMPLITUDE   {SERVO_AMPLITUDE}")
    print(f"  SERVO_DIRECTION   {SERVO_DIRECTION}")
    print(f"  SERVO_CENTER_US   {SERVO_CENTER_US} us")
    print(f"  SERVO_OFFSET_US   {SERVO_OFFSET_US} us")
    print(f"  pulse range       {SERVO_MIN_US}-{SERVO_MAX_US} us")
    print(f"  sim max_steer     {SIM_MAX_STEER_DEG:.1f} deg")
    print("-" * 62)

    subprocess.run(["pinctrl", "set", str(SERVO_GPIO), "a0", "pd"], check=True)
    servo = SysfsHardwarePWM(
        SERVO_PWM_CHANNEL, SERVO_PWM_FREQ, initial_pulse_ns=SERVO_CENTER_US * 1000
    )
    time.sleep(0.5)  # let it reach centre before the first hold

    try:
        for value in SWEEP:
            pulse_ns = Drive._steering_pulse_ns(value)
            pulse_us = pulse_ns / 1000.0
            servo.set_pulse_ns(pulse_ns)
            clipped = " (CLIPPED at the servo limit)" if pulse_us in (SERVO_MIN_US, SERVO_MAX_US) else ""
            print(
                f"\nsteer = {value:+.2f}  ->  {pulse_us:.0f} us "
                f"({pulse_us - SERVO_CENTER_US - SERVO_OFFSET_US:+.0f} us from centre){clipped}"
            )
            _prompt("    measure the front-wheel angle, then press Enter... ")
    finally:
        servo.set_pulse_ns(int((SERVO_CENTER_US + SERVO_OFFSET_US) * 1000))
        time.sleep(0.4)
        servo.stop()
        subprocess.run(["pinctrl", "set", str(SERVO_GPIO), "a0", "pd"], check=False)
        print("\nRe-centred and stopped.")

    print("\nWhat to do with the number you just measured (delta_max, degrees):")
    print(f"  curvature at full lock = tan(delta_max) / {REAL_WHEELBASE_M} m")
    for delta in (15.0, 18.0, 21.0, 24.0, 28.0):
        real = math.tan(math.radians(delta)) / REAL_WHEELBASE_M
        print(f"    {delta:4.0f} deg -> {real:5.2f} 1/m   (turn radius {1 / real:.2f} m)")
    sim_curv = math.tan(0.488) / SIM_WHEELBASE_M
    print(
        f"  sim: {SIM_MAX_STEER_DEG:.1f} deg over a {SIM_WHEELBASE_M} m wheelbase "
        f"= {sim_curv:.2f} 1/m (turn radius {1 / sim_curv:.2f} m)"
    )
    print(
        "  Match the CURVATURE, not the angle: the sim's wheelbase is "
        f"{SIM_WHEELBASE_M / REAL_WHEELBASE_M:.2f}x the car's, so equal angles are not equal turns."
    )
    return 0


if __name__ == "__main__":
    if os.geteuid() != 0:
        # Bail here rather than letting the first sysfs write raise: a clear line
        # is more use than a PermissionError traceback out of SysfsHardwarePWM.
        print("Needs root for sysfs PWM and pinctrl — re-run with sudo.", file=sys.stderr)
        sys.exit(1)
    sys.exit(main())
