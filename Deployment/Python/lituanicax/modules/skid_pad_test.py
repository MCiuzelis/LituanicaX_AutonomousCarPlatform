"""Constant-steer, constant-throttle circle — the lateral half of the calibration.

`acceleration_test.py` matched the sim's drivetrain to the car along ONE axis:
straight-line accel and braking at full throttle.  That is the axis that works.
The axis that fails was never measured, and the sim's numbers for it were never
checked against anything:

    sim   max_steer_rad 0.488 (28 deg) over a 0.224 m wheelbase -> 2.37 1/m
    real  recovered from depth-image flow in a logged run       -> 1.1-1.7 1/m

i.e. a minimum turn radius somewhere between 40 % and 100 % larger than the one
the policy learned to plan with, on a car whose measured wheelbase is 0.187 m and
whose ground friction is nothing like the sim's effective mu = 2.0.  A policy that
plans a line the car cannot draw runs out of road, and no amount of throttle
capping fixes the shape of that error.

This module is the direct measurement.  Full steer, fixed throttle, hold, brake.
It reports the distance the wheels covered over a settled window; you count the
laps, and the radius follows:

    radius = distance / (2 * pi * laps)
    curvature = 1 / radius              <- the number to match against the sim
    delta_max = atan(curvature * wheelbase)

Run it on the largest flat area you have, from a standstill, with nothing within
a couple of metres — at full lock the car will spiral out to its steady circle
before it settles.  Chalk on a front tyre, or a marker taped to the chassis,
makes the laps countable.

    Deployment/Python/run --disable-policy --enable skid_pad_test

Sweep THROTTLE across runs: the radius growing with speed IS the understeer
gradient, and it is what tells you whether the gap to the sim is steering
geometry (radius constant, just too big) or grip (radius growing with speed).
"""

from __future__ import annotations

import math
import time
from typing import TYPE_CHECKING

from ..module import Module
from .drive import Drive
from .monitoring import Monitoring

if TYPE_CHECKING:
    from ..robot import Robot

#: Measured wheelbase of this car, metres.  The sim's scaled MuSHR USD is 0.224 m,
#: so equal steering angles do NOT give equal turn radii between the two.
REAL_WHEELBASE_M = 0.187


class SkidPadTest(Module):
    # --- configuration (hardcoded, test-only — same convention as AccelerationTest) ---
    #: Full lock. The point of the test is the limit, not a comfortable arc.
    STEER = 1.0
    #: Throttle held through the whole circle. Sweep this across runs; 0.35 is a
    #: sane first pass (~1.9 m/s steady state at MAX_THROTTLE = 1.0, close to the
    #: 1.82 m/s the car actually averaged under the policy).
    THROTTLE = 0.35
    #: Not measured: the car is still spiralling out to its steady radius.
    SETTLE_S = 2.5
    #: Measured window. Long enough to count several laps at ~2 m/s.
    MEASURE_S = 6.0
    BRAKE_THROTTLE = -0.5
    VELOCITY_EPSILON = 0.1
    #: Stop waiting for a standstill after this long. Without it a stuck or failed
    #: encoder leaves the test latched in BRAKING with the brake applied and the
    #: autopilot never released — the state machine's one way to hang the car.
    BRAKE_TIMEOUT_S = 4.0
    MONITORING_TIMEOUT_S = 0.2
    TICK_HZ = 300
    STARTUP_DELAY_S = 5.0

    def __init__(self, robot: "Robot", drive: Drive | None, monitoring: Monitoring | None) -> None:
        super().__init__("skid_pad_test", self.TICK_HZ)

        self.robot = robot
        self.drive = drive
        self.monitoring = monitoring

        self.state = "IDLE"
        self.start_distance = 0.0
        self.settle_end_distance = 0.0
        self.measure_end_distance = 0.0
        self.peak_velocity = 0.0
        self.velocity_sum = 0.0
        self.velocity_samples = 0
        self.turn_start_time = 0.0
        self.measure_start_time = 0.0
        self.measure_duration = 0.0
        self.brake_start_time = 0.0
        self._module_start_time = 0.0

    def setup(self) -> None:
        if self.drive is None or self.monitoring is None:
            raise RuntimeError("Skid pad test needs the drive and monitoring modules")

        self._module_start_time = time.monotonic()
        self.log.info(
            "Skid pad test initialized "
            f"(steer={self.STEER}, throttle={self.THROTTLE}, "
            f"settle={self.SETTLE_S}s, measure={self.MEASURE_S}s, "
            f"start in {self.STARTUP_DELAY_S:.0f}s)"
        )
        self.log.info("  Clear a circle several metres across — the car spirals out before it settles.")

    def shutdown(self) -> None:
        if self.state not in ("IDLE", "DONE"):
            self.log.warning("Stopped mid-test")
        self.drive.command(0.0, 0.0)
        self.robot.set_autopilot_enabled(False)

    # --- state machine ---

    def update(self) -> None:
        if self.state == "DONE":
            return

        if self.state == "IDLE":
            if time.monotonic() - self._module_start_time >= self.STARTUP_DELAY_S:
                self._begin_test()
            return

        if time.monotonic() - self.monitoring.last_update_time > self.MONITORING_TIMEOUT_S:
            self.log.error("Monitoring data stale — aborting test")
            self._finish(aborted=True)
            return

        if self.state == "SETTLING":
            self._settling()
        elif self.state == "MEASURING":
            self._measuring()
        elif self.state == "BRAKING":
            self._braking()

    def _begin_test(self) -> None:
        self.robot.set_autopilot_enabled(True)
        self.state = "SETTLING"
        self.start_distance = self.monitoring.distance_travelled
        self.turn_start_time = time.monotonic()
        self.peak_velocity = 0.0
        self.drive.command(self.THROTTLE, self.STEER)
        self.log.info(
            f"Test started — steer={self.STEER}, throttle={self.THROTTLE}; "
            f"settling for {self.SETTLE_S}s"
        )

    def _hold(self) -> None:
        """Same command every tick: the drive's command timeout is what stops the
        car if this thread dies, so the hold has to be re-asserted, not set once."""
        self.drive.command(self.THROTTLE, self.STEER)
        self.peak_velocity = max(self.peak_velocity, self.monitoring.linear_velocity)

    def _settling(self) -> None:
        self._hold()
        if time.monotonic() - self.turn_start_time >= self.SETTLE_S:
            self.state = "MEASURING"
            self.settle_end_distance = self.monitoring.distance_travelled
            self.measure_start_time = time.monotonic()
            self.velocity_sum = 0.0
            self.velocity_samples = 0
            self.log.info(
                f"Settled at {self.monitoring.linear_velocity:.2f} m/s — "
                f"measuring for {self.MEASURE_S}s (start counting laps now)"
            )

    def _measuring(self) -> None:
        self._hold()
        self.velocity_sum += self.monitoring.linear_velocity
        self.velocity_samples += 1
        if time.monotonic() - self.measure_start_time >= self.MEASURE_S:
            self.measure_end_distance = self.monitoring.distance_travelled
            self.measure_duration = time.monotonic() - self.measure_start_time
            self._begin_braking()

    def _braking(self) -> None:
        self.drive.command(self.BRAKE_THROTTLE, 0.0)
        if abs(self.monitoring.linear_velocity) <= self.VELOCITY_EPSILON:
            self._finish(aborted=False)
        elif time.monotonic() - self.brake_start_time >= self.BRAKE_TIMEOUT_S:
            self.log.warning(
                f"Still reading {self.monitoring.linear_velocity:.2f} m/s after "
                f"{self.BRAKE_TIMEOUT_S:.1f}s of braking — releasing anyway"
            )
            self._finish(aborted=False)

    def _begin_braking(self) -> None:
        self.state = "BRAKING"
        self.brake_start_time = time.monotonic()
        self.drive.command(self.BRAKE_THROTTLE, 0.0)
        self.log.info("Braking...")

    def _finish(self, aborted: bool) -> None:
        self.state = "DONE"
        self.drive.command(0.0, 0.0)
        self.robot.set_autopilot_enabled(False)

        total_distance = self.monitoring.distance_travelled - self.start_distance
        measured_distance = self.measure_end_distance - self.settle_end_distance
        mean_velocity = (
            self.velocity_sum / self.velocity_samples if self.velocity_samples else 0.0
        )

        if aborted:
            self.log.error("--- Skid Pad Test ABORTED ---")
        else:
            self.log.info("--- Skid Pad Test Complete ---")

        self.log.info(f"  Steer / throttle:   {self.STEER:+.2f} / {self.THROTTLE:.2f}")
        self.log.info(f"  Peak velocity:      {self.peak_velocity:.3f} m/s")
        self.log.info(f"  Mean velocity:      {mean_velocity:.3f} m/s  (measured window)")
        self.log.info(f"  Measured distance:  {measured_distance:.3f} m")
        self.log.info(f"  Measured duration:  {self.measure_duration:.2f} s")
        self.log.info(f"  Total distance:     {total_distance:.3f} m  (incl. settling)")

        if aborted or measured_distance <= 0.0:
            return

        # No gyro on this car, so laps cannot be counted on-board — the distance is
        # real, the number of times it went round is not. Print the radius for each
        # plausible lap count and let the operator pick the one they watched.
        self.log.info("  Laps counted in the measured window -> steady-state circle:")
        self.log.info("     laps   radius     curvature   implied delta_max")
        for laps in range(1, 9):
            radius = measured_distance / (2.0 * math.pi * laps)
            if radius < 0.15 or radius > 20.0:
                continue
            curvature = 1.0 / radius
            delta = math.degrees(math.atan(curvature * REAL_WHEELBASE_M))
            lat_g = (mean_velocity**2) / radius / 9.81 if radius > 0 else 0.0
            self.log.info(
                f"      {laps:2d}    {radius:5.2f} m   {curvature:5.2f} 1/m    "
                f"{delta:5.1f} deg   ({lat_g:.2f} g lateral)"
            )
        sim_curvature = math.tan(0.488) / 0.224
        self.log.info(
            f"  Sim for comparison: {sim_curvature:.2f} 1/m "
            f"(radius {1.0 / sim_curvature:.2f} m) at max_steer_rad=0.488, wheelbase 0.224 m"
        )
