"""Straight-line acceleration and braking test.

Replaces `acceleration_test_node.py`.  Enable it instead of the policy: it
takes over the drive with a fixed throttle, brakes at the target velocity or
timeout, and reports the distances.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

from ..module import Module
from .drive import Drive
from .monitoring import Monitoring

if TYPE_CHECKING:
    from ..robot import Robot


class AccelerationTest(Module):
    # --- configuration (hardcoded, test-only) ---
    TARGET_VELOCITY_M_S = 45
    ACCEL_TIMEOUT_S = 10
    ACCEL_THROTTLE = 0.5  # 1
    BRAKE_THROTTLE = -0.5
    VELOCITY_EPSILON = 0.1
    #: Give up waiting for a standstill after this long. The braking phase ends on
    #: the encoder reading ~zero, so an encoder that stops updating (I2C error,
    #: unplugged magnet) leaves the test latched in BRAKING with the brake held and
    #: the autopilot never released. Timing out reports the run instead of hanging.
    BRAKE_TIMEOUT_S = 4.0
    MONITORING_TIMEOUT_S = 0.2
    TICK_HZ = 300
    STARTUP_DELAY_S = 5.0

    def __init__(self, robot: "Robot", drive: Drive | None, monitoring: Monitoring | None) -> None:
        super().__init__("acceleration_test", self.TICK_HZ)

        self.robot = robot
        self.drive = drive
        self.monitoring = monitoring

        self.state = "IDLE"
        self.start_distance = 0.0
        self.accel_end_distance = 0.0
        self.peak_velocity = 0.0
        self.accel_start_time = 0.0
        self.brake_start_time = 0.0
        self._module_start_time = 0.0

    def setup(self) -> None:
        if self.drive is None or self.monitoring is None:
            raise RuntimeError("Acceleration test needs the drive and monitoring modules")

        self._module_start_time = time.monotonic()
        self.log.info(
            "Acceleration test initialized "
            f"(target={self.TARGET_VELOCITY_M_S} m/s, "
            f"timeout={self.ACCEL_TIMEOUT_S}s, "
            f"throttle={self.ACCEL_THROTTLE}, "
            f"start in {self.STARTUP_DELAY_S:.0f}s)"
        )

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

        if self.state == "ACCELERATING":
            self._accelerating()
        elif self.state == "BRAKING":
            self._braking()

    def _begin_test(self) -> None:
        self.robot.set_autopilot_enabled(True)
        self.state = "ACCELERATING"
        self.start_distance = self.monitoring.distance_travelled
        self.accel_start_time = time.monotonic()
        self.peak_velocity = 0.0
        self.drive.command(self.ACCEL_THROTTLE, 0.0)
        self.log.info(
            f"Test started — throttle={self.ACCEL_THROTTLE}, "
            f"target={self.TARGET_VELOCITY_M_S} m/s, "
            f"timeout={self.ACCEL_TIMEOUT_S}s"
        )

    def _accelerating(self) -> None:
        self.drive.command(self.ACCEL_THROTTLE, 0.0)
        elapsed = time.monotonic() - self.accel_start_time

        if self.monitoring.linear_velocity >= self.TARGET_VELOCITY_M_S:
            self._record_accel_end()
            self.log.info(
                f"Target velocity reached: {self.peak_velocity:.3f} m/s at {elapsed:.2f}s"
            )
            self._begin_braking()
        elif elapsed >= self.ACCEL_TIMEOUT_S:
            self._record_accel_end()
            self.log.info(
                f"Timeout reached ({elapsed:.2f}s) — velocity={self.peak_velocity:.3f} m/s"
            )
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

    def _record_accel_end(self) -> None:
        self.peak_velocity = self.monitoring.linear_velocity
        self.accel_end_distance = self.monitoring.distance_travelled

    def _begin_braking(self) -> None:
        self.state = "BRAKING"
        self.brake_start_time = time.monotonic()
        self.drive.command(self.BRAKE_THROTTLE, 0.0)
        self.log.info("Braking...")

    def _finish(self, aborted: bool) -> None:
        self.state = "DONE"
        self.drive.command(0.0, 0.0)
        self.robot.set_autopilot_enabled(False)

        distance = self.monitoring.distance_travelled
        brake_distance = distance - self.accel_end_distance
        accel_distance = self.accel_end_distance - self.start_distance
        total_distance = distance - self.start_distance

        if aborted:
            self.log.error("--- Acceleration Test ABORTED ---")
        else:
            self.log.info("--- Acceleration Test Complete ---")

        self.log.info(f"  Peak velocity:     {self.peak_velocity:.3f} m/s")
        self.log.info(f"  Accel distance:    {accel_distance:.3f} m")
        self.log.info(f"  Brake distance:    {brake_distance:.3f} m")
        self.log.info(f"  Total distance:    {total_distance:.3f} m")
