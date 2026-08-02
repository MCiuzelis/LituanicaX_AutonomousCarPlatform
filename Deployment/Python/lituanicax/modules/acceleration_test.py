"""Straight-line acceleration and braking test.

Replaces `acceleration_test_node.py`.  Enable it instead of the policy: it
takes over the drive with a fixed throttle, brakes at the target velocity or
timeout, and reports the distances.

    Deployment/Python/run --disable-policy --enable-accel-test

Enabling it turns the recorder on (see `main.settings_from_args`), so the run
always leaves a full record in `Deployment/Logs/run_<timestamp>/`: this module's
own 300 Hz trace in `test.jsonl`, its phase changes and final numbers in
`events.jsonl`, and everything else the car measures — battery, current, CPU,
encoder, drive state, ToF frames — in the usual files alongside them.
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
    TARGET_VELOCITY_M_S = 4.5
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
        #: Velocity at the moment the accel phase ended — the reported figure, and
        #: the terminal velocity when the phase ended on the timeout.
        self.peak_velocity = 0.0
        #: Running maximum over the whole run. Equal to `peak_velocity` on a
        #: timeout, higher than it if the car was already slowing at the target.
        self.max_velocity = 0.0
        self.accel_start_time = 0.0
        self.accel_duration = 0.0
        self.brake_start_time = 0.0
        self.throttle_cmd = 0.0
        self.steer_cmd = 0.0
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
            self._event("stopped_mid_test", state=self.state)
        self._command(0.0, 0.0)
        self.robot.set_autopilot_enabled(False)

    # --- recording ---

    def parameters(self) -> dict:
        """The hardcoded constants this run was made with, for `meta.json`."""
        return {
            "target_velocity_m_s": self.TARGET_VELOCITY_M_S,
            "accel_timeout_s": self.ACCEL_TIMEOUT_S,
            "accel_throttle": self.ACCEL_THROTTLE,
            "brake_throttle": self.BRAKE_THROTTLE,
            "velocity_epsilon": self.VELOCITY_EPSILON,
            "brake_timeout_s": self.BRAKE_TIMEOUT_S,
            "startup_delay_s": self.STARTUP_DELAY_S,
            "tick_hz": self.TICK_HZ,
        }

    def telemetry(self) -> dict:
        """What the test is doing right now: goes into every `test.jsonl` row and
        every `telemetry.jsonl` row, so either file can be sliced by phase."""
        row = {
            "test": self.name,
            "test_state": self.state,
            "test_throttle_cmd": self.throttle_cmd,
            "test_steer_cmd": self.steer_cmd,
            "test_peak_velocity": self.max_velocity,
            "test_elapsed": (
                time.monotonic() - self.accel_start_time if self.accel_start_time else 0.0
            ),
        }
        if self.monitoring is not None:
            row["test_velocity"] = self.monitoring.linear_velocity
            row["test_distance"] = self.monitoring.distance_travelled - self.start_distance
        return row

    def _command(self, throttle: float, steer: float) -> None:
        """Every command goes through here so the recorded row says what the test
        asked for, next to what `Drive` actually did with it after the ramp and
        the current limiter."""
        self.throttle_cmd = throttle
        self.steer_cmd = steer
        self.drive.command(throttle, steer)

    def _record_tick(self) -> None:
        recorder = self.robot.recorder
        if recorder is None:
            return
        row = {"t": time.monotonic(), "t_wall": time.time()}
        row.update(self.telemetry())
        recorder.log_test_tick(row)

    def _event(self, kind: str, **fields) -> None:
        recorder = self.robot.recorder
        if recorder is None:
            return
        row = {"t": time.monotonic(), "t_wall": time.time(), "test": self.name, "event": kind}
        row.update(fields)
        recorder.log_event(row)

    # --- state machine ---

    def update(self) -> None:
        if self.state == "DONE":
            return
        self._step()
        # After the step, so the transition into DONE lands in the file too.
        self._record_tick()

    def _step(self) -> None:
        if self.state == "IDLE":
            if time.monotonic() - self._module_start_time >= self.STARTUP_DELAY_S:
                self._begin_test()
            return

        if time.monotonic() - self.monitoring.last_update_time > self.MONITORING_TIMEOUT_S:
            self.log.error("Monitoring data stale — aborting test")
            self._event(
                "monitoring_stale",
                state=self.state,
                monitoring_age=time.monotonic() - self.monitoring.last_update_time,
            )
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
        self.max_velocity = 0.0
        self._command(self.ACCEL_THROTTLE, 0.0)
        self.log.info(
            f"Test started — throttle={self.ACCEL_THROTTLE}, "
            f"target={self.TARGET_VELOCITY_M_S} m/s, "
            f"timeout={self.ACCEL_TIMEOUT_S}s"
        )
        self._event("started", start_distance=self.start_distance, **self.parameters())

    def _accelerating(self) -> None:
        self._command(self.ACCEL_THROTTLE, 0.0)
        elapsed = time.monotonic() - self.accel_start_time
        self.max_velocity = max(self.max_velocity, self.monitoring.linear_velocity)

        if self.monitoring.linear_velocity >= self.TARGET_VELOCITY_M_S:
            self._record_accel_end(elapsed)
            self.log.info(
                f"Target velocity reached: {self.peak_velocity:.3f} m/s at {elapsed:.2f}s"
            )
            self._begin_braking("target reached")
        elif elapsed >= self.ACCEL_TIMEOUT_S:
            self._record_accel_end(elapsed)
            self.log.info(
                f"Timeout reached ({elapsed:.2f}s) — velocity={self.peak_velocity:.3f} m/s"
            )
            self._begin_braking("timeout")

    def _braking(self) -> None:
        self._command(self.BRAKE_THROTTLE, 0.0)
        if abs(self.monitoring.linear_velocity) <= self.VELOCITY_EPSILON:
            self._finish(aborted=False)
        elif time.monotonic() - self.brake_start_time >= self.BRAKE_TIMEOUT_S:
            self.log.warning(
                f"Still reading {self.monitoring.linear_velocity:.2f} m/s after "
                f"{self.BRAKE_TIMEOUT_S:.1f}s of braking — releasing anyway"
            )
            self._event("brake_timeout", velocity=self.monitoring.linear_velocity)
            self._finish(aborted=False)

    def _record_accel_end(self, elapsed: float) -> None:
        self.peak_velocity = self.monitoring.linear_velocity
        self.accel_end_distance = self.monitoring.distance_travelled
        self.accel_duration = elapsed

    def _begin_braking(self, reason: str) -> None:
        self.state = "BRAKING"
        self.brake_start_time = time.monotonic()
        self._command(self.BRAKE_THROTTLE, 0.0)
        self.log.info("Braking...")
        self._event(
            "braking",
            reason=reason,
            accel_duration=self.accel_duration,
            peak_velocity=self.peak_velocity,
            accel_distance=self.accel_end_distance - self.start_distance,
        )

    def _finish(self, aborted: bool) -> None:
        brake_duration = (
            time.monotonic() - self.brake_start_time if self.brake_start_time else 0.0
        )
        self.state = "DONE"
        self._command(0.0, 0.0)
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

        # The same numbers the lines above print, in a form a script can read.
        self._event(
            "finished",
            aborted=aborted,
            peak_velocity=self.peak_velocity,
            max_velocity=self.max_velocity,
            accel_distance=accel_distance,
            accel_duration=self.accel_duration,
            brake_distance=brake_distance,
            brake_duration=brake_duration,
            total_distance=total_distance,
            mean_accel=(
                self.peak_velocity / self.accel_duration if self.accel_duration else 0.0
            ),
            mean_decel=(self.peak_velocity / brake_duration if brake_duration else 0.0),
        )
