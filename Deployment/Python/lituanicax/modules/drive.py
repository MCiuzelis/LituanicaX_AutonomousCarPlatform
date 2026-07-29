"""Motor and steering output — the only module that touches GPIO/PWM.

Replaces `drive_node.py`.  Other modules call `command(throttle, steer)`
instead of publishing `/cmd_drive`; the safety rules are unchanged:

* autopilot must be enabled,
* monitoring telemetry must be fresh and fault-free,
* commands must keep arriving, or output falls back to zero.
"""

from __future__ import annotations

import os
import subprocess
import threading
import time

import RPi.GPIO as GPIO

from ..config import (
    COMMAND_TIMEOUT_SEC,
    MAX_THROTTLE,
    MONITORING_TIMEOUT_SEC,
    MOTOR_DIRECTION,
    MOTOR_GPIO,
    MOTOR_PWM_CHANNEL,
    MOTOR_PWM_FREQ,
    PIN_INA,
    PIN_INB,
    SERVO_AMPLITUDE,
    SERVO_CENTER_US,
    SERVO_DIRECTION,
    SERVO_GPIO,
    SERVO_MAX_US,
    SERVO_MIN_US,
    SERVO_OFFSET_US,
    SERVO_PWM_CHANNEL,
    SERVO_PWM_FREQ,
    TELEMETRY_HZ,
)
from ..module import Module
from .monitoring import Monitoring

PWM_CHIP_PATH = "/sys/class/pwm/pwmchip0"


class SysfsHardwarePWM:
    """One hardware PWM channel driven through /sys/class/pwm."""

    def __init__(self, channel: int, frequency_hz: int, initial_pulse_ns: int = 0) -> None:
        self.channel = channel
        self.period_ns = int(1_000_000_000 / frequency_hz)
        self.path = f"{PWM_CHIP_PATH}/pwm{self.channel}"

        if not os.path.exists(self.path):
            self._write(f"{PWM_CHIP_PATH}/export", str(self.channel))
            time.sleep(0.1)

        self._write("enable", "0")
        self._write("duty_cycle", "0")
        self._write("period", str(self.period_ns))
        self.set_pulse_ns(initial_pulse_ns)
        self._write("enable", "1")

    def _write(self, name_or_path: str, value: str) -> None:
        path = name_or_path if name_or_path.startswith("/") else f"{self.path}/{name_or_path}"
        with open(path, "w") as f:
            f.write(value)

    def set_duty_fraction(self, duty_fraction: float) -> None:
        duty_fraction = max(0.0, min(1.0, duty_fraction))
        self.set_pulse_ns(int(self.period_ns * duty_fraction))

    def set_pulse_ns(self, pulse_ns: int) -> None:
        pulse_ns = max(0, min(self.period_ns, int(pulse_ns)))
        self._write("duty_cycle", str(pulse_ns))

    def stop(self) -> None:
        try:
            self._write("enable", "0")
        except OSError:
            pass


class Drive(Module):
    # PWM writes must stay on time even while the cameras and the policy load
    # the CPU, so this module's thread runs at real-time priority.
    realtime = True

    def __init__(
        self,
        monitoring: Monitoring | None = None,
        command_timeout_sec: float = COMMAND_TIMEOUT_SEC,
        monitoring_timeout_sec: float = MONITORING_TIMEOUT_SEC,
        rate_hz: float = TELEMETRY_HZ,
    ) -> None:
        super().__init__("drive", rate_hz)

        self.monitoring = monitoring
        self.command_timeout_sec = float(command_timeout_sec)
        self.monitoring_timeout_sec = float(monitoring_timeout_sec)

        self.enabled = False
        self.undervoltage = False
        self.overcurrent = False
        self.linear_velocity = 0.0
        self.last_throttle = 0.0
        self.last_steer = 0.0
        self.last_command_time = 0.0

        self.motor_pwm: SysfsHardwarePWM | None = None
        self.servo_pwm: SysfsHardwarePWM | None = None
        self._lock = threading.RLock()
        self._gpio_ready = False

    # --- lifecycle ---

    def setup(self) -> None:
        GPIO.setmode(GPIO.BCM)
        GPIO.setup(PIN_INA, GPIO.OUT)
        GPIO.setup(PIN_INB, GPIO.OUT)
        self._gpio_ready = True

        # Motor: hardware PWM via sysfs (channel 0, 20 kHz)
        self._pinctrl_set(MOTOR_GPIO, "a0", "pd")
        self.motor_pwm = SysfsHardwarePWM(MOTOR_PWM_CHANNEL, MOTOR_PWM_FREQ, initial_pulse_ns=0)

        # Servo: sysfs hardware PWM (channel 1, 50 Hz).  Software PWM from
        # RPi.GPIO conflicts with the RP1 PWM block on the Pi 5.
        self._pinctrl_set(SERVO_GPIO, "a0", "pd")
        self.servo_pwm = SysfsHardwarePWM(
            SERVO_PWM_CHANNEL, SERVO_PWM_FREQ, initial_pulse_ns=SERVO_CENTER_US * 1000
        )

        if self.monitoring is None:
            self.log.warning("No monitoring module: drive output stays blocked")
        self.log.info("Drive initialized")

    def update(self) -> None:
        """Watchdog tick: cut output when commands stop or safety trips."""
        with self._lock:
            self._refresh_monitoring()
            command_stale = time.monotonic() - self.last_command_time > self.command_timeout_sec
            if command_stale or not self._can_drive():
                self._apply(0.0, 0.0)

    def shutdown(self) -> None:
        """Idempotent, and safe to call when setup() only got part way."""
        with self._lock:
            if not self._gpio_ready and self.motor_pwm is None and self.servo_pwm is None:
                return

            self._apply(0.0, 0.0)  # the setters no-op once the PWMs are gone
            for pwm_attr in ("servo_pwm", "motor_pwm"):
                pwm = getattr(self, pwm_attr)
                if pwm is not None:
                    pwm.stop()
                    setattr(self, pwm_attr, None)

            if self._gpio_ready:
                GPIO.cleanup()
                self._pinctrl_set(MOTOR_GPIO, "a0", "pd")
                self._pinctrl_set(SERVO_GPIO, "a0", "pd")
                self._gpio_ready = False
        self.log.info("Drive stopped")

    # --- public API (called from the policy, dashboard and test modules) ---

    def command(self, throttle: float, steer: float) -> None:
        """Request a drive output. Ignored unless every safety check passes."""
        with self._lock:
            self.last_command_time = time.monotonic()
            self._refresh_monitoring()
            if self._can_drive():
                self._apply(throttle, steer)
            else:
                self._apply(0.0, 0.0)

    def set_autopilot_enabled(self, enabled: bool) -> None:
        with self._lock:
            self.enabled = bool(enabled)
            if not self.enabled:
                # Brake request; _apply() turns it into a coast once stopped.
                self._refresh_monitoring()
                self._apply(-1.0, 0.0)

    @property
    def safety_blocked(self) -> bool:
        return self.undervoltage or self.overcurrent

    # --- internals ---

    def _refresh_monitoring(self) -> None:
        if self.monitoring is None:
            return
        self.undervoltage = self.monitoring.undervoltage
        self.overcurrent = self.monitoring.overcurrent
        self.linear_velocity = self.monitoring.linear_velocity

    def _monitoring_is_fresh(self) -> bool:
        if self.monitoring is None:
            return False
        return time.monotonic() - self.monitoring.last_update_time <= self.monitoring_timeout_sec

    def _can_drive(self) -> bool:
        return (
            self.enabled
            and self._monitoring_is_fresh()
            and not self.undervoltage
            and not self.overcurrent
        )

    def _apply(self, throttle: float, steer: float) -> None:
        if throttle < 0.0 and self.linear_velocity <= 0.01:
            throttle = 0.0
        self.last_throttle = throttle
        self.last_steer = steer
        self._set_throttle(throttle)
        self._set_steering(steer)

    # --- actuators ---

    @staticmethod
    def _pinctrl_set(pin: int, *args: str) -> None:
        subprocess.run(["pinctrl", "set", str(pin), *args], check=True)

    def _set_throttle(self, value: float) -> None:
        if self.motor_pwm is None:
            return
        value = max(-1.0, min(1.0, value))
        value *= max(0.0, min(1.0, MAX_THROTTLE))
        value *= MOTOR_DIRECTION

        GPIO.output(PIN_INA, GPIO.HIGH if value >= 0 else GPIO.LOW)
        GPIO.output(PIN_INB, GPIO.HIGH if value < 0 else GPIO.LOW)
        self.motor_pwm.set_duty_fraction(abs(value))

    @staticmethod
    def _steering_pulse_ns(value: float) -> int:
        value = max(-1.0, min(1.0, value))
        value = value * SERVO_DIRECTION * SERVO_AMPLITUDE

        span = (SERVO_MAX_US - SERVO_MIN_US) / 2
        pulse_us = SERVO_CENTER_US + SERVO_OFFSET_US + (value * span)
        pulse_us = max(SERVO_MIN_US, min(SERVO_MAX_US, pulse_us))
        return int(pulse_us * 1000)

    def _set_steering(self, value: float) -> None:
        if self.servo_pwm is None:
            return
        self.servo_pwm.set_pulse_ns(self._steering_pulse_ns(value))
