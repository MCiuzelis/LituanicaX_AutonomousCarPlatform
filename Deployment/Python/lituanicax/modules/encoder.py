"""AS5600 magnetic encoder: wheel velocity and distance travelled.

Replaces `encoder_node.py`.  Instead of publishing `/monitoring/encoder`, it
exposes `linear_velocity` / `distance_travelled` for other modules to read.
"""

from __future__ import annotations

import math
import time
from typing import Callable

import smbus2

from ..config import (
    AS5600_ADDR,
    AS5600_ANGLE_REG,
    AS5600_I2C_BUS,
    ENCODER_HZ,
    ENCODER_ROTATIONS_PER_WHEEL_ROTATION,
    WHEEL_DIAMETER_M,
)
from ..module import Module


class Encoder(Module):
    def __init__(self, rate_hz: float = ENCODER_HZ) -> None:
        super().__init__("encoder", max(200.0, float(rate_hz)))

        self.linear_velocity = 0.0
        self.distance_travelled = 0.0
        self.last_update_time = 0.0

        #: Optional per-sample hook, called with the new velocity at the full
        #: encoder rate.  The policy filters wheel velocity here so its filter
        #: sees every sample, exactly as it did when subscribed to the topic.
        self.on_sample: Callable[[float], None] | None = None

        self.i2c: smbus2.SMBus | None = None
        self._last_angle: float | None = None
        self._last_time: float | None = None
        self._error_logged = False

    def setup(self) -> None:
        self.i2c = smbus2.SMBus(AS5600_I2C_BUS)
        self.log.info(f"Encoder initialized (rate={self.rate_hz:.1f} Hz)")

    def update(self) -> None:
        try:
            self._update_linear_velocity()
            if self._error_logged:
                self.log.info("Encoder velocity read recovered")
                self._error_logged = False
        except Exception as exc:
            if not self._error_logged:
                self.log.warning(f"Encoder velocity read failed: {exc}")
                self._error_logged = True
            self.linear_velocity = 0.0
            self._last_angle = None
            self._last_time = None

        self.last_update_time = time.monotonic()

        if self.on_sample is not None:
            self.on_sample(self.linear_velocity)

    def shutdown(self) -> None:
        if self.i2c is not None:
            self.i2c.close()
            self.i2c = None

    # --- AS5600 ---

    def _read_angle_degrees(self) -> float:
        data = self.i2c.read_i2c_block_data(AS5600_ADDR, AS5600_ANGLE_REG, 2)
        raw_value = (data[0] << 8) | data[1]
        degrees = -(raw_value * 360.0) / 4096.0
        if degrees <= -180.0:
            degrees += 360.0
        elif degrees > 180.0:
            degrees -= 360.0
        return degrees

    def _update_linear_velocity(self) -> None:
        now = time.monotonic()
        angle = self._read_angle_degrees()

        if self._last_angle is None or self._last_time is None:
            self._last_angle = angle
            self._last_time = now
            self.linear_velocity = 0.0
            return

        dt = now - self._last_time
        delta = angle - self._last_angle

        if delta > 180.0:
            delta -= 360.0
        elif delta < -180.0:
            delta += 360.0

        self._last_angle = angle
        self._last_time = now

        if dt <= 0.0:
            self.linear_velocity = 0.0
            return

        encoder_rotations_per_sec = (delta / 360.0) / dt
        wheel_rotations_per_sec = encoder_rotations_per_sec / ENCODER_ROTATIONS_PER_WHEEL_ROTATION
        self.linear_velocity = wheel_rotations_per_sec * math.pi * WHEEL_DIAMETER_M
        self.distance_travelled += abs(self.linear_velocity) * dt
