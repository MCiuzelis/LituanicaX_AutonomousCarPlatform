from __future__ import annotations

import math
import time

import rclpy
import smbus2
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray

from .common import (
    AS5600_ADDR,
    AS5600_ANGLE_REG,
    AS5600_I2C_BUS,
    ENCODER_HZ,
    ENCODER_ROTATIONS_PER_WHEEL_ROTATION,
    WHEEL_DIAMETER_M,
)


class EncoderNode(Node):
    def __init__(self) -> None:
        super().__init__("encoder_node")

        self.declare_parameter("encoder_hz", ENCODER_HZ)
        self.encoder_hz = max(200.0, float(self.get_parameter("encoder_hz").value))

        self.linear_velocity = 0.0
        self.distance_travelled = 0.0
        self._last_encoder_angle: float | None = None
        self._last_encoder_time: float | None = None
        self._encoder_error_logged = False
        self._stopped = False

        self.i2c = smbus2.SMBus(AS5600_I2C_BUS)
        self.encoder_pub = self.create_publisher(Float32MultiArray, "/monitoring/encoder", 1)

        encoder_period = 1.0 / self.encoder_hz
        self.encoder_timer = self.create_timer(encoder_period, self._encoder_tick)

        self.get_logger().info(f"Encoder node initialized (encoder={self.encoder_hz:.1f} Hz)")

    def _read_encoder_angle(self) -> float:
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
        angle = self._read_encoder_angle()

        if self._last_encoder_angle is None or self._last_encoder_time is None:
            self._last_encoder_angle = angle
            self._last_encoder_time = now
            self.linear_velocity = 0.0
            return

        dt = now - self._last_encoder_time
        delta = angle - self._last_encoder_angle

        if delta > 180.0:
            delta -= 360.0
        elif delta < -180.0:
            delta += 360.0

        self._last_encoder_angle = angle
        self._last_encoder_time = now

        if dt <= 0.0:
            self.linear_velocity = 0.0
            return

        encoder_rotations_per_sec = (delta / 360.0) / dt
        wheel_rotations_per_sec = encoder_rotations_per_sec / ENCODER_ROTATIONS_PER_WHEEL_ROTATION
        self.linear_velocity = wheel_rotations_per_sec * math.pi * WHEEL_DIAMETER_M
        self.distance_travelled += abs(self.linear_velocity) * dt

    def _encoder_tick(self) -> None:
        try:
            self._update_linear_velocity()
            if self._encoder_error_logged:
                self.get_logger().info("Encoder velocity read recovered")
                self._encoder_error_logged = False
        except Exception as exc:
            if not self._encoder_error_logged:
                self.get_logger().warning(f"Encoder velocity read failed: {exc}")
                self._encoder_error_logged = True
            self.linear_velocity = 0.0
            self._last_encoder_angle = None
            self._last_encoder_time = None
        
        msg = Float32MultiArray()
        msg.data = [
            float(self.linear_velocity),
            float(self.distance_travelled),
        ]
        self.encoder_pub.publish(msg)

    def destroy_node(self) -> bool:
        try:
            if not self._stopped:
                self.i2c.close()
                self._stopped = True
        finally:
            return super().destroy_node()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = EncoderNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
