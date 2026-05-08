from __future__ import annotations

import time

import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool, Float32MultiArray

from .common import FPS_TARGET
from .paths import discover_deployment_dir, import_robot_class


class HardwareNode(Node):
    def __init__(self) -> None:
        super().__init__("hardware_node")

        self.declare_parameter("deployment_dir", "")
        self.declare_parameter("command_timeout_sec", 0.25)
        self.declare_parameter("telemetry_hz", FPS_TARGET)

        deployment_dir = discover_deployment_dir(str(self.get_parameter("deployment_dir").value))
        Robot = import_robot_class(deployment_dir)
        self.robot = Robot()

        self.enabled = False
        self.last_command_time = 0.0
        self.last_throttle = 0.0
        self.last_steer = 0.0
        self.command_timeout_sec = float(self.get_parameter("command_timeout_sec").value)

        self.telemetry_pub = self.create_publisher(Float32MultiArray, "/hardware/telemetry", 1)
        self.create_subscription(Float32MultiArray, "/cmd_drive", self._cmd_cb, 1)
        self.create_subscription(Bool, "/autopilot/enabled", self._enabled_cb, 1)

        telemetry_period = 1.0 / float(self.get_parameter("telemetry_hz").value)
        self.timer = self.create_timer(telemetry_period, self._telemetry_tick)

        self.get_logger().info(f"Hardware initialized using Deployment dir: {deployment_dir}")

    def _enabled_cb(self, msg: Bool) -> None:
        self.enabled = bool(msg.data)
        if not self.enabled:
            self._apply_drive(0.0, 0.0)

    def _cmd_cb(self, msg: Float32MultiArray) -> None:
        if len(msg.data) < 2:
            return

        throttle = float(msg.data[0])
        steer = float(msg.data[1])
        self.last_command_time = time.monotonic()

        if self.enabled:
            self._apply_drive(throttle, steer)
        else:
            self._apply_drive(0.0, 0.0)

    def _apply_drive(self, throttle: float, steer: float) -> None:
        self.last_throttle = throttle
        self.last_steer = steer
        self.robot.set_throttle(throttle)
        self.robot.set_steering(steer)

    def _telemetry_tick(self) -> None:
        if time.monotonic() - self.last_command_time > self.command_timeout_sec:
            self._apply_drive(0.0, 0.0)

        try:
            battery_voltage = float(self.robot.get_battery_voltage())
            motor_current = float(self.robot.get_motor_current())
        except Exception as exc:
            self.get_logger().warning(f"Telemetry read failed: {exc}")
            battery_voltage = 0.0
            motor_current = 0.0

        msg = Float32MultiArray()
        msg.data = [
            float(self.last_throttle),
            float(self.last_steer),
            battery_voltage,
            motor_current,
            1.0 if self.enabled else 0.0,
        ]
        self.telemetry_pub.publish(msg)

    def destroy_node(self) -> bool:
        try:
            self._apply_drive(0.0, 0.0)
            self.robot.stop()
        finally:
            return super().destroy_node()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = HardwareNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
