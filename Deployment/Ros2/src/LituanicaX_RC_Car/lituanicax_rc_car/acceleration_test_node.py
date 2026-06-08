from __future__ import annotations

import time

import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from std_msgs.msg import Bool, Float32MultiArray


class AccelerationTestNode(Node):
    # --- configuration (hardcoded, test-only) ---
    TARGET_VELOCITY_M_S = 45
    ACCEL_TIMEOUT_S = 10 
    ACCEL_THROTTLE = 0.5 #1
    BRAKE_THROTTLE = -0.5
    VELOCITY_EPSILON = 0.1
    MONITORING_TIMEOUT_S = 0.2
    TICK_HZ = 300
    STARTUP_DELAY_S = 5.0

    def __init__(self) -> None:
        super().__init__("acceleration_test_node")

        self.state = "IDLE"
        self.linear_velocity = 0.0
        self.distance_travelled = 0.0
        self.start_distance = 0.0
        self.accel_end_distance = 0.0
        self.peak_velocity = 0.0
        self.accel_start_time = 0.0
        self.last_monitoring_time = 0.0
        self._stopped = False

        self.cmd_pub = self.create_publisher(Float32MultiArray, "/cmd_drive", 1)
        self.enable_pub = self.create_publisher(Bool, "/autopilot/enabled", 1)
        self.create_subscription(
            Float32MultiArray, "/monitoring/telemetry", self._monitoring_cb, 1
        )

        self.create_timer(1.0 / self.TICK_HZ, self._tick)
        self._start_timer = self.create_timer(self.STARTUP_DELAY_S, self._begin_test)

        self.get_logger().info(
            "Acceleration test node initialized "
            f"(target={self.TARGET_VELOCITY_M_S} m/s, "
            f"timeout={self.ACCEL_TIMEOUT_S}s, "
            f"throttle={self.ACCEL_THROTTLE})"
        )

    def _monitoring_cb(self, msg: Float32MultiArray) -> None:
        if len(msg.data) < 8:
            return
        self.linear_velocity = float(msg.data[6])
        self.distance_travelled = float(msg.data[7])
        self.last_monitoring_time = time.monotonic()

    def _begin_test(self) -> None:
        if self.state != "IDLE":
            return

        self._start_timer.cancel()
        self._start_timer = None

        self.enable_pub.publish(Bool(data=True))
        self.state = "ACCELERATING"
        self.start_distance = self.distance_travelled
        self.accel_start_time = time.monotonic()
        self.peak_velocity = 0.0
        self._publish_cmd(self.ACCEL_THROTTLE, 0.0)
        self.get_logger().info(
            f"Test started — throttle={self.ACCEL_THROTTLE}, "
            f"target={self.TARGET_VELOCITY_M_S} m/s, "
            f"timeout={self.ACCEL_TIMEOUT_S}s"
        )

    def _tick(self) -> None:
        if self.state == "IDLE":
            return

        monitoring_stale = (
            time.monotonic() - self.last_monitoring_time > self.MONITORING_TIMEOUT_S
        )
        if monitoring_stale:
            self.get_logger().error("Monitoring data stale — aborting test")
            self._finish(aborted=True)
            return

        now = time.monotonic()

        if self.state == "ACCELERATING":
            self._publish_cmd(self.ACCEL_THROTTLE, 0.0)
            elapsed = now - self.accel_start_time
            if self.linear_velocity >= self.TARGET_VELOCITY_M_S:
                self.peak_velocity = self.linear_velocity
                self.accel_end_distance = self.distance_travelled
                self.get_logger().info(
                    f"Target velocity reached: {self.peak_velocity:.3f} m/s "
                    f"at {elapsed:.2f}s"
                )
                self._begin_braking()
            elif elapsed >= self.ACCEL_TIMEOUT_S:
                self.peak_velocity = self.linear_velocity
                self.accel_end_distance = self.distance_travelled
                self.get_logger().info(
                    f"Timeout reached ({elapsed:.2f}s) — "
                    f"velocity={self.peak_velocity:.3f} m/s"
                )
                self._begin_braking()

        elif self.state == "BRAKING":
            self._publish_cmd(self.BRAKE_THROTTLE, 0.0)
            if abs(self.linear_velocity) <= self.VELOCITY_EPSILON:
                self._finish(aborted=False)

    def _begin_braking(self) -> None:
        self.state = "BRAKING"
        self._publish_cmd(self.BRAKE_THROTTLE, 0.0)
        self.get_logger().info("Braking...")

    def _finish(self, aborted: bool) -> None:
        self.state = "DONE"
        self._publish_cmd(0.0, 0.0)
        self.enable_pub.publish(Bool(data=False))

        brake_distance = self.distance_travelled - self.accel_end_distance
        accel_distance = self.accel_end_distance - self.start_distance
        total_distance = self.distance_travelled - self.start_distance

        if aborted:
            self.get_logger().error("--- Acceleration Test ABORTED ---")
        else:
            self.get_logger().info("--- Acceleration Test Complete ---")

        self.get_logger().info(f"  Peak velocity:     {self.peak_velocity:.3f} m/s")
        self.get_logger().info(f"  Accel distance:    {accel_distance:.3f} m")
        self.get_logger().info(f"  Brake distance:    {brake_distance:.3f} m")
        self.get_logger().info(f"  Total distance:    {total_distance:.3f} m")

    def _publish_cmd(self, throttle: float, steer: float) -> None:
        msg = Float32MultiArray()
        msg.data = [throttle, steer]
        self.cmd_pub.publish(msg)

    def destroy_node(self) -> bool:
        try:
            if not self._stopped:
                self._publish_cmd(0.0, 0.0)
                self.enable_pub.publish(Bool(data=False))
                self._stopped = True
        finally:
            return super().destroy_node()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = AccelerationTestNode()
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
