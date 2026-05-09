from __future__ import annotations

import os
import time

import RPi.GPIO as GPIO
import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rpi_hardware_pwm import HardwarePWM
from std_msgs.msg import Bool, Float32MultiArray

from .common import (
    COMMAND_TIMEOUT_SEC,
    FPS_TARGET,
    MAX_THROTTLE,
    MONITORING_TIMEOUT_SEC,
    MOTOR_DIRECTION,
    MOTOR_GPIO,
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
    SERVO_PWM_FREQ,
)


class DriveNode(Node):
    def __init__(self) -> None:
        super().__init__("drive_node")

        self.declare_parameter("command_timeout_sec", COMMAND_TIMEOUT_SEC)
        self.declare_parameter("monitoring_timeout_sec", MONITORING_TIMEOUT_SEC)
        self.declare_parameter("telemetry_hz", FPS_TARGET)

        self.command_timeout_sec = float(self.get_parameter("command_timeout_sec").value)
        self.monitoring_timeout_sec = float(self.get_parameter("monitoring_timeout_sec").value)
        self.enabled = False
        self.undervoltage = False
        self.overcurrent = False
        self.last_monitoring_time = 0.0
        self.last_command_time = 0.0
        self.last_throttle = 0.0
        self.last_steer = 0.0
        self._stopped = False

        motor_chip = self._find_pwmchip(MOTOR_GPIO)
        servo_chip = self._find_pwmchip(SERVO_GPIO)

        GPIO.setmode(GPIO.BCM)
        GPIO.setup(PIN_INA, GPIO.OUT)
        GPIO.setup(PIN_INB, GPIO.OUT)

        self.motor_pwm = HardwarePWM(pwm_channel=0, hz=MOTOR_PWM_FREQ, chip=motor_chip)
        self.servo_pwm = HardwarePWM(pwm_channel=0, hz=SERVO_PWM_FREQ, chip=servo_chip)
        self.motor_pwm.start(0)
        self.servo_pwm.start(7.5)

        self.telemetry_pub = self.create_publisher(Float32MultiArray, "/drive/telemetry", 1)
        self.create_subscription(Float32MultiArray, "/cmd_drive", self._cmd_cb, 1)
        self.create_subscription(Float32MultiArray, "/monitoring/telemetry", self._monitoring_cb, 1)
        self.create_subscription(Bool, "/autopilot/enabled", self._enabled_cb, 1)

        telemetry_period = 1.0 / float(self.get_parameter("telemetry_hz").value)
        self.timer = self.create_timer(telemetry_period, self._telemetry_tick)

        self.get_logger().info("Drive node initialized")

    @staticmethod
    def _find_pwmchip(gpio_pin: int) -> int:
        for entry in os.listdir("/sys/class/pwm"):
            if not entry.startswith("pwmchip"):
                continue
            chip = int(entry[7:])
            try:
                with open(f"/sys/class/pwm/{entry}/device/uevent") as f:
                    for line in f:
                        if line.startswith("OF_FULLNAME="):
                            addr = line.strip().split("@")[-1]
                            if int(addr, 16) == gpio_pin:
                                return chip
            except OSError:
                continue
        raise RuntimeError(f"No pwmchip found for GPIO{gpio_pin}")

    def _enabled_cb(self, msg: Bool) -> None:
        self.enabled = bool(msg.data)
        if not self.enabled:
            self._apply_drive(0.0, 0.0)

    def _monitoring_cb(self, msg: Float32MultiArray) -> None:
        if len(msg.data) < 6:
            return
        self.last_monitoring_time = time.monotonic()
        self.undervoltage = bool(msg.data[4])
        self.overcurrent = bool(msg.data[5])
        if self.undervoltage or self.overcurrent:
            self._apply_drive(0.0, 0.0)

    def _cmd_cb(self, msg: Float32MultiArray) -> None:
        if len(msg.data) < 2:
            return

        self.last_command_time = time.monotonic()
        throttle = float(msg.data[0])
        steer = float(msg.data[1])

        if self._can_drive():
            self._apply_drive(throttle, steer)
        else:
            self._apply_drive(0.0, 0.0)

    def _can_drive(self) -> bool:
        monitoring_fresh = time.monotonic() - self.last_monitoring_time <= self.monitoring_timeout_sec
        return self.enabled and monitoring_fresh and not self.undervoltage and not self.overcurrent

    def _apply_drive(self, throttle: float, steer: float) -> None:
        self.last_throttle = throttle
        self.last_steer = steer
        self._set_throttle(throttle)
        self._set_steering(steer)

    def _set_throttle(self, value: float) -> None:
        value = max(-1.0, min(1.0, value))
        value *= max(0.0, min(1.0, MAX_THROTTLE))
        value *= MOTOR_DIRECTION

        GPIO.output(PIN_INA, GPIO.HIGH if value >= 0 else GPIO.LOW)
        GPIO.output(PIN_INB, GPIO.HIGH if value < 0 else GPIO.LOW)
        self.motor_pwm.change_duty_cycle(abs(value) * 100.0)

    def _set_steering(self, value: float) -> None:
        value = max(-1.0, min(1.0, value))
        value = value * SERVO_DIRECTION * SERVO_AMPLITUDE

        span = (SERVO_MAX_US - SERVO_MIN_US) / 2
        pulse = SERVO_CENTER_US + SERVO_OFFSET_US + (value * span)
        pulse = max(SERVO_MIN_US, min(SERVO_MAX_US, pulse))

        duty = (pulse / 20000.0) * 100.0
        self.servo_pwm.change_duty_cycle(duty)

    def _telemetry_tick(self) -> None:
        if time.monotonic() - self.last_command_time > self.command_timeout_sec or not self._can_drive():
            self._apply_drive(0.0, 0.0)

        msg = Float32MultiArray()
        msg.data = [
            float(self.last_throttle),
            float(self.last_steer),
            1.0 if self.enabled else 0.0,
            1.0 if self.undervoltage or self.overcurrent else 0.0,
        ]
        self.telemetry_pub.publish(msg)

    def destroy_node(self) -> bool:
        try:
            if not self._stopped:
                self._apply_drive(0.0, 0.0)
                self.motor_pwm.stop()
                self.servo_pwm.stop()
                GPIO.cleanup()
                self._stopped = True
        finally:
            return super().destroy_node()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = DriveNode()
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
