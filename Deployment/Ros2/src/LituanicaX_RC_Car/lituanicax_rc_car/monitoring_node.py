from __future__ import annotations

import rclpy
import spidev
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray

from .common import (
    ADC_BUS,
    ADC_CHANNEL_BATTERY,
    ADC_CHANNEL_CURRENT,
    ADC_DEVICE,
    ADC_MAX,
    ADC_SPI_MAX_SPEED_HZ,
    ADC_V_REF,
    FPS_TARGET,
    K_RATIO,
    MOTOR_CURRENT_LPF_ALPHA,
    OVERCURRENT_THRESHOLD,
    R_DIVIDER_BOTTOM,
    R_DIVIDER_TOP,
    R_SENSE,
    UNDERVOLTAGE_THRESHOLD,
)


class MonitoringNode(Node):
    def __init__(self) -> None:
        super().__init__("monitoring_node")

        self.declare_parameter("telemetry_hz", FPS_TARGET)

        self.battery_voltage = 0.0
        self.motor_current = 0.0
        self._motor_current_filtered = 0.0
        self.cpu_usage = 0.0
        self.cpu_temperature = 0.0
        self.undervoltage = False
        self.overcurrent = False
        self._last_cpu_total: int | None = None
        self._last_cpu_idle: int | None = None
        self._stopped = False

        self.spi = spidev.SpiDev()
        self.spi.open(ADC_BUS, ADC_DEVICE)
        self.spi.max_speed_hz = ADC_SPI_MAX_SPEED_HZ

        self.telemetry_pub = self.create_publisher(Float32MultiArray, "/monitoring/telemetry", 1)

        telemetry_period = 1.0 / float(self.get_parameter("telemetry_hz").value)
        self.timer = self.create_timer(telemetry_period, self._telemetry_tick)

        self.get_logger().info(
            "Monitoring node initialized "
            f"(undervoltage<{UNDERVOLTAGE_THRESHOLD:.2f} V, overcurrent>{OVERCURRENT_THRESHOLD:.2f} A)"
        )

    def _read_adc(self, channel: int) -> int:
        config = 0x80 if channel == 0 else 0xC0
        adc_resp = self.spi.xfer2([1, config, 0])
        return ((adc_resp[1] & 0x0F) << 8) + adc_resp[2]

    @staticmethod
    def _adc_to_voltage(raw: int) -> float:
        return (raw / ADC_MAX) * ADC_V_REF

    def _read_motor_current(self) -> float:
        raw = self._read_adc(ADC_CHANNEL_CURRENT)
        v_sense = self._adc_to_voltage(raw)
        return (v_sense / R_SENSE) * K_RATIO

    def _read_battery_voltage(self) -> float:
        raw = self._read_adc(ADC_CHANNEL_BATTERY)
        v_adc = self._adc_to_voltage(raw)
        return v_adc * (R_DIVIDER_TOP + R_DIVIDER_BOTTOM) / R_DIVIDER_BOTTOM

    def _read_cpu_usage(self) -> float:
        try:
            with open("/proc/stat") as f:
                fields = [int(value) for value in f.readline().split()[1:]]
        except OSError:
            return 0.0

        idle = fields[3] + fields[4]
        total = sum(fields)
        if self._last_cpu_total is None or self._last_cpu_idle is None:
            self._last_cpu_total = total
            self._last_cpu_idle = idle
            return 0.0

        total_delta = total - self._last_cpu_total
        idle_delta = idle - self._last_cpu_idle
        self._last_cpu_total = total
        self._last_cpu_idle = idle
        if total_delta <= 0:
            return 0.0
        return max(0.0, min(100.0, (1.0 - idle_delta / total_delta) * 100.0))

    @staticmethod
    def _read_cpu_temperature() -> float:
        try:
            with open("/sys/class/thermal/thermal_zone0/temp") as f:
                return float(f.read().strip()) / 1000.0
        except (OSError, ValueError):
            return 0.0

    def _telemetry_tick(self) -> None:
        try:
            self.battery_voltage = self._read_battery_voltage()
            raw_current = self._read_motor_current()
            self._motor_current_filtered = (
                MOTOR_CURRENT_LPF_ALPHA * raw_current
                + (1.0 - MOTOR_CURRENT_LPF_ALPHA) * self._motor_current_filtered
            )
            self.motor_current = self._motor_current_filtered
        except Exception as exc:
            self.get_logger().warning(f"ADC telemetry read failed: {exc}")
            self.battery_voltage = 0.0
            self.motor_current = 0.0
            self._motor_current_filtered = 0.0

        self.cpu_usage = self._read_cpu_usage()
        self.cpu_temperature = self._read_cpu_temperature()

        self.undervoltage = self.battery_voltage > 0.0 and self.battery_voltage < UNDERVOLTAGE_THRESHOLD
        self.overcurrent = self.motor_current > OVERCURRENT_THRESHOLD

        msg = Float32MultiArray()
        msg.data = [
            float(self.battery_voltage),
            float(self.motor_current),
            float(self.cpu_usage),
            float(self.cpu_temperature),
            1.0 if self.undervoltage else 0.0,
            1.0 if self.overcurrent else 0.0,
        ]
        self.telemetry_pub.publish(msg)

    def destroy_node(self) -> bool:
        try:
            if not self._stopped:
                self.spi.close()
                self._stopped = True
        finally:
            return super().destroy_node()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = MonitoringNode()
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
