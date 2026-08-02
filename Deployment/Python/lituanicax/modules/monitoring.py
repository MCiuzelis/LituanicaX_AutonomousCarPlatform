"""Battery, motor current, and CPU health monitoring.

Replaces `monitoring_node.py`.  Reads the SPI ADC, folds in the encoder state,
and raises the `undervoltage` / `overcurrent` flags that `Drive` checks before
it lets the motor run.
"""

from __future__ import annotations

import time

import spidev

from ..config import (
    ADC_BURST_SAMPLES,
    ADC_BUS,
    ADC_CHANNEL_BATTERY,
    ADC_CHANNEL_CURRENT,
    ADC_DEVICE,
    ADC_MAX,
    ADC_SPI_MAX_SPEED_HZ,
    ADC_V_REF,
    BATTERY_SAG_WARN_V,
    K_RATIO,
    MOTOR_CURRENT_LPF_ALPHA,
    OVERCURRENT_THRESHOLD,
    OVERCURRENT_TRIP_SEC,
    R_DIVIDER_BOTTOM,
    R_DIVIDER_TOP,
    R_SENSE,
    TELEMETRY_HZ,
    UNDERVOLTAGE_THRESHOLD,
)
from ..module import Module
from .encoder import Encoder


class Monitoring(Module):
    def __init__(self, encoder: Encoder | None = None, rate_hz: float = TELEMETRY_HZ) -> None:
        super().__init__("monitoring", rate_hz)

        self.encoder = encoder

        self.battery_voltage = 0.0
        #: Lowest battery sample seen in the last tick, not the tick average.
        #: A sag deep enough to trip the motor driver can be over in a few ms,
        #: so the mean (and anything downstream of it) averages it away.
        self.battery_min = 0.0
        self.motor_current = 0.0
        #: Highest current sample seen in the last tick, same reasoning.
        self.motor_current_peak = 0.0
        self.cpu_usage = 0.0
        self.cpu_temperature = 0.0
        self.linear_velocity = 0.0
        self.distance_travelled = 0.0
        self.undervoltage = False
        self.overcurrent = False
        self.last_update_time = 0.0

        self.spi: spidev.SpiDev | None = None
        self._motor_current_filtered = 0.0
        self._last_cpu_total: int | None = None
        self._last_cpu_idle: int | None = None
        self._last_sag_warn_time = 0.0
        self._over_since: float | None = None

    def setup(self) -> None:
        self.spi = spidev.SpiDev()
        self.spi.open(ADC_BUS, ADC_DEVICE)
        self.spi.max_speed_hz = ADC_SPI_MAX_SPEED_HZ
        self.log.info(
            "Monitoring initialized "
            f"(undervoltage<{UNDERVOLTAGE_THRESHOLD:.2f} V, overcurrent>{OVERCURRENT_THRESHOLD:.2f} A)"
        )

    def update(self) -> None:
        try:
            voltages, currents = self._read_burst()
            self.battery_voltage = sum(voltages) / len(voltages)
            self.battery_min = min(voltages)
            self.motor_current_peak = max(currents)

            raw_current = sum(currents) / len(currents)
            self._motor_current_filtered = (
                MOTOR_CURRENT_LPF_ALPHA * raw_current
                + (1.0 - MOTOR_CURRENT_LPF_ALPHA) * self._motor_current_filtered
            )
            self.motor_current = self._motor_current_filtered
            self._warn_on_sag()
        except Exception as exc:
            self.log.warning(f"ADC telemetry read failed: {exc}")
            self.battery_voltage = 0.0
            self.battery_min = 0.0
            self.motor_current = 0.0
            self.motor_current_peak = 0.0
            self._motor_current_filtered = 0.0

        self.cpu_usage = self._read_cpu_usage()
        self.cpu_temperature = self._read_cpu_temperature()

        if self.encoder is not None:
            self.linear_velocity = self.encoder.linear_velocity
            self.distance_travelled = self.encoder.distance_travelled

        now = time.monotonic()
        self.undervoltage = 0.0 < self.battery_voltage < UNDERVOLTAGE_THRESHOLD
        self.overcurrent = self._debounced_overcurrent(now)
        self.last_update_time = now

    def shutdown(self) -> None:
        if self.spi is not None:
            self.spi.close()
            self.spi = None

    # --- ADC ---

    def _read_burst(self) -> tuple[list[float], list[float]]:
        """Interleaved battery/current samples across one tick.

        Both channels are read in the same pass so a current spike and the
        voltage sag it causes line up in time.
        """
        voltages: list[float] = []
        currents: list[float] = []
        for _ in range(max(1, ADC_BURST_SAMPLES)):
            voltages.append(self._read_battery_voltage())
            currents.append(self._read_motor_current())
        return voltages, currents

    def _debounced_overcurrent(self, now: float) -> bool:
        """True only after the current has stayed over the limit for a while."""
        if self.motor_current <= OVERCURRENT_THRESHOLD:
            self._over_since = None
            return False
        if self._over_since is None:
            self._over_since = now
            return False
        held_for = now - self._over_since
        if held_for < OVERCURRENT_TRIP_SEC:
            return False
        if not self.overcurrent:
            self.log.warning(
                f"Overcurrent: {self.motor_current:.1f} A sustained for "
                f"{held_for:.2f} s (limit {OVERCURRENT_THRESHOLD:.1f} A) — drive blocked"
            )
        return True

    def _warn_on_sag(self) -> None:
        """Surface the sag in the console; the dashboard averages are too slow."""
        if self.battery_min >= BATTERY_SAG_WARN_V:
            return
        now = time.monotonic()
        if now - self._last_sag_warn_time < 1.0:
            return
        self._last_sag_warn_time = now
        self.log.warning(
            f"Battery sagged to {self.battery_min:.2f} V "
            f"(mean {self.battery_voltage:.2f} V, peak current "
            f"{self.motor_current_peak:.1f} A) — at or below the motor driver's "
            "own undervoltage shutdown, which cuts the motor without the "
            "software knowing"
        )

    def _read_adc(self, channel: int) -> int:
        config = 0x80 if channel == 0 else 0xC0
        adc_resp = self.spi.xfer2([1, config, 0])
        return ((adc_resp[1] & 0x0F) << 8) + adc_resp[2]

    @staticmethod
    def _adc_to_voltage(raw: int) -> float:
        return (raw / ADC_MAX) * ADC_V_REF

    def _read_motor_current(self) -> float:
        v_sense = self._adc_to_voltage(self._read_adc(ADC_CHANNEL_CURRENT))
        return (v_sense / R_SENSE) * K_RATIO

    def _read_battery_voltage(self) -> float:
        v_adc = self._adc_to_voltage(self._read_adc(ADC_CHANNEL_BATTERY))
        return v_adc * (R_DIVIDER_TOP + R_DIVIDER_BOTTOM) / R_DIVIDER_BOTTOM

    # --- host health ---

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
