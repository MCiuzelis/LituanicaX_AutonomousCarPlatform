"""MCP3202 12-bit 2-channel SPI ADC via the kernel IIO driver (mcp320x, SPI0 CS0 / GPIO8).

config.txt:  dtoverlay=mcp3202,spi0-0-present
sysfs:       /sys/bus/iio/devices/iio:deviceN/in_voltage{0,1}_raw   (0..4095)
VREF = 3.3 V is not declared in the device tree, so scaling is done here.

Only CH1 is populated: it reads the battery through a 15k (top) / 4.7k (bottom) divider,
so V_batt = V_adc * (15k + 4.7k) / 4.7k = V_adc * 4.191.
"""
import glob
import os
import time
from dataclasses import dataclass

from .base import Subsystem, read_sysfs, HardwareError

IIO_GLOB = "/sys/bus/iio/devices/iio:device*"


class MCP3202(Subsystem):
    name = "mcp3202"
    RESOLUTION = 4095  # 12-bit full scale

    def __init__(self, vref=3.3, device_name="mcp3202"):
        self.vref = float(vref)
        self.device_name = device_name
        self._dev = None

    @property
    def device(self):
        if self._dev is None:
            for d in sorted(glob.glob(IIO_GLOB)):
                if read_sysfs(os.path.join(d, "name"), default="") == self.device_name:
                    self._dev = d
                    break
            if self._dev is None:
                raise HardwareError("no IIO device named %r (is dtoverlay=mcp3202,spi0-0-present active?)"
                                    % self.device_name)
        return self._dev

    def available(self):
        try:
            return os.path.exists(self.device)
        except HardwareError:
            return False

    def _read(self, fname, retries=5):
        path = os.path.join(self.device, fname)
        last = None
        for _ in range(retries):
            try:
                return int(read_sysfs(path))
            except (HardwareError, ValueError) as e:
                last = e
                time.sleep(0.005)
        raise HardwareError("cannot read %s: %s" % (path, last))

    def read_raw(self, channel):
        """Single-ended 12-bit count for channel 0 or 1."""
        if channel not in (0, 1):
            raise ValueError("MCP3202 has channels 0 and 1")
        return self._read("in_voltage%d_raw" % channel)

    def read_raw_differential(self, positive=1):
        """Differential count: CH1-CH0 (positive=1) or CH0-CH1 (positive=0)."""
        return self._read("in_voltage%d-voltage%d_raw" % (positive, 1 - positive))

    def raw_to_volts(self, raw):
        return raw * self.vref / self.RESOLUTION

    def read_voltage(self, channel):
        return self.raw_to_volts(self.read_raw(channel))

    def read_average(self, channel, samples=16, interval_s=0.002):
        """(avg_raw, min_raw, max_raw) over `samples` reads."""
        vals = []
        for _ in range(samples):
            vals.append(self.read_raw(channel))
            if interval_s:
                time.sleep(interval_s)
        return sum(vals) / len(vals), min(vals), max(vals)

    def status(self):
        return {
            "device": self.device,
            "vref": self.vref,
            "ch0_raw": self.read_raw(0),
            "ch1_raw": self.read_raw(1),
            "ch0_v": round(self.read_voltage(0), 4),
            "ch1_v": round(self.read_voltage(1), 4),
        }


@dataclass
class BatteryReading:
    raw: float
    raw_min: int
    raw_max: int
    adc_volts: float
    battery_volts: float
    samples: int

    def as_dict(self):
        return {k: (round(v, 4) if isinstance(v, float) else v) for k, v in self.__dict__.items()}


class BatteryMonitor(Subsystem):
    """Battery voltage through the resistor divider on one MCP3202 channel."""
    name = "battery"

    def __init__(self, adc=None, channel=1, r_top=15_000.0, r_bottom=4_700.0, vref=3.3):
        self.adc = adc or MCP3202(vref=vref)
        self.channel = channel
        self.r_top = float(r_top)
        self.r_bottom = float(r_bottom)

    @property
    def divider_ratio(self):
        """V_batt / V_adc"""
        return (self.r_top + self.r_bottom) / self.r_bottom

    @property
    def max_battery_volts(self):
        """Battery voltage that puts the ADC input at VREF (full scale)."""
        return self.adc.vref * self.divider_ratio

    def available(self):
        return self.adc.available()

    def read(self, samples=16):
        avg, lo, hi = self.adc.read_average(self.channel, samples)
        adc_v = self.adc.raw_to_volts(avg)
        return BatteryReading(avg, lo, hi, adc_v, adc_v * self.divider_ratio, samples)

    def voltage(self, samples=16):
        return self.read(samples).battery_volts

    def status(self):
        r = self.read()
        d = r.as_dict()
        d.update({"channel": self.channel, "divider_ratio": round(self.divider_ratio, 4),
                  "full_scale_v": round(self.max_battery_volts, 2)})
        return d
