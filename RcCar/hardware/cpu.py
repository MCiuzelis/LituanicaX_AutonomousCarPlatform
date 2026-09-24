"""CPU temperature / clock / throttling of the Raspberry Pi 5 (BCM2712)."""
import os

from .base import Subsystem, read_sysfs, run_cmd

THERMAL_ZONE = "/sys/class/thermal/thermal_zone0/temp"
CPUFREQ_CUR = "/sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq"

# bit meanings of `vcgencmd get_throttled`
THROTTLE_FLAGS = {
    0: "under_voltage_now",
    1: "arm_freq_capped_now",
    2: "throttled_now",
    3: "soft_temp_limit_now",
    16: "under_voltage_occurred",
    17: "arm_freq_capped_occurred",
    18: "throttled_occurred",
    19: "soft_temp_limit_occurred",
}


class CpuMonitor(Subsystem):
    name = "cpu"

    def available(self):
        return os.path.exists(THERMAL_ZONE)

    def temperature_c(self):
        """SoC temperature in degrees C (kernel thermal zone, no vcgencmd needed)."""
        return int(read_sysfs(THERMAL_ZONE)) / 1000.0

    def arm_freq_mhz(self):
        try:
            return int(read_sysfs(CPUFREQ_CUR)) / 1000.0
        except Exception:
            _, out = run_cmd(["vcgencmd", "measure_clock", "arm"])
            return int(out.strip().split("=")[1]) / 1e6

    def throttled_raw(self):
        _, out = run_cmd(["vcgencmd", "get_throttled"])
        return int(out.strip().split("=")[1], 16)

    def throttled(self):
        """Decoded get_throttled bits -> {flag_name: bool}."""
        raw = self.throttled_raw()
        return {name: bool(raw >> bit & 1) for bit, name in THROTTLE_FLAGS.items()}

    def load_average(self):
        return os.getloadavg()

    def status(self):
        raw = self.throttled_raw()
        return {
            "temp_c": round(self.temperature_c(), 1),
            "arm_freq_mhz": round(self.arm_freq_mhz()),
            "throttled_hex": "0x%x" % raw,
            "throttled_any": raw != 0,
            "load_1m": round(self.load_average()[0], 2),
        }
