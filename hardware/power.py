"""Board power measurement through the Pi 5 PMIC (vcgencmd pmic_read_adc).

The PMIC reports current and voltage for every rail it generates. Summing I*V over
the rails gives the power delivered by the PMIC to the board (SoC, RAM, 3V3/1V8 rails,
and therefore the 40-pin header + Hailo HAT's 3V3 draw). It does not include regulator
losses or the 5 V rail feeding USB/PCIe directly, so treat total_w as a close lower
bound of what the 5 V input supply is delivering.
"""
import csv
import re
import threading
import time
from dataclasses import dataclass, field, asdict

from .base import Subsystem, run_cmd, HardwareError
from .cpu import CpuMonitor

_LINE = re.compile(r"^\s*(?P<rail>[0-9A-Z_]+?)_(?P<kind>[AV])\s+(?:current|volt)\(\d+\)=(?P<val>[0-9.]+)[AV]")


@dataclass
class PowerSample:
    timestamp: float
    total_w: float
    core_w: float
    core_v: float
    core_a: float
    ext5v_v: float
    temp_c: float
    arm_freq_mhz: float
    throttled_hex: str
    rails: dict = field(default_factory=dict)

    def as_dict(self):
        return asdict(self)


class BoardPower(Subsystem):
    name = "power"

    def __init__(self):
        self._cpu = CpuMonitor()

    def available(self):
        try:
            rc, _ = run_cmd(["vcgencmd", "pmic_read_adc"], check=False)
            return rc == 0
        except HardwareError:
            return False

    def rails(self):
        """{rail_name: {"current_a": float|None, "voltage_v": float|None}}"""
        _, out = run_cmd(["vcgencmd", "pmic_read_adc"])
        rails = {}
        for line in out.splitlines():
            m = _LINE.match(line)
            if not m:
                continue
            r = rails.setdefault(m["rail"], {"current_a": None, "voltage_v": None})
            r["current_a" if m["kind"] == "A" else "voltage_v"] = float(m["val"])
        if not rails:
            raise HardwareError("could not parse pmic_read_adc output")
        return rails

    @staticmethod
    def _power_of(rails, name):
        r = rails.get(name, {})
        if r.get("current_a") is None or r.get("voltage_v") is None:
            return 0.0
        return r["current_a"] * r["voltage_v"]

    @staticmethod
    def total_power_w(rails):
        return sum(BoardPower._power_of(rails, n) for n in rails)

    def sample(self):
        rails = self.rails()
        core = rails.get("VDD_CORE", {})
        return PowerSample(
            timestamp=time.time(),
            total_w=round(self.total_power_w(rails), 3),
            core_w=round(self._power_of(rails, "VDD_CORE"), 3),
            core_v=core.get("voltage_v") or 0.0,
            core_a=core.get("current_a") or 0.0,
            ext5v_v=rails.get("EXT5V", {}).get("voltage_v") or 0.0,
            temp_c=round(self._cpu.temperature_c(), 1),
            arm_freq_mhz=round(self._cpu.arm_freq_mhz()),
            throttled_hex="0x%x" % self._cpu.throttled_raw(),
            rails={k: dict(v) for k, v in rails.items()},
        )

    def status(self):
        s = self.sample()
        return {
            "total_w": s.total_w,
            "core_w": s.core_w,
            "core_v": s.core_v,
            "ext5v_v": s.ext5v_v,
            "temp_c": s.temp_c,
            "arm_freq_mhz": s.arm_freq_mhz,
            "throttled_hex": s.throttled_hex,
        }

    def logger(self, interval_s=1.0, csv_path=None):
        return PowerLogger(self, interval_s, csv_path)


class PowerLogger:
    """Background sampler (the Python equivalent of tests/pmic_sampler.sh).

        with power.logger(1.0, "power_idle.csv") as log:
            ...do work...
        print(log.stats())
    """

    CSV_FIELDS = ["elapsed", "core_i_A", "core_v_V", "core_w_W", "total_w_W", "ext5v_V",
                  "temp_C", "arm_freq_MHz", "throttled"]

    def __init__(self, power, interval_s=1.0, csv_path=None):
        self.power = power
        self.interval = interval_s
        self.csv_path = csv_path
        self.samples = []
        self._stop = threading.Event()
        self._thread = None
        self._t0 = None

    def start(self):
        self.samples = []
        self._stop.clear()
        self._t0 = time.time()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=self.interval + 5)
        if self.csv_path:
            self._write_csv()
        return self.samples

    def _run(self):
        while not self._stop.is_set():
            try:
                self.samples.append(self.power.sample())
            except HardwareError:
                pass
            self._stop.wait(self.interval)

    def _write_csv(self):
        with open(self.csv_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(self.CSV_FIELDS)
            for s in self.samples:
                w.writerow([round(s.timestamp - self._t0, 1), s.core_a, s.core_v, s.core_w,
                            s.total_w, s.ext5v_v, s.temp_c, s.arm_freq_mhz, s.throttled_hex])

    def stats(self):
        if not self.samples:
            return {"samples": 0}
        tw = [s.total_w for s in self.samples]
        cw = [s.core_w for s in self.samples]
        t = [s.temp_c for s in self.samples]
        f = [s.arm_freq_mhz for s in self.samples]
        return {
            "samples": len(self.samples),
            "duration_s": round(self.samples[-1].timestamp - self.samples[0].timestamp, 1),
            "total_w": {"avg": round(sum(tw) / len(tw), 3), "min": min(tw), "max": max(tw)},
            "core_w": {"avg": round(sum(cw) / len(cw), 3), "min": min(cw), "max": max(cw)},
            "temp_c": {"min": min(t), "max": max(t)},
            "arm_freq_mhz": {"min": min(f), "max": max(f)},
            "throttled_samples": sum(1 for s in self.samples if s.throttled_hex != "0x0"),
        }

    def __enter__(self):
        return self.start()

    def __exit__(self, *exc):
        self.stop()
        return False
