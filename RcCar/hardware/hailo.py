"""Hailo-8 (26 TOPS) AI accelerator on the Pi AI HAT, driven through hailortcli.

The python `hailo_platform` bindings are not installed on this image, so this wraps the
CLI that the original hardware tests used (identify / scan / benchmark / measure-power).
"""
import glob
import os
import re
from dataclasses import dataclass, field

from .base import Subsystem, run_cmd, HardwareError

MODEL_DIR = "/usr/share/hailo-models"
DEFAULT_HEF = os.path.join(MODEL_DIR, "yolov8s_h8.hef")

_FPS_HW = re.compile(r"FPS\s+\(hw_only\)\s*=\s*([0-9.]+)")
_FPS_STREAM = re.compile(r"\(streaming\)\s*=\s*([0-9.]+)")
_LATENCY = re.compile(r"Latency\s+\(hw\)\s*=\s*([0-9.]+)\s*ms")
_PROGRESS_FPS = re.compile(r"FPS:\s*([0-9.]+)")


@dataclass
class BenchmarkResult:
    hef: str
    seconds: int
    hw_only_fps: float = 0.0
    streaming_fps: float = 0.0
    latency_ms: float = 0.0
    raw: str = field(default="", repr=False)

    def as_dict(self):
        return {"hef": self.hef, "seconds": self.seconds, "hw_only_fps": self.hw_only_fps,
                "streaming_fps": self.streaming_fps, "latency_ms": self.latency_ms}


class HailoBoard(Subsystem):
    name = "hailo"

    def __init__(self, device_id=None):
        self.device_id = device_id
        self._identity = None

    # ---- discovery -------------------------------------------------------
    def available(self):
        return os.path.exists("/dev/hailo0") and bool(self.scan())

    def scan(self):
        """List of device ids (PCIe BDFs) reported by `hailortcli scan`."""
        rc, out = run_cmd(["hailortcli", "scan"], check=False)
        if rc != 0:
            return []
        return re.findall(r"Device:\s*(\S+)", out)

    def identify(self, refresh=False):
        """Parsed output of `hailortcli fw-control identify` as a dict."""
        if self._identity and not refresh:
            return self._identity
        _, out = run_cmd(self._dev(["hailortcli", "fw-control", "identify"]))
        info = {}
        for line in out.splitlines():
            if ":" in line and not line.startswith("Executing") and not line.startswith("Identifying"):
                k, v = line.split(":", 1)
                # firmware pads strings with NULs (e.g. "Hailo-8\x00\x00...")
                info[k.strip().lower().replace(" ", "_")] = v.replace("\x00", "").strip()
        if not info:
            raise HardwareError("hailortcli identify returned nothing: %s" % out[:200])
        self._identity = info
        return info

    def models(self):
        """Available compiled networks. *_h8.hef files target this Hailo-8; *_h8l are for
        the 13 TOPS Hailo-8L (they still run here); *_h10 are Hailo-10 only."""
        return sorted(glob.glob(os.path.join(MODEL_DIR, "*.hef")))

    def _dev(self, args):
        if self.device_id:
            args = args + ["-s", self.device_id]
        return args

    # ---- workloads -------------------------------------------------------
    def benchmark(self, hef=DEFAULT_HEF, seconds=10, batch_size=1, power_mode="performance"):
        """Run `hailortcli benchmark`: measures hw-only fps, streaming fps and hw latency.
        Total runtime is ~3 x seconds (one pass per measurement mode)."""
        if not os.path.exists(hef):
            raise HardwareError("HEF not found: %s" % hef)
        args = self._dev(["hailortcli", "benchmark", hef, "-t", str(seconds),
                          "--batch-size", str(batch_size), "--power-mode", power_mode])
        _, out = run_cmd(args, timeout=seconds * 3 + 60)
        out = out.replace("\r", "\n")
        res = BenchmarkResult(hef=hef, seconds=seconds, raw=out)
        m = _FPS_HW.search(out)
        if m:
            res.hw_only_fps = float(m.group(1))
        m = _FPS_STREAM.search(out)
        if m:
            res.streaming_fps = float(m.group(1))
        m = _LATENCY.search(out)
        if m:
            res.latency_ms = float(m.group(1))
        if not res.hw_only_fps:
            # summary missing (e.g. killed early): fall back to last progress line
            fps = _PROGRESS_FPS.findall(out)
            if fps:
                res.hw_only_fps = float(fps[-1])
            else:
                raise HardwareError("benchmark produced no FPS figures: %s" % out[-400:])
        return res

    def measure_power(self, duration_s=5):
        """On-chip power telemetry via `hailortcli measure-power`.
        NOTE: this Hailo-8 module answers HAILO_UNSUPPORTED_OPCODE (no current monitor on the
        AI HAT), so this raises HardwareError there; use BoardPower for system power instead."""
        rc, out = run_cmd(self._dev(["hailortcli", "measure-power", "--duration", str(duration_s)]),
                          timeout=duration_s + 30, check=False)
        if rc != 0 or "UNSUPPORTED" in out:
            raise HardwareError("measure-power unsupported on this board: %s"
                                % out.strip().splitlines()[-1][:160])
        result = {}
        for line in out.splitlines():
            m = re.match(r"\s*(Average|Min|Max)\s*(?:power)?\s*[:=]\s*([0-9.]+)\s*(\w+)", line, re.I)
            if m:
                result[m.group(1).lower()] = (float(m.group(2)), m.group(3))
        result["raw"] = out
        return result

    def status(self):
        ok = self.available()
        st = {"available": ok, "devices": self.scan(), "models": len(self.models())}
        if ok:
            i = self.identify()
            st.update({
                "board": i.get("board_name"),
                "arch": i.get("device_architecture"),
                "firmware": i.get("firmware_version"),
                "serial": i.get("serial_number"),
            })
        return st
