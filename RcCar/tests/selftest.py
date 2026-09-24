#!/usr/bin/env python3
"""End-to-end self-test of the hardware framework against the real hardware.

    python3 tests/selftest.py [--batt-min 6.4] [--batt-max 8.4] [--skip camera,hailo,...]
                        [--bench-seconds 3] [--out results/selftest.json]
"""
import argparse
import json
import os
import sys
import time
import traceback
from datetime import datetime

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # rc_car/
sys.path.insert(0, ROOT)
from hardware import RP5Board, Steering, HardwareError, PWM_PINS  # noqa: E402
from hardware.base import run_cmd, read_sysfs  # noqa: E402
from hardware.pwm import find_pwm_chip  # noqa: E402

RESULTS = {}


def check(name, cond, detail=""):
    RESULTS.setdefault("checks", []).append({"name": name, "pass": bool(cond), "detail": detail})
    print("  [%s] %s %s" % ("PASS" if cond else "FAIL", name, detail))
    return bool(cond)


def section(title):
    print("\n=== %s ===" % title)


def test_cpu(b):
    section("CPU")
    st = b.cpu.status()
    print(" ", st)
    RESULTS["cpu"] = st
    check("cpu temp plausible", 15 < st["temp_c"] < 95, "%.1f C" % st["temp_c"])
    check("arm clock read", st["arm_freq_mhz"] > 100, "%d MHz" % st["arm_freq_mhz"])
    check("throttle flags read", "throttled_hex" in st, st["throttled_hex"])


def test_power(b):
    section("Board power (PMIC)")
    s = b.power.sample()
    print("  total=%.3f W core=%.3f W (%.3f V x %.3f A) ext5v=%.3f V" %
          (s.total_w, s.core_w, s.core_v, s.core_a, s.ext5v_v))
    check("total power > 0", s.total_w > 0.5, "%.3f W" % s.total_w)
    check("EXT5V ~ 5 V", 4.5 < s.ext5v_v < 5.5, "%.3f V" % s.ext5v_v)
    with b.power.logger(0.5) as log:
        time.sleep(2.2)
    st = log.stats()
    print("  logger:", st)
    RESULTS["power"] = {"sample": s.as_dict(), "logger_stats": st}
    check("power logger collected samples", st["samples"] >= 3, "%d samples" % st["samples"])


def test_hailo(b, bench_seconds):
    section("Hailo-8 (26 TOPS)")
    st = b.hailo.status()
    print(" ", st)
    RESULTS["hailo"] = {"status": st}
    check("hailo device present", st["available"], str(st.get("devices")))
    check("identified as Hailo-8", (st.get("board") or "").startswith("Hailo-8"), "%s / %s" % (st.get("board"), st.get("arch")))
    try:
        b.hailo.measure_power(1)
        RESULTS["hailo"]["measure_power"] = "supported"
        print("  measure-power: supported")
    except HardwareError as e:
        RESULTS["hailo"]["measure_power"] = "unsupported: " + str(e)
        print("  measure-power: unsupported on this module (expected) ->", str(e)[:80])
    r = b.hailo.benchmark(seconds=bench_seconds)
    print("  benchmark yolov8s: hw_only=%.1f fps streaming=%.1f fps latency=%.2f ms" %
          (r.hw_only_fps, r.streaming_fps, r.latency_ms))
    RESULTS["hailo"]["benchmark"] = r.as_dict()
    check("benchmark fps > 100", r.hw_only_fps > 100, "%.1f fps" % r.hw_only_fps)


def test_battery(b, vmin, vmax):
    section("MCP3202 ADC + battery")
    st = b.adc.status()
    print("  adc:", st)
    r = b.battery.read(samples=32)
    print("  battery: raw=%.1f (%d..%d) adc=%.4f V x %.4f -> %.3f V" %
          (r.raw, r.raw_min, r.raw_max, r.adc_volts, b.battery.divider_ratio, r.battery_volts))
    RESULTS["battery"] = b.battery.status()
    check("adc CH1 non-zero", r.raw > 0, "raw=%.1f" % r.raw)
    check("adc CH1 stable", (r.raw_max - r.raw_min) < 40, "spread=%d counts" % (r.raw_max - r.raw_min))
    check("battery within %.1f..%.1f V (2S LiPo)" % (vmin, vmax), vmin <= r.battery_volts <= vmax,
          "measured %.3f V" % r.battery_volts)


def _pin_high_fraction(gpio, n=300):
    hi = 0
    for _ in range(n):
        _, out = run_cmd(["pinctrl", "get", str(gpio)])
        if "| hi" in out:
            hi += 1
    return hi / n


def test_pwm(b):
    """Steering servo on GPIO12 is exercised over its calibrated range. GPIO13 is the ESC
    throttle: it is only inspected read-only (esc-neutral.service keeps it at 1500 us) so the
    test can never spin the motor or drop the ESC's neutral signal."""
    section("Hardware PWM: steering servo (GPIO12) + ESC throttle pin (GPIO13, read-only)")
    steer = Steering()
    try:
        st = steer.servo.status()
        print("  GPIO12: %s" % {k: st[k] for k in ("chip", "channel", "frequency_hz", "pulse_us",
                                                    "sysfs_period", "sysfs_duty", "sysfs_enable")})
        RESULTS.setdefault("pwm", {})[12] = st
        check("GPIO12 period = 20 ms (50 Hz)", st["sysfs_period"] == "20000000", st["sysfs_period"] + " ns")
        check("GPIO12 duty = steering centre", st["sysfs_duty"] == str(int(steer.center_us * 1000)), st["sysfs_duty"] + " ns")
        check("GPIO12 enabled", st["sysfs_enable"] == "1")
        _, pc = run_cmd(["pinctrl", "get", "12"])
        check("GPIO12 muxed to %s" % PWM_PINS[12][2], PWM_PINS[12][2] in pc, pc.strip())
        # crude waveform check: sample the pad level; expect ~7.5 % high at 1.5 ms / 20 ms
        frac = _pin_high_fraction(12)
        check("GPIO12 output toggling (~7.5% high)", 0.01 < frac < 0.25, "sampled high %.1f%%" % (frac * 100))
        steer.set(1.0)
        frac_hi = _pin_high_fraction(12, 150)
        steer.set(-1.0)
        frac_lo = _pin_high_fraction(12, 150)
        check("GPIO12 duty follows steering", frac_hi > frac_lo, "right: %.1f%%  left: %.1f%%" % (frac_hi * 100, frac_lo * 100))
        for pos in (-0.5, 0.0, 0.5, 0.0):
            steer.set(pos)
            time.sleep(0.25)
        print("  steering sweep left/centre/right done; centred.")
    finally:
        steer.close()

    esc_pwm = os.path.join(find_pwm_chip(), "pwm%d" % PWM_PINS[13][0])
    period, duty, enable = (read_sysfs(os.path.join(esc_pwm, f), default="?") for f in ("period", "duty_cycle", "enable"))
    print("  GPIO13 (ESC): %s period=%s duty=%s enable=%s" % (esc_pwm, period, duty, enable))
    RESULTS["pwm"][13] = {"period_ns": period, "duty_ns": duty, "enable": enable}
    check("GPIO13 ESC pin exported", os.path.isdir(esc_pwm), esc_pwm)
    check("GPIO13 ESC pin at 50 Hz", period == "20000000", period + " ns")
    check("GPIO13 ESC pin at neutral 1500 us", duty == "1500000", duty + " ns")
    check("GPIO13 ESC pin enabled", enable == "1")


def test_camera(b):
    section("Camera (IMX708)")
    cam = b.camera
    cam.open()
    modes = cam.sensor_modes()
    print("  sensor modes:", ["%dx%d@%.1f" % (m.width, m.height, m.max_fps) for m in modes])
    RESULTS["camera"] = {"sensor_modes": [m.__dict__ for m in modes], "runs": []}
    check("3 native modes", len(modes) == 3, "%d modes" % len(modes))
    plan = [(640, 480, 30), (1280, 720, 60), (1536, 864, 120), (1920, 1080, 30), (2304, 1296, 56), (4608, 2592, 14)]
    for w, h, fps in plan:
        t0 = time.monotonic()
        m = cam.set_mode(w, h, fps)
        t_switch = time.monotonic() - t0
        frame = cam.capture_frame()
        got = cam.measure_fps(1.5)
        run = {"req": [w, h, fps], "sensor": "%dx%d" % m.sensor.size, "eff_fps": m.fps,
               "measured_fps": got, "frame_shape": list(frame.shape), "switch_s": round(t_switch, 2)}
        RESULTS["camera"]["runs"].append(run)
        print("  %4dx%-4d @%3d -> raw %-9s fps=%6.2f measured=%6.2f shape=%s switch=%.2fs" %
              (w, h, fps, run["sensor"], m.fps, got, frame.shape, t_switch))
        check("%dx%d frame shape" % (w, h), tuple(frame.shape[:2]) == (h, w), str(frame.shape))
        check("%dx%d@%d achieved fps" % (w, h, fps), abs(got - m.fps) <= max(1.0, 0.05 * m.fps), "%.2f vs %.2f" % (got, m.fps))
    # fps change without reconfigure
    cam.set_mode(1280, 720, 60)
    cam.set_fps(15)
    got = cam.measure_fps(1.5)
    check("set_fps 60->15 live", abs(got - 15) <= 1.0, "%.2f fps" % got)
    out = os.path.join(ROOT, "results", "selftest_frame.jpg")
    cam.capture_file(out)
    check("jpeg written", os.path.getsize(out) > 10_000, out)
    cam.stop()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batt-min", type=float, default=6.4, help="2S LiPo: 3.2 V/cell empty")
    ap.add_argument("--batt-max", type=float, default=8.4, help="2S LiPo: 4.2 V/cell full")
    ap.add_argument("--skip", default="", help="comma list: cpu,power,hailo,battery,pwm,camera")
    ap.add_argument("--bench-seconds", type=int, default=3)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    skip = set(filter(None, a.skip.split(",")))
    RESULTS["started"] = datetime.now().isoformat(timespec="seconds")

    with RP5Board() as b:
        for name, fn in (("cpu", lambda: test_cpu(b)), ("power", lambda: test_power(b)),
                         ("hailo", lambda: test_hailo(b, a.bench_seconds)),
                         ("battery", lambda: test_battery(b, a.batt_min, a.batt_max)),
                         ("pwm", lambda: test_pwm(b)), ("camera", lambda: test_camera(b))):
            if name in skip:
                print("\n=== %s: skipped ===" % name)
                continue
            try:
                fn()
            except Exception as e:  # noqa: BLE001
                traceback.print_exc()
                check("%s section ran" % name, False, str(e).splitlines()[0][:200])

    checks = RESULTS.get("checks", [])
    npass = sum(1 for c in checks if c["pass"])
    print("\n========== %d/%d checks passed ==========" % (npass, len(checks)))
    for c in checks:
        if not c["pass"]:
            print("  FAILED:", c["name"], c["detail"])
    out = a.out or os.path.join(ROOT, "results",
                                "selftest_%s.json" % datetime.now().strftime("%Y%m%d_%H%M%S"))
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump(RESULTS, f, indent=2, default=str)
    print("results ->", out)
    return 0 if npass == len(checks) else 1


if __name__ == "__main__":
    sys.exit(main())
