"""Tests the module switches and the acceleration test state machine."""
import json
import shutil
import sys
import time
from pathlib import Path

TESTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(TESTS_DIR / "stubs"))
sys.path.insert(0, str(TESTS_DIR.parent))

from lituanicax.modules import drive as drive_mod


class FakePWM:
    def __init__(self, channel, frequency_hz, initial_pulse_ns=0):
        self.channel, self.period_ns = channel, int(1e9 / frequency_hz)
        self.pulse_ns = initial_pulse_ns

    def set_duty_fraction(self, f): self.pulse_ns = int(self.period_ns * max(0.0, min(1.0, f)))
    def set_pulse_ns(self, ns): self.pulse_ns = int(ns)
    def stop(self): pass


drive_mod.SysfsHardwarePWM = FakePWM
drive_mod.Drive._pinctrl_set = staticmethod(lambda *a: None)

from lituanicax.main import main, settings_from_args, parse_args
from lituanicax.modules.acceleration_test import AccelerationTest
from lituanicax.robot import Robot

failures = []


def check(name, ok, detail=""):
    print(f"{'PASS' if ok else 'FAIL'}  {name}  {detail}")
    if not ok:
        failures.append(name)


# --- CLI -> settings ---
s = settings_from_args(parse_args(["--disable-tof", "--disable-policy"]))
check("legacy launch flags", not s.tof_cameras and not s.policy and s.drive and s.dashboard,
      str(s.enabled_modules()))

s = settings_from_args(parse_args(["--disable", "policy", "--enable", "acceleration_test",
                                   "--port", "9090"]))
check("generic enable/disable", not s.policy and s.acceleration_test and s.port == 9090,
      str(s.enabled_modules()))

check("--list-modules exits 0", main(["--list-modules"]) == 0)

# --- a stripped-down robot: only encoder + monitoring ---
s = settings_from_args(parse_args(
    ["--disable", "tof_cameras", "--disable", "policy", "--disable", "dashboard",
     "--disable", "drive"]))
robot = Robot(s)
check("only enabled modules built", [m.name for m in robot.modules] == ["encoder", "monitoring"],
      str([m.name for m in robot.modules]))
robot.start()
time.sleep(0.7)
check("stripped robot runs", robot.monitoring.battery_voltage > 7.0 and robot.encoder.linear_velocity != 0)
check("autopilot toggle without drive/policy", (robot.set_autopilot_enabled(True) or True)
      and robot.autopilot_enabled)
robot.stop()

# --- drive without monitoring stays blocked (matches ROS: no telemetry, no motion) ---
s = settings_from_args(parse_args(
    ["--disable", "tof_cameras", "--disable", "policy", "--disable", "dashboard",
     "--disable", "monitoring", "--disable", "encoder"]))
robot = Robot(s)
robot.start()
robot.set_autopilot_enabled(True)
robot.drive.command(1.0, 0.5)
check("drive blocked without monitoring", robot.drive.last_throttle == 0.0,
      f"throttle={robot.drive.last_throttle}")
robot.stop()

# --- a module that fails setup must not leave the rest running ---
s = settings_from_args(parse_args(
    ["--disable", "tof_cameras", "--disable", "policy", "--disable", "dashboard"]))
robot = Robot(s)
robot.drive.setup = lambda: (_ for _ in ()).throw(RuntimeError("PWM export denied"))
try:
    robot.start()
    check("failed setup raises", False)
except RuntimeError as exc:
    check("failed setup raises", "PWM export denied" in str(exc), str(exc))
check("failed start stops the others",
      not any(m._thread and m._thread.is_alive() for m in robot.modules),
      str([m.name for m in robot.modules if m._thread and m._thread.is_alive()]))

# --- acceleration test end to end ---
AccelerationTest.STARTUP_DELAY_S = 0.5
AccelerationTest.ACCEL_TIMEOUT_S = 1.5
AccelerationTest.TARGET_VELOCITY_M_S = 45

# The test brings the recorder with it: a calibration run whose numbers only
# reach the console is a run that has to be repeated. `test_logging.py` checks
# what it writes; here it only has to be present and out of the way.
RUN_DIR = TESTS_DIR / "accelrun"
shutil.rmtree(RUN_DIR, ignore_errors=True)

check("accel test implies --log",
      settings_from_args(parse_args(["--enable-accel-test"])).recorder)
check("skid test implies --log",
      settings_from_args(parse_args(["--enable", "skid_pad_test"])).recorder)
check("--no-log still wins over an implied one",
      not settings_from_args(parse_args(["--enable-skid-test", "--no-log"])).recorder)
check("--disable recorder still wins over an implied one",
      not settings_from_args(parse_args(["--enable-skid-test",
                                         "--disable", "recorder"])).recorder)

s = settings_from_args(parse_args(
    ["--disable", "tof_cameras", "--disable", "dashboard", "--disable-policy",
     "--enable-accel-test", "--log-dir", str(RUN_DIR)]))
robot = Robot(s)
check("accel test config", [m.name for m in robot.modules] ==
      ["recorder", "encoder", "monitoring", "drive", "acceleration_test"],
      str([m.name for m in robot.modules]))
robot.start()

states = []
throttles = []
deadline = time.monotonic() + 6.0
while time.monotonic() < deadline and robot.acceleration_test.state != "DONE":
    states.append(robot.acceleration_test.state)
    throttles.append(robot.drive.last_throttle)
    time.sleep(0.05)

check("accel test reached DONE", robot.acceleration_test.state == "DONE",
      f"state={robot.acceleration_test.state}")
check("accel test sequence", "IDLE" in states and "ACCELERATING" in states,
      f"states={sorted(set(states))}")
check("accel throttle applied", max(throttles, default=0) > 0.4,
      f"peak throttle={max(throttles, default=0):.2f}")
check("accel test disabled autopilot at end", not robot.autopilot_enabled)
robot.stop()

result = next(
    (json.loads(line) for line in (RUN_DIR / "events.jsonl").read_text().splitlines()
     if json.loads(line)["event"] == "finished"), None)
check("accel result recorded, not just printed",
      result is not None and not result["aborted"] and result["accel_distance"] > 0
      and result["total_distance"] >= result["accel_distance"],
      str(result))
shutil.rmtree(RUN_DIR, ignore_errors=True)

print()
print("FAILURES:", failures if failures else "none")
sys.exit(1 if failures else 0)
