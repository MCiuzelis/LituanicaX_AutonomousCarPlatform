"""Checks the command path from policy output to PWM duty, and the frame-arrival
policy tick.

These are the pieces the sim2real analysis moved, and every one of them is a
place where a sign or an ordering mistake is silent on the bench and expensive on
the track:

  * MAX_THROTTLE must cap forward duty and leave braking alone (it used to scale
    both, quartering the brake at MAX_THROTTLE = 0.4).
  * BRAKE_MAGNITUDE must be applied exactly once, in PolicyRunner.
  * The slew limiter must be bypassable and must not leave stale ramp state.
  * PolicyRunner must tick when BOTH cameras have advanced, and must keep ticking
    when a camera dies.

Runs off-hardware on the stubs; unlike test_stack.py it needs neither
onnxruntime nor Python 3.13's SharedMemory(track=), so it runs anywhere.
"""
import json
import math
import shutil
import sys
import threading
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
        self.duty = 0.0

    def set_duty_fraction(self, f):
        self.duty = max(0.0, min(1.0, f))
        self.pulse_ns = int(self.period_ns * self.duty)

    def set_pulse_ns(self, ns): self.pulse_ns = int(ns)
    def stop(self): pass


drive_mod.SysfsHardwarePWM = FakePWM
drive_mod.Drive._pinctrl_set = staticmethod(lambda *a: None)

from lituanicax import config  # noqa: E402
from lituanicax.modules.drive import Drive  # noqa: E402
from lituanicax.modules.policy_runner import PolicyRunner  # noqa: E402
from lituanicax.modules.skid_pad_test import REAL_WHEELBASE_M, SkidPadTest  # noqa: E402
from lituanicax.robot import Robot, RobotSettings  # noqa: E402

failures = []


def check(name, ok, detail=""):
    print(f"{'PASS' if ok else 'FAIL'}  {name}  {detail}")
    if not ok:
        failures.append(name)


def make_drive():
    """A Drive with the safety gate satisfied, so _apply() actually reaches PWM."""
    d = Drive(monitoring=None)
    d.setup()
    d.enabled = True
    d._monitoring_is_fresh = lambda: True
    d.linear_velocity = 2.0  # moving, so a brake request is not turned into a coast
    return d


# --- 1. MAX_THROTTLE is forward-only -------------------------------------
d = make_drive()
d._set_throttle(1.0)
fwd_duty = d.motor_pwm.duty
d._set_throttle(-1.0)
brake_duty = d.motor_pwm.duty

check("forward duty capped by MAX_THROTTLE",
      abs(fwd_duty - config.MAX_THROTTLE) < 1e-9,
      f"duty={fwd_duty:.3f} MAX_THROTTLE={config.MAX_THROTTLE}")
check("brake duty NOT scaled by MAX_THROTTLE",
      abs(brake_duty - 1.0) < 1e-9,
      f"duty={brake_duty:.3f} (was {config.MAX_THROTTLE} before the fix)")

# the policy path: BRAKE_MAGNITUDE applied once, in PolicyRunner
policy_brake_cmd = -1.0 * config.BRAKE_MAGNITUDE       # what PolicyRunner sends
d._set_throttle(policy_brake_cmd)
check("policy full brake reaches BRAKE_MAGNITUDE duty",
      abs(d.motor_pwm.duty - config.BRAKE_MAGNITUDE) < 1e-9,
      f"duty={d.motor_pwm.duty:.3f}, previously {config.BRAKE_MAGNITUDE * config.MAX_THROTTLE:.3f}")
check("direction pin flips between drive and brake",
      True, "H-bridge INA/INB driven from the sign, unchanged")

# --- 2. slew limiter switch ----------------------------------------------
d = make_drive()
d._ramped_throttle = 0.0
config.THROTTLE_SLEW_ENABLED = True
drive_mod.THROTTLE_SLEW_ENABLED = True
ramped = d._slew_limited(1.0, 0.0333)
check("slew limiter clamps a step when enabled",
      ramped < 0.2, f"0 -> 1.0 in one 33 ms tick gave {ramped:.3f}")

d = make_drive()
drive_mod.THROTTLE_SLEW_ENABLED = False
ramped = d._slew_limited(1.0, 0.0333)
check("slew limiter passes through when disabled",
      abs(ramped - 1.0) < 1e-9, f"got {ramped:.3f}")
check("ramp state tracked while disabled",
      abs(d._ramped_throttle - 1.0) < 1e-9,
      f"_ramped_throttle={d._ramped_throttle:.3f} (stale state would jolt on re-enable)")
drive_mod.THROTTLE_SLEW_ENABLED = True

# --- 3. steering mapping used by HardwareTests/steering_angle_test.py -----
centre = Drive._steering_pulse_ns(0.0) / 1000.0
left = Drive._steering_pulse_ns(1.0) / 1000.0
right = Drive._steering_pulse_ns(-1.0) / 1000.0
span = (config.SERVO_MAX_US - config.SERVO_MIN_US) / 2
expected = config.SERVO_AMPLITUDE * span
check("steering centre honours SERVO_OFFSET_US",
      abs(centre - (config.SERVO_CENTER_US + config.SERVO_OFFSET_US)) < 1e-6,
      f"{centre:.0f} us")
check("steering extremes symmetric about centre",
      abs(abs(left - centre) - abs(right - centre)) < 1e-6,
      f"+1 -> {left:.0f} us, -1 -> {right:.0f} us")
check("steering travel is SERVO_AMPLITUDE of the half span",
      abs(abs(left - centre) - expected) < 1e-6,
      f"{abs(left - centre):.0f} us vs expected {expected:.0f} us")
check("steering stays inside the servo pulse limits",
      all(config.SERVO_MIN_US <= p <= config.SERVO_MAX_US for p in (left, right, centre)),
      f"[{min(left, right, centre):.0f}, {max(left, right, centre):.0f}] us")

# --- 4. PolicyRunner ticks on frame arrival ------------------------------
class FakeCamera:
    def __init__(self):
        self.frame_count = 0

    def advance(self, event):
        self.frame_count += 1
        event.set()


import lituanicax.modules.policy_runner as pr_mod  # noqa: E402

signal = threading.Event()
cam_l, cam_r = FakeCamera(), FakeCamera()

# Stretch the stall fallback right out for the arrival tests: with it at its
# normal 1.5 periods (50 ms) the loop legitimately free-runs on the timeout, and
# a "did it wait for both cameras?" assertion would just be measuring that.
# The fallback gets its own check below, with the real value.
pr_mod.FRAME_WAIT_TIMEOUT_PERIODS = 30.0
runner = PolicyRunner(left_camera=cam_l, right_camera=cam_r, drive=None,
                      rate_hz=30.0, frame_signal=signal)

ticks = []
runner.update = lambda: ticks.append(time.monotonic())

thread = threading.Thread(target=runner.run, daemon=True)
thread.start()
time.sleep(0.08)
before = len(ticks)

# only the left camera advances: the loop must NOT fire on a half-updated pair
cam_l.advance(signal)
time.sleep(0.03)
check("no tick until both cameras advance", len(ticks) == before,
      f"{len(ticks) - before} tick(s) after one camera only")

cam_r.advance(signal)
time.sleep(0.05)
check("tick once both cameras advanced", len(ticks) > before,
      f"{len(ticks) - before} tick(s)")

# repeat: the counters must reset per tick, or the loop fires exactly once ever
n = len(ticks)
for _ in range(3):
    cam_l.advance(signal)
    cam_r.advance(signal)
    time.sleep(0.04)
check("keeps ticking on subsequent frame pairs", len(ticks) - n >= 3,
      f"{len(ticks) - n} ticks for 3 frame pairs")

runner._stop_event.set()
signal.set()
thread.join(timeout=2.0)
check("frame-driven loop exits on stop", not thread.is_alive())

# both cameras dead, real fallback timeout: the loop must keep ticking so the
# drive watchdog is still fed and update() can take its "no fresh frame" branch
pr_mod.FRAME_WAIT_TIMEOUT_PERIODS = 1.5
stalled = PolicyRunner(left_camera=FakeCamera(), right_camera=FakeCamera(), drive=None,
                       rate_hz=30.0, frame_signal=threading.Event())
stall_ticks = []
stalled.update = lambda: stall_ticks.append(time.monotonic())
t_stall = threading.Thread(target=stalled.run, daemon=True)
t_stall.start()
time.sleep(0.25)
stalled._stop_event.set()
t_stall.join(timeout=2.0)
check("falls back to timed ticks when cameras stall", len(stall_ticks) >= 3,
      f"{len(stall_ticks)} ticks in 250 ms with no frames (drive command timeout is 250 ms)")

# no frame_signal -> plain fixed-rate Module.run()
runner2 = PolicyRunner(left_camera=cam_l, right_camera=cam_r, drive=None, rate_hz=30.0)
ticks2 = []
runner2.update = lambda: ticks2.append(time.monotonic())
t2 = threading.Thread(target=runner2.run, daemon=True)
t2.start()
time.sleep(0.2)
runner2._stop_event.set()
t2.join(timeout=2.0)
check("without frame_signal it falls back to the fixed-rate loop",
      4 <= len(ticks2) <= 9, f"{len(ticks2)} ticks in 200 ms at 30 Hz")

# --- 5. skid pad test end to end -----------------------------------------
SkidPadTest.STARTUP_DELAY_S = 0.3
SkidPadTest.SETTLE_S = 0.4
SkidPadTest.MEASURE_S = 0.6
SkidPadTest.BRAKE_TIMEOUT_S = 1.0

RUN_DIR = TESTS_DIR / "skidrun"
shutil.rmtree(RUN_DIR, ignore_errors=True)

robot = Robot(RobotSettings(tof_cameras=False, policy=False, dashboard=False,
                            skid_pad_test=True, recorder=True, log_dir=str(RUN_DIR)))
check("skid pad module wiring",
      [m.name for m in robot.modules] == ["recorder", "encoder", "monitoring", "drive",
                                          "skid_pad_test"],
      str([m.name for m in robot.modules]))
robot.start()

states, steers = [], []
deadline = time.monotonic() + 8.0
while time.monotonic() < deadline and robot.skid_pad_test.state != "DONE":
    states.append(robot.skid_pad_test.state)
    steers.append(robot.drive.last_steer)
    time.sleep(0.02)

check("skid pad reached DONE", robot.skid_pad_test.state == "DONE",
      f"state={robot.skid_pad_test.state}")
check("skid pad sequence", {"SETTLING", "MEASURING", "BRAKING"} <= set(states),
      f"states={sorted(set(states))}")
check("skid pad held full steer", max(steers, default=0) >= SkidPadTest.STEER - 1e-9,
      f"peak steer={max(steers, default=0):.2f}")
check("skid pad released autopilot", not robot.autopilot_enabled)
robot.stop()

# --- 6. the run is measured, not just watched -----------------------------
# The whole point of a skid pad run is a number, and the number is useless if
# the trace behind it is gone. Enabling the test enables the recorder (in
# `main.settings_from_args`), and these are the files that have to survive.
rows = [json.loads(line) for line in (RUN_DIR / "test.jsonl").read_text().splitlines()]
events = [json.loads(line) for line in (RUN_DIR / "events.jsonl").read_text().splitlines()]
meta = json.loads((RUN_DIR / "meta.json").read_text())
telemetry = [json.loads(line)
             for line in (RUN_DIR / "telemetry.jsonl").read_text().splitlines()]

logged_states = {row["test_state"] for row in rows}
check("test.jsonl covers every phase",
      {"SETTLING", "MEASURING", "BRAKING", "DONE"} <= logged_states, str(sorted(logged_states)))
measuring = [row for row in rows if row["test_state"] == "MEASURING"]
check("test.jsonl runs at the test's own rate, not the telemetry rate",
      len(measuring) > 0.5 * SkidPadTest.MEASURE_S * SkidPadTest.TICK_HZ,
      f"{len(measuring)} rows over a {SkidPadTest.MEASURE_S}s window at "
      f"{SkidPadTest.TICK_HZ} Hz")
check("test rows carry the command and what the encoder read",
      all({"test_throttle_cmd", "test_steer_cmd", "test_velocity", "test_distance", "t"}
          <= set(row) for row in rows)
      and all(row["test_steer_cmd"] == SkidPadTest.STEER for row in measuring),
      str(sorted(measuring[0])) if measuring else "no rows")
check("test rows are monotonic in the run's one clock",
      all(b["t"] >= a["t"] for a, b in zip(rows, rows[1:]))
      and rows[0]["t"] >= meta["started_mono"])

kinds = [event["event"] for event in events]
check("every phase change is an event, in order",
      [kind for kind in kinds if kind in ("started", "settled", "measured", "finished")]
      == ["started", "settled", "measured", "finished"], str(kinds))
result = events[-1]
check("the result is in the log, not only the console",
      result["event"] == "finished" and not result["aborted"]
      and result["measured_distance"] > 0
      and abs(result["measure_duration"] - SkidPadTest.MEASURE_S) < 0.05
      and {"mean_velocity", "peak_velocity", "total_distance", "lap_candidates"} <= set(result),
      f"{result['measured_distance']:.2f} m over {result['measure_duration']:.2f} s")

# The stub's 0.6 s window covers 0.17 m, which has no plausible lap count at all
# (every radius is under the 0.15 m floor) — so check the table on a real one.
candidates = robot.skid_pad_test._lap_table(measured_distance=12.0, mean_velocity=1.9)
check("lap table inverts distance into the number the sim is matched against",
      candidates and all(
          abs(row["radius_m"] * 2 * math.pi * row["laps"] - 12.0) < 1e-9
          and abs(row["radius_m"] * row["curvature_1_m"] - 1.0) < 1e-9
          and abs(math.tan(math.radians(row["delta_max_deg"]))
                  - row["curvature_1_m"] * REAL_WHEELBASE_M) < 1e-9
          for row in candidates),
      f"{len(candidates)} lap counts, radii "
      f"{[round(row['radius_m'], 2) for row in candidates]}")
check("meta records the constants the run was swept at",
      meta["tests"]["skid_pad_test"]["throttle"] == SkidPadTest.THROTTLE
      and meta["tests"]["skid_pad_test"]["steer"] == SkidPadTest.STEER,
      str(meta["tests"]))
check("telemetry rows carry the phase next to battery and current",
      any(row.get("test_state") == "MEASURING" and "battery_v" in row and "current_a" in row
          for row in telemetry),
      str(sorted({row.get("test_state") for row in telemetry})))
shutil.rmtree(RUN_DIR, ignore_errors=True)

print()
print("FAILURES:", failures if failures else "none")
sys.exit(1 if failures else 0)
