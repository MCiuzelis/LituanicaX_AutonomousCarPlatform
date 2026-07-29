"""End-to-end smoke test of the vanilla-Python stack with stubbed hardware."""
import json
import sys
import threading
import time
import urllib.request
from pathlib import Path

TESTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(TESTS_DIR / "stubs"))
sys.path.insert(0, str(TESTS_DIR.parent))

from lituanicax.modules import drive as drive_mod

# --- fake out the sysfs PWM + pinctrl that only exist on the Pi ---
pwm_log = []


class FakePWM:
    def __init__(self, channel, frequency_hz, initial_pulse_ns=0):
        self.channel = channel
        self.period_ns = int(1_000_000_000 / frequency_hz)
        self.pulse_ns = initial_pulse_ns
        self.stopped = False

    def set_duty_fraction(self, f):
        self.set_pulse_ns(int(self.period_ns * max(0.0, min(1.0, f))))

    def set_pulse_ns(self, ns):
        self.pulse_ns = max(0, min(self.period_ns, int(ns)))
        pwm_log.append((self.channel, self.pulse_ns))

    def stop(self):
        self.stopped = True


drive_mod.SysfsHardwarePWM = FakePWM
drive_mod.Drive._pinctrl_set = staticmethod(lambda *a: None)

from lituanicax.robot import Robot, RobotSettings

settings = RobotSettings(port=8099, screenshot_grabber=True)
robot = Robot(settings)
robot.screenshot_grabber.save_dir = TESTS_DIR / "shots"  # keep the repo clean
robot.start()

failures = []


def check(name, ok, detail=""):
    print(f"{'PASS' if ok else 'FAIL'}  {name}  {detail}")
    if not ok:
        failures.append(name)


time.sleep(2.5)

# 1. sensors producing data
check("encoder velocity", robot.encoder.linear_velocity != 0.0,
      f"v={robot.encoder.linear_velocity:.3f} m/s d={robot.encoder.distance_travelled:.2f} m")
check("monitoring battery", 7.0 < robot.monitoring.battery_voltage < 8.5,
      f"{robot.monitoring.battery_voltage:.2f} V")
check("monitoring no fault",
      not robot.monitoring.undervoltage and not robot.monitoring.overcurrent,
      f"I={robot.monitoring.motor_current:.2f} A cpu={robot.monitoring.cpu_usage:.0f}%")
check("cameras streaming",
      all(c.frame_count > 10 for c in robot.cameras.values()),
      f"frames={[c.frame_count for c in robot.cameras.values()]}")
check("camera previews", all(c.preview_jpeg for c in robot.cameras.values()),
      f"jpeg bytes={[len(c.preview_jpeg) for c in robot.cameras.values()]}")

# 2. autopilot disabled -> drive stays at zero throttle
check("idle throttle zero", robot.drive.last_throttle == 0.0,
      f"throttle={robot.drive.last_throttle}")

# 3. dashboard HTTP
html = urllib.request.urlopen("http://127.0.0.1:8099/", timeout=5).read()
check("dashboard index", b"<html" in html.lower(), f"{len(html)} bytes")

stream = urllib.request.urlopen("http://127.0.0.1:8099/stream/left", timeout=5)
chunk = stream.read(2000)
stream.close()
check("mjpeg stream", b"--frame" in chunk and b"image/jpeg" in chunk, f"{len(chunk)} bytes")

# 4. websocket: enable autopilot the way the browser does
import asyncio

import aiohttp


async def ws_session():
    telemetry = []
    async with aiohttp.ClientSession() as session:
        async with session.ws_connect("http://127.0.0.1:8099/ws") as ws:
            await ws.send_str(json.dumps({"type": "space", "pressed": True}))
            deadline = time.monotonic() + 3.0
            while time.monotonic() < deadline:
                try:
                    msg = await asyncio.wait_for(ws.receive(), timeout=1.0)
                except asyncio.TimeoutError:
                    break
                if msg.type == aiohttp.WSMsgType.TEXT:
                    telemetry.append(json.loads(msg.data))
            await ws.send_str(json.dumps({"type": "space", "pressed": False}))
            await asyncio.sleep(0.3)
    return telemetry


enabled_seen = {"drive": False, "policy": False, "throttle": 0.0, "steer": 0.0}


def watch_drive():
    deadline = time.monotonic() + 3.0
    while time.monotonic() < deadline:
        if robot.autopilot_enabled:
            enabled_seen["drive"] = robot.drive.enabled
            enabled_seen["policy"] = robot.policy.enabled
            if robot.drive.last_throttle != 0.0:
                enabled_seen["throttle"] = robot.drive.last_throttle
            if robot.drive.last_steer != 0.0:
                enabled_seen["steer"] = robot.drive.last_steer
        time.sleep(0.02)


watcher = threading.Thread(target=watch_drive)
watcher.start()
telemetry = asyncio.run(ws_session())
watcher.join()

check("websocket telemetry", len(telemetry) > 10, f"{len(telemetry)} messages")
if telemetry:
    last = telemetry[-1]
    check("telemetry fields",
          all(k in last for k in ("throttle", "steer", "battery", "velocity", "cpu_temp",
                                  "active", "undervoltage", "overcurrent")),
          str(sorted(last.keys())))
    check("telemetry active flag", any(t["active"] for t in telemetry),
          f"active seen in {sum(t['active'] for t in telemetry)} msgs")
    check("telemetry velocity", any(t["velocity"] != 0 for t in telemetry),
          f"last v={last['velocity']:.3f}")

check("autopilot reached drive+policy",
      enabled_seen["drive"] and enabled_seen["policy"], str(enabled_seen))
check("policy drove the motor",
      enabled_seen["throttle"] != 0.0 or enabled_seen["steer"] != 0.0,
      f"throttle={enabled_seen['throttle']:.3f} steer={enabled_seen['steer']:.3f}")
check("pwm writes happened", len(pwm_log) > 50, f"{len(pwm_log)} writes")

# 5. disabling autopilot stops the motor
time.sleep(0.5)
check("disabled -> zero throttle", robot.drive.last_throttle == 0.0,
      f"throttle={robot.drive.last_throttle}")

# 6. screenshots written
shots = sorted((TESTS_DIR / "shots").glob("*.png"))
check("screenshot grabber", len(shots) >= 2, f"{len(shots)} png files")

robot.stop()
time.sleep(0.2)
check("threads joined", not any(m._thread.is_alive() for m in robot.modules if m._thread),
      str([m.name for m in robot.modules if m._thread and m._thread.is_alive()]))
check("pwm released on shutdown",
      robot.drive.motor_pwm is None and robot.drive.servo_pwm is None, "")

print()
print("FAILURES:", failures if failures else "none")
sys.exit(1 if failures else 0)
