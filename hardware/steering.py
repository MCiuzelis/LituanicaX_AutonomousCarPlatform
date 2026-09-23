"""Steering servo on GPIO12 with a persisted calibration (center + amplitude).

    from hardware import Steering
    with Steering() as s:        # loads config/steering.json (or defaults)
        s.set(-1.0)              # full left  (center_us - amplitude_us)
        s.set(0.0)               # straight   (center_us)
        s.set(0.5)               # half right
        s.center()

Calibrate with tools/steering_cal.py; it writes the JSON, this class only reads it.
"""
import fcntl
import json
import os

from .base import Subsystem, HardwareError
from .pwm import Servo

STEERING_GPIO = 12
CONFIG_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           "config", "steering.json")
LOCK_PATH = "/tmp/rc_car_steering.lock"   # one owner at a time, or two writers fight over the PWM
DEFAULTS = {"gpio": STEERING_GPIO, "center_us": 1500.0, "amplitude_us": 300.0, "invert": False}


def load_config(path=CONFIG_PATH):
    cfg = dict(DEFAULTS)
    if os.path.exists(path):
        with open(path) as f:
            cfg.update(json.load(f))
    return cfg


def save_config(cfg, path=CONFIG_PATH):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(cfg, f, indent=2)
        f.flush()
        os.fsync(f.fileno())      # the SD card has already eaten un-synced files once
    os.replace(tmp, path)


def _acquire_lock():
    f = open(LOCK_PATH, "a+")
    try:
        fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        f.seek(0)
        pid = f.read().strip()
        f.close()
        try:
            with open("/proc/%s/cmdline" % pid) as c:
                who = c.read().replace("\0", " ").strip()
        except (OSError, ValueError):
            who = "?"
        raise HardwareError("steering servo already in use by pid %s (%s)" % (pid, who))
    f.seek(0)
    f.truncate()
    f.write(str(os.getpid()))
    f.flush()
    return f


class Steering(Subsystem):
    name = "steering"

    def __init__(self, config_path=CONFIG_PATH):
        self.cfg = load_config(config_path)
        self._lock = _acquire_lock()
        try:
            self.servo = Servo(int(self.cfg["gpio"]), min_us=500, max_us=2500)
        except Exception:
            self._lock.close()
            raise
        self._position = 0.0
        self.center()

    @property
    def center_us(self):
        return float(self.cfg["center_us"])

    @property
    def amplitude_us(self):
        return float(self.cfg["amplitude_us"])

    @property
    def position(self):
        """Last commanded position, -1 (left) .. +1 (right)."""
        return self._position

    def set(self, position):
        p = max(-1.0, min(1.0, float(position)))
        sign = -1.0 if self.cfg.get("invert") else 1.0
        self.servo.set_pulse_us(self.center_us + sign * p * self.amplitude_us)
        self._position = p
        return p

    def center(self):
        return self.set(0.0)

    def set_pulse_us(self, us):
        """Raw pulse for calibration; position is reported relative to the current config."""
        us = self.servo.set_pulse_us(us)
        self._position = (us - self.center_us) / self.amplitude_us if self.amplitude_us else 0.0
        return us

    def available(self):
        return self.servo.available()

    def status(self):
        return {"position": round(self._position, 3), "pulse_us": self.servo.pulse_us,
                "center_us": self.center_us, "amplitude_us": self.amplitude_us}

    def close(self):
        self.center()
        self.servo.close()
        self._lock.close()   # closing the fd drops the flock
