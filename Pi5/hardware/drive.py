"""Drive: motor (Hobbywing 1080 G2, forward/brake mode) + AS5600 encoder on the motor shaft
+ steering servo on GPIO12 (calibration from config/steering.json, see tools/steering_cal.py).

    from hardware import Drive
    with Drive() as d:            # neutral on GPIO13, steering centred, encoder sampling in the background
        d.steer(-0.5)             # -1 left .. +1 right
        d.forward(0.3)            # 0..1 -> 1500..1000 us
        time.sleep(1)
        print(d.rps, d.rpm, d.turns)
        d.brake(1.0)              # 0..1 -> 1500..2000 us (reverse is disabled in the ESC)
        d.stop()                  # neutral; also done on exit

rps/rpm are median-filtered over the last 15 encoder samples (~1.5 kHz), so single-sample
glitches at high speed are ignored. Positive speed = forward.
"""
import statistics
import threading
import time
from collections import deque

from .base import Subsystem
from .encoder import AS5600
from .esc import HobbywingESC
from .steering import Steering


class Drive(Subsystem):
    name = "drive"

    def __init__(self, max_throttle=1.0, filter_samples=15, sample_hz=None):
        self.esc = HobbywingESC()
        self.enc = AS5600()
        self.enc.zero()
        self.steering = Steering()
        self.max_throttle = float(max_throttle)
        self._period = 1.0 / sample_hz if sample_hz else 0.0   # None: sample as fast as the bus allows (~1.5 kHz)
        self._filter = deque(maxlen=filter_samples)
        self._dps = 0.0
        self._turns = 0.0
        self._stop_evt = threading.Event()
        self._thread = threading.Thread(target=self._sample, daemon=True)
        self._thread.start()
        self.stop()

    def _sample(self):
        nxt = time.monotonic()
        while not self._stop_evt.is_set():
            if self._period:
                nxt += self._period
                time.sleep(max(0.0, nxt - time.monotonic()))
            _, self._turns, vel = self.enc.read_fast()
            self._filter.append(vel)
            if len(self._filter) >= 3:
                self._dps = statistics.median(self._filter)

    # ---- control ------------------------------------------------------------
    def forward(self, fraction):
        """Forward throttle 0..1 (capped by max_throttle)."""
        self.esc.throttle(min(max(0.0, float(fraction)), self.max_throttle))

    def brake(self, fraction=1.0):
        """Brake strength 0..1."""
        self.esc.throttle(-min(max(0.0, float(fraction)), 1.0))

    def stop(self):
        """Neutral (motor coasts)."""
        self.esc.neutral()

    def steer(self, position):
        """Steering -1 (full left) .. 0 (straight) .. +1 (full right)."""
        return self.steering.set(position)

    # ---- readings -----------------------------------------------------------
    @property
    def rps(self):
        return self._dps / 360.0

    @property
    def rpm(self):
        return self.rps * 60.0

    @property
    def turns(self):
        return self._turns

    def reset_odometer(self):
        self.enc.zero()

    # ---- Subsystem ----------------------------------------------------------
    def available(self):
        return self.esc.available() and self.enc.available() and self.steering.available()

    def status(self):
        return {"pulse_us": self.esc.current_us, "rps": round(self.rps, 2),
                "rpm": round(self.rpm), "turns": round(self.turns, 2),
                "steer": self.steering.position}

    def close(self):
        self.stop()
        self._stop_evt.set()
        self._thread.join(timeout=1.0)
        self.esc.close()   # pin stays at neutral
        self.steering.close()
        self.enc.close()
