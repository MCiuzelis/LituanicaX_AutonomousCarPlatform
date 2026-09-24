"""AS5600 12-bit magnetic rotary encoder over I2C (address 0x36, bus 1 = GPIO2/3).

config.txt: dtparam=i2c_arm=on   (bus 1 on header pins 3 = SDA, 5 = SCL)

Registers (datasheet): STATUS 0x0B (MD magnet detected, ML too weak, MH too strong),
RAW_ANGLE 0x0C/0x0D (12-bit, unscaled), ANGLE 0x0E/0x0F (after ZPOS/MPOS scaling),
AGC 0x1A, MAGNITUDE 0x1B/0x1C, CONF 0x07/0x08, ZPOS 0x01/0x02, MPOS 0x03/0x04.
"""
import time
from dataclasses import dataclass

from .base import Subsystem, HardwareError

AS5600_ADDR = 0x36
REG_ZMCO, REG_ZPOS, REG_MPOS, REG_MANG, REG_CONF = 0x00, 0x01, 0x03, 0x05, 0x07
REG_STATUS, REG_RAW_ANGLE, REG_ANGLE, REG_AGC, REG_MAGNITUDE = 0x0B, 0x0C, 0x0E, 0x1A, 0x1B
COUNTS = 4096


@dataclass
class EncoderReading:
    raw: int              # 0..4095, unscaled
    angle_deg: float      # raw * 360 / 4096
    turns: float          # cumulative revolutions since open()/zero() (multi-turn tracking)
    velocity_dps: float   # degrees per second, from the last two readings
    magnet: str           # "ok", "weak", "strong", "none"
    agc: int
    magnitude: int
    timestamp: float

    def as_dict(self):
        return {k: (round(v, 3) if isinstance(v, float) else v) for k, v in self.__dict__.items()}


class AS5600(Subsystem):
    name = "as5600"

    def __init__(self, bus=1, address=AS5600_ADDR):
        try:
            from smbus2 import SMBus
        except ImportError as e:
            raise HardwareError("python3-smbus2 not installed: %s" % e)
        self.bus_num, self.address = bus, address
        try:
            self._bus = SMBus(bus)
        except OSError as e:
            raise HardwareError("cannot open /dev/i2c-%d (dtparam=i2c_arm=on?): %s" % (bus, e))
        self._last_raw = None
        self._last_t = None
        self._turns = 0.0
        self._offset_raw = 0

    # ---- low level -------------------------------------------------------
    def _r8(self, reg):
        try:
            return self._bus.read_byte_data(self.address, reg)
        except OSError as e:
            raise HardwareError("AS5600 not responding at 0x%02x on i2c-%d: %s" % (self.address, self.bus_num, e))

    def _r12(self, reg):
        try:
            hi, lo = self._bus.read_i2c_block_data(self.address, reg, 2)
        except OSError as e:
            raise HardwareError("AS5600 not responding at 0x%02x on i2c-%d: %s" % (self.address, self.bus_num, e))
        return ((hi & 0x0F) << 8) | lo

    def available(self):
        try:
            self._r8(REG_STATUS)
            return True
        except HardwareError:
            return False

    # ---- registers ------------------------------------------------------
    def status_bits(self):
        s = self._r8(REG_STATUS)
        return {"magnet_detected": bool(s & 0x20), "magnet_weak": bool(s & 0x10), "magnet_strong": bool(s & 0x08)}

    def magnet(self):
        b = self.status_bits()
        if not b["magnet_detected"]:
            return "none"
        if b["magnet_weak"]:
            return "weak"
        if b["magnet_strong"]:
            return "strong"
        return "ok"

    def raw_angle(self):
        return self._r12(REG_RAW_ANGLE)

    def scaled_angle_raw(self):
        return self._r12(REG_ANGLE)

    def agc(self):
        return self._r8(REG_AGC)

    def magnitude(self):
        return self._r12(REG_MAGNITUDE)

    def conf(self):
        return self._r12(REG_CONF) | ((self._r8(REG_CONF) & 0x3F) << 8)

    # ---- angles ---------------------------------------------------------
    @staticmethod
    def raw_to_deg(raw):
        return raw * 360.0 / COUNTS

    def zero(self):
        """Make the current position 0 deg / 0 turns (software offset, nothing burned)."""
        self._offset_raw = self.raw_angle()
        self._turns = 0.0
        self._last_raw = 0  # relative position is 0 after zeroing
        self._last_t = time.monotonic()

    def angle_deg(self):
        return self.raw_to_deg((self.raw_angle() - self._offset_raw) % COUNTS)

    def _track(self):
        """One RAW_ANGLE transaction + wrap tracking. Returns (raw, rel, velocity_dps, t)."""
        t = time.monotonic()
        raw = self.raw_angle()
        rel = (raw - self._offset_raw) % COUNTS
        vel = 0.0
        if self._last_raw is not None:
            d = rel - self._last_raw
            if d > COUNTS / 2:
                d -= COUNTS
            elif d < -COUNTS / 2:
                d += COUNTS
            self._turns += d / COUNTS
            dt = t - self._last_t
            if dt > 0:
                vel = self.raw_to_deg(d) / dt
        self._last_raw, self._last_t = rel, t
        return raw, rel, vel, t

    def read_fast(self):
        """(angle_deg, turns, velocity_dps): a single 2-byte I2C transaction per call, so it
        can run at several hundred Hz. Unambiguous up to half a revolution per sample."""
        _, rel, vel, _ = self._track()
        return self.raw_to_deg(rel), self._turns, vel

    def read(self):
        """Full reading (angle, turns, velocity, magnet status, AGC, magnitude) = 4 I2C
        transactions. Use read_fast() in tight loops; call this at >2x the expected
        rotation rate so the wrap-around detection stays unambiguous."""
        raw, rel, vel, t = self._track()
        return EncoderReading(raw, self.raw_to_deg(rel), self._turns, vel, self.magnet(),
                              self.agc(), self.magnitude(), t)

    def sample_rate(self, seconds=1.0):
        """Measured read_fast() rate in Hz (and the unambiguous speed limit it implies)."""
        n = 0
        t0 = time.monotonic()
        while time.monotonic() - t0 < seconds:
            self.read_fast()
            n += 1
        hz = n / (time.monotonic() - t0)
        return hz, hz * 180.0  # max deg/s = half a turn per sample

    def status(self):
        r = self.read()
        d = r.as_dict()
        d.update({"bus": self.bus_num, "address": "0x%02x" % self.address})
        return d

    def close(self):
        try:
            self._bus.close()
        except Exception:
            pass
