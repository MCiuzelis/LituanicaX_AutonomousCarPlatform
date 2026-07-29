"""Fake smbus2: AS5600 angle sweeping, optionally coasting to a stop.

Set FAKE_ENCODER_STOP_AFTER=<seconds> to make the wheel stop turning after
that many seconds, so braking-to-standstill logic can be exercised.
"""
import os
import time

_START = time.monotonic()


class SMBus:
    def __init__(self, bus): self.bus = bus

    def read_i2c_block_data(self, addr, reg, length):
        elapsed = time.monotonic() - _START
        stop_after = float(os.environ.get("FAKE_ENCODER_STOP_AFTER", "inf"))
        moving_time = min(elapsed, stop_after)
        raw = int((moving_time * 20000) % 4096)
        return [(raw >> 8) & 0xFF, raw & 0xFF]

    def close(self): pass
