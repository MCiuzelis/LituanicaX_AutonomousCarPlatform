"""AS5600 on the motor shaft, read the way hardware/drive.py reads it.

Drive runs a thread calling AS5600.read_fast() as fast as I2C allows (~1.5 kHz): each call reads a
12-bit angle, unwraps it against the previous one (so > half a turn between samples aliases, as on
the car) and turns the difference into deg/s. Drive.rps is the median of the last 15 of those.

sim.py feeds the true wheel angle (mean of the driven wheels, i.e. the differential) once per
physics step; the model reconstructs the ~1.5 kHz sample stream in between by interpolation, then
quantises and filters exactly like the car. Pure numpy.
"""
from __future__ import annotations

import math

import numpy as np

from .config import EncoderCfg
from .interface import EncoderReading


class EncoderModel:
    def __init__(self, cfg: EncoderCfg, rng: np.random.Generator):
        self.cfg, self.rng = cfg, rng
        self.reset(0.0, 0.0)

    def reset(self, t: float, wheel_angle_rad: float) -> None:
        self._t_prev, self._a_prev = t, wheel_angle_rad
        self._next_sample = t
        self._offset = self.rng.uniform(0, self.cfg.counts)   # magnet angle at power-on
        self._last_counts: int | None = None
        self._last_t = t
        self._turns = 0.0
        self._vel = np.zeros(0)                                # last median_samples velocities, turns/s
        self._rps = 0.0

    def step(self, t: float, wheel_angle_rad: float) -> None:
        """Advance to time t, where the (unwrapped) mean driven-wheel angle is wheel_angle_rad."""
        c, rng = self.cfg, self.rng
        period = 1.0 / c.sample_hz
        # sample instants in (t_prev, t], each one period later with I2C timing jitter
        n = max(0, int((t - self._next_sample) / period) + 2)
        ts = self._next_sample + np.cumsum(np.r_[0.0, period * (1.0 + c.sample_jitter * rng.uniform(-1, 1, n))])
        k = int(np.searchsorted(ts, t, side="right"))
        ts, self._next_sample = ts[:k], ts[k]
        if k:
            span = t - self._t_prev
            frac = (ts - self._t_prev) / span if span > 0 else np.ones(k)
            ang = self._a_prev + frac * (wheel_angle_rad - self._a_prev)
            raw = ang / (2 * math.pi) * c.motor_turns_per_wheel_turn * c.counts + self._offset
            counts = np.floor(raw + rng.normal(0.0, c.noise_lsb, k)).astype(np.int64) % c.counts
            prev = np.r_[self._last_counts if self._last_counts is not None else counts[0], counts[:-1]]
            d = counts - prev
            d = np.where(d > c.counts / 2, d - c.counts, np.where(d < -c.counts / 2, d + c.counts, d))  # read_fast() unwrap
            dt = np.diff(np.r_[self._last_t, ts])
            if self._last_counts is None:
                d, dt = d[1:], dt[1:]
            self._turns += d.sum() / c.counts
            self._vel = np.r_[self._vel, d / c.counts / np.maximum(dt, 1e-9)][-c.median_samples:]
            if len(self._vel) >= 3:
                self._rps = float(np.median(self._vel))
            self._last_counts, self._last_t = int(counts[-1]), float(ts[-1])
        self._t_prev, self._a_prev = t, wheel_angle_rad

    def reading(self, t: float) -> EncoderReading:
        return EncoderReading(t, self._rps, self._turns)
