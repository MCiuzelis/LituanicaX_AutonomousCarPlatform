"""BNO055 / BNO08x IMU model on top of ideal kinematics.

sim.py feeds it the true angular velocity, specific force (acceleration + gravity reaction, what an
accelerometer measures) and orientation of the IMU frame on EVERY physics step. The model averages
them over its output period (the chip's own decimation / low-pass) and adds what a real part adds:

    gyro       mount misalignment, constant bias + random walk, white noise
    accel      mount misalignment, constant bias, white noise
    quat       game rotation vector: roll/pitch from the fusion (small error), yaw relative to the
               power-on heading with a slow drift (no magnetometer)
    lin_accel  accel minus gravity as the FUSION believes it points (so tilt error leaks in)

Pure numpy, no Isaac imports.
"""
from __future__ import annotations

import math

import numpy as np

from .config import ImuCfg
from .interface import ImuSample

G = 9.81


def quat_to_matrix(q) -> np.ndarray:
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def quat_to_rpy(q) -> tuple[float, float, float]:
    w, x, y, z = q
    roll = math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    pitch = math.asin(max(-1.0, min(1.0, 2 * (w * y - z * x))))
    yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    return roll, pitch, yaw


def rpy_to_quat(roll: float, pitch: float, yaw: float) -> tuple[float, float, float, float]:
    cr, sr = math.cos(roll / 2), math.sin(roll / 2)
    cp, sp = math.cos(pitch / 2), math.sin(pitch / 2)
    cy, sy = math.cos(yaw / 2), math.sin(yaw / 2)
    return (cr * cp * cy + sr * sp * sy, sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy, cr * cp * sy - sr * sp * cy)


def rpy_to_matrix(roll: float, pitch: float, yaw: float) -> np.ndarray:
    return quat_to_matrix(rpy_to_quat(roll, pitch, yaw))


class ImuModel:
    def __init__(self, cfg: ImuCfg, rng: np.random.Generator):
        self.cfg, self.rng = cfg, rng
        self.period = 1.0 / cfg.rate_hz
        bw = cfg.rate_hz / 2.0
        self.gyro_sigma = math.radians(cfg.gyro_noise_density_dps_rthz) * math.sqrt(bw)
        self.accel_sigma = cfg.accel_noise_density_ug_rthz * 1e-6 * G * math.sqrt(bw)
        self.reset(0.0, (1.0, 0.0, 0.0, 0.0))

    def reset(self, t: float, quat_w) -> None:
        c, r = self.cfg, self.rng
        self.mount = rpy_to_matrix(*np.radians(r.normal(0.0, c.mount_error_deg, 3)))
        self.gyro_bias = np.radians(r.normal(0.0, c.gyro_bias_dps, 3))
        self.accel_bias = r.normal(0.0, c.accel_bias_mg * 1e-3 * G, 3)
        self.yaw_drift = math.radians(c.yaw_drift_deg_per_min / 60.0) * r.uniform(0.5, 1.5) * r.choice((-1.0, 1.0))
        self.yaw0 = quat_to_rpy(quat_w)[2]
        self.t0 = self.t_last = t
        self._acc_sum = np.zeros(3)
        self._gyro_sum = np.zeros(3)
        self._n = 0
        self._quat = quat_w

    def step(self, t: float, gyro_b, specific_force_b, quat_w) -> ImuSample | None:
        """Feed one physics step; returns a sample when the output period has elapsed."""
        self._gyro_sum += np.asarray(gyro_b, dtype=np.float64)
        self._acc_sum += np.asarray(specific_force_b, dtype=np.float64)
        self._n += 1
        self._quat = quat_w
        if t - self.t_last < self.period - 1e-9:
            return None
        dt = t - self.t_last
        self.t_last = t
        c, r = self.cfg, self.rng
        gyro = self._gyro_sum / self._n
        acc = self._acc_sum / self._n
        self._gyro_sum[:] = 0.0
        self._acc_sum[:] = 0.0
        self._n = 0

        self.gyro_bias += math.radians(c.gyro_bias_walk_dps_rts) * math.sqrt(dt) * r.standard_normal(3)
        gyro_m = self.mount.T @ gyro + self.gyro_bias + self.gyro_sigma * r.standard_normal(3)
        acc_m = self.mount.T @ acc + self.accel_bias + self.accel_sigma * r.standard_normal(3)

        roll, pitch, yaw = quat_to_rpy(self._quat)
        tilt = math.radians(c.tilt_noise_deg)
        roll_e, pitch_e = roll + r.normal(0.0, tilt), pitch + r.normal(0.0, tilt)
        yaw_e = yaw - self.yaw0 + self.yaw_drift * (t - self.t0)
        yaw_e = math.atan2(math.sin(yaw_e), math.cos(yaw_e))
        g_body = rpy_to_matrix(roll_e, pitch_e, 0.0).T @ np.array([0.0, 0.0, G])
        lin = acc_m - g_body
        return ImuSample(t, tuple(gyro_m), tuple(acc_m), tuple(lin), rpy_to_quat(roll_e, pitch_e, yaw_e))
