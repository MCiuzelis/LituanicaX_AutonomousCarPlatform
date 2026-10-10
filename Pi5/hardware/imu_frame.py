"""CarFrame: BNO085 board frame -> car frame (x forward, y left, z up; right-handed).

The IMU is not mounted flat: on this car the breakout stands on its edge with the chip face
(+Z) pointing forward and -X pointing up. `forward` / `up` name those board axes; the left
axis follows. A level reference (the gravity vector measured with the car still in its
nominal pose) adds the small rotation that absorbs the mount's tilt, so roll = pitch = 0 in
that pose and what moves afterwards (suspension, slopes) is real body attitude.

    from hardware import BNO085, CarFrame
    frame = CarFrame.load()                       # config/imu.json
    with BNO085() as imu:
        imu.enable_default()
        frame.attitude(imu.game_quaternion)       # (roll, pitch, yaw) deg, car frame
        frame.to_car(imu.linear_acceleration)     # (forward, left, up) m/s^2

Sign conventions (all reported in the car frame):
    roll   + = right side down          pitch  + = nose up
    yaw    + = turned left (CCW seen from above), relative: game_rv has no absolute heading
    to_car(gyro) = (roll rate, pitch rate, yaw rate) rad/s about x fwd / y left / z up
                   -> yaw rate + = turning left, pitch rate + = nose DOWN (right-hand rule)
"""
import json
import math
import os
import time

from .base import HardwareError

CONFIG = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config", "imu.json")
_AXES = {"+x": (1, 0, 0), "-x": (-1, 0, 0), "+y": (0, 1, 0), "-y": (0, -1, 0), "+z": (0, 0, 1), "-z": (0, 0, -1)}


def _cross(a, b):
    return (a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0])


def _dot(a, b):
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _norm(v):
    return math.sqrt(_dot(v, v))


def _matmul(a, b):
    return [[sum(a[i][k] * b[k][j] for k in range(3)) for j in range(3)] for i in range(3)]


def _mv(m, v):
    return tuple(_dot(row, v) for row in m)


def _transpose(m):
    return [list(r) for r in zip(*m)]


def _align(u, t):
    """Smallest rotation matrix taking unit vector u onto unit vector t (Rodrigues)."""
    v, c = _cross(u, t), _dot(u, t)
    s = _norm(v)
    if s < 1e-9:
        if c > 0:
            return [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
        raise HardwareError("level reference is upside down relative to the configured 'up' axis")
    k = [[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]]
    k2 = _matmul(k, k)
    f = (1 - c) / (s * s)
    return [[(i == j) + k[i][j] + f * k2[i][j] for j in range(3)] for i in range(3)]


def quat_to_matrix(q):
    """(i, j, k, real) -> rotation matrix, board frame -> world frame."""
    x, y, z, w = q[:4]
    return [[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]]


class CarFrame:
    def __init__(self, forward="+z", up="-x", level_gravity=None, level_time=None, path=CONFIG):
        if forward not in _AXES or up not in _AXES:
            raise ValueError("axes are one of %s" % ", ".join(_AXES))
        f, u = _AXES[forward], _AXES[up]
        if _dot(f, u):
            raise ValueError("forward (%s) and up (%s) must be perpendicular" % (forward, up))
        self.forward, self.up, self.path = forward, up, path
        self.mount = [list(f), list(_cross(u, f)), list(u)]      # rows: car x/y/z in board coords
        self.level_gravity, self.level_time = None, None
        self.R = self.mount                                     # car <- board
        if level_gravity:
            self.set_level(level_gravity, level_time)

    # ---- level reference -----------------------------------------------------------------
    def set_level(self, gravity_board, when=None):
        """gravity_board: accelerometer / gravity reading (board frame) with the car still in
        its nominal pose; that pose becomes roll = pitch = 0."""
        g = _mv(self.mount, gravity_board)
        n = _norm(g)
        if n < 5.0:
            raise HardwareError("level reference |g| = %.2f m/s^2 is implausible" % n)
        self.R = _matmul(_align(tuple(x / n for x in g), (0.0, 0.0, 1.0)), self.mount)
        self.level_gravity = [round(x, 4) for x in gravity_board]
        self.level_time = when or time.strftime("%Y-%m-%dT%H:%M:%S")

    def mount_tilt_deg(self):
        """Angle between the configured 'up' board axis and the level reference (mount error)."""
        if not self.level_gravity:
            return None
        g = _mv(self.mount, self.level_gravity)
        return math.degrees(math.acos(max(-1.0, min(1.0, g[2] / _norm(g)))))

    @staticmethod
    def capture_gravity(imu, seconds=1.5, max_gyro=0.03, max_std=0.15):
        """Mean gravity (board frame) over `seconds`; HardwareError if the car is moving.
        Needs the BNO085's gravity and gyro sensors enabled (enable_default does)."""
        gs, peak_w = [], 0.0
        t_end = time.monotonic() + seconds
        while time.monotonic() < t_end:
            r = imu.wait("gravity", timeout=0.5)
            if r is None:
                raise HardwareError("no gravity samples from the IMU")
            gs.append(r.values[:3])
            w = imu.gyro
            peak_w = max(peak_w, _norm(w) if w else 0.0)
        mean = [sum(g[i] for g in gs) / len(gs) for i in range(3)]
        std = math.sqrt(sum((g[i] - mean[i]) ** 2 for g in gs for i in range(3)) / len(gs))
        if peak_w > max_gyro or std > max_std:
            raise HardwareError("car is moving (gyro peak %.3f rad/s, gravity std %.3f m/s^2): hold it still"
                                % (peak_w, std))
        return mean

    # ---- transforms ------------------------------------------------------------------------
    def to_car(self, v):
        """Board-frame 3-vector -> car frame (forward, left, up). None passes through."""
        return None if v is None else _mv(self.R, v[:3])

    def attitude(self, q):
        """Board orientation quaternion (game_rv / rv: board -> world, z up) ->
        (roll, pitch, yaw) degrees of the car, signs as in the module docstring."""
        if q is None:
            return None
        m = _matmul(quat_to_matrix(q), _transpose(self.R))     # car -> world
        roll = math.degrees(math.atan2(m[2][1], m[2][2]))
        pitch = math.degrees(math.asin(max(-1.0, min(1.0, m[2][0]))))
        yaw = math.degrees(math.atan2(m[1][0], m[0][0]))
        return roll, pitch, yaw

    def tilt(self, q):
        """Angle (deg) between the car's up axis and vertical: 0 level, 90 on its side."""
        if q is None:
            return None
        m = _matmul(quat_to_matrix(q), _transpose(self.R))
        return math.degrees(math.atan2(math.hypot(m[2][0], m[2][1]), m[2][2]))   # acos(m22) is noisy near 0

    # ---- persistence -----------------------------------------------------------------------
    @classmethod
    def load(cls, path=CONFIG):
        """From config/imu.json; defaults (this car's mount, no level reference) if missing."""
        try:
            with open(path) as f:
                c = json.load(f)
        except FileNotFoundError:
            return cls(path=path)
        return cls(c.get("forward", "+z"), c.get("up", "-x"), c.get("level_gravity"), c.get("level_time"), path)

    def save(self):
        d = {"forward": self.forward, "up": self.up, "level_gravity": self.level_gravity,
             "level_time": self.level_time, "mount_tilt_deg": self.mount_tilt_deg() and round(self.mount_tilt_deg(), 2),
             "car_from_board": [[round(x, 6) for x in r] for r in self.R]}
        tmp = self.path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(d, f, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, self.path)       # the README warns about power cuts zeroing files

    def status(self):
        t = self.mount_tilt_deg()
        return {"forward": self.forward, "up": self.up, "levelled": bool(self.level_gravity),
                "level_time": self.level_time, "mount_tilt_deg": None if t is None else round(t, 2)}
