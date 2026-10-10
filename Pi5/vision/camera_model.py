"""Calibrated IMX708 camera: lens distortion + mounting pose -> 3D positions on the floor.

numpy only (the vision side runs without OpenCV). Calibration lives in config/camera.json and is
made by tools/camera_cal.py; this module only reads it.

Pixels are handled in *sensor* coordinates (full 4608x2592 array), so one calibration serves every
stream: an output frame is the ScalerCrop rectangle of the sensor scaled to the output size
(picamera2 metadata "ScalerCrop"; the dashboard's 1280x720 modes and the detector's 640x360 lores
copy both see crop (768, 432, 3072, 1728), the 2304x1296 / full-res modes see the whole sensor).

World frame: origin on the floor directly under the camera, X forward (optical axis projected on
the floor), Y left, Z up, millimetres. The camera sits at (0, 0, height_mm).
    pitch_deg + = looking down          roll_deg + = camera's left side down

    from vision.camera_model import CameraModel, CROP_1536
    cam = CameraModel.load()
    for x1, y1, x2, y2, score, cls in boxes:              # AppleDetector boxes, 0..1 of the frame
        p = cam.box_ground((x1, y1, x2, y2), CROP_1536)   # (X fwd, Y left) mm where the box
                                                          # bottom-centre touches the floor
    cam.ground(u, v)                 # sensor pixel -> (X, Y) on the floor, None above the horizon
    cam.ground(u, v, z=40)           # ... on the plane Z = 40 mm (e.g. an apple's centre)
    cam.project([X, Y, Z])           # world mm -> sensor pixel (overlays, checks)
    mx, my = cam.undistort_maps(CROP_1536, (1280, 720))   # for cv2.remap: distortion-free image

Sign conventions match hardware/imu_frame.py (x forward, y left, z up). The extrinsics are the
static mount with the car on level ground; pass pitch_deg / roll_deg to ground() to follow body
motion (e.g. static pitch + IMU pitch change: IMU pitch is nose-up +, camera pitch is down +).
"""
import json
import math
import os

import numpy as np

CONFIG = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config", "camera.json")
SENSOR = (4608, 2592)
CROP_FULL = (0, 0, 4608, 2592)        # raw modes 2304x1296 (2x2 binned) and 4608x2592
CROP_1536 = (768, 432, 3072, 1728)    # raw mode 1536x864 (binned + centre crop): 1280x720, 640x360 lores
CROP_BY_RAW = {(1536, 864): CROP_1536, (2304, 1296): CROP_FULL, (4608, 2592): CROP_FULL}

# level camera looking along +X: camera x = -Y (right), camera y = -Z (down), camera z = X
_B = np.array([[0.0, -1.0, 0.0], [0.0, 0.0, -1.0], [1.0, 0.0, 0.0]])


def _world_to_cam(pitch_deg, roll_deg):
    p, r = math.radians(pitch_deg), math.radians(roll_deg)
    rx = np.array([[1, 0, 0], [0, math.cos(p), -math.sin(p)], [0, math.sin(p), math.cos(p)]])
    rz = np.array([[math.cos(r), -math.sin(r), 0], [math.sin(r), math.cos(r), 0], [0, 0, 1]])
    return rz @ rx @ _B


class CameraModel:
    def __init__(self, fx, fy, cx, cy, dist, height_mm, pitch_deg, roll_deg, info=None):
        self.fx, self.fy, self.cx, self.cy = float(fx), float(fy), float(cx), float(cy)
        self.dist = np.zeros(5)
        self.dist[:len(dist)] = dist                   # OpenCV order k1, k2, p1, p2, k3
        self.height_mm, self.pitch_deg, self.roll_deg = float(height_mm), float(pitch_deg), float(roll_deg)
        self.info = info or {}
        self._R = _world_to_cam(self.pitch_deg, self.roll_deg)

    @classmethod
    def load(cls, path=CONFIG):
        with open(path) as f:
            c = json.load(f)
        i, e = c["intrinsics"], c["extrinsics"]
        return cls(i["fx"], i["fy"], i["cx"], i["cy"], i["dist"], e["height_mm"], e["pitch_deg"],
                   e["roll_deg"], c)

    # ---- pixel spaces ------------------------------------------------------------------------
    @staticmethod
    def to_sensor(x, y, crop, size=None):
        """Frame coords -> sensor pixels. size=None: x, y are normalised 0..1 (detector boxes);
        else pixels of a size=(w, h) frame."""
        x0, y0, w, h = crop
        sx, sy = (w, h) if size is None else (w / size[0], h / size[1])
        return x0 + np.asarray(x, float) * sx, y0 + np.asarray(y, float) * sy

    @staticmethod
    def from_sensor(u, v, crop, size=None):
        x0, y0, w, h = crop
        sx, sy = (w, h) if size is None else (w / size[0], h / size[1])
        return (np.asarray(u, float) - x0) / sx, (np.asarray(v, float) - y0) / sy

    def K(self, crop=CROP_FULL, size=None):
        """3x3 intrinsic matrix for a stream (pixels of `size`, default: the crop at 1:1), to use
        with self.dist in OpenCV."""
        x0, y0, w, h = crop
        size = size or (w, h)
        sx, sy = size[0] / w, size[1] / h
        return np.array([[self.fx * sx, 0, (self.cx - x0) * sx], [0, self.fy * sy, (self.cy - y0) * sy], [0, 0, 1]])

    # ---- distortion ----------------------------------------------------------------------------
    def _distort(self, x, y):
        k1, k2, p1, p2, k3 = self.dist
        r2 = x * x + y * y
        rad = 1 + r2 * (k1 + r2 * (k2 + r2 * k3))
        return (x * rad + 2 * p1 * x * y + p2 * (r2 + 2 * x * x),
                y * rad + p1 * (r2 + 2 * y * y) + 2 * p2 * x * y)

    def undistort(self, u, v, iters=20):
        """Sensor pixels -> ideal normalised image coords (x, y) = (X/Z, Y/Z) in the camera frame."""
        xd = (np.asarray(u, float) - self.cx) / self.fx
        yd = (np.asarray(v, float) - self.cy) / self.fy
        x, y = xd, yd
        for _ in range(iters):                         # fixed point; converges in < 10 on this lens
            ex, ey = self._distort(x, y)
            x, y = x + (xd - ex), y + (yd - ey)
        return x, y

    def distort(self, x, y):
        """Ideal normalised coords -> sensor pixels."""
        xd, yd = self._distort(np.asarray(x, float), np.asarray(y, float))
        return self.cx + self.fx * xd, self.cy + self.fy * yd

    # ---- geometry ------------------------------------------------------------------------------
    def _rot(self, pitch_deg, roll_deg):
        if pitch_deg is None and roll_deg is None:
            return self._R
        return _world_to_cam(self.pitch_deg if pitch_deg is None else pitch_deg,
                             self.roll_deg if roll_deg is None else roll_deg)

    def ray(self, u, v, pitch_deg=None, roll_deg=None):
        """Sensor pixel(s) -> unit direction(s) in the world frame, shape (..., 3)."""
        x, y = self.undistort(u, v)
        d = np.stack([x, y, np.ones_like(x)], -1) @ self._rot(pitch_deg, roll_deg)   # = R.T @ d
        return d / np.linalg.norm(d, axis=-1, keepdims=True)

    def ground(self, u, v, z=0.0, pitch_deg=None, roll_deg=None):
        """Sensor pixel -> (X, Y) mm where its ray meets the horizontal plane Z = z.
        Scalar in -> tuple or None (ray does not reach the plane); arrays in -> (N, 2) with NaN."""
        d = self.ray(u, v, pitch_deg, roll_deg)
        dz = d[..., 2]
        with np.errstate(divide="ignore", invalid="ignore"):
            t = (z - self.height_mm) / dz
        t = np.where(t > 0, t, np.nan)
        xy = d[..., :2] * t[..., None]
        if xy.ndim == 1:
            return None if np.isnan(xy).any() else (float(xy[0]), float(xy[1]))
        return xy

    def box_ground(self, box, crop, size=None, z=0.0, **att):
        """Detector box (x1, y1, x2, y2; normalised 0..1 unless size is given) -> (X, Y) mm of its
        bottom-centre on the plane Z = z. For an object standing on the floor use z=0."""
        x1, y1, x2, y2 = box[:4]
        u, v = self.to_sensor((x1 + x2) / 2, y2, crop, size)
        return self.ground(u, v, z, **att)

    def project(self, pts, pitch_deg=None, roll_deg=None):
        """World points (N, 3) mm -> sensor pixels (N, 2); NaN behind the camera."""
        p = np.atleast_2d(np.asarray(pts, float)) - [0, 0, self.height_mm]
        c = p @ self._rot(pitch_deg, roll_deg).T
        with np.errstate(divide="ignore", invalid="ignore"):
            x, y = np.where(c[:, 2] > 0, c[:, 0] / c[:, 2], np.nan), np.where(c[:, 2] > 0, c[:, 1] / c[:, 2], np.nan)
        return np.stack(self.distort(x, y), 1)

    def range_mm(self, X, Y, Z=0.0):
        """Straight-line distance from the camera centre."""
        return math.sqrt(X * X + Y * Y + (Z - self.height_mm) ** 2)

    def undistort_maps(self, crop, size, out_size=None, f_scale=1.0):
        """(map_x, map_y) float32 for cv2.remap(frame, map_x, map_y, cv2.INTER_LINEAR): turns a
        `size` frame of `crop` into a distortion-free pinhole image of out_size (default size),
        same centre, focal length = calibrated * f_scale (< 1 keeps more of the bent-in edges).
        The pinhole K of that output is self.K(crop, size) with fx, fy scaled by f_scale."""
        out_size = out_size or size
        K = self.K(crop, size)
        ox, oy = out_size[0] / size[0], out_size[1] / size[1]
        fx, fy = K[0, 0] * ox * f_scale, K[1, 1] * oy * f_scale
        cx, cy = K[0, 2] * ox, K[1, 2] * oy
        jj, ii = np.meshgrid(np.arange(out_size[0], dtype=float), np.arange(out_size[1], dtype=float))
        u, v = self.distort((jj - cx) / fx, (ii - cy) / fy)
        mx, my = self.from_sensor(u, v, crop, size)
        return mx.astype(np.float32), my.astype(np.float32)
