"""Camera geometry an algorithm needs: the lens model, and box -> object position on the ground.

CameraModel      pinhole or equidistant (f-theta) projection. The car's Arducam 120 deg M12 lens is
                 close to equidistant: 120 deg H and 66.7 deg V over the full 16:9 sensor are exactly
                 proportional to the sensor dimensions, which is the f-theta signature (a pinhole
                 would give 120 x 88 deg). The simulator renders with the same model, and the real
                 car should replace it with a calibrated one (e.g. OpenCV's fisheye model).

GroundProjector  two range estimates for an object of known size and shape ("ellipsoid" - the
                 apples / spheres of the current track - or "cylinder"):
  * size    - the distance at which the object's projected height matches the box height. The best
              one: 1 px of box jitter moves it a few % of the range.
  * ground  - the ray through the box's bottom-centre intersected with the ground plane. Much more
              sensitive to pixel jitter and to pitch error; used only when the box is cut by the
              top/bottom of the frame (its height is then meaningless).
The bearing comes from the box centre. With check_width > 0 a box whose width implies a different
range than its height (partly occluded, merged with a neighbour, cut by the frame side) is rejected.
"""
import math

import numpy as np

from ..interface import CameraInfo, Detection


def camera_rotation(pitch_deg: float, yaw_deg: float = 0.0) -> np.ndarray:
    """Columns = camera x (right), y (down), z (forward) expressed in the car frame (FLU), for a
    camera looking along +x, pitched nose-down by pitch_deg and yawed left by yaw_deg."""
    p, y = math.radians(pitch_deg), math.radians(yaw_deg)
    fwd = np.array([math.cos(p), 0.0, -math.sin(p)])
    right = np.array([0.0, -1.0, 0.0])
    down = np.cross(fwd, right)
    rz = np.array([[math.cos(y), -math.sin(y), 0.0], [math.sin(y), math.cos(y), 0.0], [0.0, 0.0, 1.0]])
    return rz @ np.column_stack([right, down, fwd])


class CameraModel:
    """Projection between the car frame and image pixels (origin top-left, principal point centred)."""

    def __init__(self, cam: CameraInfo):
        self.cam = cam
        self.f = cam.fx
        self.cx, self.cy = cam.width_px / 2.0, cam.height_px / 2.0
        self.R = camera_rotation(cam.pitch_deg, cam.yaw_deg)
        self.o = np.array([cam.x_m, cam.y_m, cam.z_m])
        self.fisheye = cam.projection == "equidistant"

    def project_cam(self, pc: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Camera-frame points [..., 3] (x right, y down, z forward) -> pixel (u, v)."""
        x, y, z = pc[..., 0], pc[..., 1], pc[..., 2]
        if not self.fisheye:
            z = np.maximum(z, 1e-6)
            return self.f * x / z + self.cx, self.f * y / z + self.cy
        r = np.hypot(x, y)
        rho = self.f * np.arctan2(r, z) / np.maximum(r, 1e-12)            # image radius per unit r
        return self.cx + rho * x, self.cy + rho * y

    def ray_cam(self, u, v) -> np.ndarray:
        """Pixel (u, v) -> unit viewing direction in the camera frame [..., 3]."""
        dx, dy = (np.asarray(u, dtype=np.float64) - self.cx) / self.f, (np.asarray(v, dtype=np.float64) - self.cy) / self.f
        if not self.fisheye:
            d = np.stack([dx, dy, np.ones_like(dx)], -1)
        else:
            theta = np.hypot(dx, dy)
            s = np.sin(theta) / np.maximum(theta, 1e-12)
            d = np.stack([dx * s, dy * s, np.cos(theta)], -1)
        return d / np.linalg.norm(d, axis=-1, keepdims=True)

    def project(self, p_car: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Car-frame points [..., 3] -> pixel (u, v)."""
        return self.project_cam((np.asarray(p_car) - self.o) @ self.R)

    def ray(self, u, v) -> np.ndarray:
        """Pixel (u, v) -> unit viewing direction in the car frame."""
        return self.ray_cam(u, v) @ self.R.T


class GroundProjector:
    def __init__(self, cam: CameraInfo, object_radius_m: float = 0.05, object_height_m: float = 0.0875,
                 object_shape: str = "ellipsoid"):
        if object_shape not in ("ellipsoid", "cylinder"):
            raise ValueError(f"object_shape must be 'ellipsoid' or 'cylinder', not {object_shape!r}")
        self.cam, self.r_obj, self.h_obj, self.shape = cam, object_radius_m, object_height_m, object_shape
        self.model = CameraModel(cam)
        self.o = self.model.o
        a = np.linspace(0.0, 2 * math.pi, 12, endpoint=False)
        self._ell = np.stack([np.cos(a), np.sin(a)], 1)       # outline samples in the viewing plane
        self._grid = np.geomspace(0.15, 12.0, 120)            # candidate ranges, m

    def ray(self, u_norm: float, v_norm: float) -> np.ndarray:
        return self.model.ray(u_norm * self.cam.width_px, v_norm * self.cam.height_px)

    def ground_point(self, u_norm: float, v_norm: float) -> tuple[float, float] | None:
        d = self.ray(u_norm, v_norm)
        if d[2] >= -1e-6:
            return None                   # at or above the horizon
        p = self.o + (-self.o[2] / d[2]) * d
        return float(p[0]), float(p[1])

    def project_v(self, p: np.ndarray) -> np.ndarray:
        """Image rows (normalised 0..1) of car-frame points [..., 3]."""
        return self.model.project(p)[1] / self.cam.height_px

    def _height_range(self, det: Detection, bearing: np.ndarray) -> float | None:
        """Horizontal distance camera -> object centre whose projected height equals the box's.
        The projected height falls monotonically with distance, so it is evaluated on a distance
        grid in one vectorised pass and inverted by interpolation."""
        d = self._grid[:, None]                                            # [G, 1]
        ctr = self.o[:2] + d * bearing                                     # [G, 2]
        if self.shape == "cylinder":                                       # rims: near bottom, both tops
            off = np.array([-1.0, -1.0, 1.0])[None, :, None] * self.r_obj * bearing
            z = np.array([0.0, self.h_obj, self.h_obj])
        else:                                                              # ellipse in the viewing plane
            off = self.r_obj * self._ell[None, :, 0:1] * bearing
            z = 0.5 * self.h_obj * (1.0 + self._ell[:, 1])
        pts = np.concatenate([ctr[:, None, :] + off, np.broadcast_to(z, (len(d), len(z)))[..., None]], -1)
        v = self.project_v(pts)
        if self.shape == "cylinder":
            h = v[:, 0] - np.minimum(v[:, 1], v[:, 2])
        else:
            h = v.max(1) - v.min(1)
        if not h[-1] < det.height < h[0]:
            return None
        return float(np.interp(det.height, h[::-1], self._grid[::-1]))

    def width_range(self, det: Detection) -> float:
        """Range to the object's centre from the angle its box width subtends (diameter 2 * r)."""
        vc = 0.5 * (det.y1 + det.y2)
        a, b = self.ray(det.x1, vc), self.ray(det.x2, vc)
        half = 0.5 * math.acos(max(-1.0, min(1.0, float(a @ b))))
        return self.r_obj / max(math.sin(half), 1e-6)

    def object_position(self, det: Detection, max_range: float = 6.0, check_width: float = 0.0
                        ) -> tuple[float, float] | None:
        if det.x1 > 0.003 and det.x2 < 0.997 and self.width_range(det) > 1.3 * max_range:
            return None                                                # cheap early out for far boxes
        d = self.ray(det.cx, 0.5 * (det.y1 + det.y2))
        bearing = d[:2] / max(np.hypot(*d[:2]), 1e-9)
        cut = det.y1 <= 0.002 or det.y2 >= 0.998
        rng = None if cut else self._height_range(det, bearing)
        if rng is not None and check_width > 0.0 and not (0.003 < det.x1 and det.x2 < 0.997):
            return None                                                # cut by the frame side
        if rng is not None and check_width > 0.0:
            if abs(self.width_range(det) - rng) > check_width * rng:
                return None
        if rng is None:                                                # fall back to the ground plane
            p = self.ground_point(det.cx, det.y2)
            if p is None:
                return None
            rng = float(np.hypot(p[0] - self.o[0], p[1] - self.o[1])) + self.r_obj
        if rng > max_range:
            return None
        return float(self.o[0] + rng * bearing[0]), float(self.o[1] + rng * bearing[1])
