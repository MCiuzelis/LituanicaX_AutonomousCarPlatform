"""Live view: camera frame with boxes + top-down map, shown by a separate process.

Omniverse Kit owns the GUI event loop of the simulator process, so (as in the old repo's
visualize.py) frames are written atomically to a PNG and a child process running the venv's
GUI-capable OpenCV shows them. Without $DISPLAY the PNG is still written: watch it with
`feh --reload 0.2 /tmp/rc_sim/view.png`.

Camera panel   green = detection (score), red = false positive, grey = object in view the
               detector missed this frame, yellow line = expected horizon
Map panel      car-up, 1 grid square = 1 m. Objects: blue = left side, orange = right side,
               ringed green when detected this frame. Cyan = algorithm landmarks, magenta =
               its left (magenta) / right (yellow) side assignment, white = its path, yellow dot = pursuit target, green
               trail = driven path.
"""
from __future__ import annotations

import math
import os
import subprocess
import sys

import cv2
import numpy as np

OUT_DIR = "/tmp/rc_sim"
OUT = os.path.join(OUT_DIR, "view.png")

_DISPLAY = r"""
import sys, os, time, signal
sys.path = [p for p in sys.path if "pip_prebundle" not in p and "omni" not in p]
import cv2
signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
path = sys.argv[1]
while not os.path.exists(path):
    time.sleep(0.05)
while True:
    img = cv2.imread(path)
    if img is not None:
        cv2.imshow("rc_sim", img)
    if cv2.waitKey(30) & 0xFF == ord("q"):
        break
"""


class Viewer:
    def __init__(self, track, cam_info, show: bool = True, record_dir: str | None = None, remap=None,
                 map_px: int = 480, px_per_m: float = 70.0):
        from .algorithms.geometry import CameraModel

        self.track, self.cam, self.remap = track, cam_info, remap
        # expected horizon: far points at camera height, projected through the lens (curved for a fisheye)
        model = CameraModel(cam_info)
        a = np.radians(np.linspace(-80, 80, 81))
        hu, hv = model.project(np.stack([100 * np.cos(a) + cam_info.x_m, 100 * np.sin(a), np.full_like(a, cam_info.z_m)], 1))
        self.horizon = np.stack([hu, hv], 1).astype(np.int32)
        self.record_dir, self.n_saved = record_dir, 0
        if record_dir:
            os.makedirs(record_dir, exist_ok=True)
        self.map_px, self.ppm = map_px, px_per_m
        self.trail: list = []
        os.makedirs(OUT_DIR, exist_ok=True)
        if os.path.exists(OUT):
            os.remove(OUT)
        self.proc = None
        if show and os.environ.get("DISPLAY"):
            env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
            self.proc = subprocess.Popen([sys.executable, "-c", _DISPLAY, OUT], env=env)
        else:
            print(f"[viewer] writing {OUT} (no window: {'no $DISPLAY' if show else 'disabled'})")
        # wall polylines (approximate: centre line offset by the wall's mean lateral distance)
        cl = track.cl
        t = np.roll(cl, -1, 0) - np.roll(cl, 1, 0)
        n = np.stack([-t[:, 1], t[:, 0]], 1) / np.linalg.norm(t, axis=1, keepdims=True)
        self.walls = [cl + track.wall_offset * n, cl - track.wall_offset * n]

    def reset(self):
        self.trail = []

    # ------------------------------------------------------------------------------------------
    def _camera_panel(self, snap):
        W, H = self.cam.width_px, self.cam.height_px
        if snap.rgb is None:
            img = np.zeros((H, W, 3), np.uint8)
        else:
            img = cv2.cvtColor(snap.rgb, cv2.COLOR_RGB2BGR)
            img = cv2.remap(img, *self.remap, cv2.INTER_LINEAR) if self.remap else cv2.resize(img, (W, H))
        cv2.polylines(img, [self.horizon], False, (0, 200, 255), 1)
        dbg = snap.frame_debug
        if dbg is not None:
            detected = set(t for t in dbg.truth if t >= 0)
            for i, oid in enumerate(dbg.view.ids):
                if oid not in detected and dbg.view.vis_px[i] > 0:
                    b = dbg.view.amodal[i]
                    cv2.rectangle(img, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (110, 110, 110), 1)
        if snap.frame is not None and dbg is not None:
            for d, tr in zip(snap.frame.detections, dbg.truth):
                col = (60, 60, 255) if tr < 0 else (60, 220, 60)
                p1, p2 = (int(d.x1 * W), int(d.y1 * H)), (int(d.x2 * W), int(d.y2 * H))
                cv2.rectangle(img, p1, p2, col, 2)
                cv2.putText(img, f"{d.score:.2f}", (p1[0], max(9, p1[1] - 2)), cv2.FONT_HERSHEY_SIMPLEX, 0.3, col, 1)
        elif dbg is not None:
            cv2.putText(img, "frame dropped", (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        return img

    def _map_panel(self, snap, algo_dbg):
        st, S, ppm = snap.status, self.map_px, self.ppm
        img = np.full((S, S, 3), 30, np.uint8)
        cx, cy = S / 2, S * 0.72                     # car sits low in the panel, looking up
        c, s = math.cos(st.yaw), math.sin(st.yaw)

        def world(p):
            dx, dy = p[..., 0] - st.x, p[..., 1] - st.y
            fx, fy = c * dx + s * dy, -s * dx + c * dy          # car frame
            return np.stack([cx - fy * ppm, cy - fx * ppm], -1)

        def car(p):
            p = np.asarray(p, dtype=np.float64).reshape(-1, 2)
            return np.stack([cx - p[:, 1] * ppm, cy - p[:, 0] * ppm], -1)

        for g in range(-8, 9):                        # 1 m grid in the car frame
            cv2.line(img, (int(cx + g * ppm), 0), (int(cx + g * ppm), S), (42, 42, 42), 1)
            cv2.line(img, (0, int(cy + g * ppm)), (S, int(cy + g * ppm)), (42, 42, 42), 1)
        for w in self.walls:
            cv2.polylines(img, [world(w).astype(np.int32)], True, (80, 80, 80), 1)
        cv2.polylines(img, [world(self.track.cl).astype(np.int32)], True, (55, 55, 55), 1)
        self.trail.append((st.x, st.y))
        self.trail = self.trail[-600:]
        if len(self.trail) > 1:
            cv2.polylines(img, [world(np.array(self.trail)).astype(np.int32)], False, (0, 150, 0), 1)
        detected = set(t for t in snap.frame_debug.truth if t >= 0) if snap.frame_debug else set()
        pts = world(self.track.xy)
        for i, (u, v) in enumerate(pts):
            if -10 < u < S + 10 and -10 < v < S + 10:
                col = (230, 140, 40) if self.track.side[i] > 0 else (40, 150, 240)
                cv2.circle(img, (int(u), int(v)), max(2, int(self.track.radius[i] * ppm)), col, -1)
                if i in detected:
                    cv2.circle(img, (int(u), int(v)), max(5, int(self.track.radius[i] * ppm) + 3), (60, 220, 60), 1)
        # camera field of view
        half = math.radians(self.cam.hfov_deg / 2)
        o = car([self.cam.x_m, 0.0])[0]
        for sgn in (-1, 1):
            e = car([self.cam.x_m + 4 * math.cos(half), sgn * 4 * math.sin(half)])[0]
            cv2.line(img, (int(o[0]), int(o[1])), (int(e[0]), int(e[1])), (70, 70, 0), 1)
        # algorithm debug (car frame)
        for q in car(algo_dbg.get("landmarks") or np.zeros((0, 2))):
            cv2.circle(img, (int(q[0]), int(q[1])), 3, (255, 255, 0), 1)
        for key, col in (("left", (255, 0, 255)), ("right", (0, 255, 255))):
            for q in car(algo_dbg.get(key) or np.zeros((0, 2))):
                cv2.drawMarker(img, (int(q[0]), int(q[1])), col, cv2.MARKER_CROSS, 7, 1)
        path = algo_dbg.get("path") or []
        if len(path) > 1:
            cv2.polylines(img, [car(path).astype(np.int32)], False, (255, 255, 255), 2)
        if algo_dbg.get("target") is not None:
            q = car(algo_dbg["target"])[0]
            cv2.circle(img, (int(q[0]), int(q[1])), 5, (0, 255, 255), -1)
        # the car: 0.26 x 0.19 m box from 5 cm behind the rear axle
        body = car([[-0.05, 0.095], [0.21, 0.095], [0.21, -0.095], [-0.05, -0.095]]).astype(np.int32)
        cv2.polylines(img, [body], True, (255, 255, 255), 1)
        return img

    def update(self, snap, cmd, algo_dbg: dict, extra: str = "") -> None:
        cam = self._camera_panel(snap)
        scale = self.map_px / cam.shape[0]
        cam = cv2.resize(cam, (int(cam.shape[1] * scale), self.map_px), interpolation=cv2.INTER_NEAREST)
        img = np.hstack([cam, self._map_panel(snap, algo_dbg)])
        st = snap.status
        lines = [
            f"t {st.t:6.2f}s  v {st.speed:4.2f} m/s  yaw rate {math.degrees(st.yaw_rate):6.1f} deg/s  "
            f"lap {st.laps}  progress {st.progress:6.1f}/{self.track.length:.1f} m  lateral {st.lateral:+.2f} m",
            f"cmd throttle {cmd.throttle:+.2f} steer {cmd.steer:+.2f} | applied {snap.applied.throttle:+.2f} "
            f"{snap.applied.steer:+.2f} | boxes {len(snap.frame.detections) if snap.frame else 0}"
            + (f" | laps {', '.join(f'{x:.2f}s' for x in st.lap_times)}" if st.lap_times else ""),
            (algo_dbg.get("text") or "") + ("  " + extra if extra else ""),
        ]
        bar = np.zeros((18 * len(lines) + 6, img.shape[1], 3), np.uint8)
        for i, t in enumerate(lines):
            cv2.putText(bar, t, (6, 16 + 18 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (230, 230, 230), 1, cv2.LINE_AA)
        img = np.vstack([img, bar])
        if self.record_dir:
            cv2.imwrite(os.path.join(self.record_dir, f"{self.n_saved:05d}_t{st.t:07.3f}.jpg"), img)
            self.n_saved += 1
        tmp = OUT[:-4] + "_tmp.png"
        cv2.imwrite(tmp, img)
        os.replace(tmp, OUT)

    def close(self) -> None:
        if self.proc is not None and self.proc.poll() is None:
            self.proc.terminate()
            self.proc.wait()
