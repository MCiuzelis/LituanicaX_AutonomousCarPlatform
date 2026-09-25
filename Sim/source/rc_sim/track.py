"""Track geometry from assets/export/track.json: object positions, centre line, lap progress.

Used by the simulator (spawn pose, progress, lap timing) and the viewer. NOT available to an
algorithm - the real car has no map.
"""
from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, field

import numpy as np

from .config import EXPORT_DIR


class TrackMap:
    def __init__(self, path: str = os.path.join(EXPORT_DIR, "track.json")):
        with open(path) as f:
            d = json.load(f)
        self.raw = d
        self.object_class = d.get("object_class", "object")
        objs = d["objects"]
        self.names = [o["name"] for o in objs]
        self.xy = np.array([[o["x"], o["y"]] for o in objs], dtype=np.float64)
        self.z0 = np.array([o["z0"] for o in objs], dtype=np.float64)
        self.radius = np.array([o["radius"] for o in objs], dtype=np.float64)
        self.height = np.array([o["height"] for o in objs], dtype=np.float64)
        self.side = np.array([1 if o["side"] == "left" else -1 for o in objs], dtype=np.int8)
        self.name_to_id = {n: i for i, n in enumerate(self.names)}

        cl = d["centerline"]
        self.closed = bool(cl["closed"])
        p = np.array(cl["points"], dtype=np.float64)
        self.cl = p
        nxt = np.roll(p, -1, axis=0) if self.closed else np.vstack([p[1:], p[-1:]])
        self.seg_a, self.seg_d = p, nxt - p
        self.seg_len = np.linalg.norm(self.seg_d, axis=1)
        self.seg_s = np.concatenate([[0.0], np.cumsum(self.seg_len)[:-1]])
        self.length = float(self.seg_len.sum())
        self.wall_offset = d["walls"]["lateral"]["left"]["mean"] if d.get("walls") else 0.44

    @property
    def n_objects(self) -> int:
        return len(self.names)

    def project(self, x: float, y: float) -> tuple[float, float, float]:
        """(arc length s, signed lateral offset + = left, centre-line heading) of a point."""
        ap = np.array([x, y]) - self.seg_a
        L2 = np.maximum(self.seg_len ** 2, 1e-12)
        t = np.clip((ap * self.seg_d).sum(1) / L2, 0.0, 1.0)
        diff = ap - t[:, None] * self.seg_d
        d2 = (diff ** 2).sum(1)
        i = int(np.argmin(d2))
        dx, dy = self.seg_d[i]
        cross = dx * ap[i, 1] - dy * ap[i, 0]
        return (float(self.seg_s[i] + t[i] * self.seg_len[i]), math.copysign(math.sqrt(d2[i]), cross),
                math.atan2(dy, dx))

    def pose_at(self, s: float, lateral: float = 0.0, reverse: bool = False) -> tuple[float, float, float]:
        """(x, y, yaw) on the centre line at arc length s, offset `lateral` to the left of travel."""
        s = s % self.length if self.closed else min(max(s, 0.0), self.length)
        i = int(np.searchsorted(self.seg_s, s, side="right") - 1)
        i = max(0, min(i, len(self.seg_s) - 1))
        t = (s - self.seg_s[i]) / max(self.seg_len[i], 1e-9)
        x, y = self.seg_a[i] + t * self.seg_d[i]
        yaw = math.atan2(self.seg_d[i, 1], self.seg_d[i, 0])
        if reverse:
            yaw += math.pi
        return x - math.sin(yaw) * lateral, y + math.cos(yaw) * lateral, yaw


@dataclass
class ProgressTracker:
    """Unwrapped progress along the centre line in the driving direction, laps and lap times."""
    track: TrackMap
    reverse: bool = False
    progress: float = 0.0
    best: float = 0.0
    laps: int = 0
    lap_times: list[float] = field(default_factory=list)
    _last_s: float | None = None
    _lap_start_t: float = 0.0
    _best_t: float = 0.0

    def reset(self, x: float, y: float, t: float = 0.0) -> None:
        self._last_s = self.track.project(x, y)[0]
        self.progress = self.best = 0.0
        self.laps = 0
        self.lap_times = []
        self._lap_start_t = self._best_t = t

    def update(self, x: float, y: float, t: float) -> float:
        s, lateral, _ = self.track.project(x, y)
        ds = s - self._last_s
        L = self.track.length
        if self.track.closed:
            ds = (ds + L / 2) % L - L / 2      # unwrap across the start
        self._last_s = s
        self.progress += -ds if self.reverse else ds
        if self.progress > self.best + 0.05:
            self.best, self._best_t = self.progress, t
        while self.progress >= (self.laps + 1) * L:
            self.laps += 1
            self.lap_times.append(t - self._lap_start_t)
            self._lap_start_t = t
        return lateral

    def since_progress(self, t: float) -> float:
        return t - self._best_t
