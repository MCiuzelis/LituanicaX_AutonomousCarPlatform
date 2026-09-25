"""Synthetic YOLO: renderer ground truth -> boxes that behave like the car's Hailo detector.

Input per camera frame, for every track object that projects into the view (sim.py measures it):
    amodal   full projected box of the object, px, NOT clipped to the frame (from its 3D mesh)
    area_px  pixel area of its full projected silhouette (convex hull), NOT clipped
    vis_px   number of pixels of the object the renderer actually drew (instance segmentation):
             other objects, the car body and the frame edge all reduce it
    vis_box  tight box around those pixels
From those the model derives how visible / truncated / small / blurred each object is and turns
that into a detection probability and a box:

    p = recall_max * size(shorter side px) * visibility * in_frame * blur * upper_band

  * misses are drawn from a per-object AR(1) latent, so they come in bursts (flicker) instead of
    being independent every frame; same for the box jitter (centre + log-size noise ~ box size)
  * a mildly occluded object gets its full box (detectors complete objects), a heavily occluded one
    only the visible part
  * false positives appear as short-lived ghosts in the ground band
  * greedy NMS at nms_iou, so objects lined up along a straight merge the way real ones do
  * boxes are delivered after a latency drawn around the measured 24.5 ms, some frames drop

No Isaac / torch imports: this file is plain numpy and unit-tested in tests/.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .config import DetectorCfg
from .interface import Detection, DetectionFrame


def ramp(x, lo, hi):
    """0 below lo, 1 above hi, linear in between (vectorised)."""
    return np.clip((np.asarray(x, dtype=np.float64) - lo) / max(hi - lo, 1e-9), 0.0, 1.0)


def iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """IoU of box a [4] against boxes b [K, 4]."""
    ix = np.clip(np.minimum(a[2], b[:, 2]) - np.maximum(a[0], b[:, 0]), 0, None)
    iy = np.clip(np.minimum(a[3], b[:, 3]) - np.maximum(a[1], b[:, 1]), 0, None)
    inter = ix * iy
    area = lambda r: (r[..., 2] - r[..., 0]) * (r[..., 3] - r[..., 1])
    return inter / np.maximum(area(a) + area(b) - inter, 1e-9)


def nms(boxes: np.ndarray, scores: np.ndarray, thr: float) -> np.ndarray:
    order = np.argsort(-scores)
    keep = []
    while order.size:
        i = order[0]
        keep.append(i)
        if order.size == 1:
            break
        rest = order[1:]
        order = rest[iou(boxes[i], boxes[rest]) <= thr]
    return np.array(keep, dtype=np.int64)


def _phi(z: np.ndarray) -> np.ndarray:
    """Standard normal CDF."""
    return np.array([0.5 * (1.0 + math.erf(v / math.sqrt(2.0))) for v in np.ravel(z)]).reshape(np.shape(z))


@dataclass
class ObjectView:
    """Ground truth for one frame (arrays of equal length M, one row per candidate object)."""
    ids: np.ndarray        # [M] object index
    amodal: np.ndarray     # [M, 4] x1 y1 x2 y2 px, unclipped
    area_px: np.ndarray    # [M] silhouette area px, unclipped
    vis_px: np.ndarray     # [M] visible pixel count
    vis_box: np.ndarray    # [M, 4] px (undefined where vis_px == 0)
    depth: np.ndarray      # [M] m, camera to object centre


@dataclass
class FrameDebug:
    """What the model decided, for the viewer and the logs."""
    view: ObjectView
    p_detect: np.ndarray       # [M]
    visibility: np.ndarray     # [M]
    detected: np.ndarray       # [M] bool, before NMS
    truth: list[int]           # per output box: object id, or -1 for a false positive
    blur_px: float


class SyntheticDetector:
    def __init__(self, cfg: DetectorCfg, n_objects: int, width_px: int, height_px: int, fx: float,
                 exposure_s: float, rng: np.random.Generator):
        self.cfg, self.W, self.H, self.fx, self.exposure = cfg, width_px, height_px, fx, exposure_s
        self.rng = rng
        self.n = n_objects
        self.reset()

    def reset(self) -> None:
        self._miss_z = self.rng.standard_normal(self.n)
        self._jit = self.rng.standard_normal((self.n, 4))
        self._t_obj = np.full(self.n, -1e9)
        self._ghosts: list[dict] = []   # false positives: box, score, frames left

    # ------------------------------------------------------------------------------------------
    def _ar1(self, state: np.ndarray, ids: np.ndarray, dt: np.ndarray, tau: float) -> np.ndarray:
        rho = np.exp(-dt / max(tau, 1e-6))
        shape = (len(ids),) + state.shape[1:]
        if state.ndim == 2:
            rho = rho[:, None]
        state[ids] = rho * state[ids] + np.sqrt(1.0 - rho ** 2) * self.rng.standard_normal(shape)
        return state[ids]

    def process(self, t: float, view: ObjectView, yaw_rate: float = 0.0) -> tuple[DetectionFrame | None, FrameDebug]:
        """One camera frame -> boxes (None if the frame was dropped) + debug info."""
        c, W, H = self.cfg, self.W, self.H
        M = len(view.ids)
        blur_px = abs(yaw_rate) * self.exposure * self.fx

        boxes, scores, truth = np.zeros((0, 4)), np.zeros(0), []
        p = vis = np.zeros(M)
        detected = np.zeros(M, dtype=bool)
        if M:
            am = view.amodal.astype(np.float64)
            amc = np.stack([np.clip(am[:, 0], 0, W), np.clip(am[:, 1], 0, H),
                            np.clip(am[:, 2], 0, W), np.clip(am[:, 3], 0, H)], 1)
            area_full = np.maximum((am[:, 2] - am[:, 0]) * (am[:, 3] - am[:, 1]), 1e-6)
            area_in = np.maximum((amc[:, 2] - amc[:, 0]) * (amc[:, 3] - amc[:, 1]), 0.0)
            in_frame = area_in / area_full
            vis = np.clip(view.vis_px / np.maximum(view.area_px * in_frame, 1.0), 0.0, 1.0)
            vis[view.vis_px <= 0] = 0.0
            short_side = np.minimum(amc[:, 2] - amc[:, 0], amc[:, 3] - amc[:, 1])
            blur_f = 1.0 - ramp(blur_px, c.blur_ok_px, c.blur_max_px)
            upper = np.where(amc[:, 3] < c.upper_band_frac * H, c.upper_band_recall, 1.0)
            p = (c.recall_max * ramp(short_side, c.min_side_px, c.full_side_px) * ramp(vis, c.vis_min, c.vis_full)
                 * ramp(in_frame, c.in_frame_min, c.in_frame_full) * blur_f * upper)

            dt = t - self._t_obj[view.ids]
            self._t_obj[view.ids] = t
            z = self._ar1(self._miss_z, view.ids, dt, c.miss_tau_s)
            detected = (_phi(z) < p) & (view.vis_px > 0)
            jit = self._ar1(self._jit, view.ids, dt, c.jitter_tau_s)

            k = np.nonzero(detected)[0]
            if k.size:
                w_am = ramp(vis[k], c.amodal_vis_lo, c.amodal_vis_hi)[:, None]
                box = w_am * amc[k] + (1.0 - w_am) * view.vis_box[k]
                cx, cy = (box[:, 0] + box[:, 2]) / 2, (box[:, 1] + box[:, 3]) / 2
                bw, bh = box[:, 2] - box[:, 0], box[:, 3] - box[:, 1]
                cx = cx + jit[k, 0] * (c.jitter_center_frac * bw + c.jitter_center_px)
                cy = cy + jit[k, 1] * (c.jitter_center_frac * bh + c.jitter_center_px)
                bw = bw * np.exp(c.jitter_scale_frac * jit[k, 2]) + 0.5 * blur_px
                bh = bh * np.exp(c.jitter_scale_frac * jit[k, 3])
                boxes = np.stack([cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2], 1)
                scores = np.clip(c.score_threshold + (0.97 - c.score_threshold) * p[k]
                                 + 0.04 * self.rng.standard_normal(k.size), c.score_threshold + 1e-3, 0.99)
                truth = [int(view.ids[i]) for i in k]

        # false-positive ghosts: spawn, age, jitter
        for _ in range(self.rng.poisson(c.fp_rate_per_frame)):
            h = math.exp(self.rng.uniform(math.log(6), math.log(40)))
            w = h * self.rng.uniform(0.35, 0.9)
            x, y = self.rng.uniform(0, W), self.rng.uniform(c.upper_band_frac * H, 0.95 * H)
            self._ghosts.append({"box": np.array([x - w / 2, y - h / 2, x + w / 2, y + h / 2]),
                                 "score": self.rng.uniform(c.score_threshold, 0.6),
                                 "left": self.rng.geometric(1.0 / max(c.fp_mean_life_frames, 1.0))})
        live = []
        for g in self._ghosts:
            g["left"] -= 1
            if g["left"] < 0:
                continue
            live.append(g)
            wh = g["box"][2:] - g["box"][:2]
            b = g["box"] + np.tile(self.rng.normal(0, 0.03, 2) * wh, 2)
            boxes = np.vstack([boxes, b[None]])
            scores = np.append(scores, g["score"])
            truth.append(-1)
        self._ghosts = live

        # clip, drop degenerate, NMS
        if len(scores):
            boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, W)
            boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, H)
            ok = ((boxes[:, 2] - boxes[:, 0]) >= 1.0) & ((boxes[:, 3] - boxes[:, 1]) >= 1.0)
            boxes, scores, truth = boxes[ok], scores[ok], [tr for tr, o in zip(truth, ok) if o]
            keep = nms(boxes, scores, c.nms_iou) if len(scores) else np.zeros(0, dtype=np.int64)
            boxes, scores, truth = boxes[keep], scores[keep], [truth[i] for i in keep]

        dbg = FrameDebug(view, p, vis, detected, truth, blur_px)
        if self.rng.random() < c.frame_drop_prob:
            dbg.truth = []
            return None, dbg
        latency = max(c.latency_min_s, self.rng.normal(c.latency_mean_s, c.latency_std_s))
        dets = [Detection(float(b[0] / W), float(b[1] / H), float(b[2] / W), float(b[3] / H), float(s), 0)
                for b, s in zip(boxes, scores)]
        return DetectionFrame(t, t + latency, dets), dbg
