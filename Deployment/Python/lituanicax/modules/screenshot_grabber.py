"""Saves colorized depth screenshots to Deployment/Screenshots/.

Replaces `screenshot_grabber.py`.  Applies the same orientation and crop as
the policy pipeline, once per second, whenever both cameras have a new frame.
"""

from __future__ import annotations

import time
from pathlib import Path

import cv2
import numpy as np

from ..module import Module
from .tof_camera import (
    CONF_THRESHOLD,
    CROP_BOTTOM_ROWS,
    CROP_TOP_ROWS,
    DEPTH_RAW_H,
    MAX_DISTANCE_MM,
    TofCamera,
)

SAVE_DIR = Path(__file__).resolve().parents[3] / "Screenshots"


def colorize_depth(depth: np.ndarray, confidence: np.ndarray) -> np.ndarray:
    """Convert a depth slice to a color BGR image (same as the dashboard preview)."""
    depth = np.nan_to_num(depth)
    depth[confidence < CONF_THRESHOLD] = 0.0
    vis = (depth * (255.0 / MAX_DISTANCE_MM)).astype(np.uint8)
    vis = cv2.applyColorMap(vis, cv2.COLORMAP_RAINBOW)
    vis[confidence < CONF_THRESHOLD] = (0, 0, 0)
    return vis


class ScreenshotGrabber(Module):
    def __init__(
        self,
        left_camera: TofCamera | None,
        right_camera: TofCamera | None,
        rate_hz: float = 1.0,
        save_dir: Path = SAVE_DIR,
    ) -> None:
        super().__init__("screenshot_grabber", rate_hz)

        self.left_camera = left_camera
        self.right_camera = right_camera
        self.save_dir = Path(save_dir)
        self._last_saved_frames = (-1, -1)

    def setup(self) -> None:
        if self.left_camera is None or self.right_camera is None:
            raise RuntimeError("Screenshot grabber needs both ToF camera modules")
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.log.info(f"Ready – saving cropped screenshots to {self.save_dir}")

    def update(self) -> None:
        depth_l, conf_l, _ = self.left_camera.snapshot()
        depth_r, conf_r, _ = self.right_camera.snapshot()
        if depth_l is None or depth_r is None:
            return

        # Only save once per pair of new frames, like consuming the ROS packets did.
        frames = (self.left_camera.frame_count, self.right_camera.frame_count)
        if frames == self._last_saved_frames:
            return
        self._last_saved_frames = frames

        crop = slice(CROP_TOP_ROWS, DEPTH_RAW_H - CROP_BOTTOM_ROWS)
        image_l = colorize_depth(depth_l[crop, :], conf_l[crop, :])
        image_r = colorize_depth(depth_r[crop, :], conf_r[crop, :])

        stamp = time.strftime("%Y%m%d_%H%M%S")
        cv2.imwrite(str(self.save_dir / f"left_cropped_{stamp}.png"), image_l)
        cv2.imwrite(str(self.save_dir / f"right_cropped_{stamp}.png"), image_r)
        self.log.info(f"Saved cropped screenshots at {stamp}")
