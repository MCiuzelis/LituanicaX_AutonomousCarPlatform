"""Arducam ToF depth camera.

Replaces `tof_camera_node.py`.  One instance per camera; each owns a thread
that pulls frames from the SDK, keeps the newest depth/confidence pair for the
policy, and renders a JPEG preview for the dashboard at a lower rate.

The depth pipeline constants and helpers below are shared with the policy and
the screenshot grabber, exactly as they were in the ROS package.
"""

from __future__ import annotations

import threading
import time
from typing import Tuple

import cv2
import numpy as np

from ..config import (
    CAMERA_LEFT_FLIP_VERTICAL,
    CAMERA_LEFT_INDEX,
    CAMERA_RIGHT_FLIP_VERTICAL,
    CAMERA_RIGHT_INDEX,
    CAMERA_SWAP_LEFT_RIGHT,
    DASHBOARD_HZ,
)
from ..module import Module

MAX_DISTANCE_MM = 2000
CONF_THRESHOLD = 30

DEPTH_RAW_H = 180
DEPTH_RAW_W = 240
CROP_TOP_FRAC = 0.375
CROP_BOTTOM_FRAC = 0.45
POLICY_W = 120
CROP_TOP_ROWS = int(CROP_TOP_FRAC * DEPTH_RAW_H)
CROP_BOTTOM_ROWS = int(CROP_BOTTOM_FRAC * DEPTH_RAW_H)
CROPPED_H = DEPTH_RAW_H - CROP_TOP_ROWS - CROP_BOTTOM_ROWS
POLICY_H = round(POLICY_W * CROPPED_H / DEPTH_RAW_W)

PER_CAM_DIM = POLICY_W * POLICY_H
ACTION_HIST = 30
EXTRA_OBS = 1
OBS_DIM = PER_CAM_DIM * 2 + EXTRA_OBS  # Temporarily changed from + ACTION_HIST


def camera_index_for_side(side: str) -> int:
    if side not in ("left", "right"):
        raise ValueError(f"Unknown camera side: {side}")
    left_index = CAMERA_RIGHT_INDEX if CAMERA_SWAP_LEFT_RIGHT else CAMERA_LEFT_INDEX
    right_index = CAMERA_LEFT_INDEX if CAMERA_SWAP_LEFT_RIGHT else CAMERA_RIGHT_INDEX
    return left_index if side == "left" else right_index


def vertical_flip_for_side(side: str) -> bool:
    if side == "left":
        return CAMERA_LEFT_FLIP_VERTICAL
    if side == "right":
        return CAMERA_RIGHT_FLIP_VERTICAL
    raise ValueError(f"Unknown camera side: {side}")


def orient_camera_frame(
    depth_buf: np.ndarray, conf_buf: np.ndarray, vertical_flip: bool
) -> Tuple[np.ndarray, np.ndarray]:
    depth = np.flip(depth_buf, (0, 1))
    confidence = np.flip(conf_buf, (0, 1))
    if vertical_flip:
        depth = np.flip(depth, 0)
        confidence = np.flip(confidence, 0)
    return np.ascontiguousarray(depth), np.ascontiguousarray(confidence)


def depth_to_jpeg(depth_buf: np.ndarray, conf_buf: np.ndarray, depth_range: float) -> bytes:
    depth = np.nan_to_num(depth_buf)
    depth[conf_buf < CONF_THRESHOLD] = 0
    color = (depth * (255.0 / depth_range)).astype(np.uint8)
    color = cv2.applyColorMap(color, cv2.COLORMAP_RAINBOW)
    color[conf_buf < CONF_THRESHOLD] = (0, 0, 0)
    ok, jpeg = cv2.imencode(".jpg", color, [cv2.IMWRITE_JPEG_QUALITY, 80])
    if not ok:
        return b""
    return jpeg.tobytes()


def process_for_policy(depth_buf: np.ndarray, conf_buf: np.ndarray) -> np.ndarray:
    depth_m = np.nan_to_num(depth_buf).astype(np.float32) / 1000.0
    depth_m[conf_buf < CONF_THRESHOLD] = MAX_DISTANCE_MM / 1000.0

    cropped = depth_m[CROP_TOP_ROWS : DEPTH_RAW_H - CROP_BOTTOM_ROWS, :]
    resized = cv2.resize(cropped, (POLICY_W, POLICY_H), interpolation=cv2.INTER_LINEAR)
    normalized = np.clip(1.0 - resized / (MAX_DISTANCE_MM / 1000.0), 0.0, 1.0)
    return normalized.flatten().astype(np.float32)


class TofCamera(Module):
    """Streams one Arducam ToF camera. Overrides run(): the SDK blocks for us."""

    def __init__(
        self,
        side: str,
        camera_index: int | None = None,
        max_distance_mm: int = MAX_DISTANCE_MM,
        preview_hz: float = DASHBOARD_HZ,
    ) -> None:
        super().__init__(f"tof_camera_{side}", preview_hz)

        self.side = side
        self.camera_index = camera_index if camera_index is not None else camera_index_for_side(side)
        self.vertical_flip = vertical_flip_for_side(side)
        self.frame_id = f"tof_{side}"
        self.max_distance_mm = int(max_distance_mm)
        self.preview_period = 1.0 / float(preview_hz)

        self.frame_count = 0
        self.preview_jpeg: bytes | None = None

        self._lock = threading.Lock()
        self._depth: np.ndarray | None = None
        self._confidence: np.ndarray | None = None
        self._frame_time = 0.0
        self._last_preview_time = 0.0

    def setup(self) -> None:
        self.log.info(
            f"Starting {self.side} ToF camera: CSI index={self.camera_index}, "
            f"vertical_flip={self.vertical_flip}"
        )

    def snapshot(self) -> Tuple[np.ndarray | None, np.ndarray | None, float]:
        """Newest (depth, confidence, monotonic timestamp). Arrays are shared read-only."""
        with self._lock:
            return self._depth, self._confidence, self._frame_time

    def run(self) -> None:
        import ArducamDepthCamera as ac

        cam = ac.ArducamCamera()
        opened = False
        try:
            if cam.open(ac.Connection.CSI, self.camera_index) != 0:
                self.log.error(f"Failed to open Arducam CSI index {self.camera_index}")
                return
            opened = True

            cam.start(ac.FrameType.DEPTH)
            cam.setControl(ac.Control.RANGE, self.max_distance_mm)
            depth_range = cam.getControl(ac.Control.RANGE)

            while not self.stopping:
                try:
                    self._grab_frame(cam, ac, depth_range)
                except Exception as exc:
                    if self.stopping:
                        break
                    self.log.error(f"Camera {self.camera_index} loop error: {exc}")
                    self.sleep(1.0)
        finally:
            if opened:
                for method_name in ("stop", "close"):
                    method = getattr(cam, method_name, None)
                    if callable(method):
                        try:
                            method()
                        except Exception:
                            pass

    def _grab_frame(self, cam, ac, depth_range: float) -> None:
        frame = cam.requestFrame(1000)
        if frame is None or not isinstance(frame, ac.DepthData):
            return
        if self.stopping:
            cam.releaseFrame(frame)
            return

        # Deep-copy before releaseFrame. This is the critical guard against
        # SDK-owned shared buffers bleeding between the two cameras.
        depth = np.array(frame.depth_data, copy=True).astype(np.float32)
        confidence = np.array(frame.confidence_data, copy=True)
        cam.releaseFrame(frame)
        if self.stopping:
            return

        depth, confidence = orient_camera_frame(depth, confidence, self.vertical_flip)

        now = time.monotonic()
        with self._lock:
            self._depth = depth
            self._confidence = confidence
            self._frame_time = now
            self.frame_count += 1

        if now - self._last_preview_time >= self.preview_period:
            self.preview_jpeg = depth_to_jpeg(depth, confidence, depth_range)
            self._last_preview_time = now
