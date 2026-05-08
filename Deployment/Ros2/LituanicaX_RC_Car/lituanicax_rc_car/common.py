from __future__ import annotations

from array import array
from typing import Tuple

import cv2
import numpy as np
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import MultiArrayDimension, UInt8MultiArray


CAMERA_INDICES = {"left": 0, "right": 8}
MAX_DISTANCE_MM = 2000
CONF_THRESHOLD = 30
FPS_TARGET = 15
POLICY_HZ = 30

DEPTH_RAW_H = 180
DEPTH_RAW_W = 240
CROP_TOP_FRAC = 0.35
CROP_BOTTOM_FRAC = 0.3
POLICY_W = 120
CROP_TOP_ROWS = int(CROP_TOP_FRAC * DEPTH_RAW_H)
CROP_BOTTOM_ROWS = int(CROP_BOTTOM_FRAC * DEPTH_RAW_H)
CROPPED_H = DEPTH_RAW_H - CROP_TOP_ROWS - CROP_BOTTOM_ROWS
POLICY_H = round(POLICY_W * CROPPED_H / DEPTH_RAW_W)

PER_CAM_DIM = POLICY_W * POLICY_H
ACTION_HIST = 30
OBS_DIM = PER_CAM_DIM * 2 + ACTION_HIST


def depth_to_jpeg(depth_buf: np.ndarray, conf_buf: np.ndarray, depth_range: float) -> bytes:
    """Create the dashboard preview image from copied Arducam buffers."""
    depth = np.nan_to_num(depth_buf)
    depth[conf_buf < CONF_THRESHOLD] = 0
    color = (depth * (255.0 / depth_range)).astype(np.uint8)
    color = cv2.applyColorMap(color, cv2.COLORMAP_RAINBOW)
    color[conf_buf < CONF_THRESHOLD] = (0, 0, 0)
    color = cv2.flip(color, -1)
    ok, jpeg = cv2.imencode(".jpg", color, [cv2.IMWRITE_JPEG_QUALITY, 80])
    if not ok:
        return b""
    return jpeg.tobytes()


def process_for_policy(depth_buf: np.ndarray, conf_buf: np.ndarray) -> np.ndarray:
    """
    Match the original training-time depth pipeline used by Deployment/autopilot.py.
    The input arrays should already be flipped into the policy orientation.
    """
    depth_m = np.nan_to_num(depth_buf).astype(np.float32) / 1000.0
    depth_m[conf_buf < CONF_THRESHOLD] = MAX_DISTANCE_MM / 1000.0

    cropped = depth_m[CROP_TOP_ROWS : DEPTH_RAW_H - CROP_BOTTOM_ROWS, :]
    resized = cv2.resize(cropped, (POLICY_W, POLICY_H), interpolation=cv2.INTER_LINEAR)
    normalized = np.clip(1.0 - resized / (MAX_DISTANCE_MM / 1000.0), 0.0, 1.0)
    return normalized.flatten().astype(np.float32)


def make_tof_packet(depth_buf: np.ndarray, conf_buf: np.ndarray) -> UInt8MultiArray:
    """
    Pack one depth/confidence pair into a single ROS message.

    Payload layout:
      depth: little-endian float32, DEPTH_RAW_H * DEPTH_RAW_W values
      confidence: uint8, DEPTH_RAW_H * DEPTH_RAW_W values
    """
    depth = np.ascontiguousarray(depth_buf.astype("<f4", copy=False))
    confidence = np.ascontiguousarray(conf_buf.astype(np.uint8, copy=False))
    payload = depth.tobytes() + confidence.tobytes()

    msg = UInt8MultiArray()
    msg.layout.dim = [
        MultiArrayDimension(label="height", size=depth.shape[0], stride=depth.shape[0] * depth.shape[1]),
        MultiArrayDimension(label="width", size=depth.shape[1], stride=depth.shape[1]),
        MultiArrayDimension(label="depth_bytes", size=depth.nbytes, stride=depth.nbytes),
        MultiArrayDimension(label="confidence_bytes", size=confidence.nbytes, stride=confidence.nbytes),
    ]
    msg.data = array("B", payload)
    return msg


def unpack_tof_packet(msg: UInt8MultiArray) -> Tuple[np.ndarray, np.ndarray]:
    if len(msg.layout.dim) < 4:
        raise ValueError("ToF packet is missing layout dimensions")

    height = int(msg.layout.dim[0].size)
    width = int(msg.layout.dim[1].size)
    depth_bytes = int(msg.layout.dim[2].size)
    confidence_bytes = int(msg.layout.dim[3].size)
    expected_depth_bytes = height * width * np.dtype("<f4").itemsize
    expected_confidence_bytes = height * width

    if depth_bytes != expected_depth_bytes:
        raise ValueError(f"Depth payload has {depth_bytes} bytes, expected {expected_depth_bytes}")
    if confidence_bytes != expected_confidence_bytes:
        raise ValueError(
            f"Confidence payload has {confidence_bytes} bytes, expected {expected_confidence_bytes}"
        )

    payload = bytes(msg.data)
    expected_total = depth_bytes + confidence_bytes
    if len(payload) != expected_total:
        raise ValueError(f"ToF packet has {len(payload)} bytes, expected {expected_total}")

    depth = np.frombuffer(payload[:depth_bytes], dtype="<f4").reshape((height, width)).copy()
    confidence = np.frombuffer(payload[depth_bytes:], dtype=np.uint8).reshape((height, width)).copy()
    return depth, confidence


def make_compressed_image(jpeg: bytes, frame_id: str, stamp) -> CompressedImage:
    msg = CompressedImage()
    msg.header.stamp = stamp
    msg.header.frame_id = frame_id
    msg.format = "jpeg"
    msg.data = array("B", jpeg)
    return msg
