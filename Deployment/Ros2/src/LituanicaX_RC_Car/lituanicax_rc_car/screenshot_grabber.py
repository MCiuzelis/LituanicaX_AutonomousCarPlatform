#!/usr/bin/env python3
"""
ROS2 node that subscribes to both ToF camera packets,
applies the same orientation and crop as the policy pipeline,
and saves colorized depth screenshots to Deployment/Screenshots/.
"""

import sys
import time
from pathlib import Path

_VENV_SITE = (
    Path(__file__).resolve().parents[3] / ".venv" / "lib" / f"python{sys.version_info.major}.{sys.version_info.minor}" / "site-packages"
)
if _VENV_SITE.is_dir():
    sys.path.insert(0, str(_VENV_SITE))

import cv2
import numpy as np
import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import UInt8MultiArray

# ── constants (mirrors tof_camera_node.py) ──────────────────────────────
MAX_DISTANCE_MM = 2000
CONF_THRESHOLD = 30
DEPTH_RAW_H = 180
DEPTH_RAW_W = 240
CROP_TOP_FRAC = 0.35  # keep in step with tof_camera_node / ConeTrackEnvCfg
CROP_BOTTOM_FRAC = 0.45
CROP_TOP_ROWS = int(CROP_TOP_FRAC * DEPTH_RAW_H)
CROP_BOTTOM_ROWS = int(CROP_BOTTOM_FRAC * DEPTH_RAW_H)


def sensor_qos() -> QoSProfile:
    return QoSProfile(
        reliability=ReliabilityPolicy.BEST_EFFORT,
        history=HistoryPolicy.KEEP_LAST,
        depth=1,
    )

SAVE_DIR = Path(__file__).resolve().parents[4] / "Screenshots"


def orient_frame(depth: np.ndarray, confidence: np.ndarray):
    """180-degree rotation applied to raw camera frames."""
    depth = np.flip(depth, (0, 1))
    confidence = np.flip(confidence, (0, 1))
    return np.ascontiguousarray(depth), np.ascontiguousarray(confidence)


def colorize_depth(depth: np.ndarray, confidence: np.ndarray) -> np.ndarray:
    """Convert a depth slice to a color BGR image (same as dashboard preview)."""
    depth = np.nan_to_num(depth)
    depth[confidence < CONF_THRESHOLD] = 0.0
    vis = (depth * (255.0 / MAX_DISTANCE_MM)).astype(np.uint8)
    vis = cv2.applyColorMap(vis, cv2.COLORMAP_RAINBOW)
    vis[confidence < CONF_THRESHOLD] = (0, 0, 0)
    return vis


def unpack_packet(msg: UInt8MultiArray):
    dim = msg.layout.dim
    height = int(dim[0].size)
    width = int(dim[1].size)
    depth_bytes = int(dim[2].size)
    payload = bytes(msg.data)
    depth = np.frombuffer(payload[:depth_bytes], dtype="<f4").reshape((height, width)).copy()
    confidence = np.frombuffer(payload[depth_bytes:], dtype=np.uint8).reshape((height, width)).copy()
    return depth, confidence


class CameraScreenshotNode(Node):
    def __init__(self):
        super().__init__("camera_screenshot_node")

        self.left_packet: UInt8MultiArray | None = None
        self.right_packet: UInt8MultiArray | None = None

        self.create_subscription(UInt8MultiArray, "/camera/left/tof_packet", self._left_cb, sensor_qos())
        self.create_subscription(UInt8MultiArray, "/camera/right/tof_packet", self._right_cb, sensor_qos())

        SAVE_DIR.mkdir(parents=True, exist_ok=True)
        self.timer = self.create_timer(1.0, self._tick)
        self.get_logger().info(f"Ready – saving cropped screenshots to {SAVE_DIR}")

    def _left_cb(self, msg: UInt8MultiArray):
        self.left_packet = msg

    def _right_cb(self, msg: UInt8MultiArray):
        self.right_packet = msg

    def _tick(self):
        if self.left_packet is None or self.right_packet is None:
            return

        left_msg = self.left_packet
        right_msg = self.right_packet
        self.left_packet = None
        self.right_packet = None

        try:
            depth_l, conf_l = unpack_packet(left_msg)
            depth_r, conf_r = unpack_packet(right_msg)

            depth_l, conf_l = orient_frame(depth_l, conf_l)
            depth_r, conf_r = orient_frame(depth_r, conf_r)

            # Apply the same vertical crop as process_for_policy()
            crop = slice(CROP_TOP_ROWS, DEPTH_RAW_H - CROP_BOTTOM_ROWS)
            depth_l = depth_l[crop, :]
            conf_l = conf_l[crop, :]
            depth_r = depth_r[crop, :]
            conf_r = conf_r[crop, :]

            img_l = colorize_depth(depth_l, conf_l)
            img_r = colorize_depth(depth_r, conf_r)

            ts = time.strftime("%Y%m%d_%H%M%S")
            cv2.imwrite(str(SAVE_DIR / f"left_cropped_{ts}.png"), img_l)
            cv2.imwrite(str(SAVE_DIR / f"right_cropped_{ts}.png"), img_r)
            self.get_logger().info(f"Saved cropped screenshots at {ts}")
        except Exception as e:
            self.get_logger().error(f"Screenshot failed: {e}")


def main(args=None):
    rclpy.init(args=args)
    node = CameraScreenshotNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
