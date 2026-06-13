from __future__ import annotations

import threading
import time
from array import array
from typing import Tuple

import cv2
import numpy as np
import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import MultiArrayDimension, UInt8MultiArray

from .common import (
    CAMERA_LEFT_FLIP_VERTICAL,
    CAMERA_LEFT_INDEX,
    CAMERA_RIGHT_FLIP_VERTICAL,
    CAMERA_RIGHT_INDEX,
    CAMERA_SWAP_LEFT_RIGHT,
)

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


def depth_to_jpeg(
    depth_buf: np.ndarray, conf_buf: np.ndarray, depth_range: float
) -> bytes:
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


def make_tof_packet(depth_buf: np.ndarray, conf_buf: np.ndarray) -> UInt8MultiArray:
    depth = np.ascontiguousarray(depth_buf.astype("<f4", copy=False))
    confidence = np.ascontiguousarray(conf_buf.astype(np.uint8, copy=False))
    payload = depth.tobytes() + confidence.tobytes()

    msg = UInt8MultiArray()
    msg.layout.dim = [
        MultiArrayDimension(
            label="height", size=depth.shape[0], stride=depth.shape[0] * depth.shape[1]
        ),
        MultiArrayDimension(label="width", size=depth.shape[1], stride=depth.shape[1]),
        MultiArrayDimension(
            label="depth_bytes", size=depth.nbytes, stride=depth.nbytes
        ),
        MultiArrayDimension(
            label="confidence_bytes", size=confidence.nbytes, stride=confidence.nbytes
        ),
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
        raise ValueError(
            f"Depth payload has {depth_bytes} bytes, expected {expected_depth_bytes}"
        )
    if confidence_bytes != expected_confidence_bytes:
        raise ValueError(
            f"Confidence payload has {confidence_bytes} bytes, expected {expected_confidence_bytes}"
        )

    payload = bytes(msg.data)
    expected_total = depth_bytes + confidence_bytes
    if len(payload) != expected_total:
        raise ValueError(
            f"ToF packet has {len(payload)} bytes, expected {expected_total}"
        )

    depth = (
        np.frombuffer(payload[:depth_bytes], dtype="<f4")
        .reshape((height, width))
        .copy()
    )
    confidence = (
        np.frombuffer(payload[depth_bytes:], dtype=np.uint8)
        .reshape((height, width))
        .copy()
    )
    return depth, confidence


def make_compressed_image(jpeg: bytes, frame_id: str, stamp) -> CompressedImage:
    msg = CompressedImage()
    msg.header.stamp = stamp
    msg.header.frame_id = frame_id
    msg.format = "jpeg"
    msg.data = array("B", jpeg)
    return msg


def sensor_qos() -> QoSProfile:
    return QoSProfile(
        reliability=ReliabilityPolicy.BEST_EFFORT,
        history=HistoryPolicy.KEEP_LAST,
        depth=1,
    )


class TofCameraNode(Node):
    def __init__(self) -> None:
        super().__init__("tof_camera_node")

        self.declare_parameter("side", "left")
        self.declare_parameter("camera_index", -1)
        self.declare_parameter("frame_id", "")
        self.declare_parameter("max_distance_mm", MAX_DISTANCE_MM)
        self.declare_parameter("preview_topic", "")
        self.declare_parameter("packet_topic", "")

        self.side = str(self.get_parameter("side").value)
        configured_index = int(self.get_parameter("camera_index").value)
        self.camera_index = (
            configured_index
            if configured_index >= 0
            else camera_index_for_side(self.side)
        )
        self.vertical_flip = vertical_flip_for_side(self.side)
        self.frame_id = str(self.get_parameter("frame_id").value) or f"tof_{self.side}"
        self.max_distance_mm = int(self.get_parameter("max_distance_mm").value)

        packet_topic = (
            str(self.get_parameter("packet_topic").value)
            or f"/camera/{self.side}/tof_packet"
        )
        preview_topic = (
            str(self.get_parameter("preview_topic").value)
            or f"/camera/{self.side}/preview/compressed"
        )

        self.packet_pub = self.create_publisher(
            UInt8MultiArray, packet_topic, sensor_qos()
        )
        self.preview_pub = self.create_publisher(
            CompressedImage, preview_topic, sensor_qos()
        )

        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._camera_loop, daemon=True)
        self._thread.start()

        self.get_logger().info(
            f"Starting {self.side} ToF camera node: CSI index={self.camera_index}, "
            f"vertical_flip={self.vertical_flip}, packet={packet_topic}, preview={preview_topic}"
        )

    def _camera_loop(self) -> None:
        import ArducamDepthCamera as ac

        cam = ac.ArducamCamera()
        opened = False
        try:
            if cam.open(ac.Connection.CSI, self.camera_index) != 0:
                self.get_logger().error(
                    f"Failed to open Arducam CSI index {self.camera_index}"
                )
                return
            opened = True

            cam.start(ac.FrameType.DEPTH)
            cam.setControl(ac.Control.RANGE, self.max_distance_mm)
            depth_range = cam.getControl(ac.Control.RANGE)

            while rclpy.ok() and not self._stop.is_set():
                try:
                    frame = cam.requestFrame(1000)
                    if frame is None or not isinstance(frame, ac.DepthData):
                        continue
                    if not rclpy.ok() or self._stop.is_set():
                        cam.releaseFrame(frame)
                        break

                    # Deep-copy before releaseFrame. This is the critical guard against
                    # SDK-owned shared buffers bleeding between the two cameras.
                    depth = np.array(frame.depth_data, copy=True).astype(np.float32)
                    confidence = np.array(frame.confidence_data, copy=True)
                    cam.releaseFrame(frame)
                    if not rclpy.ok() or self._stop.is_set():
                        break

                    depth_oriented, confidence_oriented = orient_camera_frame(
                        depth,
                        confidence,
                        self.vertical_flip,
                    )
                    jpeg = depth_to_jpeg(
                        depth_oriented, confidence_oriented, depth_range
                    )

                    if not rclpy.ok() or self._stop.is_set():
                        break
                    stamp = self.get_clock().now().to_msg()
                    self.packet_pub.publish(
                        make_tof_packet(depth_oriented, confidence_oriented)
                    )
                    self.preview_pub.publish(
                        make_compressed_image(jpeg, self.frame_id, stamp)
                    )
                except Exception as exc:
                    if not rclpy.ok() or self._stop.is_set():
                        break
                    self.get_logger().error(
                        f"Camera {self.camera_index} loop error: {exc}"
                    )
                    time.sleep(1.0)
        finally:
            if opened:
                for method_name in ("stop", "close"):
                    method = getattr(cam, method_name, None)
                    if callable(method):
                        try:
                            method()
                        except Exception:
                            pass

    def destroy_node(self) -> bool:
        self._stop.set()
        if self._thread.is_alive():
            self._thread.join(timeout=2.0)
        return super().destroy_node()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = TofCameraNode()
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
