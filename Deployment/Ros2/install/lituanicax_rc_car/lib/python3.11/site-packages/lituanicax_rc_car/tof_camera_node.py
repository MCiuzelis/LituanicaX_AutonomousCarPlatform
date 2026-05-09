from __future__ import annotations

import threading
import time

import numpy as np
import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import UInt8MultiArray

from .common import CAMERA_INDICES, MAX_DISTANCE_MM, depth_to_jpeg, make_compressed_image, make_tof_packet


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
        self.camera_index = configured_index if configured_index >= 0 else CAMERA_INDICES[self.side]
        self.frame_id = str(self.get_parameter("frame_id").value) or f"tof_{self.side}"
        self.max_distance_mm = int(self.get_parameter("max_distance_mm").value)

        packet_topic = str(self.get_parameter("packet_topic").value) or f"/camera/{self.side}/tof_packet"
        preview_topic = str(self.get_parameter("preview_topic").value) or f"/camera/{self.side}/preview/compressed"

        self.packet_pub = self.create_publisher(UInt8MultiArray, packet_topic, sensor_qos())
        self.preview_pub = self.create_publisher(CompressedImage, preview_topic, sensor_qos())

        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._camera_loop, daemon=True)
        self._thread.start()

        self.get_logger().info(
            f"Starting {self.side} ToF camera node: CSI index={self.camera_index}, "
            f"packet={packet_topic}, preview={preview_topic}"
        )

    def _camera_loop(self) -> None:
        import ArducamDepthCamera as ac

        cam = ac.ArducamCamera()
        opened = False
        try:
            if cam.open(ac.Connection.CSI, self.camera_index) != 0:
                self.get_logger().error(f"Failed to open Arducam CSI index {self.camera_index}")
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

                    # Deep-copy before releaseFrame. This is the critical guard against
                    # SDK-owned shared buffers bleeding between the two cameras.
                    depth = np.array(frame.depth_data, copy=True).astype(np.float32)
                    confidence = np.array(frame.confidence_data, copy=True)
                    cam.releaseFrame(frame)

                    jpeg = depth_to_jpeg(depth, confidence, depth_range)

                    depth_flipped = np.ascontiguousarray(np.flip(depth, (0, 1)))
                    confidence_flipped = np.ascontiguousarray(np.flip(confidence, (0, 1)))

                    stamp = self.get_clock().now().to_msg()
                    self.packet_pub.publish(make_tof_packet(depth_flipped, confidence_flipped))
                    self.preview_pub.publish(make_compressed_image(jpeg, self.frame_id, stamp))
                except Exception as exc:
                    self.get_logger().error(f"Camera {self.camera_index} loop error: {exc}")
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
