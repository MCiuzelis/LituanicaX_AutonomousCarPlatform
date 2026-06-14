from __future__ import annotations

import importlib
import os
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import Bool, Float32MultiArray, UInt8MultiArray

from .common import BRAKE_MAGNITUDE, POLICY_HZ
from .tof_camera_node import ACTION_HIST, OBS_DIM, process_for_policy, unpack_tof_packet


@contextmanager
def _suppress_native_stderr():
    """Temporarily silence native-library stderr during noisy imports only."""
    stderr_fd = 2
    saved_stderr = os.dup(stderr_fd)
    try:
        with open(os.devnull, "w") as devnull:
            os.dup2(devnull.fileno(), stderr_fd)
            yield
    finally:
        os.dup2(saved_stderr, stderr_fd)
        os.close(saved_stderr)


def _load_onnxruntime():
    os.environ.setdefault("ORT_LOG_SEVERITY_LEVEL", "3")
    with _suppress_native_stderr():
        ort = importlib.import_module("onnxruntime")
    if hasattr(ort, "set_default_logger_severity"):
        ort.set_default_logger_severity(3)
    return ort


def discover_policy_path(configured_policy_path: str = "") -> Path:
    if configured_policy_path:
        return Path(configured_policy_path).expanduser()
    return Path(__file__).with_name("policy.onnx")


def sensor_qos() -> QoSProfile:
    return QoSProfile(
        reliability=ReliabilityPolicy.BEST_EFFORT,
        history=HistoryPolicy.KEEP_LAST,
        depth=1,
    )


@dataclass
class CameraState:
    depth: np.ndarray | None = None
    confidence: np.ndarray | None = None
    monotonic_time: float = 0.0


class PolicyRunnerNode(Node):
    def __init__(self) -> None:
        super().__init__("policy_runner_node")

        self.declare_parameter("policy_path", "")
        self.declare_parameter("policy_hz", POLICY_HZ)
        self.declare_parameter("camera_timeout_sec", 0.5)
        # Small MLPs suffer from huge thread spin-wait overhead if > 1
        self.declare_parameter("onnx_threads", 1)

        policy_path = discover_policy_path(str(self.get_parameter("policy_path").value))
        if not policy_path.exists():
            raise FileNotFoundError(f"Policy file does not exist: {policy_path}")

        ort = _load_onnxruntime()
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = int(self.get_parameter("onnx_threads").value)
        opts.log_severity_level = 3
        self.session = ort.InferenceSession(
            str(policy_path),
            sess_options=opts,
            providers=["CPUExecutionProvider"],
        )
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

        self.left = CameraState()
        self.right = CameraState()
        self.enabled = False
        self.action_history = np.zeros(ACTION_HIST, dtype=np.float32)
        self.wheel_vel_norm = 0.0
        self._wheel_vel_filtered = 0.0
        self._wheel_vel_alpha = 0.1
        self.camera_timeout_sec = float(self.get_parameter("camera_timeout_sec").value)

        self.cmd_pub = self.create_publisher(Float32MultiArray, "/cmd_drive", 1)
        self.action_pub = self.create_publisher(Float32MultiArray, "/policy/action", 1)

        self.create_subscription(UInt8MultiArray, "/camera/left/tof_packet", self._left_cb, sensor_qos())
        self.create_subscription(UInt8MultiArray, "/camera/right/tof_packet", self._right_cb, sensor_qos())
        self.create_subscription(Bool, "/autopilot/enabled", self._enabled_cb, 1)
        self.create_subscription(Float32MultiArray, "/monitoring/encoder", self._encoder_cb, 1)

        period = 1.0 / float(self.get_parameter("policy_hz").value)
        self.timer = self.create_timer(period, self._tick)

        self.get_logger().info(f"Policy loaded from {policy_path}; obs={OBS_DIM}, target={1.0 / period:.1f} Hz")

    def _left_cb(self, msg: UInt8MultiArray) -> None:
        self._store_camera(self.left, msg, "left")

    def _right_cb(self, msg: UInt8MultiArray) -> None:
        self._store_camera(self.right, msg, "right")

    def _store_camera(self, state: CameraState, msg: UInt8MultiArray, side: str) -> None:
        try:
            state.depth, state.confidence = unpack_tof_packet(msg)
            state.monotonic_time = time.monotonic()
        except Exception as exc:
            self.get_logger().warning(f"Dropping malformed {side} ToF packet: {exc}")

    def _enabled_cb(self, msg: Bool) -> None:
        self.enabled = bool(msg.data)
        if not self.enabled:
            self.action_history[:] = 0.0
            self.wheel_vel_norm = 0.0
            self._wheel_vel_filtered = 0.0

    def _encoder_cb(self, msg: Float32MultiArray) -> None:
        if len(msg.data) >= 2:
            raw = max(0.0, min(1.0, float(msg.data[0]) / 6.7))
            self._wheel_vel_filtered = self._wheel_vel_alpha * raw + (1.0 - self._wheel_vel_alpha) * self._wheel_vel_filtered
            self.wheel_vel_norm = self._wheel_vel_filtered

    def _tick(self) -> None:
        now = time.monotonic()
        cameras_ready = (
            self.left.depth is not None
            and self.left.confidence is not None
            and self.right.depth is not None
            and self.right.confidence is not None
            and now - self.left.monotonic_time <= self.camera_timeout_sec
            and now - self.right.monotonic_time <= self.camera_timeout_sec
        )

        if not self.enabled or not cameras_ready:
            self.action_history[:] = 0.0
            self._publish_action(0.0, 0.0, 0.0)
            return

        left_obs = process_for_policy(self.left.depth, self.left.confidence)
        right_obs = process_for_policy(self.right.depth, self.right.confidence)
        
        # Temporarily replacing action_history with wheel_vel_norm
        wheel_vel_arr = np.array([self.wheel_vel_norm], dtype=np.float32)
        obs = np.concatenate([left_obs, right_obs, wheel_vel_arr])[np.newaxis, :]
        # obs = np.concatenate([left_obs, right_obs, self.action_history])[np.newaxis, :]

        action = self.session.run([self.output_name], {self.input_name: obs})[0].clip(-1.0, 1.0)[0]
        throttle_raw = float(action[0])
        steer = float(action[1])
        throttle_motor = throttle_raw if throttle_raw >= 0 else throttle_raw * BRAKE_MAGNITUDE

        self.action_history = np.roll(self.action_history, -2)
        self.action_history[-2] = throttle_raw
        self.action_history[-1] = steer

        self._publish_action(throttle_motor, steer, throttle_raw)

    def _publish_action(self, throttle_motor: float, steer: float, throttle_raw: float) -> None:
        cmd = Float32MultiArray()
        cmd.data = [float(throttle_motor), float(steer)]
        self.cmd_pub.publish(cmd)

        action = Float32MultiArray()
        action.data = [float(throttle_motor), float(steer), float(throttle_raw), 1.0 if self.enabled else 0.0]
        self.action_pub.publish(action)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = PolicyRunnerNode()
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
