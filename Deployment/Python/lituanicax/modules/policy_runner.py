"""ONNX policy inference loop.

Replaces `policy_runner_node.py`.  Reads the newest frame from both cameras
plus the wheel velocity, runs the network, and hands the result straight to
`Drive.command()`.
"""

from __future__ import annotations

import importlib
import os
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np

from ..config import BRAKE_MAGNITUDE, POLICY_HZ
from ..module import Module
from .drive import Drive
from .tof_camera import ACTION_HIST, OBS_DIM, TofCamera, process_for_policy

MAX_WHEEL_VELOCITY_M_S = 6.7
WHEEL_VELOCITY_LPF_ALPHA = 0.1


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
    return Path(__file__).resolve().parents[1] / "policy.onnx"


class PolicyRunner(Module):
    # Inference sits on the control path, so keep it off the normal scheduler.
    realtime = True

    def __init__(
        self,
        left_camera: TofCamera | None,
        right_camera: TofCamera | None,
        drive: Drive | None = None,
        policy_path: str = "",
        rate_hz: float = POLICY_HZ,
        camera_timeout_sec: float = 0.5,
        onnx_threads: int = 1,
    ) -> None:
        super().__init__("policy_runner", rate_hz)

        self.left_camera = left_camera
        self.right_camera = right_camera
        self.drive = drive
        self.policy_path = discover_policy_path(policy_path)
        self.camera_timeout_sec = float(camera_timeout_sec)
        # Small MLPs suffer from huge thread spin-wait overhead if > 1
        self.onnx_threads = int(onnx_threads)

        self.enabled = False
        self.action_history = np.zeros(ACTION_HIST, dtype=np.float32)
        self.wheel_vel_norm = 0.0
        self.last_action = (0.0, 0.0, 0.0)

        self.session = None
        self.input_name = ""
        self.output_name = ""
        self._wheel_vel_filtered = 0.0

    def setup(self) -> None:
        if not self.policy_path.exists():
            raise FileNotFoundError(f"Policy file does not exist: {self.policy_path}")
        if self.left_camera is None or self.right_camera is None:
            self.log.warning("Cameras are disabled: the policy will never command the drive")

        ort = _load_onnxruntime()
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = self.onnx_threads
        opts.log_severity_level = 3
        self.session = ort.InferenceSession(
            str(self.policy_path), sess_options=opts, providers=["CPUExecutionProvider"]
        )
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

        self.log.info(
            f"Policy loaded from {self.policy_path}; obs={OBS_DIM}, target={self.rate_hz:.1f} Hz"
        )

    def set_autopilot_enabled(self, enabled: bool) -> None:
        self.enabled = bool(enabled)
        if not self.enabled:
            self.action_history[:] = 0.0
            self.wheel_vel_norm = 0.0
            self._wheel_vel_filtered = 0.0

    def update_wheel_velocity(self, linear_velocity: float) -> None:
        """Called by the encoder for every sample it takes (see Encoder.on_sample)."""
        raw = max(0.0, min(1.0, linear_velocity / MAX_WHEEL_VELOCITY_M_S))
        self._wheel_vel_filtered = (
            WHEEL_VELOCITY_LPF_ALPHA * raw
            + (1.0 - WHEEL_VELOCITY_LPF_ALPHA) * self._wheel_vel_filtered
        )
        self.wheel_vel_norm = self._wheel_vel_filtered

    def update(self) -> None:
        left = self._fresh_frame(self.left_camera)
        right = self._fresh_frame(self.right_camera)

        if not self.enabled or left is None or right is None:
            self.action_history[:] = 0.0
            self._send(0.0, 0.0, 0.0)
            return

        left_obs = process_for_policy(*left)
        right_obs = process_for_policy(*right)

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

        self._send(throttle_motor, steer, throttle_raw)

    # --- internals ---

    def _fresh_frame(self, camera: TofCamera | None):
        """(depth, confidence) if the camera has a recent frame, else None."""
        if camera is None:
            return None
        depth, confidence, frame_time = camera.snapshot()
        if depth is None or confidence is None:
            return None
        if time.monotonic() - frame_time > self.camera_timeout_sec:
            return None
        return depth, confidence

    def _send(self, throttle_motor: float, steer: float, throttle_raw: float) -> None:
        self.last_action = (throttle_motor, steer, throttle_raw)
        if self.drive is not None:
            self.drive.command(throttle_motor, steer)
