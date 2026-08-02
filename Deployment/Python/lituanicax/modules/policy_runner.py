"""ONNX policy inference loop.

Replaces `policy_runner_node.py`.  Reads the newest frame from both cameras
plus the wheel velocity, runs the network, and hands the result straight to
`Drive.command()`.
"""

from __future__ import annotations

import importlib
import os
import threading
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np

from ..config import BRAKE_MAGNITUDE, POLICY_HZ
from ..module import Module
from .drive import Drive
from .tof_camera import (
    ACTION_HIST,
    OBS_DIM,
    PER_CAM_DIM,
    WHEEL_VEL_OBS,
    TofCamera,
    process_for_policy,
)

MAX_WHEEL_VELOCITY_M_S = 6.7
WHEEL_VELOCITY_LPF_ALPHA = 0.1

#: How long to wait for both cameras before ticking anyway, as a multiple of the
#: nominal policy period.  The fallback exists so a dead or stalled camera cannot
#: wedge the loop: the tick still runs, `update()` takes its "no fresh frame"
#: branch and commands zero, and the drive watchdog keeps being fed.
FRAME_WAIT_TIMEOUT_PERIODS = 1.5
#: Never start a tick sooner than this fraction of the nominal period after the
#: last one.  The ToF sensors are hardware-locked near 30 fps so this should
#: never bite, but the policy was trained at 30 Hz *with an action history* —
#: running the loop faster would quietly shorten the temporal context the network
#: reads, which is a behavioural change, not a speed-up.
MIN_TICK_INTERVAL_PERIODS = 0.8


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
        recorder=None,
        frame_signal: threading.Event | None = None,
    ) -> None:
        super().__init__("policy_runner", rate_hz)

        self.left_camera = left_camera
        self.right_camera = right_camera
        self.drive = drive
        #: Shared with both cameras: set whenever either delivers a depth frame.
        #: When present, `run()` ticks on frame arrival instead of on a timer.
        #: None restores the plain fixed-rate `Module.run()` loop.
        self.frame_signal = frame_signal
        #: Frame counter of the newest frame each camera had at the last tick, so
        #: the loop can tell "a new frame arrived" from "the event was left set".
        self._consumed_frames = {"left": -1, "right": -1}
        #: Optional Recorder (`--log`). Every tick is handed to it whole, so the
        #: images, the observation and the resulting command stay one record.
        self.recorder = recorder
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
        self._last_tick_time = 0.0

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
        model_input = self.session.get_inputs()[0]
        self.input_name = model_input.name
        self.output_name = self.session.get_outputs()[0].name

        # The exported policy hard-codes its input width, so a stale constant here
        # is a silent sim-to-real mismatch.  Catch it at startup, not on the track.
        expected_dim = model_input.shape[-1] if model_input.shape else None
        if isinstance(expected_dim, int) and expected_dim != OBS_DIM:
            raise ValueError(
                f"Policy expects {expected_dim} observations but this build assembles "
                f"{OBS_DIM} ({PER_CAM_DIM} px/camera × 2 + {WHEEL_VEL_OBS} wheel vel + "
                f"{ACTION_HIST} action history).  Check that the crop fractions, "
                f"policy width and prev_action_history_len in tof_camera.py still match "
                f"ConeTrackEnvCfg in the training repo."
            )

        self.log.info(
            f"Policy loaded from {self.policy_path}; obs={OBS_DIM}, target={self.rate_hz:.1f} Hz"
        )

    def set_autopilot_enabled(self, enabled: bool) -> None:
        self.enabled = bool(enabled)
        if not self.enabled:
            self.action_history[:] = 0.0
            self.wheel_vel_norm = 0.0
            self._wheel_vel_filtered = 0.0

    def run(self) -> None:
        """Tick when the cameras have both delivered, not when a timer says so.

        The free-running 30 Hz timer this replaces was measurably costing the
        control loop: with the two sensors free-running near 29.33 Hz and the
        loop at 30 Hz, the frames the policy read were on average 23.2 ms (left)
        and 14.3 ms (right) old at inference time — both very steady (std
        1.4 ms), i.e. the loop had settled into a fixed, arbitrary phase against
        the sensors and was simply reading late.  Waiting for the SECOND of the
        two arrivals instead puts inference right behind the newer frame and
        leaves the older one only the inter-sensor skew (~9 ms) stale, cutting
        the mean age from ~18.8 ms to ~4.5 ms.  That is ~14 ms off a loop whose
        measured command-to-yaw latency is 100 ms against the 33 ms the policy
        was trained with.

        Falls back to `Module.run()`'s fixed-rate loop when there is no signal to
        wait on (no cameras, or constructed without `frame_signal`).
        """
        if self.frame_signal is None or self.left_camera is None or self.right_camera is None:
            super().run()
            return

        period = 1.0 / self.rate_hz
        timeout = FRAME_WAIT_TIMEOUT_PERIODS * period
        min_interval = MIN_TICK_INTERVAL_PERIODS * period
        last_tick = 0.0

        while not self.stopping:
            self._wait_for_frames(timeout)
            if self.stopping:
                break

            since_last = time.monotonic() - last_tick
            if since_last < min_interval:
                self.sleep(min_interval - since_last)
                if self.stopping:
                    break
            last_tick = time.monotonic()

            self._mark_frames_consumed()
            try:
                self.update()
            except Exception as exc:
                self.log.error(f"update() failed: {exc}", exc_info=True)

    def _wait_for_frames(self, timeout: float) -> None:
        """Block until both cameras hold a frame newer than the last tick's.

        Returns early on timeout so a stalled camera degrades the loop to a timed
        tick rather than stopping it.  The event is cleared BEFORE the condition
        is tested: a frame landing in the gap between the test and the wait
        leaves the event set, so `wait()` returns immediately instead of missing
        the wakeup.
        """
        deadline = time.monotonic() + timeout
        while not self.stopping:
            self.frame_signal.clear()
            if self._both_cameras_advanced():
                return
            remaining = deadline - time.monotonic()
            if remaining <= 0.0:
                return
            self.frame_signal.wait(remaining)

    def _both_cameras_advanced(self) -> bool:
        """True once every camera holds a frame newer than the last tick's.

        Deliberately free of side effects: the counters are only advanced by
        `_mark_frames_consumed()` at tick time.  Marking a camera as seen the
        moment it delivers would make it compare equal on the next pass through
        the wait loop, and the condition could then never be satisfied by two
        sensors that arrive at different times — which is the entire case this
        exists to handle.
        """
        for side, camera in (("left", self.left_camera), ("right", self.right_camera)):
            if camera is None:
                continue
            if camera.frame_count == self._consumed_frames[side]:
                return False
        return True

    def _mark_frames_consumed(self) -> None:
        for side, camera in (("left", self.left_camera), ("right", self.right_camera)):
            if camera is not None:
                self._consumed_frames[side] = camera.frame_count

    def update_wheel_velocity(self, linear_velocity: float) -> None:
        """Called by the encoder for every sample it takes (see Encoder.on_sample)."""
        raw = max(0.0, min(1.0, linear_velocity / MAX_WHEEL_VELOCITY_M_S))
        self._wheel_vel_filtered = (
            WHEEL_VELOCITY_LPF_ALPHA * raw
            + (1.0 - WHEEL_VELOCITY_LPF_ALPHA) * self._wheel_vel_filtered
        )
        self.wheel_vel_norm = self._wheel_vel_filtered

    def update(self) -> None:
        tick_time = time.monotonic()
        left = self._fresh_frame(self.left_camera)
        right = self._fresh_frame(self.right_camera)

        if not self.enabled or left is None or right is None:
            self.action_history[:] = 0.0
            history = self.action_history
            self._send(0.0, 0.0, 0.0)
            self._record(
                tick_time,
                left,
                right,
                history,
                obs=None,
                infer_ms=None,
                skipped="autopilot disabled" if not self.enabled else "no fresh frame",
            )
            return

        left_obs = process_for_policy(left[0], left[1])
        right_obs = process_for_policy(right[0], right[1])

        # Layout must match ConeTrackEnv._get_observations exactly:
        # [depth_left | depth_right | wheel_vel | action history oldest→newest].
        wheel_vel_arr = np.array([self.wheel_vel_norm], dtype=np.float32)
        history = self.action_history
        obs = np.concatenate([left_obs, right_obs, wheel_vel_arr, history])[np.newaxis, :]

        inference_start = time.monotonic()
        action = self.session.run([self.output_name], {self.input_name: obs})[0].clip(-1.0, 1.0)[0]
        infer_ms = (time.monotonic() - inference_start) * 1000.0
        throttle_raw = float(action[0])
        steer = float(action[1])
        # The one and only place the brake is scaled. BRAKE_MAGNITUDE mirrors the
        # sim's brake_throttle_scale, so this line reproduces the actuator model
        # the policy was trained against; `Drive` now leaves the negative side
        # alone rather than scaling it a second time by MAX_THROTTLE.
        throttle_motor = throttle_raw if throttle_raw >= 0 else throttle_raw * BRAKE_MAGNITUDE

        # Push this tick's raw command onto the history FIFO, newest last — the sim
        # does the same in _pre_physics_step, so the next observation carries it as
        # the final pair.  Note it is the RAW policy output that is remembered, not
        # the brake-scaled value handed to the motor.  `np.roll` returns a new
        # array, so `history` still holds what this tick was given.
        self.action_history = np.roll(self.action_history, -2)
        self.action_history[-2] = throttle_raw
        self.action_history[-1] = steer

        self._send(throttle_motor, steer, throttle_raw)
        self._record(tick_time, left, right, history, obs[0], infer_ms, None)

    # --- internals ---

    def _fresh_frame(self, camera: TofCamera | None):
        """(depth, confidence, frame time, frame id) if recent, else None."""
        if camera is None:
            return None
        depth, confidence, frame_time, frame_id = camera.snapshot_with_id()
        if depth is None or confidence is None:
            return None
        if time.monotonic() - frame_time > self.camera_timeout_sec:
            return None
        return depth, confidence, frame_time, frame_id

    def _send(self, throttle_motor: float, steer: float, throttle_raw: float) -> None:
        self.last_action = (throttle_motor, steer, throttle_raw)
        if self.drive is not None:
            self.drive.command(throttle_motor, steer)

    def _record(self, tick_time, left, right, history, obs, infer_ms, skipped) -> None:
        """Hand this tick — frames, observation, action, and what the drive did
        with it — to the recorder as one record. No-op without `--log`."""
        if self.recorder is None:
            return

        throttle_motor, steer, throttle_raw = self.last_action
        row = {
            "t": tick_time,
            "t_wall": tick_time + self.recorder.clock_offset,
            "dt": tick_time - self._last_tick_time if self._last_tick_time else None,
            "enabled": self.enabled,
            "skipped": skipped,
            "wheel_vel_norm": self.wheel_vel_norm,
            "action_history_in": [float(value) for value in history],
            "action_throttle_raw": throttle_raw,
            "action_steer": steer,
            "action_throttle_motor": throttle_motor,
            "infer_ms": infer_ms,
            "tick_ms": (time.monotonic() - tick_time) * 1000.0,
        }
        self._last_tick_time = tick_time

        frames = {}
        for side, frame in (("left", left), ("right", right)):
            # Always present, so every row has the same columns: null means the
            # camera had nothing fresh and the policy commanded zero.
            row[f"{side}_frame_id"] = None
            row[f"{side}_frame_t"] = None
            row[f"{side}_frame_age_ms"] = None
            row[f"{side}_image_index"] = None
            if frame is not None:
                depth, confidence, frame_time, frame_id = frame
                row[f"{side}_frame_id"] = frame_id
                row[f"{side}_frame_t"] = frame_time
                row[f"{side}_frame_age_ms"] = (tick_time - frame_time) * 1000.0
                frames[side] = (frame_id, frame_time, depth, confidence)

        if self.drive is not None:
            row.update(
                {
                    "drive_enabled": self.drive.enabled,
                    "drive_throttle": self.drive.last_throttle,
                    "drive_steer": self.drive.last_steer,
                    "drive_duty": self.drive.last_duty_fraction,
                }
            )

        self.recorder.log_policy_step(row, frames, obs)
