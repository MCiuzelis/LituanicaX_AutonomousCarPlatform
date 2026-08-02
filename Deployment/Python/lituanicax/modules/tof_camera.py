"""Arducam ToF depth camera.

Replaces `tof_camera_node.py`.  One instance per camera; each owns a *subprocess*
that pulls frames from the SDK, keeps the newest depth/confidence pair for the
policy, and renders a JPEG preview for the dashboard at a lower rate.

The depth pipeline constants and helpers below are shared with the policy and
the screenshot grabber, exactly as they were in the ROS package.

Why a subprocess and not a thread
---------------------------------
`ArducamDepthCamera.requestFrame()` blocks until the next frame *without
releasing the GIL*.  In-process that freezes every other Python thread for a
whole frame period: measured on the Pi, a pure-Python heartbeat that should tick
every 1 ms stalled to 33 ms median with a single camera grabbing, and the 30 Hz
policy loop collapsed to 12 Hz with two.  Since the policy was trained at 30 Hz
with an action history, that is a behavioural change, not just a slow log.

So each camera runs in its own process — which is what the ROS build did — and
publishes frames through shared memory.  The parent keeps a light poll thread
that copies the newest frame out; `snapshot()`, `frame_count` and
`preview_jpeg` behave exactly as they did when this was a thread.
"""

from __future__ import annotations

import multiprocessing as mp
import signal
import threading
import time
from multiprocessing import shared_memory
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
from ..logging_utils import get_logger
from ..module import Module

# ── Shared-memory transport between the camera process and this one ──────────
#: Ceiling for one preview JPEG. A 240×180 rainbow frame at quality 80 is ~25 kB.
PREVIEW_MAX_BYTES = 512 * 1024
#: How often the parent poll thread checks for a new frame. Well above the
#: ~30 Hz frame rate so a frame is picked up promptly, and cheap: the fast path
#: is two float reads from shared memory.
CAMERA_POLL_HZ = 400.0

# Slots in the shared metadata array (float64; counters stay exact well past any
# runtime we care about).
_M_FRAME_COUNT = 0
_M_FRAME_TIME = 1
_M_PREVIEW_LEN = 2
_M_PREVIEW_COUNT = 3
_M_DEPTH_RANGE = 4
_M_OPENED = 5
_M_SLOTS = 8

# ─────────────────────────────────────────────────────────────────────────────
# Observation layout — MUST mirror ConeTrackEnvCfg in
# Training/source/cone_track/env.py.  The policy is a fixed-width MLP, so any
# disagreement here is either an onnxruntime shape error at startup (best case)
# or a silently misread observation (worst case).
#
#   env.py depth_max_m           → MAX_DISTANCE_MM / 1000
#   env.py depth_crop_top_frac   → CROP_TOP_FRAC
#   env.py depth_crop_bottom_frac→ CROP_BOTTOM_FRAC
#   env.py policy_depth_width    → POLICY_W
#   env.py include_wheel_vel     → WHEEL_VEL_OBS
#   env.py prev_action_history_len → ACTION_HIST_LEN
# ─────────────────────────────────────────────────────────────────────────────
MAX_DISTANCE_MM = 2000
MAX_DISTANCE_M = MAX_DISTANCE_MM / 1000.0  # == env.py depth_max_m
CONF_THRESHOLD = 30
# Sim convention for "nothing there": ConeTrackEnv._process_single_depth replaces
# every non-finite depth with depth_max_m, and depth_invalid_pixel_prob sets its
# dropouts to the same value.  So anything the sensor cannot measure — NaN, or a
# pixel below CONF_THRESHOLD — must read as FAR here, never as near.
INVALID_DEPTH_MM = MAX_DISTANCE_MM

DEPTH_RAW_H = 180
DEPTH_RAW_W = 240
CROP_TOP_FRAC = 0.35
CROP_BOTTOM_FRAC = 0.45
POLICY_W = 120
CROP_TOP_ROWS = int(CROP_TOP_FRAC * DEPTH_RAW_H)
CROP_BOTTOM_ROWS = int(CROP_BOTTOM_FRAC * DEPTH_RAW_H)
CROPPED_H = DEPTH_RAW_H - CROP_TOP_ROWS - CROP_BOTTOM_ROWS
POLICY_H = round(POLICY_W * CROPPED_H / DEPTH_RAW_W)

# Dashboard-only: how the discarded rows are drawn in the preview stream.  These
# affect the JPEG the browser sees and nothing the policy reads.
CROP_DIM_FACTOR = 0.25
CROP_EDGE_COLOR = (255, 255, 255)

PER_CAM_DIM = POLICY_W * POLICY_H  # 120 × 18 = 2160
WHEEL_VEL_OBS = 1  # normalised wheel speed, one scalar
ACTION_HIST_LEN = 5  # remembered (throttle, steer) commands
ACTION_HIST = 2 * ACTION_HIST_LEN  # flattened width of that history
# Order matters as much as width: [left | right | wheel_vel | action history].
OBS_DIM = PER_CAM_DIM * 2 + WHEEL_VEL_OBS + ACTION_HIST


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
    _mark_policy_crop(color)
    ok, jpeg = cv2.imencode(".jpg", color, [cv2.IMWRITE_JPEG_QUALITY, 80])
    if not ok:
        return b""
    return jpeg.tobytes()


def _mark_policy_crop(color: np.ndarray) -> None:
    """Dim the rows the policy never sees, in place.

    The dashboard streams the whole sensor frame so the camera can be aimed, but
    the policy only gets `CROP_TOP_ROWS : H - CROP_BOTTOM_ROWS`.  Darkening
    instead of cutting keeps both readable at once: full frame for aiming, bright
    band for what actually reaches the network.  The frame is already oriented
    when this runs, so these row indices are the same ones `process_for_policy`
    slices with.
    """
    top = CROP_TOP_ROWS
    bottom = color.shape[0] - CROP_BOTTOM_ROWS
    if top <= 0 and bottom >= color.shape[0]:
        return

    color[:top] = (color[:top] * CROP_DIM_FACTOR).astype(np.uint8)
    color[bottom:] = (color[bottom:] * CROP_DIM_FACTOR).astype(np.uint8)

    # Low-confidence pixels are already black, so dimming alone leaves no visible
    # edge in dead regions — draw the boundary explicitly.
    if 0 < top < color.shape[0]:
        color[top] = CROP_EDGE_COLOR
    if 0 < bottom < color.shape[0]:
        color[bottom - 1] = CROP_EDGE_COLOR


def process_for_policy(depth_buf: np.ndarray, conf_buf: np.ndarray) -> np.ndarray:
    """Sensor frame → the 2160 depth values the policy expects for one camera.

    Verified bit-exact against ConeTrackEnv._process_single_depth (noise off) on
    in-range frames, which is why the step order below is not free to change:
    clamp to [0.01, depth_max] happens BEFORE the resize, exactly as the sim
    clamps before F.interpolate.  Clipping after the resize instead lets an
    out-of-range pixel drag its in-range neighbour toward "far" when the two are
    averaged by the 2:1 downsample.
    """
    # nan → far (see INVALID_DEPTH_MM): nan_to_num's default of 0 would report a
    # dead pixel as an obstacle touching the lens.
    depth_m = np.nan_to_num(
        depth_buf,
        nan=INVALID_DEPTH_MM,
        posinf=INVALID_DEPTH_MM,
        neginf=INVALID_DEPTH_MM,
    ).astype(np.float32) / 1000.0
    depth_m[conf_buf < CONF_THRESHOLD] = MAX_DISTANCE_M

    cropped = depth_m[CROP_TOP_ROWS : DEPTH_RAW_H - CROP_BOTTOM_ROWS, :]
    clamped = np.clip(cropped, 0.01, MAX_DISTANCE_M)
    resized = cv2.resize(clamped, (POLICY_W, POLICY_H), interpolation=cv2.INTER_LINEAR)
    normalized = np.clip(1.0 - resized / MAX_DISTANCE_M, 0.0, 1.0)
    return normalized.flatten().astype(np.float32)


def _attach(names: dict) -> tuple:
    """Attach the child to the parent's shared memory and build numpy views."""
    # track=False: the parent created these and unlinks them, so this process
    # must not also register them with the resource tracker.
    blocks = {
        key: shared_memory.SharedMemory(name=name, create=False, track=False)
        for key, name in names.items()
    }
    meta = np.ndarray(_M_SLOTS, dtype=np.float64, buffer=blocks["meta"].buf)
    depth = np.ndarray((DEPTH_RAW_H, DEPTH_RAW_W), dtype=np.float32, buffer=blocks["depth"].buf)
    conf = np.ndarray((DEPTH_RAW_H, DEPTH_RAW_W), dtype=np.uint8, buffer=blocks["conf"].buf)
    preview = np.ndarray(PREVIEW_MAX_BYTES, dtype=np.uint8, buffer=blocks["preview"].buf)
    return blocks, meta, depth, conf, preview


def _camera_worker(
    side: str,
    camera_index: int,
    vertical_flip: bool,
    max_distance_mm: int,
    preview_period: float,
    names: dict,
    ipc_lock,
    stop_event,
) -> None:
    """Camera process entry point: SDK → shared memory. Never returns early on
    a transient frame error, only on stop or a failure to open."""
    # The parent owns Ctrl-C; a SIGINT here would race the orderly shutdown.
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    signal.signal(signal.SIGTERM, signal.SIG_DFL)

    log = get_logger(f"tof_camera_{side}")
    import ArducamDepthCamera as ac

    blocks, meta, depth_view, conf_view, preview_view = _attach(names)

    cam = ac.ArducamCamera()
    opened = False
    try:
        if cam.open(ac.Connection.CSI, camera_index) != 0:
            log.error(f"Failed to open Arducam CSI index {camera_index}")
            return
        opened = True

        cam.start(ac.FrameType.DEPTH)
        cam.setControl(ac.Control.RANGE, max_distance_mm)
        depth_range = cam.getControl(ac.Control.RANGE)
        meta[_M_DEPTH_RANGE] = float(depth_range)
        meta[_M_OPENED] = 1.0

        frame_count = 0
        preview_count = 0
        last_preview = 0.0

        while not stop_event.is_set():
            try:
                frame = cam.requestFrame(1000)
                if frame is None or not isinstance(frame, ac.DepthData):
                    continue
                if stop_event.is_set():
                    cam.releaseFrame(frame)
                    break

                # Deep-copy before releaseFrame. This is the critical guard
                # against SDK-owned shared buffers bleeding between frames.
                depth = np.array(frame.depth_data, copy=True).astype(np.float32)
                confidence = np.array(frame.confidence_data, copy=True)
                cam.releaseFrame(frame)

                depth, confidence = orient_camera_frame(depth, confidence, vertical_flip)
                now = time.monotonic()  # CLOCK_MONOTONIC is system-wide: the
                # parent compares this against its own clock for staleness.
                frame_count += 1

                jpeg = b""
                if now - last_preview >= preview_period:
                    jpeg = depth_to_jpeg(depth, confidence, depth_range)
                    last_preview = now

                with ipc_lock:
                    depth_view[:] = depth
                    conf_view[:] = confidence
                    meta[_M_FRAME_TIME] = now
                    meta[_M_FRAME_COUNT] = float(frame_count)
                    if jpeg:
                        n = min(len(jpeg), PREVIEW_MAX_BYTES)
                        preview_view[:n] = np.frombuffer(jpeg[:n], dtype=np.uint8)
                        preview_count += 1
                        meta[_M_PREVIEW_LEN] = float(n)
                        meta[_M_PREVIEW_COUNT] = float(preview_count)
            except Exception as exc:
                if stop_event.is_set():
                    break
                log.error(f"Camera {camera_index} loop error: {exc}")
                stop_event.wait(1.0)
    finally:
        if opened:
            for method_name in ("stop", "close"):
                method = getattr(cam, method_name, None)
                if callable(method):
                    try:
                        method()
                    except Exception:
                        pass
        for block in blocks.values():
            try:
                block.close()
            except Exception:
                pass


class TofCamera(Module):
    """Streams one Arducam ToF camera from its own process.

    `run()` is the parent-side poll thread; the SDK blocking happens in the
    child, where it cannot stall the policy loop.
    """

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

        # "fork" keeps the child from re-importing __main__, which would re-run
        # any unguarded caller (the test scripts). Cameras are the first modules
        # Robot starts, so the fork happens before the rest of the stack exists.
        self._ctx = mp.get_context("fork")
        self._proc = None
        self._ipc_lock = self._ctx.Lock()
        self._child_stop = self._ctx.Event()
        self._blocks: dict = {}
        self._meta = None
        self._depth_view = None
        self._conf_view = None
        self._preview_view = None
        self._seen_frame = -1
        self._seen_preview = -1
        self._death_logged = False

    # --- lifecycle ---

    def setup(self) -> None:
        self.log.info(
            f"Starting {self.side} ToF camera: CSI index={self.camera_index}, "
            f"vertical_flip={self.vertical_flip}"
        )

        sizes = {
            "meta": _M_SLOTS * 8,
            "depth": DEPTH_RAW_H * DEPTH_RAW_W * 4,
            "conf": DEPTH_RAW_H * DEPTH_RAW_W,
            "preview": PREVIEW_MAX_BYTES,
        }
        self._blocks = {
            key: shared_memory.SharedMemory(create=True, size=size) for key, size in sizes.items()
        }
        names = {key: block.name for key, block in self._blocks.items()}

        self._meta = np.ndarray(_M_SLOTS, dtype=np.float64, buffer=self._blocks["meta"].buf)
        self._meta[:] = 0.0
        self._depth_view = np.ndarray(
            (DEPTH_RAW_H, DEPTH_RAW_W), dtype=np.float32, buffer=self._blocks["depth"].buf
        )
        self._conf_view = np.ndarray(
            (DEPTH_RAW_H, DEPTH_RAW_W), dtype=np.uint8, buffer=self._blocks["conf"].buf
        )
        self._preview_view = np.ndarray(
            PREVIEW_MAX_BYTES, dtype=np.uint8, buffer=self._blocks["preview"].buf
        )

        self._child_stop.clear()
        self._proc = self._ctx.Process(
            target=_camera_worker,
            name=f"tof_camera_{self.side}",
            args=(
                self.side,
                self.camera_index,
                self.vertical_flip,
                self.max_distance_mm,
                self.preview_period,
                names,
                self._ipc_lock,
                self._child_stop,
            ),
            daemon=True,
        )
        self._proc.start()

    def snapshot(self) -> Tuple[np.ndarray | None, np.ndarray | None, float]:
        """Newest (depth, confidence, monotonic timestamp). Arrays are read-only."""
        return self.snapshot_with_id()[:3]

    def snapshot_with_id(self) -> Tuple[np.ndarray | None, np.ndarray | None, float, int]:
        """`snapshot()` plus the frame's sequence number, read under the same
        lock so a log can name exactly which frame an observation came from."""
        with self._lock:
            return self._depth, self._confidence, self._frame_time, self.frame_count

    def run(self) -> None:
        period = 1.0 / CAMERA_POLL_HZ
        while not self.stopping:
            try:
                self._pull()
            except Exception as exc:
                self.log.error(f"Camera {self.camera_index} poll error: {exc}")
                self.sleep(1.0)
                continue
            if not self._death_logged and self._proc is not None and not self._proc.is_alive():
                self.log.error(f"Camera process for {self.side} exited (index {self.camera_index})")
                self._death_logged = True
            self.sleep(period)

    def _pull(self) -> None:
        """Copy a new frame and/or preview out of shared memory, if there is one."""
        frame_count = int(self._meta[_M_FRAME_COUNT])
        preview_count = int(self._meta[_M_PREVIEW_COUNT])
        if frame_count == self._seen_frame and preview_count == self._seen_preview:
            return

        depth = confidence = jpeg = None
        frame_time = 0.0
        with self._ipc_lock:
            frame_count = int(self._meta[_M_FRAME_COUNT])
            if frame_count != self._seen_frame:
                depth = self._depth_view.copy()
                confidence = self._conf_view.copy()
                frame_time = float(self._meta[_M_FRAME_TIME])
            preview_count = int(self._meta[_M_PREVIEW_COUNT])
            if preview_count != self._seen_preview:
                jpeg = bytes(self._preview_view[: int(self._meta[_M_PREVIEW_LEN])])

        with self._lock:
            if depth is not None:
                self._depth = depth
                self._confidence = confidence
                self._frame_time = frame_time
                self.frame_count = frame_count
                self._seen_frame = frame_count
            if jpeg is not None:
                self.preview_jpeg = jpeg
                self._seen_preview = preview_count

    def shutdown(self) -> None:
        self._child_stop.set()
        proc, self._proc = self._proc, None
        if proc is not None:
            proc.join(timeout=2.0)
            if proc.is_alive():
                self.log.warning(f"Camera process for {self.side} did not exit; terminating")
                proc.terminate()
                proc.join(timeout=1.0)

        # Drop the numpy views before closing the buffers they borrow from.
        self._meta = self._depth_view = self._conf_view = self._preview_view = None
        for block in self._blocks.values():
            try:
                block.close()
                block.unlink()
            except Exception:
                pass
        self._blocks = {}
