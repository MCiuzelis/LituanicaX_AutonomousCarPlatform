"""The Robot: builds, wires, starts and stops every module.

This is the replacement for `ros_autopilot.launch.py`.  `RobotSettings` says
which modules run and with which parameters — flip a flag to disable a module
exactly like dropping a node from the launch file.

Wiring (who reads whom):

    TofCamera(left), TofCamera(right)  ─▶ PolicyRunner ─▶ Drive ─▶ motor + servo
    Encoder ─▶ PolicyRunner (wheel velocity observation)
    Encoder ─▶ Monitoring ─▶ Drive (safety gate: voltage, current, freshness)
    Dashboard ─▶ Robot.set_autopilot_enabled() ─▶ Drive + PolicyRunner
"""

from __future__ import annotations

import signal
import threading
from dataclasses import dataclass
from typing import Dict, List

from .config import DASHBOARD_HZ, ENCODER_HZ, POLICY_HZ, TELEMETRY_HZ
from .logging_utils import get_logger
from .module import Module
from .modules.acceleration_test import AccelerationTest
from .modules.dashboard import Dashboard
from .modules.drive import Drive
from .modules.encoder import Encoder
from .modules.monitoring import Monitoring
from .modules.policy_runner import PolicyRunner
from .modules.recorder import DEFAULT_MAX_MB, Recorder
from .modules.screenshot_grabber import ScreenshotGrabber
from .modules.skid_pad_test import SkidPadTest
from .modules.tof_camera import TofCamera

#: Everything that can be switched on or off, in start-up order.
MODULE_NAMES = (
    "tof_cameras",
    "recorder",
    "encoder",
    "monitoring",
    "drive",
    "policy",
    "dashboard",
    "screenshot_grabber",
    "acceleration_test",
    "skid_pad_test",
)


@dataclass
class RobotSettings:
    """Which modules run, and the parameters the launch file used to pass."""

    # --- module switches ---
    tof_cameras: bool = True
    recorder: bool = False
    encoder: bool = True
    monitoring: bool = True
    drive: bool = True
    policy: bool = True
    dashboard: bool = True
    screenshot_grabber: bool = False
    acceleration_test: bool = False
    skid_pad_test: bool = False

    # --- camera ---
    camera_sides: tuple = ("left", "right")
    preview_hz: float = DASHBOARD_HZ

    # --- policy ---
    policy_path: str = ""
    policy_hz: float = POLICY_HZ
    camera_timeout_sec: float = 0.5
    onnx_threads: int = 1

    # --- drive ---
    command_timeout_sec: float = 0.25
    monitoring_timeout_sec: float = 0.5
    telemetry_hz: float = TELEMETRY_HZ

    # --- encoder ---
    encoder_hz: float = ENCODER_HZ

    # --- dashboard ---
    host: str = "0.0.0.0"
    port: int = 8080
    stream_fps: float = 15.0

    # --- recorder (--log) ---
    log_dir: str = ""
    log_max_mb: float = DEFAULT_MAX_MB

    def enabled_modules(self) -> List[str]:
        return [name for name in MODULE_NAMES if getattr(self, name)]


class Robot:
    def __init__(self, settings: RobotSettings | None = None) -> None:
        self.settings = settings or RobotSettings()
        self.log = get_logger("robot")

        self.autopilot_enabled = False
        self.cameras: Dict[str, TofCamera] = {}
        self.recorder: Recorder | None = None
        self.encoder: Encoder | None = None
        self.monitoring: Monitoring | None = None
        self.drive: Drive | None = None
        self.policy: PolicyRunner | None = None
        self.dashboard: Dashboard | None = None
        self.screenshot_grabber: ScreenshotGrabber | None = None
        self.acceleration_test: AccelerationTest | None = None
        self.skid_pad_test: SkidPadTest | None = None

        self.modules: List[Module] = []
        self._started: List[Module] = []
        self._shutdown_event = threading.Event()
        #: Raised by whichever ToF camera has just published a depth frame, waited
        #: on by PolicyRunner so inference runs on frame arrival rather than on a
        #: timer that drifts into an arbitrary phase against the sensors.
        self._frame_signal = threading.Event()

        self._build()

    # --- construction ---

    def _build(self) -> None:
        settings = self.settings

        if settings.tof_cameras:
            for side in settings.camera_sides:
                self.cameras[side] = TofCamera(
                    side=side,
                    preview_hz=settings.preview_hz,
                    frame_event=self._frame_signal,
                )
            self.modules.extend(self.cameras.values())

        if settings.recorder:
            # Second in start order and therefore second-to-last to stop: it is
            # up before anything it logs, and drains after they have gone quiet.
            # It stays behind the cameras so their fork does not inherit the
            # writer thread's open blob files.
            self.recorder = Recorder(
                robot=self,
                log_dir=settings.log_dir,
                max_mb=settings.log_max_mb,
                rate_hz=settings.telemetry_hz,
            )
            self.modules.append(self.recorder)

        if settings.encoder:
            self.encoder = Encoder(rate_hz=settings.encoder_hz)
            self.modules.append(self.encoder)

        if settings.monitoring:
            self.monitoring = Monitoring(encoder=self.encoder, rate_hz=settings.telemetry_hz)
            self.modules.append(self.monitoring)

        if settings.drive:
            self.drive = Drive(
                monitoring=self.monitoring,
                command_timeout_sec=settings.command_timeout_sec,
                monitoring_timeout_sec=settings.monitoring_timeout_sec,
                rate_hz=settings.telemetry_hz,
            )
            self.modules.append(self.drive)

        if settings.policy:
            self.policy = PolicyRunner(
                left_camera=self.cameras.get("left"),
                right_camera=self.cameras.get("right"),
                drive=self.drive,
                policy_path=settings.policy_path,
                rate_hz=settings.policy_hz,
                camera_timeout_sec=settings.camera_timeout_sec,
                onnx_threads=settings.onnx_threads,
                recorder=self.recorder,
                # None when the cameras are off, which drops PolicyRunner back to
                # its plain fixed-rate loop.
                frame_signal=self._frame_signal if self.cameras else None,
            )
            self.modules.append(self.policy)

            if self.encoder is not None:
                # Filter wheel velocity on every encoder sample, not once per
                # inference step, so the policy sees the same signal it did
                # when it subscribed to the encoder topic.
                self.encoder.on_sample = self.policy.update_wheel_velocity

        if settings.dashboard:
            self.dashboard = Dashboard(
                robot=self,
                host=settings.host,
                port=settings.port,
                stream_fps=settings.stream_fps,
                sample_hz=settings.telemetry_hz,
            )
            self.modules.append(self.dashboard)

        if settings.screenshot_grabber:
            self.screenshot_grabber = ScreenshotGrabber(
                left_camera=self.cameras.get("left"),
                right_camera=self.cameras.get("right"),
            )
            self.modules.append(self.screenshot_grabber)

        if settings.acceleration_test:
            self.acceleration_test = AccelerationTest(
                robot=self, drive=self.drive, monitoring=self.monitoring
            )
            self.modules.append(self.acceleration_test)

        if settings.skid_pad_test:
            self.skid_pad_test = SkidPadTest(
                robot=self, drive=self.drive, monitoring=self.monitoring
            )
            self.modules.append(self.skid_pad_test)

        # Every one of these commands the drive directly and unconditionally, so
        # any two of them running together means whichever wrote last wins.
        drivers = [
            name
            for name in ("policy", "acceleration_test", "skid_pad_test")
            if getattr(settings, name)
        ]
        if len(drivers) > 1:
            self.log.warning(
                f"{' and '.join(drivers)} are all enabled — they will fight over the drive"
            )

    # --- autopilot gate (the old /autopilot/enabled topic) ---

    def set_autopilot_enabled(self, enabled: bool) -> None:
        enabled = bool(enabled)
        self.autopilot_enabled = enabled
        if self.drive is not None:
            self.drive.set_autopilot_enabled(enabled)
        if self.policy is not None:
            self.policy.set_autopilot_enabled(enabled)

    # --- lifecycle ---

    def start(self) -> None:
        self.log.info(f"Starting modules: {', '.join(m.name for m in self.modules)}")
        for module in self.modules:
            try:
                module.start()
            except Exception as exc:
                self.log.error(f"Module {module.name} failed to start: {exc}")
                self.stop()
                raise
            self._started.append(module)

    def stop(self) -> None:
        try:
            self.set_autopilot_enabled(False)
        except Exception as exc:
            self.log.error(f"Could not disable autopilot while stopping: {exc}")

        for module in reversed(self._started):
            try:
                module.stop()
            except Exception as exc:
                self.log.error(f"Module {module.name} failed to stop cleanly: {exc}")
        self._started.clear()
        self.log.info("All modules stopped")

    def shutdown(self) -> None:
        """Ask `spin()` to return; safe to call from a signal handler."""
        self._shutdown_event.set()

    def spin(self) -> None:
        """Block until SIGINT/SIGTERM, then stop every module."""
        for sig in (signal.SIGINT, signal.SIGTERM):
            signal.signal(sig, lambda *_: self.shutdown())

        try:
            while not self._shutdown_event.is_set():
                self._shutdown_event.wait(0.5)
        except KeyboardInterrupt:
            pass
        finally:
            self.stop()

    def run(self) -> None:
        self.start()
        self.spin()
