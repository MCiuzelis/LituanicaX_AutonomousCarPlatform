"""Arducam 12MP IMX708 (CSI CAM0) through picamera2.

Native sensor modes on this unit (10-bit RAW):
    1536x864  @ 120 fps  (2x2 binned, cropped)
    2304x1296 @  56 fps  (2x2 binned, full FoV)
    4608x2592 @  14 fps  (full resolution)

Any output resolution <= the chosen raw mode is produced by the ISP; frame rate is set via
FrameDurationLimits. All 92 fps x resolution combos in sweep_modes.py were verified OK.
"""
import time
from dataclasses import dataclass, asdict

from .base import Subsystem, HardwareError

FMT_BY_DEPTH = {8: "SRGGB8_CSI2P", 10: "SRGGB10_CSI2P", 12: "SRGGB12_CSI2P", 16: "SRGGB16_CSI2P"}


@dataclass(frozen=True)
class SensorMode:
    index: int
    width: int
    height: int
    bit_depth: int
    max_fps: float

    @property
    def size(self):
        return (self.width, self.height)

    @property
    def raw_format(self):
        return FMT_BY_DEPTH.get(self.bit_depth, "SRGGB10_CSI2P")

    def fits(self, width, height):
        return width <= self.width and height <= self.height


@dataclass
class CameraMode:
    """The currently configured output: what capture_frame() will deliver."""
    width: int
    height: int
    fps: float
    fps_requested: float
    sensor: SensorMode
    pixel_format: str = "RGB888"

    def as_dict(self):
        d = asdict(self)
        d["sensor"] = asdict(self.sensor)
        return d


class Camera(Subsystem):
    name = "camera"

    def __init__(self, camera_num=0, pixel_format="RGB888"):
        self.camera_num = camera_num
        self.pixel_format = pixel_format
        self._cam = None
        self._modes = None
        self._mode = None
        self._running = False

    # ---- lifecycle -------------------------------------------------------
    def available(self):
        try:
            from picamera2 import Picamera2
            return bool(Picamera2.global_camera_info())
        except Exception:
            return False

    def open(self):
        if self._cam is not None:
            return self
        try:
            from picamera2 import Picamera2
        except ImportError as e:
            raise HardwareError("picamera2 not installed: %s" % e)
        try:
            self._cam = Picamera2(self.camera_num)
        except Exception as e:
            raise HardwareError("failed to open camera %d: %s" % (self.camera_num, e))
        if not self._cam.global_camera_info():
            self.close()
            raise HardwareError("libcamera reports no cameras (check dtoverlay=imx708,cam0)")
        return self

    def close(self):
        if self._cam is not None:
            try:
                if self._running:
                    self._cam.stop()
            finally:
                self._cam.close()
                self._cam = None
                self._running = False
                self._mode = None

    @property
    def cam(self):
        """The underlying Picamera2 object (opens on demand)."""
        if self._cam is None:
            self.open()
        return self._cam

    @property
    def is_running(self):
        return self._running

    @property
    def mode(self):
        return self._mode

    # ---- capabilities ----------------------------------------------------
    def info(self):
        return self.cam.global_camera_info()[self.camera_num]

    def sensor_modes(self):
        """Native modes reported by the driver, deduplicated by size, smallest first."""
        if self._modes is None:
            seen = set()
            modes = []
            for i, m in enumerate(self.cam.sensor_modes):
                size = tuple(m["size"])
                if size in seen:
                    continue
                seen.add(size)
                modes.append(SensorMode(i, size[0], size[1], int(m["bit_depth"]), float(m["fps"])))
            self._modes = sorted(modes, key=lambda s: s.width * s.height)
        return self._modes

    def select_sensor_mode(self, width, height, fps):
        """Smallest sensor mode that can deliver width x height at fps.
        If no mode reaches fps at that size, the smallest fitting mode is returned and the
        caller clamps fps to its max."""
        fitting = [m for m in self.sensor_modes() if m.fits(width, height)]
        if not fitting:
            raise HardwareError("no sensor mode is >= %dx%d" % (width, height))
        for m in fitting:
            if m.max_fps >= fps - 0.01:
                return m
        return max(fitting, key=lambda m: m.max_fps)

    # ---- mode switching --------------------------------------------------
    def set_mode(self, width, height, fps=30.0, sensor_mode=None, settle_s=0.3, lores=None, **controls):
        """(Re)configure output resolution + frame rate and start streaming.

        Can be called while running: the pipeline is stopped, reconfigured and restarted.
        Returns the effective CameraMode (fps may be clamped to the sensor mode maximum).
        lores=(w, h, format) adds a second, ISP-scaled stream (e.g. (640, 360, "BGR888") for the
        detector); encoders keep using the main stream.
        Extra keyword args are passed as libcamera controls (e.g. ExposureTime=15000).
        """
        cam = self.cam
        if sensor_mode is None:
            sensor_mode = self.select_sensor_mode(width, height, fps)
        eff_fps = min(fps, sensor_mode.max_fps)
        dur_us = int(round(1e6 / eff_fps))
        if self._running:
            cam.stop()
            self._running = False
        cfg = cam.create_video_configuration(
            main={"size": (width, height), "format": self.pixel_format},
            lores={"size": lores[:2], "format": lores[2]} if lores else None,
            raw={"size": sensor_mode.size, "format": sensor_mode.raw_format},
        )
        try:
            cam.configure(cfg)
            ctl = {"FrameDurationLimits": (dur_us, dur_us)}
            ctl.update(controls)
            cam.set_controls(ctl)
            cam.start()
        except Exception as e:
            raise HardwareError("configure %dx%d@%.1f (raw %s) failed: %s"
                                % (width, height, fps, sensor_mode.size, str(e).splitlines()[0]))
        self._running = True
        self._mode = CameraMode(width, height, round(eff_fps, 2), fps, sensor_mode, self.pixel_format)
        if settle_s:
            time.sleep(settle_s)
        return self._mode

    def set_fps(self, fps, wait=True, timeout_s=1.0):
        """Change frame rate without reconfiguring (must stay <= current sensor mode max).
        With wait=True, blocks until the pipeline reports the new FrameDuration (controls are
        applied a few frames after being queued)."""
        if not self._running:
            raise HardwareError("camera not started; call set_mode() first")
        eff = min(fps, self._mode.sensor.max_fps)
        dur = int(round(1e6 / eff))
        self.cam.set_controls({"FrameDurationLimits": (dur, dur)})
        self._mode.fps, self._mode.fps_requested = round(eff, 2), fps
        if wait:
            deadline = time.monotonic() + timeout_s
            while time.monotonic() < deadline:
                got = self.cam.capture_metadata().get("FrameDuration", 0)
                if abs(got - dur) <= dur * 0.02:
                    break
        return self._mode

    def start(self, width=1920, height=1080, fps=30.0):
        return self.set_mode(width, height, fps)

    def stop(self):
        if self._cam is not None and self._running:
            self._cam.stop()
            self._running = False

    # ---- acquisition -----------------------------------------------------
    def capture_frame(self):
        """One frame as a numpy array (H x W x 3 for RGB888)."""
        if not self._running:
            self.start()
        return self.cam.capture_array("main")

    def capture_frames(self, n):
        return [self.capture_frame() for _ in range(n)]

    def capture_file(self, path):
        if not self._running:
            self.start()
        self.cam.capture_file(path)
        return path

    def metadata(self):
        return self.cam.capture_metadata()

    def frames(self, max_frames=None):
        """Generator of frames; stops after max_frames if given."""
        n = 0
        while max_frames is None or n < max_frames:
            yield self.capture_frame()
            n += 1

    def measure_fps(self, seconds=2.0, warmup=2):
        """Time the capture loop and return the achieved frame rate."""
        for _ in range(warmup):
            self.capture_frame()
        t0 = time.monotonic()
        n = 0
        while time.monotonic() - t0 < seconds:
            self.capture_frame()
            n += 1
        return round(n / (time.monotonic() - t0), 2)

    def status(self):
        st = {"available": self.available(), "open": self._cam is not None, "running": self._running}
        if self._cam is not None:
            st["model"] = self.info().get("Model")
            st["sensor_modes"] = ["%dx%d@%.1f" % (m.width, m.height, m.max_fps) for m in self.sensor_modes()]
            st["mode"] = self._mode.as_dict() if self._mode else None
        return st
