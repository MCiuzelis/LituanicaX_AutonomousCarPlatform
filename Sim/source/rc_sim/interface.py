"""The contract between the car (real or simulated) and a driving algorithm.

Nothing here imports Isaac Sim or torch, so an algorithm written against these types runs
unchanged on the Pi 5. Units and sign conventions follow the car code in Pi5/:

    Detection        exactly AppleDetector.detect()'s tuple: x1, y1, x2, y2 normalised 0..1 to the
                     camera frame (origin top-left), score, class_id.
    Command          the dashboard link: throttle -1..1 (negative = brake, the ESC is forward/brake
                     only), steer -1 (full left) .. +1 (full right), as Drive.steer() takes it.
    EncoderReading   Drive.rps / Drive.turns: MOTOR shaft revolutions (AS5600 on the motor, not the
                     wheels). CarInfo.wheel_m_per_motor_turn converts to metres.
    ImuSample        vehicle body frame, x forward / y left / z up (FLU), SI units.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field


@dataclass(frozen=True)
class Detection:
    x1: float
    y1: float
    x2: float
    y2: float
    score: float
    class_id: int = 0

    @property
    def cx(self) -> float:
        return 0.5 * (self.x1 + self.x2)

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1

    def as_tuple(self):
        return (self.x1, self.y1, self.x2, self.y2, self.score, self.class_id)


@dataclass
class DetectionFrame:
    """Boxes from one camera frame. t_capture is the sensor timestamp, t_available when the boxes
    reached the control loop (t_capture + readout/ISP + Hailo + decode)."""
    t_capture: float
    t_available: float
    detections: list[Detection]


@dataclass(frozen=True)
class ImuSample:
    t: float
    gyro: tuple[float, float, float]         # rad/s, calibrated (bias remains)
    accel: tuple[float, float, float]        # m/s^2 including gravity (reads +9.81 on z at rest)
    lin_accel: tuple[float, float, float]    # m/s^2 gravity removed by the fusion (uses its own tilt)
    quat: tuple[float, float, float, float]  # w, x, y, z game rotation vector (no magnetometer: yaw drifts)

    @property
    def yaw(self) -> float:
        w, x, y, z = self.quat
        return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


@dataclass(frozen=True)
class EncoderReading:
    t: float
    rps: float     # motor revolutions per second, median-filtered like Drive.rps (+ = forward)
    turns: float   # cumulative motor revolutions since reset, like Drive.turns


@dataclass
class SensorFrame:
    """Everything that became available to the control loop since the previous call."""
    t: float
    detections: list[DetectionFrame] = field(default_factory=list)   # oldest first
    imu: list[ImuSample] = field(default_factory=list)               # oldest first
    encoder: EncoderReading | None = None

    @property
    def latest_detections(self) -> DetectionFrame | None:
        return self.detections[-1] if self.detections else None


@dataclass(frozen=True)
class Command:
    throttle: float = 0.0   # -1 (full brake) .. 0 (coast) .. +1 (full forward)
    steer: float = 0.0      # -1 full left .. +1 full right

    def clamped(self, max_throttle: float = 1.0) -> "Command":
        return Command(max(-1.0, min(max_throttle, float(self.throttle))), max(-1.0, min(1.0, float(self.steer))))


@dataclass(frozen=True)
class CameraInfo:
    """Lens model of the frame the detector sees (boxes are normalised to it). See
    algorithms/geometry.py CameraModel for the projection itself."""
    width_px: int
    height_px: int
    hfov_deg: float
    x_m: float        # optical centre, car frame (origin: rear axle centre on the ground)
    y_m: float
    z_m: float        # height above the ground
    pitch_deg: float  # + = nose down
    yaw_deg: float = 0.0
    projection: str = "equidistant"    # "equidistant" (f-theta, wide-angle M12 lens) or "pinhole"

    @property
    def fx(self) -> float:
        """Focal length in pixels: r = f * tan(theta) (pinhole) or r = f * theta (equidistant)."""
        half = math.radians(self.hfov_deg) / 2.0
        return 0.5 * self.width_px / (half if self.projection == "equidistant" else math.tan(half))

    @property
    def vfov_deg(self) -> float:
        h = 0.5 * self.height_px / self.fx
        return 2.0 * math.degrees(h if self.projection == "equidistant" else math.atan(h))


@dataclass(frozen=True)
class CarInfo:
    """Constants an algorithm is allowed to know - on the real car they come from calibration."""
    wheelbase_m: float
    max_steer_rad: float          # road-wheel angle at |steer| = 1 (nominal, unloaded)
    wheel_radius_m: float
    motor_turns_per_wheel_turn: float
    camera: CameraInfo
    imu_x_m: float                # IMU position, car frame (origin: rear axle centre on the ground)
    imu_z_m: float
    control_hz: float
    max_throttle: float

    @property
    def wheel_m_per_motor_turn(self) -> float:
        return 2.0 * math.pi * self.wheel_radius_m / self.motor_turns_per_wheel_turn
