"""Every tunable number of the simulator, in one place.

Plain dataclasses (no Isaac imports) so the sensor models and their tests run without Isaac Sim.
Values marked MEASURE are placeholders that have to be checked on the real car; everything else
says where it came from. run.py can override any field: --set detector.fp_rate_per_frame=0.5
"""
from __future__ import annotations

import dataclasses
import math
import os
from dataclasses import dataclass, field

SIM_ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
ASSET_DIR = os.path.join(SIM_ROOT, "assets")
EXPORT_DIR = os.path.join(ASSET_DIR, "export")       # written by assets/blender/export_track.py
CAR_USD = os.path.join(ASSET_DIR, "car", "mushr_nano_v2.usd")
RUNS_DIR = os.path.join(SIM_ROOT, "runs")


@dataclass
class CameraCfg:
    """Arducam 12MP IMX708 HDR with the 120 deg wide-angle M12 lens, as the detector sees it.

    Lens: 152.2 (D) x 120 (H) x 66.7 (V) deg over the full sensor. H:V = 120:66.7 = 16:9, the
    sensor's own aspect, so the lens is modelled as equidistant (f-theta): angle proportional to
    distance from the image centre (a pinhole would give 120 x 88 deg).

    Sensor mode decides the field of view the detector gets:
      720p @ up to 120 fps   1536x864 mode, 2x2 binned AND cropped to the central 2/3 of the
                             sensor -> 120 * 2/3 = 80 deg H x 44.5 deg V.  What the dashboard and the
                             YOLO training use, and the only one that meets the >= 60 fps floor.
      1080p                  2304x1296 mode, full field of view (120 x 66.7 deg) but max 56 fps.
                             -> --set camera.hfov_deg=120
    The detector input is the 640x360 lores stream; boxes are normalised, so resolution only sets
    pixel quantisation.
    """
    width_px: int = 640
    height_px: int = 360
    hfov_deg: float = 80.0
    projection: str = "equidistant"
    # Isaac renders pinhole images, so the render is a pinhole view wide enough to cover the fisheye
    # frame, with this many times the detector frame's pixels per side. It only feeds the
    # visibility measurement (box geometry comes from the meshes), so 1.0 is enough; raise it for
    # sharper small-object occlusion at the cost of render time.
    render_scale: float = 1.0
    # MEASURE: mount pose. Car frame origin = rear-axle centre on the ground, x forward, z up.
    # Height ~2.5 x the 8.75 cm spheres. Pitch: YoloTraining dropped labels whose bottom edge is
    # above 34 % of the frame height, i.e. the horizon sits ~0.16 * 360 px = 58 px above the image
    # centre -> 58 / f(458 px) = 7.2 deg nose-down with this lens.
    x_m: float = 0.21
    y_m: float = 0.0
    z_m: float = 0.22
    pitch_deg: float = 7.0
    yaw_deg: float = 0.0
    exposure_s: float = 0.006   # sets the motion-blur model; <= 8.3 ms at 120 fps
    render_rgb: bool = True     # needed for the viewer only (costs a little render time)


@dataclass
class DetectorCfg:
    """Synthetic YOLO: turns renderer ground truth into boxes that behave like the Hailo detector's.

    Reference numbers (YoloTraining/README.md, held-out clip, HEF on the car): recall 0.981 and
    precision 0.974 at conf 0.35, ~25 apples per frame, 24.5 ms sensor->boxes p50.
    Pixel thresholds are in the 640x360 detector frame.
    """
    score_threshold: float = 0.35       # AppleDetector default
    recall_max: float = 0.98            # best-case per-frame detection probability
    # size: probability ramps 0 -> 1 as the box's SHORTER side grows min_side_px -> full_side_px
    min_side_px: float = 4.0
    full_side_px: float = 10.0
    # occlusion: visible pixels / expected pixels of the whole (in-frame) object
    vis_min: float = 0.25
    vis_full: float = 0.60
    # frame truncation: fraction of the full box inside the frame
    in_frame_min: float = 0.30
    in_frame_full: float = 0.70
    # box shape: mildly occluded objects get the full (amodal) box, heavily occluded ones only the
    # visible part - blended between these visibilities
    amodal_vis_lo: float = 0.40
    amodal_vis_hi: float = 0.85
    # jitter: AR(1) noise per object (correlated frame to frame, like a real detector's wobble)
    jitter_center_frac: float = 0.03    # of box width/height, 1 sigma
    jitter_center_px: float = 0.5
    jitter_scale_frac: float = 0.05     # log-normal size noise, 1 sigma
    jitter_tau_s: float = 0.08
    miss_tau_s: float = 0.10            # misses come in bursts: correlation time of the miss latent
    # motion blur from yaw rate: blur_px = |yaw rate| * exposure * fx
    blur_ok_px: float = 3.0
    blur_max_px: float = 25.0
    # the training labels had no boxes with the bottom edge above 34 % of the frame height
    upper_band_frac: float = 0.34
    upper_band_recall: float = 0.3
    nms_iou: float = 0.70               # Ultralytics / Hailo yolov8 default
    # false positives: Poisson per frame, each ghost lives ~geometric(fp_mean_life_frames)
    fp_rate_per_frame: float = 0.3
    fp_mean_life_frames: float = 3.0
    # delivery
    latency_mean_s: float = 0.0245      # measured p50 sensor timestamp -> boxes
    latency_std_s: float = 0.004
    latency_min_s: float = 0.012
    frame_drop_prob: float = 0.005
    max_range_m: float = 12.0           # objects further than this are never projected


@dataclass
class ImuCfg:
    """BNO055 / BNO08x class 9-axis IMU on the rear axle, fusion on-chip, no magnetometer used.

    Noise from the BNO055 datasheet (gyro 0.014 deg/s/rtHz, accel 150 ug/rtHz); biases are what is
    left AFTER the chip's own calibration. Game-rotation-vector yaw drift ~0.5 deg/min (BNO085 spec).
    Output is in the vehicle frame (FLU), i.e. after the axis remap; mount_error_deg is the residual
    misalignment that remap cannot fix.
    """
    rate_hz: float = 100.0
    x_m: float = 0.0          # rear axle (car frame origin)
    y_m: float = 0.0
    z_m: float = 0.05         # MEASURE
    gyro_noise_density_dps_rthz: float = 0.014
    gyro_bias_dps: float = 0.2            # 1 sigma, constant per run
    gyro_bias_walk_dps_rts: float = 0.004 # random walk
    accel_noise_density_ug_rthz: float = 150.0
    accel_bias_mg: float = 20.0           # 1 sigma, constant per run
    tilt_noise_deg: float = 0.2           # fusion roll/pitch error
    yaw_drift_deg_per_min: float = 0.5    # game rotation vector heading drift (random sign)
    mount_error_deg: float = 0.5          # 1 sigma per axis, constant per run


@dataclass
class EncoderCfg:
    """AS5600 on the motor shaft, sampled like hardware/drive.py: read_fast() at ~1.5 kHz,
    velocity from consecutive samples, median over the last 15."""
    motor_turns_per_wheel_turn: float = 10.0   # MEASURE: push the car 1 m, read Drive.turns
    counts: int = 4096
    sample_hz: float = 1500.0
    sample_jitter: float = 0.15                # fraction of the sample period (I2C timing)
    median_samples: int = 15
    noise_lsb: float = 0.5                     # 1 sigma angle noise in counts


@dataclass
class CarCfg:
    """Vehicle dynamics. Carried over from the calibrated RL simulator (Training/ in the old repo,
    real_car_calibration.md, 2026-08-03) and re-expressed per unit of motor duty instead of per
    unit of the old policy throttle (old throttle 1.0 == duty 0.40).

    The drive/brake numbers were fitted on the OLD motor driver. With the Hobbywing ESC the mapping
    throttle -> effective duty is new: esc_duty_per_throttle is the knob, fit it from encoder logs.
    """
    mass_kg: float = 1.27
    wheel_radius_m: float = 0.037
    robot_scale: float = 0.755          # mushr_nano_v2.usd spawned at this scale -> 74 mm wheels
    wheelbase_m: float = 0.187          # measured
    # Old sim: 26 deg (steering amplitude 350 us, protractor 28.5 deg). steering.json now has 270 us,
    # which the old protractor table (250 us 18.4 deg, 300 us 23.5 deg) puts at ~20 deg. MEASURE.
    max_steer_deg: float = 28.0
    understeer_deg_per_g: float = 6.29  # skid pad
    understeer_max_reduction: float = 0.75
    servo_rate_rad_s: float = 6.51      # spec-sheet guess
    esc_duty_per_throttle: float = 1.0  # MEASURE (Hobbywing throttle -> effective duty)
    esc_deadband: float = 0.0
    stall_torque_full_duty_nm: float = 0.0955 / 0.40   # per wheel
    noload_speed_full_duty_m_s: float = 5.60 / 0.40
    rolling_resistance_m_s2: float = 1.20
    rolling_release_omega: float = 1.0
    brake_torque_base_nm: float = 0.028
    brake_torque_full_nm: float = 0.310 * 0.4          # graded part at full brake (the slip ceiling)
    brake_release_omega: float = 2.0
    ground_static_friction: float = 0.85
    ground_dynamic_friction: float = 0.75
    suspension_stiffness: float = 1e6
    steer_delay_s: float = 0.03         # servo PWM frame + mechanics; camera latency is separate
    throttle_delay_s: float = 0.03
    max_throttle: float = 1.0           # like Drive(max_throttle=...); the dashboard uses 0.2


@dataclass
class EpisodeCfg:
    laps: int = 2
    max_time_s: float = 120.0
    spawn_x: float = 0.0                # world position; (0, 0) is the middle of a straight.
    spawn_y: float = 0.0                # heading = centre-line direction at the nearest point
    reverse: bool = False               # drive the track the other way round
    spawn_lateral_jitter_m: float = 0.0
    spawn_yaw_jitter_deg: float = 0.0
    wall_force_thresh_n: float = 1.0
    wrong_way_m: float = 1.5            # end the run if progress falls this far behind its best
    stuck_time_s: float = 5.0           # ... or if it makes no progress for this long


@dataclass
class SimCfg:
    # One car: CPU PhysX is faster than GPU PhysX (no GPU->CPU sync on every small tensor read).
    device: str = "cpu"
    # RTX preset, FXAA instead of DLSS, no GI/reflections/AO. The RGB image is only for the viewer; boxes come
    # from instance segmentation, which these settings do not affect.
    render_mode: str = "performance"
    physics_hz: float = 240.0
    camera_hz: float = 60.0             # the real camera + Hailo run at 60+ fps (up to 120)
    control_hz: float = 60.0
    seed: int = 0
    camera: CameraCfg = field(default_factory=CameraCfg)
    detector: DetectorCfg = field(default_factory=DetectorCfg)
    imu: ImuCfg = field(default_factory=ImuCfg)
    encoder: EncoderCfg = field(default_factory=EncoderCfg)
    car: CarCfg = field(default_factory=CarCfg)
    episode: EpisodeCfg = field(default_factory=EpisodeCfg)

    @property
    def max_steer_rad(self) -> float:
        return math.radians(self.car.max_steer_deg)

    def set(self, dotted: str, value: str) -> None:
        """Override one field from the command line: set('detector.nms_iou', '0.5')."""
        obj, *path, last = [self] + dotted.split(".")
        for p in path:
            obj = getattr(obj, p)
        if not hasattr(obj, last):
            raise KeyError(f"no config field '{dotted}'")
        cur = getattr(obj, last)
        if isinstance(cur, bool):
            val = value.lower() in ("1", "true", "yes", "on")
        else:
            val = type(cur)(value)
        setattr(obj, last, val)

    def as_dict(self) -> dict:
        return dataclasses.asdict(self)
