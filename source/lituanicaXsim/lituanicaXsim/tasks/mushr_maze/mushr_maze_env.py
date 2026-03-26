"""MuSHR nano v2 RC car — cone-track navigation environment (DirectRLEnv).

Observation space:
    [0:N]   flattened depth image from ArduCam TOF camera, after top/bottom crop,
            with Gaussian noise added to simulate real-world 2 cm depth accuracy.
            Normalized: 1.0 = object at 0 m (danger), 0.0 = object at depth_max_m (clear).
    [N:N+L] 360° LiDAR scan (L=450 rays, 0.8° resolution, 12 m max).
            Normalized: 0.0 = wall at sensor, 1.0 = clear (≥12 m).  ±30 mm noise applied.
    [N+L]   normalized forward linear velocity  (v / vel_cap) — vel_cap grows with curriculum
    [N+L+1] normalized body-frame yaw rate  (ang_vel_z / imu_ang_vel_norm)
    [N+L+2] previous throttle command ∈ [-1, 1]   (only if include_prev_action=True)
    [N+L+3] previous steering command ∈ [-1, 1]   (only if include_prev_action=True)

Camera profile (ArduCam TOF sensor for Raspberry Pi):
    - Sensor: Time-of-Flight depth, 240×180 pixels at 30 fps
    - Diagonal FOV: 70°  →  HFOV ≈ 58.5°, VFOV ≈ 45.6°  (4:3 aspect, pinhole)
    - Depth accuracy: ±2 cm (simulated via Gaussian noise σ ≈ 0.007 m)
    - Policy runs at 30 Hz, synced with camera (sim 120 Hz, decimation=4)

Action space (2):
    [0]  throttle command in [-1, 1]  → remapped to [0, 1] inside the env:
         +1 → full acceleration at max_accel_m_s2
          0 → neutral (no velocity change)
         -1 → full deceleration at max_decel_m_s2
         The env integrates a commanded velocity state clamped to [0, vel_cap].
    [1]  steering command in [-1, 1]  → scaled to [-max_steer_rad, +max_steer_rad]

Actuation (Ackermann / AWD):
    - All 4 wheels receive a velocity target (rad/s) = commanded_vel / WHEEL_RADIUS
    - Front wheels receive a position target (rad) = tan(steering_angle)
      (tan-convention matches the joint representation in mushr_nano_v2.usd)

Why depth camera vs RGB cone mask:
    The depth camera gives the policy direct metric information about wall and obstacle
    proximity.  A shallow corner looks identical in RGB (just more cone pixels) but
    in depth the wall is visibly close.  This makes braking-before-corners learnable:
    the agent can correlate "high depth values filling the frame → wall approaching →
    reduce throttle" without needing to understand color or stripe classification.
    The previous throttle command in the observation provides temporal credit for
    learning to anticipate braking rather than reacting after a collision.

Termination conditions:
    - Wall contact: any horizontal normal force on robot links above wall_contact_force_thresh_n
    - Robot tilt > 72° from vertical (up_z < 0.3)
    - Slow driving: speed < 2% of vel_cap for > ~1.5 s after episode start

TUNING NOTES (first run):
    - All adjustable parameters are in MushrMazeEnvCfg in one place below.
    - Spawn position: set use_preset_spawns + preset_spawn_points in cfg.
    - Camera crop: depth_crop_top_fraction, depth_crop_bottom_fraction.
    - Speed curriculum: initial_vel_cap_m_s, vel_cap_max_m_s, vel_cap_step_m_s.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Sequence
import csv
import math
import os

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensor, ContactSensorCfg
from isaaclab.sensors import MultiMeshRayCaster, MultiMeshRayCasterCfg, RayCasterCfg
from isaaclab.sensors.ray_caster import patterns as ray_patterns
try:
    from isaaclab.sensors import TiledCamera, TiledCameraCfg
except Exception:  # pragma: no cover
    from isaaclab.sensors import Camera as TiledCamera
    from isaaclab.sensors import CameraCfg as TiledCameraCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import quat_from_euler_xyz


# ---------------------------------------------------------------------------
# Asset USD paths  (relative to this file → project root → assets/)
# ---------------------------------------------------------------------------
_ASSETS_DIR = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "../../../../../assets")
)

MUSHR_USD        = os.path.join(_ASSETS_DIR, "mushr_nano_v2.usd")
TRACK_USD        = os.path.join(_ASSETS_DIR, "CONES.usd")
WALL_USD         = os.path.join(_ASSETS_DIR, "WALLS.usd")
WALL_INITIAL_USD = os.path.join(_ASSETS_DIR, "WALLS_INITIAL.usd")
RAMP_USD         = os.path.join(_ASSETS_DIR, "RAMPS.usd")

# ---------------------------------------------------------------------------
# Physical constants — MuSHR nano v2 (fixed by the USD / hardware)
# These match mushr_nano_v2.usd geometry and are NOT user-adjustable without
# also modifying the robot USD.
# ---------------------------------------------------------------------------
WHEEL_RADIUS = 0.05    # m
BASE_LENGTH  = 0.325   # m  (front–rear axle distance)
BASE_WIDTH   = 0.2     # m  (left–right wheel distance)

# ---------------------------------------------------------------------------
# ArduCam TOF camera optics  (derived from 70° diagonal FOV, 240×180 sensor)
#   f (pixels)        = diag_pixels / 2 / tan(diag_half) = 150 / tan(35°) ≈ 214.2 px
#   HFOV              = 2 * arctan(120 / 214.2) ≈ 58.52°
#   VFOV              = 2 * arctan(90  / 214.2) ≈ 45.57°
#
# For Isaac PinholeCameraCfg we pick an arbitrary focal_length_mm and derive
# the aperture so the ratio h_aperture / focal_length = 2 * tan(HFOV/2).
# ---------------------------------------------------------------------------
_TOF_DIAG_FOV_DEG  = 70.0
_TOF_DIAG_PIXELS   = math.sqrt(240**2 + 180**2)    # 300 px
_TOF_FOCAL_PX      = _TOF_DIAG_PIXELS / 2.0 / math.tan(math.radians(_TOF_DIAG_FOV_DEG / 2.0))
_TOF_HFOV_DEG      = 2.0 * math.degrees(math.atan2(120.0, _TOF_FOCAL_PX))   # ≈ 58.52°
_TOF_VFOV_DEG      = 2.0 * math.degrees(math.atan2(90.0,  _TOF_FOCAL_PX))   # ≈ 45.57°
_TOF_FOCAL_MM      = 4.0   # reference focal length (mm); aperture ratio is what matters
_TOF_H_APERTURE_MM = 2.0 * _TOF_FOCAL_MM * math.tan(math.radians(_TOF_HFOV_DEG / 2.0))
_TOF_V_APERTURE_MM = 2.0 * _TOF_FOCAL_MM * math.tan(math.radians(_TOF_VFOV_DEG / 2.0))

# ---------------------------------------------------------------------------
# Sector gate system — centerline exported from Blender/curve_points.csv
# ---------------------------------------------------------------------------
SECTOR_CSV_TO_ISAAC_SCALE    = 0.3
SECTOR_CSV_TO_ISAAC_X_OFFSET = 16.0

# Robot links monitored for wall contact.
WALL_CONTACT_LINKS = (
    "base_link",
    "front_left_wheel_link",
    "front_right_wheel_link",
    "back_left_wheel_link",
    "back_right_wheel_link",
)


# ===========================================================================
# Environment Configuration  ←  ALL TUNABLE PARAMETERS LIVE HERE
# ===========================================================================

@configclass
class MushrMazeEnvCfg(DirectRLEnvCfg):
    """Single source of truth for every number you might want to change.

    Sections
    --------
    CAMERA          — sensor resolution, FOV, depth range, crop, noise
    SPAWN           — starting position, heading, and speed
    VELOCITY        — speed curriculum settings
    DYNAMICS        — acceleration limits, steering limit, IMU normalization
    TERMINATION     — contact-force threshold, slow-speed detection
    OBSERVATION     — whether to include previous action in obs
    REWARD WEIGHTS  — all reward scale factors
    """

    # ── Core env parameters ────────────────────────────────────────────────
    decimation:        int   = 4      # policy at ~30 Hz  (sim 120 Hz / 4)
    episode_length_s:  float = 900.0  # 900 s × 30 Hz = 27 000 steps
    action_space:      int   = 2
    # observation_space is patched in MushrMazeEnv.__init__ after computing
    # the exact cropped depth image size.  The placeholder below keeps
    # @configclass happy; the real value is written before super().__init__().
    observation_space: int   = 1      # patched in __init__
    state_space:       int   = 0

    sim: SimulationCfg = SimulationCfg(dt=1.0 / 120.0, render_interval=4)

    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=12,
        env_spacing=0.0,
        replicate_physics=True,
    )

    # ════════════════════════════════════════════════════════════════════════
    # CAMERA — ArduCam TOF (240×180, 30 fps, 70° diagonal FOV)
    # ════════════════════════════════════════════════════════════════════════
    # Crop fractions (fraction of raw pixel rows discarded before the policy):
    #   top  → removes ceiling / sky (depth = max / inf)
    #   bottom → removes near-ground clutter and the robot's own body shadow
    depth_crop_top_fraction:    float = 0.4
    depth_crop_bottom_fraction: float = 0.0

    # Gaussian depth noise: σ ≈ 0.007 m gives a ±2 cm (3σ) error band,
    # matching the ArduCam TOF spec.  Set to 0.0 to disable.
    depth_noise_sigma_m: float = 0.007

    # Depth range for normalization.  Readings > depth_max_m are clamped.
    depth_max_m: float = 5.0

    # ── Policy image downsampling ──────────────────────────────────────────
    # After cropping the raw depth frame (240×≈99 at defaults) the image is
    # bilinearly downsampled to this resolution before being flattened into
    # the observation vector.  Smaller = faster MLP convergence.
    # policy_depth_height < 0 → auto-compute from cropped aspect ratio.
    # Default 80×33 = 2640 dims (vs 240×99 = 23 760 without downsampling).
    policy_depth_width:  int = 100
    policy_depth_height: int = -1   # -1 = auto

    # ════════════════════════════════════════════════════════════════════════
    # SPAWN — initial robot position, heading, and speed
    # ════════════════════════════════════════════════════════════════════════
    spawn_speed_m_s:   float = 2.5   # initial forward speed at episode start (m/s)
    # Grace period after each reset: wall / flip / stuck terminations are suppressed
    # for this many policy steps so that PhysX contact forces can settle.
    # 10 steps ≈ 333 ms at 30 Hz — enough for any spawn-physics transient.
    spawn_grace_steps: int   = 10

    # ── Spawn mode ────────────────────────────────────────────────────────────
    #   use_preset_spawns=False (default)
    #       Each reset picks a uniformly random sector gate from curve_points.csv,
    #       giving every part of the circuit equal training coverage.
    #
    #   use_preset_spawns=True
    #       Each reset picks randomly from preset_spawn_points.  Use this for
    #       targeted curriculum: list the hard corners / ramp approaches you
    #       want agents to practise most.  Each entry is (x_m, y_m, theta_deg).
    #       spawn_yaw_jitter_rad is still applied in both modes.
    #
    use_preset_spawns: bool = True
    preset_spawn_points: list = [
        # (x_m, y_m, theta_deg)  — replace with real track coordinates.
        (7.6168,  -1.31077, 116.0),   # main start/finish
        (15.15,     5.0,      -90.0),   # example: tight right-hander
        (10.295,     1.847,     18.0),   # example: ramp approach
        (12.542,    9.8085,     -2.0),
        (15.247,    -7.292,     -95),
        (12.191,    -7.7997,    111),
        (8.840,     -8.089,     -93),
        (5.540,     -9.1814,    88),
        (12.326,    -3.5450,    53),
        (10.799,    -0.5676,    -121),
        (6.0156,    8.81074,    28),
        (11.60,     2.4047,     38),
    ]
    spawn_yaw_jitter_rad: float = 0.1   # ± heading noise applied in both modes

    # ════════════════════════════════════════════════════════════════════════
    # VELOCITY CURRICULUM
    # vel_cap starts at initial_vel_cap_m_s and steps up each time any agent
    # completes a full lap, up to vel_cap_max_m_s.
    # ════════════════════════════════════════════════════════════════════════
    initial_vel_cap_m_s: float = 2.5   # starting speed ceiling (m/s)
    vel_cap_max_m_s:     float = 5.0   # absolute maximum speed (m/s)
    vel_cap_step_m_s:    float = 0.25  # speed ceiling increment per lap (m/s)

    # ════════════════════════════════════════════════════════════════════════
    # DYNAMICS
    # ════════════════════════════════════════════════════════════════════════
    # Acceleration limits (AWD, rubber-on-court friction):
    #   μ_s × g = 0.8 × 9.81 ≈ 7.85 → 7.8 m/s²
    #   μ_d × g = 0.7 × 9.81 ≈ 6.87 → 6.9 m/s²
    max_accel_m_s2:  float = 7.8    # max forward acceleration
    max_decel_m_s2:  float = 6.9    # max braking deceleration
    max_steer_rad:   float = 0.488  # ≈28° max front-wheel steer angle

    # IMU angular-velocity normalization ceiling for the yaw-rate observation.
    imu_ang_vel_norm: float = 10.0  # rad/s (generous ceiling for fast cornering)

    # Steering EMA low-pass filter coefficient (simulates servo bandwidth ≈100 ms).
    # alpha = 1 − exp(−dt/τ), dt ≈ 1/30 s → alpha=0.4 gives τ ≈ 76 ms.
    action_lpf_alpha_steer: float = 0.4

    # ════════════════════════════════════════════════════════════════════════
    # TERMINATION THRESHOLDS
    # ════════════════════════════════════════════════════════════════════════
    # Horizontal contact force on any robot link that counts as a wall hit.
    wall_contact_force_thresh_n: float = 15.0  # Newtons

    # Speed fraction thresholds (two separate knobs):
    #   slow_penalty_speed_fraction — slow-penalty fires every step below this
    #     (use 0.0 to disable; do NOT set high or you penalise deliberate braking)
    #   slow_stop_speed_fraction    — episode terminates when speed falls below this
    #     (catches truly stuck / reversed agents; keep low so normal braking is fine)
    slow_penalty_speed_fraction: float = 0.0   # disabled — braking is desired behaviour
    slow_stop_speed_fraction:    float = 0.03  # terminate if speed < 3 % of vel_cap

    # ════════════════════════════════════════════════════════════════════════
    # LIDAR (360° horizontal scan — real-world analogue: RPLIDAR A1/A2)
    # ════════════════════════════════════════════════════════════════════════
    # Simulates a single-plane 360° LiDAR with specs matching the user's sensor:
    #   Range        : 0.02 mm – 12 m       → max_distance = 12.0 m
    #   Scan freq    : 10 Hz                (≈ every 3 policy steps at 30 Hz)
    #   Ranging freq : 4 500 Hz             → 450 samples/rev → 0.8°/sample
    #   Accuracy     : ±30 mm (1-σ noise)   → lidar_noise_sigma_m = 0.03
    #
    # Rays cast against /World/Walls only (cones have no CollisionAPI).
    # The 450-dim normalized distance vector is appended to the policy obs.
    lidar_cfg: MultiMeshRayCasterCfg = MultiMeshRayCasterCfg(
        prim_path="/World/envs/env_.*/Robot/mushr_nano/base_link",
        # Cones (/World/Track) and ramps (/World/Ramps) — walls excluded.
        # RayCaster reads UsdGeom.Mesh visual geometry directly; CollisionAPI is
        # NOT required, so cones stay collision-free (no jiggling).
        # Walls are invisible to the policy — ContactSensor termination only.
        mesh_prim_paths=[
            MultiMeshRayCasterCfg.RaycastTargetCfg(
                prim_expr="/World/Track",
                is_shared=True,
                track_mesh_transforms=False,
            ),
            MultiMeshRayCasterCfg.RaycastTargetCfg(
                prim_expr="/World/Ramps",
                is_shared=True,
                track_mesh_transforms=False,
            ),
        ],
        pattern_cfg=ray_patterns.LidarPatternCfg(
            channels=1,
            vertical_fov_range=(0.0, 0.0),   # single horizontal plane
            horizontal_fov_range=(0.0, 360.0),
            horizontal_res=0.8,              # 360 / 0.8 = 450 rays
        ),
        max_distance=12.0,
        ray_alignment="base",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 0.05)),  # 5 cm above base_link
    )
    lidar_noise_sigma_m: float = 0.03   # ±30 mm sensor accuracy

    # ════════════════════════════════════════════════════════════════════════
    # OBSERVATION EXTRAS
    # ════════════════════════════════════════════════════════════════════════
    # Append the previous policy action (throttle + steer, each ∈ [-1,1]) to
    # the observation.  Strongly recommended: gives the policy temporal context
    # so it can learn to anticipate braking rather than react after a collision.
    include_prev_action: bool = True

    # ════════════════════════════════════════════════════════════════════════
    # REWARD WEIGHTS
    # ════════════════════════════════════════════════════════════════════════
    # Distance: rewards physical metres covered each step (v × dt).
    distance_weight:    float = 6.0

    # Speed bonus, normalised by initial_vel_cap_m_s (fixed scale across curriculum).
    forward_weight:     float = 0.1

    # Extra penalty for wall contact (on top of the base collision cost).
    wall_penalty:       float = 20.0

    # Sector timing improvement reward.
    # Fires every time an agent crosses a sector gate faster than the global reference.
    # reward = sector_gain_weight × (ref_time - sector_time) / ref_time
    # This is the primary incentive for racing lines and faster cornering.
    sector_gain_weight: float = 200.0

    # Rear-wheel slip penalty.
    slip_weight:        float = 1.0

    # Steering deadzone penalty: only penalises steering beyond the deadzone.
    steer_deadzone_weight: float = 0.05
    steer_deadzone:        float = 0.05   # fraction of max_steer_rad that is free

    # Dense per-step proximity reward toward the next sector gate.
    sector_proximity_weight:   float = 0
    sector_proximity_max_dist: float = 10.0   # metres — reward fades to 0 beyond this

    # Lap completion bonus (scaled by lap time relative to target_lap_time_s).
    lap_completion_reward:       float = 1000.0
    target_lap_time_s:           float = 28.0   # agents at ~36s; target ahead to keep gradient
    lap_time_reward_min_scale:   float = 0.25
    lap_time_reward_max_scale:   float = 8.0    # up from 4.0 — bigger spread for faster laps

    # Ramp speed cap — prevents flying off ramp peaks at full vel_cap speed.
    # When the robot base is elevated (on_ramp), _current_vel is clamped to this value.
    ramp_max_speed_m_s: float = 3.0

    # Ramp per-step reward (while any robot link touches the ramp).
    ramp_reward_weight: float = 1

    # Slow-driving penalty (per step below slow_speed_fraction × vel_cap).
    slow_penalty_weight: float = 1.0

    # ════════════════════════════════════════════════════════════════════════
    # FIXED → RANDOM SPAWN TRANSITION
    # ════════════════════════════════════════════════════════════════════════
    fixed_spawn_reward_threshold: float = 9_999_999  # effectively disabled (fixed spawn always)
    fixed_spawn_history_len:      int   = 50

    # ════════════════════════════════════════════════════════════════════════
    # HARDWARE CONFIG  (robot articulation + camera sensor)
    # ════════════════════════════════════════════════════════════════════════

    robot_cfg: ArticulationCfg = ArticulationCfg(
        prim_path="/World/envs/env_.*/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=MUSHR_USD,
            scale=(1.0, 1.0, 1.0),
            activate_contact_sensors=True,
            mass_props=sim_utils.MassPropertiesCfg(mass=2.0),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                rigid_body_enabled=True,
                max_linear_velocity=10000.0,
                max_angular_velocity=100000.0,
                max_depenetration_velocity=1.0,
                max_contact_impulse=5.0,
                enable_gyroscopic_forces=True,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False,
                solver_position_iteration_count=4,
                solver_velocity_iteration_count=0,
                sleep_threshold=0.005,
                stabilization_threshold=0.001,
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(7.6168, -1.31077, 0.002),   # placeholder — overridden every episode in _reset_idx
            joint_pos={
                "back_left_wheel_throttle":    0.0,
                "back_right_wheel_throttle":   0.0,
                "front_left_wheel_throttle":   0.0,
                "front_right_wheel_throttle":  0.0,
                "front_left_wheel_steer":      0.0,
                "front_right_wheel_steer":     0.0,
                "front_left_wheel_suspension": 0.0,
                "front_right_wheel_suspension":0.0,
                "back_left_wheel_suspension":  0.0,
                "back_right_wheel_suspension": 0.0,
            },
        ),
        actuators={
            "steering": ImplicitActuatorCfg(
                joint_names_expr=["front_left_wheel_steer", "front_right_wheel_steer"],
                stiffness=100.0, damping=10.0,
                velocity_limit_sim=10.0, effort_limit_sim=3.2,
            ),
            "rear_throttle": ImplicitActuatorCfg(
                joint_names_expr=["back_left_wheel_throttle", "back_right_wheel_throttle"],
                stiffness=0.0, damping=1000.0,
                velocity_limit_sim=450.0, effort_limit_sim=0.5,
            ),
            "front_throttle": ImplicitActuatorCfg(
                joint_names_expr=["front_left_wheel_throttle", "front_right_wheel_throttle"],
                stiffness=0.0, damping=1000.0,
                velocity_limit_sim=450.0, effort_limit_sim=0.5,
            ),
        },
    )

    # ArduCam TOF depth camera — 240×180, 30 fps, 70° diagonal FOV.
    # data_types=["distance_to_image_plane"]: perpendicular depth in metres,
    # matching what a real TOF sensor measures.
    camera_cfg: TiledCameraCfg = TiledCameraCfg(
        prim_path="/World/envs/env_.*/Robot/mushr_nano/base_link/front_camera",
        update_period=0.0,    # render once per policy step (render_interval=4 → 30 Hz)
        height=180,
        width=240,
        data_types=["distance_to_image_plane"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=_TOF_FOCAL_MM,
            horizontal_aperture=_TOF_H_APERTURE_MM,
            vertical_aperture=_TOF_V_APERTURE_MM,
            clipping_range=(0.05, 15.0),   # TOF range + generous clip for far walls
        ),
        offset=TiledCameraCfg.OffsetCfg(
            pos=(0.0, 0.0, 0.1),
            rot=(-0.5, 0.5, -0.5, 0.5),
            convention="ros",
        ),
        debug_vis=False,
    )

    # Live debug view in training — use scripts/rsl_rl/visualize.py instead.
    show_camera_debug:    bool = False
    camera_debug_env_id:  int  = 0
    camera_debug_interval:int  = 2


# ===========================================================================
# Sector gate geometry helper
# ===========================================================================

def _load_sector_gates(device: str):
    """Load Blender/curve_points.csv and return gate tensors for sector timing."""
    csv_path = os.path.join(
        os.path.dirname(__file__),
        "../../../../../Blender/curve_points.csv",
    )
    pts: list[tuple[float, float]] = []
    with open(csv_path, newline="") as f:
        for row in csv.reader(f):
            row = [r.strip() for r in row if r.strip()]
            if len(row) >= 2:
                bx, by = float(row[0]), float(row[1])
                x = by * -SECTOR_CSV_TO_ISAAC_SCALE + SECTOR_CSV_TO_ISAAC_X_OFFSET
                y = bx *  SECTOR_CSV_TO_ISAAC_SCALE
                pts.append((x, y))

    n = len(pts)
    assert n >= 3, f"curve_points.csv must have at least 3 points, got {n}"
    print(f"[Sector] Loaded {n} sector gates from curve_points.csv")

    tangents: list[tuple[float, float]] = []
    for i in range(n):
        px, py = pts[(i - 1) % n]
        nx, ny = pts[(i + 1) % n]
        dx, dy = nx - px, ny - py
        mag = math.hypot(dx, dy)
        tangents.append((dx / mag, dy / mag))

    gates_t    = torch.tensor(pts,      dtype=torch.float32, device=device)
    tangents_t = torch.tensor(tangents, dtype=torch.float32, device=device)
    return n, gates_t, tangents_t


# ===========================================================================
# Environment Class
# ===========================================================================

class MushrMazeEnv(DirectRLEnv):
    cfg: MushrMazeEnvCfg

    def __init__(self, cfg: MushrMazeEnvCfg, render_mode: str | None = None, **kwargs):
        # ContactSensor dict must exist before super().__init__() calls _setup_scene().
        self._wall_contact_sensors: dict[str, ContactSensor] = {}

        # Compute exact depth observation size.
        # 1. Crop: remove top/bottom rows from the raw 240×180 frame.
        # 2. Downsample: bilinear resize to policy_depth_width × policy_depth_height.
        #    Smaller resolution = smaller MLP first layer = faster convergence.
        cam_h = cfg.camera_cfg.height   # 180
        cam_w = cfg.camera_cfg.width    # 240
        remaining = max(0.0, 1.0 - cfg.depth_crop_top_fraction - cfg.depth_crop_bottom_fraction)
        cropped_h = max(1, int(cam_h * remaining))
        cropped_w = cam_w

        self._policy_depth_w = cfg.policy_depth_width
        if cfg.policy_depth_height < 0:
            # Auto-compute from cropped aspect ratio
            self._policy_depth_h = max(1, round(cfg.policy_depth_width * cropped_h / cropped_w))
        else:
            self._policy_depth_h = cfg.policy_depth_height

        depth_obs_size = self._policy_depth_h * self._policy_depth_w

        # LiDAR dims: channels × horizontal_rays
        # horizontal_res=0.8° over 360° (exclusive endpoint) → 450 rays × 1 channel
        import math as _math
        lidar_h_rays = _math.ceil(360.0 / cfg.lidar_cfg.pattern_cfg.horizontal_res)
        self._lidar_n_rays = lidar_h_rays * cfg.lidar_cfg.pattern_cfg.channels

        extra_obs = 2  # lin_vel + yaw_rate
        if cfg.include_prev_action:
            extra_obs += 2  # prev_throttle + prev_steer
        cfg.observation_space = depth_obs_size + self._lidar_n_rays + extra_obs

        print(
            f"[MushrMazeEnv] Depth obs: raw {cam_w}×{cam_h} "
            f"→ crop {cropped_w}×{cropped_h} "
            f"→ policy {self._policy_depth_w}×{self._policy_depth_h} "
            f"= {depth_obs_size} px  |  LiDAR: {self._lidar_n_rays} rays  |  "
            f"total obs = {cfg.observation_space}"
        )

        super().__init__(cfg, render_mode, **kwargs)

        # Joint indices
        self._rear_throttle_ids, _ = self.robot.find_joints([
            "back_left_wheel_throttle", "back_right_wheel_throttle",
        ])
        self._front_throttle_ids, _ = self.robot.find_joints([
            "front_left_wheel_throttle", "front_right_wheel_throttle",
        ])
        self._steer_ids, _ = self.robot.find_joints([
            "front_left_wheel_steer", "front_right_wheel_steer",
        ])

        # Ackermann command buffers
        self._rear_wheel_ang_vel = torch.zeros(self.num_envs, device=self.device)
        self._steer_tan          = torch.zeros(self.num_envs, device=self.device)

        # Velocity integration state
        self._current_vel = torch.zeros(self.num_envs, device=self.device)
        self._accel_m_s2  = torch.zeros(self.num_envs, device=self.device)

        # Steering LPF state
        self._filtered_steer = torch.zeros(self.num_envs, device=self.device)
        self._steer_integral  = torch.zeros(self.num_envs, device=self.device)
        self._prev_steer      = torch.zeros(self.num_envs, device=self.device)
        self._steer_hp        = torch.zeros(self.num_envs, device=self.device)

        # Previous action for observation (throttle + steer in [-1,1])
        self._prev_action = torch.zeros(self.num_envs, 2, device=self.device)

        # Curriculum
        self._vel_cap:              float = cfg.initial_vel_cap_m_s
        self._lap_at_current_vel:   bool  = False

        # Ramp / global state
        self._total_laps_completed: int  = 0
        self._ramp_enabled:         bool = True

        # Fixed→random spawn tracking
        self._random_spawn_unlocked: bool = True
        self._ep_return_buf = torch.zeros(self.num_envs, device=self.device)
        self._ep_return_history: deque = deque(maxlen=cfg.fixed_spawn_history_len)

        # Sector gate geometry
        self._n_sectors, self._sector_gates, self._sector_tangents = (
            _load_sector_gates(self.device)
        )
        print("[Sector] First 3 world gate positions (world XY):")
        for i in range(min(3, self._n_sectors)):
            x, y = self._sector_gates[i].tolist()
            print(f"  gate[{i}]: ({x:.4f}, {y:.4f})")

        # Sector timing
        self._sector_best_times = torch.full(
            (self._n_sectors,), float("inf"), dtype=torch.float32, device=self.device,
        )
        self._sector_reference_times = torch.full(
            (self._n_sectors,), float("inf"), dtype=torch.float32, device=self.device,
        )
        self._reference_established: bool = False
        self._sectors_this_ep = torch.zeros(
            self.num_envs, dtype=torch.int32, device=self.device,
        )
        self._lap_start_step = torch.zeros(
            self.num_envs, dtype=torch.int32, device=self.device,
        )
        self._agent_spawn_vel_cap = torch.full(
            (self.num_envs,), cfg.initial_vel_cap_m_s,
            dtype=torch.float32, device=self.device,
        )
        self._current_sector = torch.zeros(
            self.num_envs, dtype=torch.int64, device=self.device,
        )
        self._sector_entry_step = torch.zeros(
            self.num_envs, dtype=torch.int32, device=self.device,
        )
        self._sector_d_prev = torch.zeros(
            self.num_envs, dtype=torch.float32, device=self.device,
        )

        # Quick-start tracking
        self._spawn_pos = torch.zeros(self.num_envs, 2, device=self.device)

        # Contact / termination flags
        self._wall_terminated    = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._on_ramp            = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._wall_ever_touched  = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._ramp_ascended      = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._slow_timer         = torch.zeros(self.num_envs, dtype=torch.int32, device=self.device)

        # Depth observation cache (policy resolution after crop + downsample)
        self._camera_obs = torch.zeros(
            self.num_envs,
            self._policy_depth_h * self._policy_depth_w,
            device=self.device,
        )
        self._last_camera_update_step = -1

    # ------------------------------------------------------------------
    # Scene setup
    # ------------------------------------------------------------------

    def _setup_scene(self):
        self.robot = Articulation(self.cfg.robot_cfg)

        track_cfg = sim_utils.UsdFileCfg(usd_path=TRACK_USD, scale=(0.003, 0.003, 0.003))
        track_cfg.func("/World/Track", track_cfg, translation=(16, 0.0, 0.0), orientation=(0.0, 0.0, 0.0, 0.0))

        wall_cfg = sim_utils.UsdFileCfg(usd_path=WALL_USD, scale=(0.003, 0.003, 0.01))
        wall_cfg.func("/World/Walls", wall_cfg, translation=(16, 0.0, 0.0), orientation=(0.0, 0.0, 0.0, 0.0))

        try:
            import omni.usd
            from pxr import UsdPhysics, Usd
            stage = omni.usd.get_context().get_stage()
            for wall_path in ("/World/Walls",):
                wall_root = stage.GetPrimAtPath(wall_path)
                if wall_root.IsValid():
                    for prim in Usd.PrimRange(wall_root):
                        if prim.HasAPI(UsdPhysics.MeshCollisionAPI):
                            UsdPhysics.MeshCollisionAPI(prim).GetApproximationAttr().Set("convexDecomposition")
                        elif prim.HasAPI(UsdPhysics.CollisionAPI):
                            mesh_api = UsdPhysics.MeshCollisionAPI.Apply(prim)
                            mesh_api.GetApproximationAttr().Set("convexDecomposition")
        except Exception as e:
            print(f"[Wall] convexDecomposition setup failed: {e}")

        ramp_cfg = sim_utils.UsdFileCfg(usd_path=RAMP_USD, scale=(0.003, 0.003, 0.003))
        ramp_cfg.func("/World/Ramps", ramp_cfg, translation=(16, 0.0, 0.0), orientation=(0.0, 0.0, 0.0, 0.0))

        import omni.usd
        from pxr import Usd, UsdGeom, UsdPhysics, UsdShade
        stage = omni.usd.get_context().get_stage()

        wheel_link_names = {
            "front_left_wheel_link", "front_right_wheel_link",
            "back_left_wheel_link",  "back_right_wheel_link",
        }
        wheel_root = stage.GetPrimAtPath("/World/envs/env_0/Robot/mushr_nano")
        if wheel_root.IsValid():
            for prim in Usd.PrimRange(wheel_root):
                if not prim.IsA(UsdGeom.Mesh):
                    continue
                if prim.GetParent().GetName() not in wheel_link_names:
                    continue
                mesh_collision = UsdPhysics.MeshCollisionAPI.Apply(prim)
                mesh_collision.GetApproximationAttr().Set("convexHull")

        for wall_path, collision_enabled in (("/World/Walls", True),):
            wall_root = stage.GetPrimAtPath(wall_path)
            if not wall_root.IsValid():
                continue
            for prim in Usd.PrimRange(wall_root):
                if not prim.IsA(UsdGeom.Mesh):
                    continue
                wall_collision = UsdPhysics.CollisionAPI.Apply(prim)
                wall_collision.GetCollisionEnabledAttr().Set(collision_enabled)
                wall_mesh_collision = UsdPhysics.MeshCollisionAPI.Apply(prim)
                wall_mesh_collision.GetApproximationAttr().Set("none")
            wall_rb = UsdPhysics.RigidBodyAPI.Apply(wall_root)
            wall_rb.GetRigidBodyEnabledAttr().Set(True)
            wall_rb.GetKinematicEnabledAttr().Set(True)
            UsdGeom.Imageable(wall_root).MakeInvisible()

        ramp_root = stage.GetPrimAtPath("/World/Ramps")
        if ramp_root.IsValid():
            UsdGeom.Imageable(ramp_root).MakeVisible()
            for prim in Usd.PrimRange(ramp_root):
                if not prim.IsA(UsdGeom.Mesh):
                    continue
                ramp_collision = UsdPhysics.CollisionAPI.Apply(prim)
                ramp_collision.GetCollisionEnabledAttr().Set(True)
                ramp_mesh_col = UsdPhysics.MeshCollisionAPI.Apply(prim)
                ramp_mesh_col.GetApproximationAttr().Set("none")

        suspension_joint_names = (
            "front_left_wheel_suspension", "front_right_wheel_suspension",
            "back_left_wheel_suspension",  "back_right_wheel_suspension",
        )
        for env_id in range(self.num_envs):
            base_link_path = f"/World/envs/env_{env_id}/Robot/mushr_nano/base_link"
            for joint_name in suspension_joint_names:
                joint_prim = stage.GetPrimAtPath(f"{base_link_path}/{joint_name}")
                if not joint_prim.IsValid():
                    continue
                for prop in list(joint_prim.GetProperties()):
                    prop_name = prop.GetName()
                    if "stiffness" in prop_name or "damping" in prop_name:
                        attr = joint_prim.GetAttribute(prop_name)
                        if attr.IsValid():
                            attr.Block()

        ground_cfg = sim_utils.GroundPlaneCfg(
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=0.8, dynamic_friction=0.7, restitution=0.05,
                friction_combine_mode="multiply", restitution_combine_mode="min",
            )
        )
        ground_cfg.func("/World/GroundPlane", ground_cfg)

        self.camera = TiledCamera(self.cfg.camera_cfg)
        self.lidar  = MultiMeshRayCaster(self.cfg.lidar_cfg)
        for link_name in WALL_CONTACT_LINKS:
            sensor_cfg = ContactSensorCfg(
                prim_path=f"/World/envs/env_.*/Robot/mushr_nano/{link_name}",
                filter_prim_paths_expr=["/World/Walls"],
                history_length=self.cfg.decimation,
                update_period=0.0,
                track_pose=False,
                debug_vis=False,
            )
            self._wall_contact_sensors[link_name] = ContactSensor(sensor_cfg)

        rubber_mat_path = "/World/PhysicsMaterials/WheelRubber"
        rubber_mat_prim = stage.DefinePrim(rubber_mat_path, "Material")
        UsdPhysics.MaterialAPI.Apply(rubber_mat_prim)
        phys_rubber = UsdPhysics.MaterialAPI(rubber_mat_prim)
        phys_rubber.CreateStaticFrictionAttr().Set(1.0)
        phys_rubber.CreateDynamicFrictionAttr().Set(1.0)
        phys_rubber.CreateRestitutionAttr().Set(0.05)

        self.scene.clone_environments(copy_from_source=False)

        rubber_mat = UsdShade.Material(rubber_mat_prim)
        for env_id in range(self.num_envs):
            wheel_root_env = stage.GetPrimAtPath(f"/World/envs/env_{env_id}/Robot/mushr_nano")
            if not wheel_root_env.IsValid():
                continue
            for prim in Usd.PrimRange(wheel_root_env):
                if not prim.IsA(UsdGeom.Mesh):
                    continue
                if prim.GetParent().GetName() not in wheel_link_names:
                    continue
                UsdShade.MaterialBindingAPI.Apply(prim).Bind(
                    rubber_mat, UsdShade.Tokens.strongerThanDescendants, "physics",
                )

        self.scene.articulations["robot"] = self.robot
        self.scene.sensors["front_camera"] = self.camera
        self.scene.sensors["lidar"] = self.lidar
        for link_name, sensor in self._wall_contact_sensors.items():
            self.scene.sensors[f"wall_contact_{link_name}"] = sensor

        # Ramp XY bounding box for contact masking
        try:
            from pxr import UsdGeom
            bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
            ramp_prim = stage.GetPrimAtPath("/World/Ramps")
            if ramp_prim.IsValid():
                bb = bbox_cache.ComputeWorldBound(ramp_prim).GetRange()
                margin = 1.0
                self._ramp_xy_min = torch.tensor(
                    [bb.GetMin()[0] - margin, bb.GetMin()[1] - margin],
                    dtype=torch.float32, device=self.device,
                )
                self._ramp_xy_max = torch.tensor(
                    [bb.GetMax()[0] + margin, bb.GetMax()[1] + margin],
                    dtype=torch.float32, device=self.device,
                )
                print(f"[Ramp] XY bbox (±{margin} m margin): "
                      f"min={self._ramp_xy_min.tolist()}  max={self._ramp_xy_max.tolist()}")
            else:
                self._ramp_xy_min = self._ramp_xy_max = None
        except Exception as e:
            print(f"[Ramp] Bounding-box computation failed: {e}")
            self._ramp_xy_min = self._ramp_xy_max = None

        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    # ------------------------------------------------------------------
    # Ramp / wall curriculum helpers
    # ------------------------------------------------------------------

    def _enable_ramp(self) -> None:
        try:
            import omni.usd
            from pxr import Usd, UsdGeom, UsdPhysics
            stage = omni.usd.get_context().get_stage()
            ramp_root = stage.GetPrimAtPath("/World/Ramps")
            if ramp_root.IsValid():
                UsdGeom.Imageable(ramp_root).MakeVisible()
                for prim in Usd.PrimRange(ramp_root):
                    if not prim.IsA(UsdGeom.Mesh):
                        continue
                    col_api = UsdPhysics.CollisionAPI(prim)
                    if col_api:
                        col_api.GetCollisionEnabledAttr().Set(True)
                print(f"[Ramp] ENABLED after {self._total_laps_completed} total laps!")
        except Exception as e:
            print(f"[Ramp] Enable failed: {e}")

    def _swap_walls(self) -> None:
        try:
            import omni.usd
            from pxr import Usd, UsdGeom, UsdPhysics
            stage = omni.usd.get_context().get_stage()
            initial_root = stage.GetPrimAtPath("/World/WallsInitial")
            if initial_root.IsValid():
                for prim in Usd.PrimRange(initial_root):
                    if not prim.IsA(UsdGeom.Mesh):
                        continue
                    col_api = UsdPhysics.CollisionAPI(prim)
                    if col_api:
                        col_api.GetCollisionEnabledAttr().Set(False)
            full_root = stage.GetPrimAtPath("/World/Walls")
            if full_root.IsValid():
                for prim in Usd.PrimRange(full_root):
                    if not prim.IsA(UsdGeom.Mesh):
                        continue
                    col_api = UsdPhysics.CollisionAPI(prim)
                    if col_api:
                        col_api.GetCollisionEnabledAttr().Set(True)
            print(f"[Walls] Swapped to WALLS.usd after {self._total_laps_completed} total laps!")
        except Exception as e:
            print(f"[Walls] Swap failed: {e}")

    # ------------------------------------------------------------------
    # Step hooks
    # ------------------------------------------------------------------

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        """Convert actions → Ackermann commands using velocity integration.

        Throttle (actions[:,0]) ∈ [-1,1]:
            +1 → max acceleration (max_accel_m_s2)
             0 → neutral (no change)
            -1 → max deceleration (max_decel_m_s2)
        Velocity is integrated and clamped to [0, vel_cap].  No throttle LPF
        is needed — the integration itself is a rate limiter.
        """
        dt = self.cfg.decimation / 120.0  # policy step ≈ 0.033 s

        throttle_norm = (actions[:, 0].clamp(-1.0, 1.0) + 1.0) * 0.5   # → [0, 1]
        accel_m_s2 = (
            throttle_norm * (self.cfg.max_accel_m_s2 + self.cfg.max_decel_m_s2)
            - self.cfg.max_decel_m_s2
        )   # → [-max_decel, +max_accel]
        self._accel_m_s2 = accel_m_s2

        self._current_vel = (
            self._current_vel + accel_m_s2 * dt
        ).clamp(0.0, self._vel_cap)

        # Ramp speed cap: prevent flying off at full speed.
        # _on_ramp is updated in _get_dones; use it from the previous step here.
        # This is one step delayed but that's fine — ramps span many steps.
        if self.cfg.ramp_max_speed_m_s < self.cfg.vel_cap_max_m_s:
            self._current_vel = torch.where(
                self._on_ramp,
                self._current_vel.clamp(max=self.cfg.ramp_max_speed_m_s),
                self._current_vel,
            )

        alpha_s = self.cfg.action_lpf_alpha_steer
        self._filtered_steer = (
            alpha_s * actions[:, 1].clamp(-1.0, 1.0)
            + (1.0 - alpha_s) * self._filtered_steer
        )
        steer_ang = self._filtered_steer * self.cfg.max_steer_rad

        self._rear_wheel_ang_vel = self._current_vel / WHEEL_RADIUS
        self._steer_tan          = torch.tan(steer_ang)

        # Store raw action for next-step observation
        self._prev_action[:, 0] = actions[:, 0].clamp(-1.0, 1.0)
        self._prev_action[:, 1] = actions[:, 1].clamp(-1.0, 1.0)

    def _apply_action(self) -> None:
        wall_hit_now = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        for sensor in self._wall_contact_sensors.values():
            fm = sensor.data.force_matrix_w
            if fm is not None:
                h = torch.linalg.vector_norm(fm[..., :2], dim=-1)
                wall_hit_now |= (
                    h.reshape(self.num_envs, -1)
                    .gt(self.cfg.wall_contact_force_thresh_n)
                    .any(dim=1)
                )
        in_grace = self.episode_length_buf < self.cfg.spawn_grace_steps
        self._wall_ever_touched |= (wall_hit_now & ~in_grace)

        wall_mask = self._wall_ever_touched
        wheel_vels = self._rear_wheel_ang_vel.unsqueeze(-1).expand(-1, 2).clone()
        wheel_vels[wall_mask] = 0.0
        self.robot.set_joint_velocity_target(wheel_vels, joint_ids=self._rear_throttle_ids)
        self.robot.set_joint_velocity_target(wheel_vels, joint_ids=self._front_throttle_ids)

        steer_positions = self._steer_tan.unsqueeze(-1).expand(-1, 2).clone()
        steer_positions[wall_mask] = 0.0
        self.robot.set_joint_position_target(steer_positions, joint_ids=self._steer_ids)

    # ------------------------------------------------------------------
    # Depth camera cache  (called once per step, shared by obs + rewards)
    # ------------------------------------------------------------------

    def _update_depth_cache(self) -> None:
        """Process the depth frame and write self._camera_obs.

        Pipeline:
        1. Read distance_to_image_plane [N, H, W, 1] from TiledCamera.
        2. Strip the channel dim → [N, H, W].
        3. Replace inf/nan with depth_max_m (no-hit pixels).
        4. Crop top and bottom rows (sky and near-ground noise).
        5. Add Gaussian noise to simulate TOF ±2 cm accuracy.
        6. Clamp to [0.01, depth_max_m] and normalise:
               depth_norm = 1.0 - depth / depth_max_m
           → 1.0 = obstacle at 0 m (danger), 0.0 = max range (clear).
        7. Flatten to [N, H_crop × W] and store as float32.
        """
        current_step = int(self.episode_length_buf.max().item())
        if self._last_camera_update_step == current_step:
            return
        self._last_camera_update_step = current_step

        depth_raw = self.camera.data.output["distance_to_image_plane"]  # [N, H, W, 1]
        depth = depth_raw[..., 0]                                        # [N, H, W]

        # Replace non-finite values (no-hit rays → max range)
        depth = torch.where(torch.isfinite(depth), depth, torch.tensor(self.cfg.depth_max_m, device=depth.device))

        # Crop top and bottom
        h = depth.shape[1]
        top  = int(self.cfg.depth_crop_top_fraction    * h)
        bot  = h - int(self.cfg.depth_crop_bottom_fraction * h)
        depth = depth[:, top:bot, :]   # [N, H_crop, W]

        # Add Gaussian depth noise (simulate 2 cm TOF accuracy)
        if self.cfg.depth_noise_sigma_m > 0.0:
            noise = torch.randn_like(depth) * self.cfg.depth_noise_sigma_m
            depth = depth + noise

        # Clamp and normalise (close = high, far = low)
        depth = depth.clamp(0.01, self.cfg.depth_max_m)
        depth_norm = 1.0 - depth / self.cfg.depth_max_m   # [N, H_crop, W]

        # Downsample to policy resolution (much smaller MLP input)
        import torch.nn.functional as F
        depth_4d = depth_norm.unsqueeze(1)   # [N, 1, H_crop, W]
        depth_4d = F.interpolate(
            depth_4d,
            size=(self._policy_depth_h, self._policy_depth_w),
            mode="bilinear",
            align_corners=False,
        )
        depth_policy = depth_4d.squeeze(1)   # [N, policy_h, policy_w]

        self._camera_obs = depth_policy.reshape(self.num_envs, -1)   # [N, policy_h*policy_w]

    # ------------------------------------------------------------------
    # Observations
    # ------------------------------------------------------------------

    def _get_observations(self) -> dict:
        self._update_depth_cache()

        # Normalised forward velocity (fraction of current vel_cap)
        lin_vel_b = self.robot.data.root_lin_vel_b[:, 0:1] / self._vel_cap   # [N, 1]

        # Normalised yaw rate
        yaw_rate = self.robot.data.root_ang_vel_b[:, 2:3] / self.cfg.imu_ang_vel_norm  # [N, 1]

        # 360° LiDAR — [N, n_rays] hit distances, normalised to [0, 1]
        # hit_distance=0 → wall at sensor origin, =1 → beyond max range (safe)
        lidar_hits = self.lidar.data.ray_hits_w         # [N, n_rays, 3] world-frame hit points
        lidar_origin = self.lidar.data.pos_w.unsqueeze(1)  # [N, 1, 3] broadcast over rays
        lidar_dist = torch.linalg.vector_norm(lidar_hits - lidar_origin, dim=-1)  # [N, n_rays]
        lidar_dist = lidar_dist.clamp(0.0, self.cfg.lidar_cfg.max_distance)
        # Simulate ±30 mm accuracy
        if self.cfg.lidar_noise_sigma_m > 0.0:
            lidar_dist = lidar_dist + torch.randn_like(lidar_dist) * self.cfg.lidar_noise_sigma_m
            lidar_dist = lidar_dist.clamp(0.0, self.cfg.lidar_cfg.max_distance)
        lidar_norm = lidar_dist / self.cfg.lidar_cfg.max_distance   # [N, n_rays] ∈ [0, 1]

        parts = [self._camera_obs, lidar_norm, lin_vel_b, yaw_rate]
        if self.cfg.include_prev_action:
            parts.append(self._prev_action)   # [N, 2]

        obs = torch.cat(parts, dim=-1)
        return {"policy": obs}

    # ------------------------------------------------------------------
    # Rewards
    # ------------------------------------------------------------------

    def _get_rewards(self) -> torch.Tensor:
        self._update_depth_cache()

        lin_vel_b = self.robot.data.root_lin_vel_b[:, 0]   # m/s forward
        dt = self.cfg.decimation * self.cfg.sim.dt          # ≈ 0.0333 s

        # Distance covered this step
        r_distance = self.cfg.distance_weight * lin_vel_b * dt

        # Speed bonus (fixed normalisation → stable across curriculum)
        r_forward = self.cfg.forward_weight * (lin_vel_b / self.cfg.initial_vel_cap_m_s)

        # Wall penalty
        r_wall = -self.cfg.wall_penalty * self._wall_terminated.float()

        pos_xy = self.robot.data.root_pos_w[:, :2]

        # ── Sector timing ─────────────────────────────────────────────────────────
        next_gate = (self._current_sector + 1) % self._n_sectors
        gate_pts  = self._sector_gates[next_gate]
        gate_tan  = self._sector_tangents[next_gate]
        d_current = ((pos_xy - gate_pts) * gate_tan).sum(dim=-1)
        near_gate = torch.linalg.vector_norm(pos_xy - gate_pts, dim=-1) < 1.8
        crossed   = (self._sector_d_prev <= 0.0) & (d_current > 0.0) & near_gate
        sector_time = (
            (self.episode_length_buf - self._sector_entry_step).float() * dt
        )
        lap_agents = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        if crossed.any():
            crossed_sectors = next_gate[crossed]
            crossed_times   = sector_time[crossed]
            candidate = torch.full(
                (self._n_sectors,), float("inf"), dtype=torch.float32, device=self.device,
            )
            candidate.scatter_reduce_(0, crossed_sectors, crossed_times, reduce="amin", include_self=False)
            torch.minimum(self._sector_best_times, candidate, out=self._sector_best_times)
            self._sectors_this_ep[crossed] += 1
            lap_agents = self._sectors_this_ep >= self._n_sectors
            if lap_agents.any():
                self._sectors_this_ep[lap_agents] = 0
                self._total_laps_completed += int(lap_agents.sum().item())

                dt_p = self.cfg.decimation * self.cfg.sim.dt
                for eid in lap_agents.nonzero(as_tuple=False).squeeze(-1).tolist():
                    ep_steps  = int(self.episode_length_buf[eid].item())
                    lap_steps = ep_steps - int(self._lap_start_step[eid].item())
                    lap_time_s = max(lap_steps, 1) * dt_p
                    scale = float(
                        torch.tensor(self.cfg.target_lap_time_s / lap_time_s).clamp(
                            self.cfg.lap_time_reward_min_scale,
                            self.cfg.lap_time_reward_max_scale,
                        )
                    )
                    print(
                        f"[Lap] env={eid:>3d}  lap_time={lap_time_s:.1f}s  "
                        f"reward={self.cfg.lap_completion_reward * scale:.0f}  "
                        f"(×{scale:.2f})  total_laps={self._total_laps_completed}"
                    )

                if not self._reference_established:
                    self._sector_reference_times = self._sector_best_times.clone()
                    self._reference_established = True
                    print(
                        f"[Sector] Reference lap established! "
                        f"{self._n_sectors} sectors, mean ref = "
                        f"{self._sector_reference_times.mean().item():.2f} s"
                    )

                fresh_lap = lap_agents & (self._agent_spawn_vel_cap == self._vel_cap)
                if fresh_lap.any() and not self._lap_at_current_vel:
                    self._lap_at_current_vel = True

        # Lap completion bonus (scaled by lap time)
        r_lap = torch.zeros(self.num_envs, device=self.device)
        if lap_agents.any():
            lap_steps_f = (
                self.episode_length_buf[lap_agents].float()
                - self._lap_start_step[lap_agents].float()
            ).clamp(min=1.0)
            lap_time_s = lap_steps_f * dt
            time_scale = (self.cfg.target_lap_time_s / lap_time_s).clamp(
                self.cfg.lap_time_reward_min_scale, self.cfg.lap_time_reward_max_scale,
            )
            r_lap[lap_agents] = self.cfg.lap_completion_reward * time_scale
            self._lap_start_step[lap_agents] = self.episode_length_buf[lap_agents].int()

        if not self._reference_established:
            r_sector = torch.zeros(self.num_envs, device=self.device)
        else:
            ref_time  = self._sector_reference_times[next_gate]
            beats_ref = crossed & (sector_time < ref_time)
            improvement = (
                (ref_time - sector_time) / ref_time.clamp(min=1e-6)
            ).clamp(0.0, 1.0)
            r_sector = self.cfg.sector_gain_weight * beats_ref.float() * improvement

        self._current_sector[crossed]    = next_gate[crossed]
        self._sector_entry_step[crossed] = self.episode_length_buf[crossed].int()
        self._sector_d_prev = d_current

        # ── Wheel slip (AWD) ──────────────────────────────────────────────────────
        all_wheel_omega = torch.cat([
            self.robot.data.joint_vel[:, self._rear_throttle_ids],
            self.robot.data.joint_vel[:, self._front_throttle_ids],
        ], dim=-1)
        wheel_vel  = all_wheel_omega.mean(dim=-1) * WHEEL_RADIUS
        slip_speed = torch.abs(wheel_vel - lin_vel_b)
        r_slip     = -self.cfg.slip_weight * slip_speed / self.cfg.initial_vel_cap_m_s * (~self._on_ramp).float()

        # ── Steering deadzone penalty ────────────────────────────────────────────
        steer_norm    = torch.abs(self._filtered_steer).clamp(0.0, 1.0)
        steer_excess  = (steer_norm - self.cfg.steer_deadzone).clamp(min=0.0)
        steer_excess_norm = steer_excess / max(1.0 - self.cfg.steer_deadzone, 1e-6)
        r_steer_shape = -self.cfg.steer_deadzone_weight * steer_excess_norm

        # ── Sector proximity ─────────────────────────────────────────────────────
        dist_to_gate = torch.linalg.vector_norm(pos_xy - gate_pts, dim=-1)
        proximity    = (1.0 - dist_to_gate / self.cfg.sector_proximity_max_dist).clamp(0.0, 1.0)
        r_proximity  = self.cfg.sector_proximity_weight * proximity

        # ── Ramp reward ──────────────────────────────────────────────────────────
        r_ramp = (
            self.cfg.ramp_reward_weight
            * self._on_ramp.float()
            * (lin_vel_b / self.cfg.initial_vel_cap_m_s).clamp(0.0, 1.0)
        )

        # ── Slow-driving penalty ─────────────────────────────────────────────────
        # Uses slow_penalty_speed_fraction (default 0.0 = disabled).
        # Deliberately NOT penalising braking — the depth camera already teaches
        # the agent to slow before walls; penalising low speed here would fight that.
        if self.cfg.slow_penalty_speed_fraction > 0.0:
            is_slow = (
                (lin_vel_b < (self.cfg.slow_penalty_speed_fraction * self._vel_cap))
                & ~self._on_ramp
            )
        else:
            is_slow = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._slow_timer = torch.where(
            is_slow,
            self._slow_timer + 1,
            torch.zeros_like(self._slow_timer),
        )
        r_slow = -self.cfg.slow_penalty_weight * is_slow.float()

        total = (
            r_distance + r_forward + r_slip + r_steer_shape
            + r_slow + r_wall + r_sector + r_ramp + r_lap + r_proximity
        )

        self._ep_return_buf += total.detach()
        return total

    # ------------------------------------------------------------------
    # Termination
    # ------------------------------------------------------------------

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        wall_terminated = self._wall_ever_touched.clone()

        for sensor in self._wall_contact_sensors.values():
            fm = sensor.data.force_matrix_w
            if fm is not None:
                h = torch.linalg.vector_norm(fm[..., :2], dim=-1)
                wall_terminated |= (
                    h.reshape(self.num_envs, -1)
                    .gt(self.cfg.wall_contact_force_thresh_n)
                    .any(dim=1)
                )

        # Ramp: robot is elevated
        self._on_ramp = self.robot.data.root_pos_w[:, 2] >= 0.001

        in_grace = self.episode_length_buf < self.cfg.spawn_grace_steps
        self._wall_ever_touched |= (wall_terminated & ~in_grace)
        wall_terminated = self._wall_ever_touched
        self._wall_terminated = wall_terminated

        # Flip detection (suppressed during grace)
        q    = self.robot.data.root_quat_w
        up_z = 1.0 - 2.0 * (q[:, 1] * q[:, 1] + q[:, 2] * q[:, 2])
        flipped = (up_z < 0.3) & ~in_grace

        # Near-stop termination — only fires if agent is essentially stationary
        # (slow_stop_speed_fraction × vel_cap, default 3% = 0.075 m/s at 2.5 m/s cap).
        # Grace period of 45 steps (~1.5 s) lets the car accelerate from spawn.
        lin_vel_b = self.robot.data.root_lin_vel_b[:, 0]
        stopped = (
            (lin_vel_b < (self.cfg.slow_stop_speed_fraction * self._vel_cap))
            & (self.episode_length_buf > 45)
        )

        terminated = wall_terminated | flipped | stopped
        time_out   = self.episode_length_buf >= self.max_episode_length - 1
        return terminated, time_out

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES

        # Episode return history
        for eid in env_ids:
            self._ep_return_history.append(float(self._ep_return_buf[eid].item()))
        self._ep_return_buf[env_ids] = 0.0

        if (
            not self._random_spawn_unlocked
            and len(self._ep_return_history) >= self.cfg.fixed_spawn_history_len
        ):
            mean_return = sum(self._ep_return_history) / len(self._ep_return_history)
            if mean_return >= self.cfg.fixed_spawn_reward_threshold:
                self._random_spawn_unlocked = True
                print(
                    f"[Spawn] Random spawning UNLOCKED! "
                    f"Mean return = {mean_return:.1f} >= {self.cfg.fixed_spawn_reward_threshold:.1f}"
                )

        # Curriculum advancement
        if self._lap_at_current_vel and self._vel_cap < self.cfg.vel_cap_max_m_s:
            self._lap_at_current_vel = False
            self._vel_cap = min(
                self._vel_cap + self.cfg.vel_cap_step_m_s, self.cfg.vel_cap_max_m_s
            )
            print(f"[Curriculum] vel_cap → {self._vel_cap:.2f} m/s  (full lap completed)")

        super()._reset_idx(env_ids)

        n = len(env_ids)
        default_root_state = self.robot.data.default_root_state[env_ids].clone()
        default_root_state[:, :3] += self.scene.env_origins[env_ids]

        speed     = self.cfg.spawn_speed_m_s
        sz        = 0.002   # just above ground — constant for all spawn modes
        env_ids_t = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)

        if self.cfg.use_preset_spawns:
            # ── Preset-point spawn: randomly pick from hand-crafted list ──────
            pts   = self.cfg.preset_spawn_points
            idx   = torch.randint(0, len(pts), (n,), device="cpu").tolist()
            xs    = torch.tensor([pts[i][0] for i in idx], dtype=torch.float32, device=self.device)
            ys    = torch.tensor([pts[i][1] for i in idx], dtype=torch.float32, device=self.device)
            yaw   = torch.tensor(
                [math.radians(pts[i][2]) for i in idx], dtype=torch.float32, device=self.device
            )
            if self.cfg.spawn_yaw_jitter_rad > 0.0:
                yaw = yaw + (torch.rand(n, device=self.device) - 0.5) * 2.0 * self.cfg.spawn_yaw_jitter_rad
            spawn_xy = torch.stack([xs, ys], dim=-1)               # [n, 2]
            # Nearest sector gate per spawn point (for sector progress tracking)
            nearest_gates = torch.zeros(n, dtype=torch.long, device=self.device)
            for j in range(n):
                diff = self._sector_gates - spawn_xy[j].unsqueeze(0)
                nearest_gates[j] = torch.linalg.vector_norm(diff, dim=-1).argmin()
            self._current_sector[env_ids_t] = nearest_gates
        else:
            # ── CSV-gate spawn: uniform random over all sector gates ──────────
            gate_idx = torch.randint(0, self._n_sectors, (n,), device=self.device)
            spawn_xy = self._sector_gates[gate_idx]                # [n, 2]
            gate_tan = self._sector_tangents[gate_idx]             # [n, 2]
            yaw      = torch.atan2(gate_tan[:, 1], gate_tan[:, 0])
            if self.cfg.spawn_yaw_jitter_rad > 0.0:
                yaw = yaw + (torch.rand(n, device=self.device) - 0.5) * 2.0 * self.cfg.spawn_yaw_jitter_rad
            self._current_sector[env_ids_t] = gate_idx

        default_root_state[:, 0] = spawn_xy[:, 0]
        default_root_state[:, 1] = spawn_xy[:, 1]
        default_root_state[:, 2] = sz

        zeros = torch.zeros(n, device=self.device)
        dq = quat_from_euler_xyz(zeros, zeros, yaw)
        default_root_state[:, 3:7] = dq

        default_root_state[:, 7:]  = 0.0
        default_root_state[:, 7]   = speed * torch.cos(yaw)
        default_root_state[:, 8]   = speed * torch.sin(yaw)

        self.robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self.robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)

        # Sector / spawn / filter state reset
        self._spawn_pos[env_ids]           = spawn_xy
        self._sectors_this_ep[env_ids]     = 0
        self._agent_spawn_vel_cap[env_ids] = self._vel_cap
        self._sector_entry_step[env_ids]   = 0
        self._lap_start_step[env_ids]      = 0

        next_gate_r = (self._current_sector[env_ids] + 1) % self._n_sectors
        gate_pts_r  = self._sector_gates[next_gate_r]
        gate_tan_r  = self._sector_tangents[next_gate_r]
        d_init      = ((spawn_xy - gate_pts_r) * gate_tan_r).sum(dim=-1)
        self._sector_d_prev[env_ids] = d_init.clamp(max=0.0)

        # Seed commanded velocity to match initial physics speed
        self._current_vel[env_ids]    = speed
        self._accel_m_s2[env_ids]     = 0.0
        self._filtered_steer[env_ids] = 0.0
        self._steer_integral[env_ids] = 0.0
        self._prev_steer[env_ids]     = 0.0
        self._prev_action[env_ids]    = 0.0

        self._slow_timer[env_ids]        = 0
        self._ramp_ascended[env_ids]     = False
        self._wall_ever_touched[env_ids] = False

        joint_pos = self.robot.data.default_joint_pos[env_ids]
        joint_vel = torch.zeros_like(joint_pos)
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)
