"""MuSHR nano v2 RC car — cone-track navigation environment (DirectRLEnv).

Observation space:
    [0:N]        flattened depth image from LEFT ArduCam TOF camera (+35° yaw), after crop.
                 Normalized: 1.0 = object at 0 m (danger), 0.0 = object at depth_max_m (clear).
    [N:2N]       flattened depth image from RIGHT ArduCam TOF camera (−35° yaw), same pipeline.
    [2N:2N+2H]   last H throttle+steer pairs, oldest→newest ∈ [−1, 1]
                 H = action_history_steps (default 15) → 30 values.
                 Only appended when include_prev_action=True.
    No velocity in observation — vehicle speed is not available from real-robot sensors.

Camera profile (ArduCam TOF sensor for Raspberry Pi):
    - Sensor: Time-of-Flight depth, 240×180 pixels at 30 fps
    - Diagonal FOV: 70°  →  HFOV ≈ 58.5°, VFOV ≈ 45.6°  (4:3 aspect, pinhole)
    - Depth accuracy: ±2 cm (simulated via Gaussian noise σ ≈ 0.007 m)
    - Policy runs at 30 Hz, synced with camera (sim 120 Hz, decimation=4)

Action space (2):
    [0]  throttle command in [-1, 1]  → remapped to [0, 1] inside the env.
         Applied as per-wheel torque via a linear torque-speed curve:
           T(ω) = throttle_norm × T_stall × max(0, 1 − ω/ω_noload)
         where T_stall and ω_noload are derived from slip_threshold + max_velocity_m_s.
    [1]  steering command in [-1, 1]  → scaled to [-max_steer_rad, +max_steer_rad]

Actuation (Ackermann / AWD, direct torque control):
    - All 4 wheels receive a torque target (Nm) via set_joint_effort_target.
      Torque follows a linear torque-speed curve: T = throttle × T_stall × (1 − ω/ω_noload).
      Wheel spin occurs naturally when applied torque exceeds PhysX traction limit.
    - Front wheel steer receives position target (rad) = tan(steering_angle)
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
    - All adjustable parameters are in ConeTrackEnvCfg in one place below.
    - Spawn position: set preset_spawn_points in cfg.
    - Camera crop: depth_crop_top_fraction, depth_crop_bottom_fraction.
    - Motor model: slip_threshold (wheel-spin sensitivity) and max_velocity_m_s (top speed).
"""

from __future__ import annotations

from collections.abc import Sequence
import math
import os

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensor, ContactSensorCfg
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
    os.path.join(os.path.dirname(__file__), "../../../assets")
)
_REPO_ROOT  = os.path.normpath(os.path.join(os.path.dirname(__file__), "../../../../"))
_EXPORT_DIR = os.path.join(_REPO_ROOT, "CAD", "Track", "Export")

MUSHR_USD        = os.path.join(_ASSETS_DIR, "mushr_nano_v2.usd")
TRACK_USD        = os.path.join(_EXPORT_DIR, "Track.usdc")
WALL_USD         = os.path.join(_EXPORT_DIR, "Walls.usdc")
RAMP_USD         = os.path.join(_ASSETS_DIR, "RAMPS.usd")

# Decorative asset sets — purely visual objects the depth cameras see.
# Index NUM_ASSET_SETS means "no decorations" (all sets hidden).
ASSET_SET_USDS = [
    os.path.join(_EXPORT_DIR, "Assets1.usdc"),
    os.path.join(_EXPORT_DIR, "Assets2.usdc"),
    os.path.join(_EXPORT_DIR, "Assets3.usdc"),
    os.path.join(_EXPORT_DIR, "Assets4.usdc"),
    os.path.join(_EXPORT_DIR, "Assets5.usdc"),
]
NUM_ASSET_SETS = len(ASSET_SET_USDS)   # randomisation draws from [0, NUM_ASSET_SETS] inclusive

# ---------------------------------------------------------------------------
# Physical constants — MuSHR nano v2 (fixed by the USD / hardware)
# These match mushr_nano_v2.usd geometry and are NOT user-adjustable without
# also modifying the robot USD.
# ---------------------------------------------------------------------------
WHEEL_RADIUS = 0.041   # m  (82 mm diameter wheels)

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
class ConeTrackEnvCfg(DirectRLEnvCfg):
    """Single source of truth for every number you might want to change.

    Sections
    --------
    CAMERA          — sensor resolution, FOV, depth range, crop, noise
    SPAWN           — starting position, heading, and speed
    SPAWN SPEED     — initial episode velocity range
    DYNAMICS        — physics-based motor model, steering limit
    TERMINATION     — contact-force threshold, slow-speed detection
    OBSERVATION     — whether to include previous action in obs
    REWARD WEIGHTS  — all reward scale factors
    """

    # ── Core env parameters ────────────────────────────────────────────────
    decimation:        int   = 4      # policy at ~30 Hz  (sim 120 Hz / 4)
    episode_length_s:  float = 150.0  # 150 s × 30 Hz = 4 500 steps
    action_space:      int   = 2
    # observation_space is patched in ConeTrackEnv.__init__ after computing
    # the exact cropped depth image size.  The placeholder below keeps
    # @configclass happy; the real value is written before super().__init__().
    observation_space: int   = 1      # patched in __init__
    state_space:       int   = 0

    sim: SimulationCfg = SimulationCfg(dt=1.0 / 120.0, render_interval=4)

    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=5,
        env_spacing=0.0,
        replicate_physics=True,
    )

    # ════════════════════════════════════════════════════════════════════════
    # CAMERA — ArduCam TOF (240×180, 30 fps, 70° diagonal FOV)
    # ════════════════════════════════════════════════════════════════════════
    # Crop fractions (fraction of raw pixel rows discarded before the policy):
    #   top  → removes ceiling / sky (depth = max / inf)
    #   bottom → removes near-ground clutter and the robot's own body shadow
    depth_crop_top_fraction:    float = 0.35
    depth_crop_bottom_fraction: float = 0.3

    # Gaussian depth noise: σ ≈ 0.007 m gives a ±2 cm (3σ) error band,
    # matching the ArduCam TOF spec.  Set to 0.0 to disable.
    depth_noise_sigma_m: float = 0.008

    # Random invalid-pixel dropout: each pixel is independently zeroed with this
    # probability, replicating the black-spec speckle noise visible in real ArduCam
    # TOF output.  Affected pixels are set to depth_max_m (treated as far/safe by the
    # policy, matching the real camera's low-confidence masking behaviour).
    # Visible as black dots in both windows of visualize.py.  Set to 0.0 to disable.
    depth_invalid_pixel_prob: float = 0.05

    # Random-depth (coloured) speckle: each pixel is set to a uniform random depth in
    # [0, depth_max_m] with this probability.  Models stray reflections and multi-path
    # interference that appear as bright or mid-range dots (unlike depth_invalid_pixel_prob
    # which always sets the pixel to the far/black depth_max_m value).
    # Set to 0.0 to disable.
    depth_color_noise_prob: float = 0.01

    # Edge blur noise — simulates the mixed-pixel / depth-bleeding artefact common in
    # real ToF sensors: pixels whose sensing area straddles a depth discontinuity report
    # a value somewhere between the two depths rather than either clean value.
    # Implementation: L1 gradient |∇depth| is computed on the CLEAN cropped depth frame
    # (before any other noise), then Gaussian noise with
    #   std = depth_edge_noise_sigma_m × clamp(|∇depth| / depth_edge_gradient_threshold_m, 0, 1)
    # is added.  Only pixels at depth edges receive meaningful extra noise.
    # Set depth_edge_noise_sigma_m = 0.0 to disable entirely.
    depth_edge_gradient_threshold_m: float = 0.15   # gradient (m/px) that gives edge_weight = 1
    # Mixed-pixel scatter: edge pixels randomly jump to 0 m or depth_max_m (binary wrong-side
    # values — models the individual black/red dots seen on real ToF sensors at boundaries).
    depth_edge_scatter_prob: float = 0.25
    # Small continuous Gaussian blur layered on top of the scatter, adds the faint soft halo
    # visible around object edges on the real camera alongside the point scatter.
    depth_edge_blur_sigma_m: float = 0.06

    # Depth range for normalization.  Readings > depth_max_m are clamped.
    # Must match autopilot.py MAX_DISTANCE / 1000.  ArduCam supports 2000 mm or 4000 mm only.
    depth_max_m: float = 2.0

    # ── Policy image downsampling ──────────────────────────────────────────
    # After cropping the raw depth frame (240×≈99 at defaults) the image is
    # bilinearly downsampled to this resolution before being flattened into
    # the observation vector.  Smaller = faster MLP convergence.
    # policy_depth_height < 0 → auto-compute from cropped aspect ratio.
    # Default 80×33 = 2640 dims (vs 240×99 = 23 760 without downsampling).
    policy_depth_width:  int = 180
    policy_depth_height: int = -1   # -1 = auto

    # ════════════════════════════════════════════════════════════════════════
    # SPAWN — initial robot position, heading, and speed
    # ════════════════════════════════════════════════════════════════════════
    # Grace period after each reset: wall / flip / stuck terminations are suppressed
    # for this many policy steps so that PhysX contact forces can settle.
    # 10 steps ≈ 333 ms at 30 Hz — enough for any spawn-physics transient.
    spawn_grace_steps: int   = 10

    # ── Spawn mode ────────────────────────────────────────────────────────────
    #   Each reset picks randomly from preset_spawn_points.
    #   Each entry is (x_m, y_m, theta_deg).
    #   spawn_yaw_jitter_rad is applied.
    preset_spawn_points: list = [
        # (x_m, y_m, theta_deg) — raw Blender coordinates
        (0.756,  5.0, 20.8),
        (3.91,  3.31, -156.0),
        (3.62,  1.97, -154.48),
        (0.0,  0.0, 0.0),
        (-5.46,  3.7, 0.0),
        (-3.6,  4.7, -217.86),
    ]
    # Coordinate transform applied to every spawn point at runtime.
    # Blender exports X/Y inverted relative to Isaac Sim's frame, so both are -1.
    spawn_x_scale: float = -1.15
    spawn_y_scale: float = -1.15

    duplicate_with_flipped_yaw: bool = True  # if True, each (x, y, θ) spawns two episodes: one with +180deg rotation around z
    spawn_yaw_jitter_rad: float = 0.15   # ± heading noise applied in both modes

    # ════════════════════════════════════════════════════════════════════════
    # SPAWN SPEED — initial robot velocity at each episode reset
    # ════════════════════════════════════════════════════════════════════════
    spawn_vel_min_m_s: float = 0.0   # minimum spawn speed (m/s)
    spawn_vel_max_m_s: float = 0.0   # maximum spawn speed (m/s)

    # ════════════════════════════════════════════════════════════════════════
    # DYNAMICS — physics-based motor model (AWD, direct torque control)
    #
    # Motor torque-speed curve (linear, same for all 4 wheels):
    #   T(throttle, ω) = throttle_norm × T_stall × max(0, 1 − ω/ω_noload)
    #
    # T_stall and ω_noload are DERIVED from two intuitive parameters:
    #
    #   slip_threshold   — throttle fraction [0,1] that just saturates wheel
    #                      traction at rest (ω=0).  At this point motor torque
    #                      equals the grip limit μ_s×(m×g/N_wheels)×r_wheel:
    #                        T_stall = μ_s × (m×g/4) × r / slip_threshold
    #
    #   max_velocity_m_s — vehicle speed where full-throttle torque drops to 0
    #                      (the motor's no-load speed times wheel radius).  This
    #                      is a SOFT physics limit — the car simply cannot
    #                      accelerate past it on a flat straight.
    #                        ω_noload = max_velocity_m_s / WHEEL_RADIUS
    #
    # Both derived values are printed at startup and stored as self._T_stall /
    # self._omega_noload.  No hard velocity clamp is applied anywhere.
    # ════════════════════════════════════════════════════════════════════════
    car_mass_kg:      float = 1.3    # total robot mass (kg); must match robot_cfg mass_props
    wheel_grip_mu_s:  float = 0.8    # effective static friction coeff (wheel × ground)
    slip_threshold:   float = 0.8   # throttle fraction [0,1] that saturates traction at rest
    max_velocity_m_s: float = 6.0   # top speed (m/s): full-throttle torque drops to 0 here
    max_steer_rad:    float = 0.488  # ≈28° max front-wheel steer angle
    # DC-motor zero-power "brake" mode (not float).
    # When policy outputs -1 (throttle_norm == 0) the ESC shorts both motor terminals,
    # creating a back-EMF braking torque proportional to wheel speed:
    #   T_brake = -brake_strength × T_stall × (ω / ω_noload)   [viscous, sign opposes rotation]
    # At any throttle above 0 there is no braking — pure drive torque.
    # At brake_strength=0.3, braking at 6 m/s ≈ 0.6 g deceleration.
    brake_strength:   float = 0.3

    # Action latency simulation — all actuator commands are delayed by this many
    # milliseconds before reaching the wheels/steering.  At 30 Hz each step ≈ 33 ms,
    # so 100 ms ≈ 3 policy steps.  Set to 0.0 to disable.
    action_delay_ms: float = 33

    # ════════════════════════════════════════════════════════════════════════
    # TERMINATION THRESHOLDS
    # ════════════════════════════════════════════════════════════════════════
    # Horizontal contact force on any robot link that counts as a wall hit.
    wall_contact_force_thresh_n: float = 1.0  # Newtonsgz

    # slow_stop_speed_fraction — episode terminates when speed falls below this
    # (catches truly stuck / reversed agents; keep low so normal braking is fine)
    slow_stop_speed_fraction:    float = 0.04  # terminate if speed < 3 % of vel_cap

    # ════════════════════════════════════════════════════════════════════════
    # OBSERVATION EXTRAS
    # ════════════════════════════════════════════════════════════════════════
    # Append the previous policy actions (throttle + steer, each ∈ [-1,1]) to
    # the observation.  With forward velocity removed, this is the policy's only
    # speed cue — longer history allows better speed inference from throttle commands.
    include_prev_action:  bool = True
    action_history_steps: int  = 15   # number of past (throttle, steer) pairs to include

    # ════════════════════════════════════════════════════════════════════════
    # REWARD WEIGHTS
    # ════════════════════════════════════════════════════════════════════════
    # Constant reward per step the agent stays alive (not crashed / stuck).
    alive_weight:       float = 0.1

    # Distance: rewards physical metres covered each step (v × dt).
    distance_weight:    float = 50.0

    # Speed bonus, normalised by max_velocity_m_s.
    forward_weight:     float = 10

    # Rear-wheel slip penalty.
    slip_weight:        float = 0.0

    # Steering deadzone penalty: only penalises steering beyond the deadzone.
    steer_deadzone_weight: float = 1
    steer_deadzone:        float = 0.05   # fraction of max_steer_rad that is free

    # Ramp per-step reward (while robot base is elevated on the ramp surface).
    ramp_reward_weight: float = 0

    # Ramp speed penalty: fires every step the robot is inside the ramp XY bounding
    # box (or within ramp_approach_radius_m of it) and its forward speed exceeds
    # ramp_speed_threshold_m_s.  Penalty scales linearly with the excess speed.
    ramp_speed_threshold_m_s:  float = 3.0
    ramp_speed_penalty_weight: float = 0.0
    ramp_approach_radius_m:    float = 4.0  # extend penalty zone this far beyond ramp bbox

    # How often to switch the active decorative asset set (in policy steps).
    # Decoupled from episode resets so the background changes at a steady rate
    # regardless of how fast agents are crashing.
    asset_randomize_every_n_steps: int = 500

    # ════════════════════════════════════════════════════════════════════════
    # HARDWARE CONFIG  (robot articulation + camera sensor)
    # ════════════════════════════════════════════════════════════════════════

    robot_cfg: ArticulationCfg = ArticulationCfg(
        prim_path="/World/envs/env_.*/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=MUSHR_USD,
            scale=(1.0, 1.0, 1.0),
            activate_contact_sensors=True,
            mass_props=sim_utils.MassPropertiesCfg(mass=1.3),
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
            pos=(0, 0, 0),   # placeholder — overridden every episode in _reset_idx
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
                # Velocity limit matches real servo: full sweep (2×0.488 rad) in 150 ms
                # → 0.976 / 0.150 = 6.51 rad/s.  PhysX enforces this as a hard cap.
                velocity_limit_sim=6.51, effort_limit_sim=3.2,
            ),
            "rear_throttle": ImplicitActuatorCfg(
                joint_names_expr=["back_left_wheel_throttle", "back_right_wheel_throttle"],
                stiffness=0.0, damping=0.0,       # direct torque control via effort target
                velocity_limit_sim=450.0, effort_limit_sim=5.0,
            ),
            "front_throttle": ImplicitActuatorCfg(
                joint_names_expr=["front_left_wheel_throttle", "front_right_wheel_throttle"],
                stiffness=0.0, damping=0.0,       # direct torque control via effort target
                velocity_limit_sim=450.0, effort_limit_sim=5.0,
            ),
        },
    )

    # Two ArduCam TOF depth cameras:
    #   Position: 122 mm forward of base_link, ±50 mm left/right, 70 mm above base_link.
    #   Orientation: ±30° yaw outward only — no pitch.
    #
    # Rotation derivation (all quaternions in (w, x, y, z) ROS convention):
    #   Start: q_forward = (-0.5, 0.5, -0.5, 0.5)  (camera level, pointing fwd in ROS)
    #   Yaw  : q_yaw_z(±30°) * q_forward
    #             left  (+30°): (-0.6124, 0.6124, -0.3536, 0.3536)
    #             right (-30°): (-0.3536, 0.3536, -0.6124, 0.6124)
    #   Result forward vectors in base_link frame:
    #             left : (cos30°, sin30°, 0) = (0.866, 0.500, 0.000)
    #             right: (cos30°, -sin30°, 0) = (0.866, -0.500, 0.000)
    camera_left_cfg: TiledCameraCfg = TiledCameraCfg(
        prim_path="/World/envs/env_.*/Robot/mushr_nano/base_link/camera_left",
        update_period=0.0,
        height=180,
        width=240,
        data_types=["distance_to_image_plane"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=_TOF_FOCAL_MM,
            horizontal_aperture=_TOF_H_APERTURE_MM,
            vertical_aperture=_TOF_V_APERTURE_MM,
            clipping_range=(0.01, 5.0),
        ),
        offset=TiledCameraCfg.OffsetCfg(
            pos=(0.122, 0.05, 0.07),   # 122 mm fwd, 50 mm left, 70 mm up from base_link
            rot=(-0.6124, 0.6124, -0.3536, 0.3536),   # 30° yaw left, no pitch
            convention="ros",
        ),
        debug_vis=False,
    )
    camera_right_cfg: TiledCameraCfg = TiledCameraCfg(
        prim_path="/World/envs/env_.*/Robot/mushr_nano/base_link/camera_right",
        update_period=0.0,
        height=180,
        width=240,
        data_types=["distance_to_image_plane"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=_TOF_FOCAL_MM,
            horizontal_aperture=_TOF_H_APERTURE_MM,
            vertical_aperture=_TOF_V_APERTURE_MM,
            clipping_range=(0.01, 5.0),
        ),
        offset=TiledCameraCfg.OffsetCfg(
            pos=(0.122, -0.05, 0.07),  # 122 mm fwd, 50 mm right, 70 mm up from base_link
            rot=(-0.3536, 0.3536, -0.6124, 0.6124),   # 30° yaw right, no pitch
            convention="ros",
        ),
        debug_vis=False,
    )

    # Live debug view in training — use scripts/rsl_rl/visualize.py instead.
    show_camera_debug:    bool = False
    camera_debug_env_id:  int  = 0
    camera_debug_interval:int  = 2

    # Spawn a pair of visible cones in the sim to verify camera placement.
    # Tip at each camera sensor origin; base (wide end) pointing along the camera's
    # look direction.  Left camera = cyan, right camera = magenta.
    # Set to True for one training run, inspect in the viewport, then set back to False.
    show_camera_cones: bool = False


# ===========================================================================
# Environment Class
# ===========================================================================

class ConeTrackEnv(DirectRLEnv):
    cfg: ConeTrackEnvCfg

    def __init__(self, cfg: ConeTrackEnvCfg, render_mode: str | None = None, **kwargs):
        # ContactSensor dict must exist before super().__init__() calls _setup_scene().
        self._wall_contact_sensors: dict[str, ContactSensor] = {}

        # Compute exact depth observation size (same for both cameras).
        # 1. Crop: remove top/bottom rows from the raw 240×180 frame.
        # 2. Downsample: bilinear resize to policy_depth_width × policy_depth_height.
        cam_h = cfg.camera_left_cfg.height   # 180
        cam_w = cfg.camera_left_cfg.width    # 240
        remaining = max(0.0, 1.0 - cfg.depth_crop_top_fraction - cfg.depth_crop_bottom_fraction)
        cropped_h = max(1, int(cam_h * remaining))
        cropped_w = cam_w

        self._policy_depth_w = cfg.policy_depth_width
        if cfg.policy_depth_height < 0:
            self._policy_depth_h = max(1, round(cfg.policy_depth_width * cropped_h / cropped_w))
        else:
            self._policy_depth_h = cfg.policy_depth_height

        depth_obs_size = self._policy_depth_h * self._policy_depth_w

        extra_obs = 0  # lin_vel removed — not available on real hardware
        if cfg.include_prev_action:
            extra_obs += cfg.action_history_steps * 2  # N × (throttle + steer)
        cfg.observation_space = depth_obs_size * 2 + extra_obs  # left + right cameras

        print(
            f"[ConeTrackEnv] Depth obs per camera: raw {cam_w}×{cam_h} "
            f"→ crop {cropped_w}×{cropped_h} "
            f"→ policy {self._policy_depth_w}×{self._policy_depth_h} "
            f"= {depth_obs_size} px  |  2 cameras  |  "
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
        self._steer_tan = torch.zeros(self.num_envs, device=self.device)

        # Precomputed tensor of all 4 driven wheel joint indices [rear_L, rear_R, front_L, front_R]
        self._all_throttle_ids = torch.tensor(
            self._rear_throttle_ids + self._front_throttle_ids,
            dtype=torch.long, device=self.device,
        )

        # Per-wheel torque commands (split rear/front for set_joint_effort_target)
        self._wheel_torque_rear  = torch.zeros(self.num_envs, 2, device=self.device)
        self._wheel_torque_front = torch.zeros(self.num_envs, 2, device=self.device)

        # Steering LPF state
        self._filtered_steer = torch.zeros(self.num_envs, device=self.device)
        self._steer_integral  = torch.zeros(self.num_envs, device=self.device)
        self._prev_steer      = torch.zeros(self.num_envs, device=self.device)
        self._steer_hp        = torch.zeros(self.num_envs, device=self.device)

        # Action history for observation: action_history_steps most-recent (throttle, steer) pairs.
        # Shape [N, H, 2]; flattened to [N, H*2] when appended to obs.
        H = cfg.action_history_steps
        self._prev_action = torch.zeros(self.num_envs, H, 2, device=self.device)

        # Action delay buffer — holds the last (delay_steps+1) raw actions so that
        # actuators receive the command issued delay_steps policy ticks ago.
        # The observation history still stores the undelayed policy output so the
        # network can reason about what it recently commanded vs what the car has done.
        policy_dt = cfg.decimation * cfg.sim.dt          # ≈ 1/30 s
        self._delay_steps = max(0, round(cfg.action_delay_ms * 1e-3 / policy_dt))
        buf_len = max(1, self._delay_steps + 1)
        self._action_delay_buf = torch.zeros(self.num_envs, buf_len, 2, device=self.device)
        print(f"[ConeTrackEnv] Action delay: {cfg.action_delay_ms:.0f} ms → {self._delay_steps} policy step(s)")

        # Derived motor parameters — computed once from physics-based config fields.
        # T_stall: stall torque per wheel = traction limit / slip_threshold
        # ω_noload: no-load angular velocity = max_velocity / wheel_radius
        _g = 9.81
        _traction_limit = cfg.wheel_grip_mu_s * (cfg.car_mass_kg * _g / 4.0) * WHEEL_RADIUS
        self._T_stall      = _traction_limit / cfg.slip_threshold
        self._omega_noload = cfg.max_velocity_m_s / WHEEL_RADIUS
        print(
            f"[ConeTrackEnv] Motor model: T_stall={self._T_stall:.4f} Nm/wheel, "
            f"ω_noload={self._omega_noload:.2f} rad/s  "
            f"(slip_threshold={cfg.slip_threshold:.2f}, μ_s={cfg.wheel_grip_mu_s}, "
            f"m={cfg.car_mass_kg} kg, v_max={cfg.max_velocity_m_s} m/s)"
        )


        # Contact / termination flags
        self._wall_terminated    = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._on_ramp            = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._wall_ever_touched  = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._ramp_ascended      = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        # Average-speed reward accumulator: sum of clamped forward velocity per env.
        self._cumulative_forward_vel = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        # Index of the currently active decorative asset set (-1 = none selected yet).
        self._current_asset_set: int = -1
        self._asset_step_counter: int = 0
        # All asset sets are left visible during scene init so RTX builds their BVH
        # structures at startup (cheap time slot). On the first physics step we hide
        # all but one selected set. This flag tracks whether that warmup hide has run.
        self._assets_warmup_done: bool = False

        # Depth observation caches (policy resolution after crop + downsample)
        _cam_flat = self._policy_depth_h * self._policy_depth_w
        self._camera_left_obs  = torch.zeros(self.num_envs, _cam_flat, device=self.device)
        self._camera_right_obs = torch.zeros(self.num_envs, _cam_flat, device=self.device)
        self._last_camera_update_step = -1

    # ------------------------------------------------------------------
    # Scene setup
    # ------------------------------------------------------------------

    def _setup_scene(self):
        self.robot = Articulation(self.cfg.robot_cfg)

        track_cfg = sim_utils.UsdFileCfg(usd_path=TRACK_USD, scale=(1, 1, 1))
        track_cfg.func("/World/Track", track_cfg, translation=(0, 0.0, 0.0), orientation=(0.0, 0.0, 0.0, 1.0))

        wall_cfg = sim_utils.UsdFileCfg(usd_path=WALL_USD, scale=(1, 1, 1))
        wall_cfg.func("/World/Walls", wall_cfg, translation=(0, 0.0, 0.0), orientation=(0.0, 0.0, 0.0, 1.0))

        for set_idx, asset_usd in enumerate(ASSET_SET_USDS):
            asset_cfg = sim_utils.UsdFileCfg(usd_path=asset_usd, scale=(1, 1, 1))
            asset_cfg.func(f"/World/Assets/Set{set_idx}", asset_cfg,
                           translation=(0.0, 0.0, 0.0), orientation=(0.0, 0.0, 0.0, 1.0))


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

        # DISABLED: ramp temporarily removed
        # ramp_cfg = sim_utils.UsdFileCfg(usd_path=RAMP_USD, scale=(0.003, 0.003, 0.003))
        # ramp_cfg.func("/World/Ramps", ramp_cfg, translation=(16, 0.0, 0.0), orientation=(0.0, 0.0, 0.0, 0.0))

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

        self.camera_left  = TiledCamera(self.cfg.camera_left_cfg)
        self.camera_right = TiledCamera(self.cfg.camera_right_cfg)
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

        # ── Debug camera-direction cones (disabled by default) ───────────────────
        # Cone tip at camera sensor origin; base (wide end) along camera look direction.
        # Cloned to all envs automatically.  Toggle: cfg.show_camera_cones = True.
        if self.cfg.show_camera_cones:
            import math as _math
            from pxr import Gf
            base_link_p = "/World/envs/env_0/Robot/mushr_nano/base_link"
            cone_h, cone_r = 0.15, 0.025   # height (m), base radius (m)
            for cam_name, cam_cfg, colour in (
                ("camera_left",  self.cfg.camera_left_cfg,  (0.0, 1.0, 1.0)),  # cyan
                ("camera_right", self.cfg.camera_right_cfg, (1.0, 0.0, 1.0)),  # magenta
            ):
                # Camera forward direction in base_link frame = R(q) · (0,0,1).
                # Camera quaternion is (w, x, y, z) in ROS convention.
                qw, qx, qy, qz = cam_cfg.offset.rot
                fwd_x = 2.0*(qx*qz + qw*qy)
                fwd_y = 2.0*(qy*qz - qw*qx)
                fwd_z = 1.0 - 2.0*(qx*qx + qy*qy)

                # Cone centre = camera_pos + (h/2) * fwd  → tip lands at camera origin.
                px, py, pz = cam_cfg.offset.pos
                ccx = px + 0.5*cone_h*fwd_x
                ccy = py + 0.5*cone_h*fwd_y
                ccz = pz + 0.5*cone_h*fwd_z

                # Quaternion rotating cone's +Y onto fwd.
                # cross((0,1,0), fwd) = (fwd_z, 0, -fwd_x); dot = fwd_y.
                dot      = fwd_y
                cross_len = _math.sqrt(fwd_z**2 + fwd_x**2)
                if cross_len < 1e-6:
                    orient = Gf.Quatf(1.0, 0.0, 0.0, 0.0) if dot > 0 else Gf.Quatf(0.0, 1.0, 0.0, 0.0)
                else:
                    w_q    = _math.sqrt(max(0.0, 0.5*(1.0 + dot)))
                    k      = _math.sqrt(max(0.0, 0.5*(1.0 - dot))) / cross_len
                    orient = Gf.Quatf(w_q, k*fwd_z, 0.0, k*(-fwd_x))

                cone_prim = UsdGeom.Cone.Define(
                    stage, f"{base_link_p}/{cam_name}_debug_cone"
                )
                cone_prim.CreateHeightAttr(cone_h)
                cone_prim.CreateRadiusAttr(cone_r)
                cone_prim.CreateAxisAttr("Y")
                cone_prim.CreateDisplayColorAttr([Gf.Vec3f(*colour)])
                xf = UsdGeom.Xformable(cone_prim.GetPrim())
                xf.ClearXformOpOrder()
                xf.AddTranslateOp().Set(Gf.Vec3d(ccx, ccy, ccz))
                xf.AddOrientOp().Set(orient)
                print(
                    f"[CamCone] {cam_name}: tip=({px:.4f}, {py:.4f}, {pz:.4f})  "
                    f"fwd=({fwd_x:.3f}, {fwd_y:.3f}, {fwd_z:.3f})"
                )

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
        self.scene.sensors["camera_left"]  = self.camera_left
        self.scene.sensors["camera_right"] = self.camera_right
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
                print(f"[Ramp] ENABLED after {self._ramp_enabled} total laps!")
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
            print(f"[Walls] Swapped to WALLS.usd after {self._ramp_enabled} total laps!")
        except Exception as e:
            print(f"[Walls] Swap failed: {e}")

    def _set_active_assets(self, set_idx: int) -> None:
        """Show asset set set_idx; hide all others."""
        try:
            import omni.usd
            from pxr import UsdGeom
            stage = omni.usd.get_context().get_stage()
            for i in range(NUM_ASSET_SETS):
                root = stage.GetPrimAtPath(f"/World/Assets/Set{i}")
                if not root.IsValid():
                    continue
                if i == set_idx:
                    UsdGeom.Imageable(root).MakeVisible()
                else:
                    UsdGeom.Imageable(root).MakeInvisible()
            self._current_asset_set = set_idx
        except Exception as e:
            print(f"[Assets] Visibility toggle failed for set {set_idx}: {e}")

    # ------------------------------------------------------------------
    # Step hooks
    # ------------------------------------------------------------------

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        """Convert actions → Ackermann commands (direct torque control, AWD).

        Throttle (actions[:,0]) ∈ [-1,1]:
            Remapped to [0,1], then applied as wheel torque following a
            linear torque-speed curve (models motor back-EMF):
                T(ω) = throttle_norm × T_stall × max(0, 1 − ω/ω_noload)
            Wheel spin occurs naturally when T > traction limit.

        Latency model: raw actions are pushed into _action_delay_buf; actuators
        receive the command from delay_steps ticks ago.  The observation history
        (_prev_action) still records the undelayed policy output so the network
        can reason about what it recently commanded.
        """
        # On the very first physics step all asset sets are still visible (RTX has
        # already built every BVH during sim init). Now hide all but one randomly
        # chosen set — hiding is cheap, it only removes from the BVH.
        if not self._assets_warmup_done:
            init_set = int(torch.randint(0, NUM_ASSET_SETS, (1,)).item())
            self._set_active_assets(init_set)
            self._assets_warmup_done = True

        # Step-based asset randomisation — decoupled from episode resets.
        self._asset_step_counter += 1
        if self._asset_step_counter >= self.cfg.asset_randomize_every_n_steps:
            self._asset_step_counter = 0
            new_set = int(torch.randint(0, NUM_ASSET_SETS, (1,)).item())
            if new_set != self._current_asset_set:
                self._set_active_assets(new_set)

        raw = actions[:, :2].clamp(-1.0, 1.0)   # [N, 2]  — undelayed policy output

        # Push into delay FIFO and read the oldest (delayed) entry
        self._action_delay_buf = self._action_delay_buf.roll(-1, dims=1)
        self._action_delay_buf[:, -1] = raw
        delayed = self._action_delay_buf[:, 0]   # [N, 2]  — what the actuators act on

        # Policy output [-1, 1] split at 0:
        #   [0, 1]  → drive  (throttle_norm = output directly)
        #   [-1, 0) → brake  (ESC shorts motor terminals; magnitude is irrelevant)
        throttle_norm = delayed[:, 0].clamp(min=0.0)          # drive: [0, 1]
        is_braking    = delayed[:, 0] < 0.0                   # brake: any negative value

        # Per-wheel speed — abs for rolloff, signed for braking direction
        wheel_vel_signed = self.robot.data.joint_vel[:, self._all_throttle_ids]
        wheel_omega      = wheel_vel_signed.abs()

        # Linear torque-speed rolloff: full torque at standstill, zero at no-load speed
        torque_rolloff = (1.0 - wheel_omega / self._omega_noload).clamp(0.0, 1.0)

        # Drive torque: positive effort, forward only
        drive_torque = throttle_norm.unsqueeze(-1) * self._T_stall * torque_rolloff

        # DC-motor zero-power "brake" mode (not float).
        # The ESC shorts both motor terminals only when the throttle command is exactly 0
        # (policy output -1). The resulting back-EMF current creates a braking torque
        # proportional to wheel speed: T = -k * ω. Using signed velocity means the
        # torque always opposes rotation and reaches 0 as ω → 0, so the car cannot
        # reverse even without an extra clamp.
        # At brake_strength=0.3, full-speed braking ≈ 0.6 g deceleration.
        brake_torque = -self.cfg.brake_strength * self._T_stall * (wheel_vel_signed / self._omega_noload)

        per_wheel_torque = torch.where(is_braking.unsqueeze(-1), brake_torque, drive_torque)
        self._wheel_torque_rear  = per_wheel_torque[:, :2]
        self._wheel_torque_front = per_wheel_torque[:, 2:]

        # Steering: direct passthrough, no LPF (matches real hardware — no filter on servo)
        self._filtered_steer = delayed[:, 1]
        steer_ang = self._filtered_steer * self.cfg.max_steer_rad
        self._steer_tan = torch.tan(steer_ang)

        # Shift observation history left and append newest RAW (undelayed) action
        self._prev_action = self._prev_action.roll(-1, dims=1)
        self._prev_action[:, -1] = raw

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
        torque_rear  = self._wheel_torque_rear.clone()
        torque_front = self._wheel_torque_front.clone()
        torque_rear[wall_mask]  = 0.0
        torque_front[wall_mask] = 0.0
        self.robot.set_joint_effort_target(torque_rear,  joint_ids=self._rear_throttle_ids)
        self.robot.set_joint_effort_target(torque_front, joint_ids=self._front_throttle_ids)

        steer_positions = self._steer_tan.unsqueeze(-1).expand(-1, 2).clone()
        steer_positions[wall_mask] = 0.0
        self.robot.set_joint_position_target(steer_positions, joint_ids=self._steer_ids)

    # ------------------------------------------------------------------
    # Depth camera cache  (called once per step, shared by obs + rewards)
    # ------------------------------------------------------------------

    def _process_single_depth(self, camera) -> torch.Tensor:
        """Shared depth pipeline for one TiledCamera → [N, policy_h*policy_w]."""
        import torch.nn.functional as F
        depth_raw = camera.data.output["distance_to_image_plane"]  # [N, H, W, 1]
        depth = depth_raw[..., 0]                                   # [N, H, W]
        depth = torch.where(
            torch.isfinite(depth),
            depth,
            torch.tensor(self.cfg.depth_max_m, device=depth.device),
        )
        h = depth.shape[1]
        top = int(self.cfg.depth_crop_top_fraction    * h)
        bot = h - int(self.cfg.depth_crop_bottom_fraction * h)
        depth = depth[:, top:bot, :]                                # [N, H', W]

        # Edge weights computed on the CLEAN signal so that only real depth
        # discontinuities (not noise spikes) drive the mixed-pixel scatter effect.
        if self.cfg.depth_edge_scatter_prob > 0.0:
            _d4  = depth.unsqueeze(1)                                        # [N,1,H',W]
            _px  = F.pad(_d4, (1, 1, 0, 0), mode="replicate")               # [N,1,H',W+2]
            _py  = F.pad(_d4, (0, 0, 1, 1), mode="replicate")               # [N,1,H'+2,W]
            _dx  = (_px[:, :, :, 2:] - _px[:, :, :, :-2]) * 0.5            # centred ∂/∂x
            _dy  = (_py[:, :, 2:, :] - _py[:, :, :-2, :]) * 0.5            # centred ∂/∂y
            _edge_w = (
                (_dx.abs() + _dy.abs()).squeeze(1)                           # L1 grad [N,H',W]
                / self.cfg.depth_edge_gradient_threshold_m
            ).clamp(0.0, 1.0)

        # ── Noise passes ──────────────────────────────────────────────────────
        if self.cfg.depth_noise_sigma_m > 0.0:
            # Broadband Gaussian noise (models ±2 cm ToF depth accuracy).
            depth = depth + torch.randn_like(depth) * self.cfg.depth_noise_sigma_m

        if self.cfg.depth_invalid_pixel_prob > 0.0:
            # Black-spec speckle: pixel → depth_max_m (far/safe, matches ArduCam masking).
            invalid = torch.rand_like(depth) < self.cfg.depth_invalid_pixel_prob
            depth = depth.masked_fill(invalid, self.cfg.depth_max_m)

        if self.cfg.depth_color_noise_prob > 0.0:
            # Coloured speckle: pixel → uniform random depth in [0, depth_max_m].
            # Models stray reflections / multi-path interference.
            color_mask = torch.rand_like(depth) < self.cfg.depth_color_noise_prob
            depth = torch.where(
                color_mask,
                torch.rand_like(depth) * self.cfg.depth_max_m,
                depth,
            )

        if self.cfg.depth_edge_scatter_prob > 0.0:
            # Mixed-pixel scatter: each edge pixel independently jumps to either 0 m (near)
            # or depth_max_m (far) with probability scaled by the local edge weight.
            # This produces the individual black dots inside objects and coloured dots in the
            # background seen on real ToF sensors — a binary point scatter, not a smooth blur.
            scatter_mask = torch.rand_like(depth) < (self.cfg.depth_edge_scatter_prob * _edge_w)
            scatter_vals = torch.where(
                torch.rand_like(depth) > 0.5,
                torch.full_like(depth, self.cfg.depth_max_m),  # far / black
                torch.zeros_like(depth),                        # near / red
            )
            depth = torch.where(scatter_mask, scatter_vals, depth)

        if self.cfg.depth_edge_blur_sigma_m > 0.0 and self.cfg.depth_edge_scatter_prob > 0.0:
            # Faint smooth halo on top of the scatter — softens the hard geometric edge
            # slightly, matching the residual blur seen on real ToF frames.
            depth = depth + torch.randn_like(depth) * (self.cfg.depth_edge_blur_sigma_m * _edge_w)

        depth = depth.clamp(0.01, self.cfg.depth_max_m)
        depth_norm = 1.0 - depth / self.cfg.depth_max_m
        depth_4d = F.interpolate(
            depth_norm.unsqueeze(1),
            size=(self._policy_depth_h, self._policy_depth_w),
            mode="bilinear",
            align_corners=False,
        )
        return depth_4d.squeeze(1).reshape(self.num_envs, -1)

    def _update_depth_cache(self) -> None:
        """Process both depth cameras and write self._camera_left_obs / _camera_right_obs."""
        current_step = int(self.episode_length_buf.max().item())
        if self._last_camera_update_step == current_step:
            return
        self._last_camera_update_step = current_step
        self._camera_left_obs  = self._process_single_depth(self.camera_left)
        self._camera_right_obs = self._process_single_depth(self.camera_right)

    # ------------------------------------------------------------------
    # Observations
    # ------------------------------------------------------------------

    def _get_observations(self) -> dict:
        self._update_depth_cache()

        parts = [self._camera_left_obs, self._camera_right_obs]
        if self.cfg.include_prev_action:
            parts.append(self._prev_action.view(self.num_envs, -1))   # [N, H*2]

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

        # Average-speed bonus: accumulate positive forward velocity each step and
        # reward the running mean, so the policy must sustain speed, not just burst.
        self._cumulative_forward_vel += lin_vel_b.clamp(min=0.0)
        avg_vel = self._cumulative_forward_vel / self.episode_length_buf.float().clamp(min=1)
        r_forward = self.cfg.forward_weight * (avg_vel / self.cfg.max_velocity_m_s)

        pos_xy = self.robot.data.root_pos_w[:, :2]

        # ── Wheel slip (AWD — all 4 wheels) ──────────────────────────────────────
        wheel_vel  = self.robot.data.joint_vel[:, self._all_throttle_ids].mean(dim=-1) * WHEEL_RADIUS
        slip_speed = torch.abs(wheel_vel - lin_vel_b)
        r_slip     = -self.cfg.slip_weight * slip_speed / self.cfg.max_velocity_m_s * (~self._on_ramp).float()

        # ── Steering deadzone penalty ────────────────────────────────────────────
        steer_norm    = torch.abs(self._filtered_steer).clamp(0.0, 1.0)
        steer_excess  = (steer_norm - self.cfg.steer_deadzone).clamp(min=0.0)
        steer_excess_norm = steer_excess / max(1.0 - self.cfg.steer_deadzone, 1e-6)
        r_steer_shape = -self.cfg.steer_deadzone_weight * steer_excess_norm

        # ── Ramp reward ──────────────────────────────────────────────────────────
        r_ramp = (
            self.cfg.ramp_reward_weight
            * self._on_ramp.float()
            * (lin_vel_b / self.cfg.max_velocity_m_s).clamp(0.0, 1.0)
        )

        # ── Ramp speed penalty ───────────────────────────────────────────────────
        # Penalise every step the robot is inside the ramp XY bbox OR within
        # ramp_approach_radius_m of it, and going faster than ramp_speed_threshold_m_s.
        if self._ramp_xy_min is not None:
            in_ramp_xy = (
                (pos_xy[:, 0] >= self._ramp_xy_min[0]) & (pos_xy[:, 0] <= self._ramp_xy_max[0]) &
                (pos_xy[:, 1] >= self._ramp_xy_min[1]) & (pos_xy[:, 1] <= self._ramp_xy_max[1])
            )
            # Distance from robot to nearest point on bbox (0 when inside)
            nearest_x = pos_xy[:, 0].clamp(self._ramp_xy_min[0], self._ramp_xy_max[0])
            nearest_y = pos_xy[:, 1].clamp(self._ramp_xy_min[1], self._ramp_xy_max[1])
            dx = pos_xy[:, 0] - nearest_x
            dy = pos_xy[:, 1] - nearest_y
            dist_to_ramp_bbox = torch.sqrt(dx * dx + dy * dy)
            near_ramp_xy = dist_to_ramp_bbox <= self.cfg.ramp_approach_radius_m
            in_ramp_xy = in_ramp_xy | near_ramp_xy
        else:
            in_ramp_xy = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        ramp_speed_excess = (lin_vel_b - self.cfg.ramp_speed_threshold_m_s).clamp(min=0.0)
        r_airborne = -self.cfg.ramp_speed_penalty_weight * ramp_speed_excess * in_ramp_xy.float()

        r_alive = torch.full((self.num_envs,), self.cfg.alive_weight, device=self.device)

        total = r_alive + r_distance + r_forward + r_slip + r_steer_shape + r_ramp + r_airborne

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
            (lin_vel_b < (self.cfg.slow_stop_speed_fraction * self.cfg.max_velocity_m_s))
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

        super()._reset_idx(env_ids)

        n = len(env_ids)
        default_root_state = self.robot.data.default_root_state[env_ids].clone()
        default_root_state[:, :3] += self.scene.env_origins[env_ids]

        speed_vec = (
            torch.rand(n, device=self.device)
            * (self.cfg.spawn_vel_max_m_s - self.cfg.spawn_vel_min_m_s)
            + self.cfg.spawn_vel_min_m_s
        )
        sz        = 0.002   # just above ground — constant for all spawn modes
        env_ids_t = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)

        pts = list(self.cfg.preset_spawn_points)
        if self.cfg.duplicate_with_flipped_yaw:
            pts = pts + [(x, y, theta + 180.0) for x, y, theta in pts]
        idx  = torch.randint(0, len(pts), (n,), device="cpu").tolist()
        xs   = torch.tensor([pts[i][0] * self.cfg.spawn_x_scale for i in idx], dtype=torch.float32, device=self.device)
        ys   = torch.tensor([pts[i][1] * self.cfg.spawn_y_scale for i in idx], dtype=torch.float32, device=self.device)
        yaw  = torch.tensor(
            [math.radians(pts[i][2]) for i in idx], dtype=torch.float32, device=self.device
        )
        if self.cfg.spawn_yaw_jitter_rad > 0.0:
            yaw = yaw + (torch.rand(n, device=self.device) - 0.5) * 2.0 * self.cfg.spawn_yaw_jitter_rad
        spawn_xy = torch.stack([xs, ys], dim=-1)

        default_root_state[:, 0] = spawn_xy[:, 0]
        default_root_state[:, 1] = spawn_xy[:, 1]
        default_root_state[:, 2] = sz

        zeros = torch.zeros(n, device=self.device)
        dq = quat_from_euler_xyz(zeros, zeros, yaw)
        default_root_state[:, 3:7] = dq

        default_root_state[:, 7:]  = 0.0
        default_root_state[:, 7]   = speed_vec * torch.cos(yaw)
        default_root_state[:, 8]   = speed_vec * torch.sin(yaw)

        self.robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self.robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)

        # Body velocity already seeded above via write_root_velocity_to_sim
        self._filtered_steer[env_ids] = 0.0
        self._steer_integral[env_ids] = 0.0
        self._prev_steer[env_ids]     = 0.0
        self._prev_action[env_ids]    = 0.0         # clears full [H, 2] history
        self._action_delay_buf[env_ids] = 0.0       # clears latency FIFO

        self._ramp_ascended[env_ids]     = False
        self._wall_ever_touched[env_ids] = False
        self._cumulative_forward_vel[env_ids] = 0.0

        joint_pos = self.robot.data.default_joint_pos[env_ids]
        joint_vel = torch.zeros_like(joint_pos)
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

