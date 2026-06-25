"""MuSHR nano v2 RC car — cone-track navigation environment (DirectRLEnv).

Observation space:
    [0:N]        flattened depth image from LEFT ArduCam TOF camera (+35° yaw), after crop.
                 Normalized: 1.0 = object at 0 m (danger), 0.0 = object at depth_max_m (clear).
    [N:2N]       flattened depth image from RIGHT ArduCam TOF camera (−35° yaw), same pipeline.
    [2N]         average wheel surface speed ∈ [0, 1], normalized by max_velocity_m_s.
                 Computed as mean(|joint_vel[all 4 throttle joints]|) × WHEEL_RADIUS / max_velocity_m_s.
                 Only appended when include_wheel_vel=True.
                 Matches real hardware: wheel encoders give rotation rate directly.

Camera profile (ArduCam TOF sensor for Raspberry Pi):
    - Sensor: Time-of-Flight depth, 240×180 pixels at 30 fps
    - Diagonal FOV: 70°  →  HFOV ≈ 58.5°, VFOV ≈ 45.6°  (4:3 aspect, pinhole)
    - Depth accuracy: ±2 cm (simulated via Gaussian noise σ ≈ 0.007 m)
    - Policy runs at 30 Hz, synced with camera (sim 240 Hz, decimation=8)

Action space (2):
    [0]  throttle command in [-1, 1] — forward-only (the car never reverses):
           (0, 1]  → DRIVE  torque = drive_torque × (1 − ω/(throttle·ω_noload))  (DC curve)
            0      → COAST  (zero torque, free roll)
           [-1, 0) → BRAKE  torque = −(brake_torque_base + brake_torque_gain × level),
                     level = |throttle| × brake_throttle_scale (policy −1 = the slip
                     ceiling).  Opposes motion, fades to 0 near standstill → stops then
                     coasts, never spinning the wheels backward.
    [1]  steering command in [-1, 1]  → scaled to [-max_steer_rad, +max_steer_rad]

Actuation (Ackermann / AWD, effort-controlled DC motor):
    - All 4 wheels receive a torque (Nm) via set_joint_effort_target.  Drive follows a
      DC torque-speed curve (tops out at the no-load speed max_velocity_m_s); braking is
      an independent, throttle-graded reverse torque.  Wheel slip occurs naturally when
      the commanded torque exceeds the PhysX contact-friction limit (μ·N).
    - Front wheel steer receives position target (rad) = tan(steering_angle)
      (tan-convention matches the joint representation in mushr_nano_v2.usd)

Why depth camera vs RGB cone mask:
    The depth camera gives the policy direct metric information about wall and obstacle
    proximity.  A shallow corner looks identical in RGB (just more cone pixels) but
    in depth the wall is visibly close.  This makes braking-before-corners learnable:
    the agent can correlate "high depth values filling the frame → wall approaching →
    reduce throttle" without needing to understand color or stripe classification.

Termination conditions:
    - Wall contact: any horizontal normal force on robot links above wall_contact_force_thresh_n
    - Robot tilt > 72° from vertical (up_z < 0.3)
    - Slow driving: speed < 2% of vel_cap for > ~1.5 s after episode start

TUNING NOTES (first run):
    - All adjustable parameters are in ConeTrackEnvCfg in one place below.
    - Spawn position: set preset_spawn_points in cfg.
    - Camera crop: depth_crop_top_fraction, depth_crop_bottom_fraction.
    - Drivetrain: drive_torque (acceleration), brake_torque_gain (braking strength),
      max_velocity_m_s (no-load top speed), brake_release_omega, suspension_stiffness.
"""

from __future__ import annotations

import math
import os
from collections.abc import Sequence

import isaaclab.sim as sim_utils
import torch
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import Articulation, ArticulationCfg
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
_REPO_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), "../../../../"))
_EXPORT_DIR = os.path.join(_REPO_ROOT, "CAD", "Track", "Export")

MUSHR_USD = os.path.join(_ASSETS_DIR, "mushr_nano_v2.usd")
TRACK_USD = os.path.join(_EXPORT_DIR, "Track.usdc")
WALL_USD = os.path.join(_EXPORT_DIR, "Walls.usdc")

# Decorative asset sets — purely visual objects the depth cameras see.
# Index NUM_ASSET_SETS means "no decorations" (all sets hidden).
ASSET_SET_USDS = [
    os.path.join(_EXPORT_DIR, "Assets1.usdc"),
    os.path.join(_EXPORT_DIR, "Assets2.usdc"),
    os.path.join(_EXPORT_DIR, "Assets3.usdc"),
    os.path.join(_EXPORT_DIR, "Assets4.usdc"),
    os.path.join(_EXPORT_DIR, "Assets5.usdc"),
]
NUM_ASSET_SETS = len(
    ASSET_SET_USDS
)  # randomisation draws from [0, NUM_ASSET_SETS] inclusive

# ---------------------------------------------------------------------------
# Physical constants — MuSHR nano v2 (fixed by the USD / hardware)
# These match mushr_nano_v2.usd geometry and are NOT user-adjustable without
# also modifying the robot USD.
# ---------------------------------------------------------------------------
WHEEL_RADIUS = 0.037  # m  (82 mm diameter wheels)

TRACK_SCALE = 0.85

# ---------------------------------------------------------------------------
# ArduCam TOF camera optics  (derived from 70° diagonal FOV, 240×180 sensor)
#   f (pixels)        = diag_pixels / 2 / tan(diag_half) = 150 / tan(35°) ≈ 214.2 px
#   HFOV              = 2 * arctan(120 / 214.2) ≈ 58.52°
#   VFOV              = 2 * arctan(90  / 214.2) ≈ 45.57°
#
# For Isaac PinholeCameraCfg we pick an arbitrary focal_length_mm and derive
# the aperture so the ratio h_aperture / focal_length = 2 * tan(HFOV/2).
# ---------------------------------------------------------------------------
_TOF_DIAG_FOV_DEG = 70.0
_TOF_DIAG_PIXELS = math.sqrt(240**2 + 180**2)  # 300 px
_TOF_FOCAL_PX = _TOF_DIAG_PIXELS / 2.0 / math.tan(math.radians(_TOF_DIAG_FOV_DEG / 2.0))
_TOF_HFOV_DEG = 2.0 * math.degrees(math.atan2(120.0, _TOF_FOCAL_PX))  # ≈ 58.52°
_TOF_VFOV_DEG = 2.0 * math.degrees(math.atan2(90.0, _TOF_FOCAL_PX))  # ≈ 45.57°
_TOF_FOCAL_MM = 4.0  # reference focal length (mm); aperture ratio is what matters
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
    OBSERVATION     — optional wheel-speed channel
    REWARD WEIGHTS  — all reward scale factors
    """

    # ── Core env parameters ────────────────────────────────────────────────
    decimation: int = 8  # policy at ~30 Hz  (sim 240 Hz / 8)
    episode_length_s: float = 90.0  # 180 s × 30 Hz = 5 400 steps
    action_space: int = 2
    # observation_space is patched in ConeTrackEnv.__init__ after computing
    # the exact cropped depth image size.  The placeholder below keeps
    # @configclass happy; the real value is written before super().__init__().
    observation_space: int = 1  # patched in __init__
    state_space: int = 0

    sim: SimulationCfg = SimulationCfg(dt=1.0 / 240.0, render_interval=8)

    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=26,
        env_spacing=0.0,
        replicate_physics=True,
    )

    # ════════════════════════════════════════════════════════════════════════
    # CAMERA — ArduCam TOF (240×180, 30 fps, 70° diagonal FOV)
    # ════════════════════════════════════════════════════════════════════════
    # Crop fractions (fraction of raw pixel rows discarded before the policy):
    #   top  → removes ceiling / sky (depth = max / inf)
    #   bottom → removes near-ground clutter and the robot's own body shadow
    depth_crop_top_fraction: float = 0.375
    depth_crop_bottom_fraction: float = 0.45

    # Gaussian depth noise: σ ≈ 0.007 m gives a ±2 cm (3σ) error band,
    # matching the ArduCam TOF spec.  Set to 0.0 to disable.
    depth_noise_sigma_m: float = 0.007

    # Random invalid-pixel dropout: each pixel is independently zeroed with this
    # probability, replicating the black-spec speckle noise visible in real ArduCam
    # TOF output.  Affected pixels are set to depth_max_m (treated as far/safe by the
    # policy, matching the real camera's low-confidence masking behaviour).
    # Visible as black dots in both windows of visualize.py.  Set to 0.0 to disable.
    depth_invalid_pixel_prob: float = 0.008

    # Random-depth (coloured) speckle: each pixel is set to a uniform random depth in
    # [0, depth_max_m] with this probability.  Models stray reflections and multi-path
    # interference that appear as bright or mid-range dots (unlike depth_invalid_pixel_prob
    # which always sets the pixel to the far/black depth_max_m value).
    # Set to 0.0 to disable.
    depth_color_noise_prob: float = 0.006

    # Edge blur noise — simulates the mixed-pixel / depth-bleeding artefact common in
    # real ToF sensors: pixels whose sensing area straddles a depth discontinuity report
    # a value somewhere between the two depths rather than either clean value.
    # Implementation: L1 gradient |∇depth| is computed on the CLEAN cropped depth frame
    # (before any other noise), then Gaussian noise with
    #   std = depth_edge_noise_sigma_m × clamp(|∇depth| / depth_edge_gradient_threshold_m, 0, 1)
    # is added.  Only pixels at depth edges receive meaningful extra noise.
    # Set depth_edge_noise_sigma_m = 0.0 to disable entirely.
    depth_edge_gradient_threshold_m: float = (
        0.15  # gradient (m/px) that gives edge_weight = 1
    )
    # Mixed-pixel scatter: edge pixels randomly jump to 0 m or depth_max_m (binary wrong-side
    # values — models the individual black/red dots seen on real ToF sensors at boundaries).
    depth_edge_scatter_prob: float = 0.25
    # Small continuous Gaussian blur layered on top of the scatter, adds the faint soft halo
    # visible around object edges on the real camera alongside the point scatter.
    depth_edge_blur_sigma_m: float = 0.085

    # Depth range for normalization.  Readings > depth_max_m are clamped.
    # Must match autopilot.py MAX_DISTANCE / 1000.  ArduCam supports 2000 mm or 4000 mm only.
    depth_max_m: float = 2.0

    # ── Policy image resolution ────────────────────────────────────────────
    # The policy now contains a lightweight CNN, so keep more spatial structure
    # than the old MLP path while avoiding excessive rollout/activation memory.
    # policy_depth_height < 0 → auto-compute from cropped aspect ratio.
    policy_depth_width: int = 160
    policy_depth_height: int = -1

    # ════════════════════════════════════════════════════════════════════════
    # SPAWN — initial robot position, heading, and speed
    # ════════════════════════════════════════════════════════════════════════
    # Grace period after each reset: wall / flip / stuck terminations are suppressed
    # for this many policy steps so that PhysX contact forces can settle.
    # 10 steps ≈ 333 ms at 30 Hz — enough for any spawn-physics transient.
    spawn_grace_steps: int = 10

    # ── Spawn mode ────────────────────────────────────────────────────────────
    #   Each reset picks randomly from preset_spawn_points.
    #   Each entry is (x_m, y_m, theta_deg).
    #   spawn_yaw_jitter_rad is applied.
    preset_spawn_points: list = [
        # (x_m, y_m, theta_deg) — raw Blender coordinates
        (0.756, 5.0, 20.8),
        (3.91, 3.31, -156.0),
        (3.62, 1.97, -154.48),
        # (0.0,  0.0, 0.0),
        (-5.46, 3.7, 0.0),
        (-3.6, 4.7, -217.86),
        # (-5.0,  0.0, 180.0),
        # (3.4,  0.0, 0.0),
    ]
    # Coordinate transform applied to every spawn point at runtime.
    # Blender exports X/Y inverted relative to Isaac Sim's frame, so both are -1.
    spawn_x_scale: float = -1.15 * TRACK_SCALE
    spawn_y_scale: float = -1.15 * TRACK_SCALE

    duplicate_with_flipped_yaw: bool = True  # if True, each (x, y, θ) spawns two episodes: one with +180deg rotation around z
    spawn_yaw_jitter_rad: float = 0.1  # ± heading noise applied in both modes

    # One-time bonus the first time the car moves farther than spawn_escape_radius_m
    # from its spawn position, scaled by how fast the car is moving at that moment:
    #   reward = spawn_escape_reward × (current_speed / max_velocity_m_s)
    # At max speed this gives the full bonus; at crawl speed it gives nearly nothing.
    # Teaches aggressive initial acceleration without rewarding slow drift.
    # Set spawn_escape_reward=0 to disable.
    spawn_escape_radius_m: float = 0.5
    spawn_escape_reward: float = 0.0

    # ════════════════════════════════════════════════════════════════════════
    # SPAWN SPEED — initial robot velocity at each episode reset
    # ════════════════════════════════════════════════════════════════════════
    spawn_vel_min_m_s: float = 0.0  # minimum spawn speed (m/s)
    # Curriculum: spawn across a velocity range so the policy practises control at
    # speed instead of over-sampling the slow start regime (which reinforces the
    # cautious / slow-acceleration local optimum).  play.py forces this to 0.
    spawn_vel_max_m_s: float = 0.0  # maximum spawn speed (m/s)

    # ════════════════════════════════════════════════════════════════════════
    # DYNAMICS — effort-controlled DC-motor drivetrain (AWD), tuned to real car
    #
    # Each step the policy throttle ∈ [-1, 1] is turned into a per-wheel TORQUE
    # (applied via set_joint_effort_target).  Three zones — forward-only:
    #
    #   throttle > 0  DRIVE: T = drive_torque · clamp(1 − ω/ω_noload(throttle), 0, 1)
    #                 where ω_noload(throttle) = throttle · max_velocity_m_s / r.
    #                 A DC-motor torque-speed curve: full drive_torque at standstill,
    #                 tapering to 0 at the (throttle-scaled) no-load speed.  So full
    #                 throttle tops out at max_velocity_m_s; half throttle at half that.
    #   throttle = 0  COAST: T = 0 (free roll; ground/contact losses only).
    #   throttle < 0  BRAKE: T = −brake_torque_gain · |throttle|, opposing rotation,
    #                 faded to 0 below brake_release_omega so the car can never be
    #                 driven backwards (brake → coast at standstill).
    #
    # Drive and brake torques are INDEPENDENT (the real car brakes far harder than it
    # accelerates), and traction-induced wheel slip emerges naturally from the PhysX
    # contact friction when the commanded torque exceeds μ·N (e.g. hard braking).
    #
    # TUNABLE DRIVETRAIN PARAMETERS — single source of truth.  Edit them here;
    # __post_init__ / _pre_physics_step read them, so train / play / visualize /
    # acceleration_test all pick them up automatically (none override these).
    # ════════════════════════════════════════════════════════════════════════
    car_mass_kg: float = 1.27  # total robot mass (kg); USD per-body masses are scaled to this in __init__
    max_velocity_m_s: float = 6.7  # no-load top speed (m/s); full-throttle drive torque → 0 here. Also the reward normaliser
    drive_torque: float = 0.068  # per-wheel stall drive torque (Nm) → sets acceleration (gentle, low-slip)
    # Braking torque per wheel = brake_torque_base + brake_torque_gain·|throttle| (Nm), opposing motion.
    # Fit to the real car: F_total ≈ 2.5 N + 26.6·|throttle|  →  per wheel 0.023 + 0.246·|throttle|.
    # The base term is the motor's regen/drag braking present at any brake input; the gain term is
    # the throttle-graded component.  Hard braking (≳|0.3|) exceeds μ·N → natural wheel slip.
    brake_torque_base: float = 0.028  # constant brake torque/wheel when braking (Nm)
    brake_torque_gain: float = (
        0.310  # additional brake torque/wheel per unit braking level (Nm)
    )
    # Policy brake remap: a policy throttle of −1 maps to this braking level (the value that
    # used to require throttle −0.5).  The tyres already slip at this level, so braking harder
    # buys no extra deceleration — capping it here gives the policy a more useful action range.
    # Effective braking level = |throttle| × brake_throttle_scale, fed into base + gain·level.
    brake_throttle_scale: float = (
        0.4  # |throttle|=1 → braking level 0.4 (the slip ceiling)
    )
    brake_release_omega: float = 2.0  # wheel speed (rad/s ≈ 0.07 m/s) below which braking fades to coast → no reverse
    # Tyre–ground friction (sets traction / wheel-slip onset).  Combined with the bound wheel-rubber
    # material (μ=1.0, "multiply") these are the effective μ.  Tuned so the car grips up to ≈|0.3|
    # braking and only slips under the hardest braking (−0.5), matching the real car.
    ground_static_friction: float = 2.0  # static μ (grip before slip); traction ≈ μ·m·g
    ground_dynamic_friction: float = 1.6  # sliding μ (once a wheel breaks traction)
    suspension_stiffness: float = (
        1e6  # suspension-joint lock stiffness (rigid; stops energy-sapping travel)
    )
    max_steer_rad: float = 0.488  # ≈28° max front-wheel steer angle

    # Action latency simulation — all actuator commands are delayed by this many
    # milliseconds before reaching the wheels/steering.  At 30 Hz each step ≈ 33 ms,
    # so 100 ms ≈ 3 policy steps.  Set to 0.0 to disable.
    action_delay_ms: float = 33.0

    # ════════════════════════════════════════════════════════════════════════
    # TERMINATION THRESHOLDS
    # ════════════════════════════════════════════════════════════════════════
    # Horizontal contact force on any robot link that counts as a wall hit.
    wall_contact_force_thresh_n: float = 1.0  # Newtonsgz

    # slow_stop_speed_fraction — episode terminates when speed falls below this
    # (catches truly stuck / reversed agents; keep low so normal braking is fine)
    slow_stop_speed_fraction: float = 0.03  # terminate if speed < 3 % of vel_cap

    # ════════════════════════════════════════════════════════════════════════
    # OBSERVATION EXTRAS
    # ════════════════════════════════════════════════════════════════════════
    # Append the current average wheel surface speed to the observation.
    # Normalized: 0 = stopped, 1 = max_velocity_m_s.
    # This is the primary speed signal — essential for braking-before-corners.
    # On real hardware: read wheel encoder RPM, average all 4, multiply by WHEEL_RADIUS,
    # divide by max_velocity_m_s, clamp to [0, 1].
    include_wheel_vel: bool = True

    # ════════════════════════════════════════════════════════════════════════
    # REWARD WEIGHTS
    # ════════════════════════════════════════════════════════════════════════
    # Constant reward per step the agent stays alive (not crashed / stuck).
    alive_weight: float = 0.06  # 0.04

    # Distance: rewards physical metres covered each step (v × dt).
    distance_weight: float = 4.0

    # Average-speed bonus: every step rewards the episode's mean forward speed
    # from reset through the current step, normalized by max_velocity_m_s.
    average_speed_weight: float = 0.0  # redundant with distance + non-Markovian; off

    # Steering deadzone penalty: only penalises steering beyond the deadzone.
    steer_deadzone_weight: float = 0.003
    steer_deadzone: float = 0.05  # fraction of max_steer_rad that is free

    # Steering-rate (smoothness) penalty: penalises the change in steering command
    # between consecutive policy steps, r = -weight × |steer_t − steer_{t-1}|.
    # Directly suppresses the jerky servo chatter; encourages smooth, committed lines.
    steer_rate_weight: float = 0.005

    # Throttle-rate penalty: penalises the change in throttle/brake command
    # between consecutive policy steps to suppress banging between full throttle and brake.
    throttle_rate_weight: float = 0.007

    # Wheel slip penalty: penalises the difference between theoretical wheel speed and actual car speed.
    slip_weight: float = 2.0

    # Proximity speed penalty: penalises going fast when a wall is close along the
    # car's current velocity direction.  Each step a 2D ray is cast from the car
    # along its world-frame linear velocity vector against /World/Walls (the same
    # invisible mesh used for termination — visibility, collision, and termination
    # behaviour are untouched, only the cached 2D footprint is read once at startup).
    # The closest intersection along the ray gives `dist_m`; the penalty is then
    #     r = -weight × speed_norm × max(0, 1 − dist_m / proximity_lookahead_m)
    # so it is largest when the car is fast AND the wall is close in front of it
    # (corner entry).  Set to 0.0 to disable entirely.
    proximity_speed_weight: float = 0.01
    proximity_lookahead_m: float = 2.0  # walls farther than this give zero signal

    # Braking reward V2: dense, instantaneous, no session state.
    #   r_braking_v2 = weight × forward_speed² × |brake_input|       (only when throttle < 0)
    # Quadratic in speed: braking at low speeds is nearly free reward-wise, while
    # braking at high speed delivers a much larger payout — incentivises braking
    # specifically where it matters (corner entries) rather than continuously.
    #   At v = max_velocity, |brake| = 1 → r_braking_v2 = weight × v_max²
    #   r_distance at v = max_velocity   ≈ distance_weight × dt × 1.0
    # With weight = 0.1, v_max = 8 m/s   → r_braking_v2 = 6.4/step at full brake at top speed.
    # At v = 2 m/s the same brake input yields only 0.4/step — 16× weaker than at top speed.
    braking_v2_weight: float = 0.0

    # ── Lap completion reward ────────────────────────────────────────────────
    # Sparse reward fired when the agent crosses the start line (the line through
    # the spawn point perpendicular to the spawn heading) while back inside the
    # `lap_return_radius_m` start circle, after having left that circle.  Reward scales
    # inversely with lap time, so faster laps yield much more reward:
    #   r_lap = lap_reward_weight × lap_max_time_s / clamp(lap_time_s, lap_min, lap_max)
    # At base=500 and a 30 s lap → +500.  A 10 s lap → +1500.  Designed to be the
    # primary improvement driver once the agent can complete laps, dwarfing the
    # dense distance reward (~25/sec at top speed → ~750 per 30 s lap).
    # Until the agent can actually finish a lap this reward stays at 0, so it has
    # no effect on early training.
    lap_reward_weight: float = 8.0  # rescaled to the new O(10) return scale; pays
    # more for faster laps (lap_max/lap_time, 20-40 s)
    lap_return_radius_m: float = (
        0.3  # start circle: leaving it starts the lap timer; crossing the
    )
    # start line while back inside closes the lap
    lap_min_time_s: float = 20.0  # lower clamp (caps maximum reward per lap)
    lap_max_time_s: float = 50.0  # upper clamp & numerator anchor

    # How often to advance to the next asset set in the round-robin cycle (in policy steps).
    # Set to num_steps_per_env / (NUM_ASSET_SETS * 2) so every set appears exactly twice
    # per PPO iteration.  At num_steps_per_env=2000: 2000 / (5*2) = 200.
    asset_randomize_every_n_steps: int = 200

    # ════════════════════════════════════════════════════════════════════════
    # HARDWARE CONFIG  (robot articulation + camera sensor)
    # ════════════════════════════════════════════════════════════════════════

    robot_cfg: ArticulationCfg = ArticulationCfg(
        prim_path="/World/envs/env_.*/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=MUSHR_USD,
            scale=(0.755, 0.755, 0.755),
            activate_contact_sensors=True,
            # NOTE: no MassPropertiesCfg here.  A blanket mass=X override stamps the SAME
            # mass onto every one of the 17 rigid bodies (→ 17× too heavy: 21.6 kg for a
            # 1.27 kg car), crippling acceleration.  Instead the USD-native per-body mass
            # distribution (~4.11 kg total) is scaled to car_mass_kg in __init__.
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
            pos=(0, 0, 0),  # placeholder — overridden every episode in _reset_idx
            joint_pos={
                "back_left_wheel_throttle": 0.0,
                "back_right_wheel_throttle": 0.0,
                "front_left_wheel_throttle": 0.0,
                "front_right_wheel_throttle": 0.0,
                "front_left_wheel_steer": 0.0,
                "front_right_wheel_steer": 0.0,
                "front_left_wheel_suspension": 0.0,
                "front_right_wheel_suspension": 0.0,
                "back_left_wheel_suspension": 0.0,
                "back_right_wheel_suspension": 0.0,
            },
        ),
        # Only the (fixed) steering actuator is defined here.  The velocity-controlled
        # DC-motor "throttle" group and the rigid "suspension" group are built in
        # __post_init__ from the tunable scalar fields above, so they have a single
        # source of truth.
        actuators={
            "steering": ImplicitActuatorCfg(
                joint_names_expr=["front_left_wheel_steer", "front_right_wheel_steer"],
                stiffness=100.0,
                damping=10.0,
                # Velocity limit matches real servo: full sweep (2×0.488 rad) in 150 ms
                # → 0.976 / 0.150 = 6.51 rad/s.  PhysX enforces this as a hard cap.
                velocity_limit_sim=6.51,
                effort_limit_sim=3.2,
            ),
        },
    )

    def __post_init__(self):
        """Build the drivetrain actuators from the tunable scalar fields so that a
        single edit in this cfg propagates to every script that uses the env."""
        # Throttle wheels: pure effort (torque) control.  Zero stiffness/damping so the
        # per-wheel torque computed in _pre_physics_step (DC drive curve + graded brake)
        # is applied verbatim via set_joint_effort_target.  effort_limit_sim is set well
        # above any commanded torque so PhysX never clips it; real traction limiting comes
        # from the ground contact friction (slip emerges when torque exceeds μ·N).
        self.robot_cfg.actuators["throttle"] = ImplicitActuatorCfg(
            joint_names_expr=[".*_wheel_throttle"],
            stiffness=0.0,
            damping=0.0,
            friction=0.0,
            effort_limit_sim=5.0,
            velocity_limit_sim=1e9,
        )
        # Suspension joints locked rigid.  mushr_nano_v2.usd ships these joints; left
        # unconfigured, Isaac Lab keeps their raw USD drive gains (compliant + dissipative)
        # which bleeds energy and caps top speed.  High stiffness welds them at zero travel.
        self.robot_cfg.actuators["suspension"] = ImplicitActuatorCfg(
            joint_names_expr=[".*_wheel_suspension"],
            stiffness=self.suspension_stiffness,
            damping=0.0,
            friction=0.0,
            effort_limit_sim=1e9,
            velocity_limit_sim=1e9,
        )

    # Two ArduCam TOF depth cameras:
    #   Position: 122 mm forward of base_link, ±50 mm left/right, 70 mm above base_link.
    #   Orientation: ±27° yaw outward only — no pitch.
    #
    # Yaw is 27° (< HFOV/2 ≈ 29.26°) so the two camera frusta OVERLAP by ~4.5°
    # dead-ahead, eliminating the forward blind cone that ±30° left between them.
    # The car can now see straight ahead — needed to sense the wall at the end of
    # a straight and to accelerate confidently out of the spawn.
    # NOTE: this is a physical mounting angle — the real cameras must be remounted
    # to ±27° for sim-to-real transfer to hold (autopilot.py uses the same pipeline).
    #
    # Rotation derivation (all quaternions in (w, x, y, z) ROS convention):
    #   Start: q_forward = (-0.5, 0.5, -0.5, 0.5)  (camera level, pointing fwd in ROS)
    #   Yaw  : q_yaw_z(±27°) * q_forward
    #             left  (+27°): (-0.6029, 0.6029, -0.3695, 0.3695)
    #             right (-27°): (-0.3695, 0.3695, -0.6029, 0.6029)
    #   Result forward vectors in base_link frame:
    #             left : (cos27°, sin27°, 0) = (0.891, 0.454, 0.000)
    #             right: (cos27°, -sin27°, 0) = (0.891, -0.454, 0.000)
    camera_left_cfg: TiledCameraCfg = TiledCameraCfg(
        prim_path="/World/envs/env_.*/Robot/mushr_nano/base_link/camera_left",
        update_period=8.0
        / 240.0,  # render only at policy steps (decimation=8, sim=240Hz)
        # Render at the sensor's native 240x180 to match the real ArduCam TOF.
        # All noise effects are applied at this full resolution before cropping
        # and policy-side resizing, mirroring how the real camera produces noisy depth.
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
            pos=(
                0.185,
                0.04,
                -0.008,
            ),  # 122 mm fwd, 50 mm left, 70 mm up from base_link
            rot=(-0.6029, 0.6029, -0.3695, 0.3695),  # 27° yaw left, no pitch
            convention="ros",
        ),
        debug_vis=False,
    )
    camera_right_cfg: TiledCameraCfg = TiledCameraCfg(
        prim_path="/World/envs/env_.*/Robot/mushr_nano/base_link/camera_right",
        update_period=8.0
        / 240.0,  # render only at policy steps (decimation=8, sim=240Hz)
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
            pos=(
                0.185,
                -0.04,
                -0.008,
            ),  # 122 mm fwd, 50 mm right, 70 mm up from base_link
            rot=(-0.3695, 0.3695, -0.6029, 0.6029),  # 27° yaw right, no pitch
            convention="ros",
        ),
        debug_vis=False,
    )

    # Live debug view in training — use scripts/rsl_rl/visualize.py instead.
    show_camera_debug: bool = False
    camera_debug_env_id: int = 0
    camera_debug_interval: int = 2

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
        # 2. Resize: bilinear resize to policy_depth_width × policy_depth_height.
        cam_h = cfg.camera_left_cfg.height  # 180
        cam_w = cfg.camera_left_cfg.width  # 240
        remaining = max(
            0.0, 1.0 - cfg.depth_crop_top_fraction - cfg.depth_crop_bottom_fraction
        )
        cropped_h = max(1, int(cam_h * remaining))
        cropped_w = cam_w

        self._policy_depth_w = cfg.policy_depth_width
        if cfg.policy_depth_height < 0:
            self._policy_depth_h = max(
                1, round(cfg.policy_depth_width * cropped_h / cropped_w)
            )
        else:
            self._policy_depth_h = cfg.policy_depth_height

        depth_obs_size = self._policy_depth_h * self._policy_depth_w

        extra_obs = 0
        if cfg.include_wheel_vel:
            extra_obs += 1  # scalar: avg wheel surface speed normalized to [0, 1]
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
        self._rear_throttle_ids, _ = self.robot.find_joints(
            [
                "back_left_wheel_throttle",
                "back_right_wheel_throttle",
            ]
        )
        self._front_throttle_ids, _ = self.robot.find_joints(
            [
                "front_left_wheel_throttle",
                "front_right_wheel_throttle",
            ]
        )
        self._steer_ids, _ = self.robot.find_joints(
            [
                "front_left_wheel_steer",
                "front_right_wheel_steer",
            ]
        )

        # Ackermann command buffers
        self._steer_tan = torch.zeros(self.num_envs, device=self.device)

        # Precomputed tensor of all 4 driven wheel joint indices [rear_L, rear_R, front_L, front_R]
        self._all_throttle_ids = torch.tensor(
            self._rear_throttle_ids + self._front_throttle_ids,
            dtype=torch.long,
            device=self.device,
        )

        # Per-wheel drive/brake torque commands (Nm), ordered to match
        # self._all_throttle_ids = [rear_L, rear_R, front_L, front_R].
        self._wheel_torque = torch.zeros(self.num_envs, 4, device=self.device)

        # ── Correct the total mass ───────────────────────────────────────────────
        # mushr_nano_v2.usd authors per-body masses summing to ~4.11 kg (sensible light
        # wheels, placeholder 1.0 kg chassis links).  Scale that native distribution
        # uniformly so the articulation total = car_mass_kg, and scale the inertia
        # tensors by the same ratio to keep them geometry-consistent (Isaac Lab's own
        # mass-randomization convention; masses/inertias live on CPU in the PhysX view).
        _pv = self.robot.root_physx_view
        _masses = _pv.get_masses()  # [N, n_bodies] (CPU)
        _native_total = _masses[0].sum().item()
        _mass_scale = cfg.car_mass_kg / _native_total
        _env_ids_cpu = torch.arange(self.num_envs, dtype=torch.int, device="cpu")
        _pv.set_masses(_masses * _mass_scale, _env_ids_cpu)
        _pv.set_inertias(_pv.get_inertias() * _mass_scale, _env_ids_cpu)
        print(
            f"[ConeTrackEnv] Mass: USD-native {_native_total:.3f} kg → scaled ×{_mass_scale:.4f} "
            f"to total {cfg.car_mass_kg:.3f} kg ({_masses.shape[1]} bodies, distribution preserved)"
        )

        # Steering LPF state
        self._filtered_steer = torch.zeros(self.num_envs, device=self.device)

        # Last two raw steering/throttle commands (undelayed), kept for rate penalties.
        self._steer_cmd_curr = torch.zeros(self.num_envs, device=self.device)
        self._steer_cmd_prev = torch.zeros(self.num_envs, device=self.device)
        self._throttle_cmd_curr = torch.zeros(self.num_envs, device=self.device)
        self._throttle_cmd_prev = torch.zeros(self.num_envs, device=self.device)

        # Action delay buffer — holds the last (delay_steps+1) raw actions so that
        # actuators receive the command issued delay_steps policy ticks ago.
        # The observation history still stores the undelayed policy output so the
        # network can reason about what it recently commanded vs what the car has done.
        policy_dt = cfg.decimation * cfg.sim.dt  # ≈ 1/30 s
        self._delay_steps = max(0, round(cfg.action_delay_ms * 1e-3 / policy_dt))
        buf_len = max(1, self._delay_steps + 1)
        self._action_delay_buf = torch.zeros(
            self.num_envs, buf_len, 2, device=self.device
        )
        print(
            f"[ConeTrackEnv] Action delay: {cfg.action_delay_ms:.0f} ms → {self._delay_steps} policy step(s)"
        )

        # No-load wheel speed (rad/s) at full throttle — drive torque rolls off to 0 here.
        self._omega_noload_full = cfg.max_velocity_m_s / WHEEL_RADIUS
        print(
            f"[ConeTrackEnv] Drivetrain: effort-controlled DC motor — "
            f"no-load {cfg.max_velocity_m_s:.2f} m/s ({self._omega_noload_full:.1f} rad/s/wheel), "
            f"drive_torque {cfg.drive_torque:.3f} Nm/wheel, brake_gain {cfg.brake_torque_gain:.3f} Nm/wheel/|u|"
        )
        # Lap-tracking state.
        # _lap_state: 0 = inside the start circle (lap timer running, awaiting departure),
        #             1 = outside the start circle (lap timer running, awaiting line crossing).
        # _lap_start_step: episode_length_buf value when the CURRENT lap's timer started —
        #     0 at episode reset, or the completion step of the previous lap.  The timer is
        #     therefore continuous from spawn, restarting on every lap closure.
        # _fastest_lap_s_global: scalar best lap time observed across all envs, all episodes.
        # _spawn_heading_{cos,sin}: unit vector along the spawn heading, captured at reset.
        # _lap_prev_forward: previous step's signed forward position (pos − spawn) · heading,
        #     used to detect start-line crossings via a sign flip (prev * now < 0).
        self._lap_state = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
        self._lap_start_step = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
        self._fastest_lap_s_global = torch.tensor(0.0, device=self.device)
        self._spawn_heading_cos = torch.ones(self.num_envs, device=self.device)
        self._spawn_heading_sin = torch.zeros(self.num_envs, device=self.device)
        self._lap_prev_forward = torch.zeros(self.num_envs, device=self.device)
        # Contact / termination flags
        self._wall_ever_touched = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )

        # Average-speed reward accumulator: sum of clamped forward velocity per env.
        self._cumulative_forward_vel = torch.zeros(
            self.num_envs, dtype=torch.float32, device=self.device
        )
        # Spawn-escape bonus tracking: world XY position at reset + one-time fired flag.
        self._spawn_pos = torch.zeros(self.num_envs, 2, device=self.device)
        self._spawn_escaped = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        # Index of the currently active decorative asset set (-1 = none selected yet).
        self._current_asset_set: int = -1
        self._asset_step_counter: int = 0
        self._asset_round_robin_idx: int = (
            0  # advances each switch; modulo NUM_ASSET_SETS gives current set
        )
        # All asset sets are left visible during scene init so RTX builds their BVH
        # structures at startup (cheap time slot). On the first physics step we hide
        # all but one selected set. This flag tracks whether that warmup hide has run.
        self._assets_warmup_done: bool = False

        # Depth observation caches (policy resolution after crop + resize)
        _cam_flat = self._policy_depth_h * self._policy_depth_w
        self._camera_left_obs = torch.zeros(
            self.num_envs, _cam_flat, device=self.device
        )
        self._camera_right_obs = torch.zeros(
            self.num_envs, _cam_flat, device=self.device
        )
        self._last_camera_update_step = -1

        # Wall geometry for the proximity_speed reward — ray-cast against the same
        # /World/Walls mesh used for termination (visibility / collision untouched).
        # Extracted once here and cached on-device for fast batched intersection.
        _wall_segs = self._extract_wall_segments_xy()
        if _wall_segs:
            import numpy as _np

            _seg_arr = torch.from_numpy(_np.asarray(_wall_segs, dtype=_np.float32)).to(
                self.device
            )
            self._wall_seg_a = _seg_arr[:, 0, :].contiguous()  # [M, 2]
            self._wall_seg_b = _seg_arr[:, 1, :].contiguous()  # [M, 2]
            print(
                f"[ConeTrackEnv] Proximity ray-cast: cached {_seg_arr.shape[0]} wall segments"
            )
        else:
            self._wall_seg_a = None
            self._wall_seg_b = None

    # ------------------------------------------------------------------
    # Scene setup
    # ------------------------------------------------------------------

    def _setup_scene(self):
        self.robot = Articulation(self.cfg.robot_cfg)

        track_cfg = sim_utils.UsdFileCfg(
            usd_path=TRACK_USD, scale=(TRACK_SCALE, TRACK_SCALE, TRACK_SCALE)
        )
        track_cfg.func(
            "/World/Track",
            track_cfg,
            translation=(0, 0.0, 0.0),
            orientation=(0.0, 0.0, 0.0, 1.0),
        )

        wall_cfg = sim_utils.UsdFileCfg(
            usd_path=WALL_USD, scale=(TRACK_SCALE, TRACK_SCALE, TRACK_SCALE)
        )
        wall_cfg.func(
            "/World/Walls",
            wall_cfg,
            translation=(0, 0.0, 0.0),
            orientation=(0.0, 0.0, 0.0, 1.0),
        )

        for set_idx, asset_usd in enumerate(ASSET_SET_USDS):
            asset_cfg = sim_utils.UsdFileCfg(
                usd_path=asset_usd, scale=(TRACK_SCALE, TRACK_SCALE, TRACK_SCALE)
            )
            asset_cfg.func(
                f"/World/Assets/Set{set_idx}",
                asset_cfg,
                translation=(0.0, 0.0, 0.0),
                orientation=(0.0, 0.0, 0.0, 1.0),
            )

        import omni.usd
        from pxr import Usd, UsdGeom, UsdPhysics, UsdShade

        stage = omni.usd.get_context().get_stage()

        wheel_link_names = {
            "front_left_wheel_link",
            "front_right_wheel_link",
            "back_left_wheel_link",
            "back_right_wheel_link",
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

        suspension_joint_names = (
            "front_left_wheel_suspension",
            "front_right_wheel_suspension",
            "back_left_wheel_suspension",
            "back_right_wheel_suspension",
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
                static_friction=self.cfg.ground_static_friction,
                dynamic_friction=self.cfg.ground_dynamic_friction,
                restitution=0.05,
                friction_combine_mode="multiply",
                restitution_combine_mode="min",
            )
        )
        ground_cfg.func("/World/GroundPlane", ground_cfg)

        self.camera_left = TiledCamera(self.cfg.camera_left_cfg)
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
            cone_h, cone_r = 0.15, 0.025  # height (m), base radius (m)
            for cam_name, cam_cfg, colour in (
                ("camera_left", self.cfg.camera_left_cfg, (0.0, 1.0, 1.0)),  # cyan
                ("camera_right", self.cfg.camera_right_cfg, (1.0, 0.0, 1.0)),  # magenta
            ):
                # Camera forward direction in base_link frame = R(q) · (0,0,1).
                # Camera quaternion is (w, x, y, z) in ROS convention.
                qw, qx, qy, qz = cam_cfg.offset.rot
                fwd_x = 2.0 * (qx * qz + qw * qy)
                fwd_y = 2.0 * (qy * qz - qw * qx)
                fwd_z = 1.0 - 2.0 * (qx * qx + qy * qy)

                # Cone centre = camera_pos + (h/2) * fwd  → tip lands at camera origin.
                px, py, pz = cam_cfg.offset.pos
                ccx = px + 0.5 * cone_h * fwd_x
                ccy = py + 0.5 * cone_h * fwd_y
                ccz = pz + 0.5 * cone_h * fwd_z

                # Quaternion rotating cone's +Y onto fwd.
                # cross((0,1,0), fwd) = (fwd_z, 0, -fwd_x); dot = fwd_y.
                dot = fwd_y
                cross_len = _math.sqrt(fwd_z**2 + fwd_x**2)
                if cross_len < 1e-6:
                    orient = (
                        Gf.Quatf(1.0, 0.0, 0.0, 0.0)
                        if dot > 0
                        else Gf.Quatf(0.0, 1.0, 0.0, 0.0)
                    )
                else:
                    w_q = _math.sqrt(max(0.0, 0.5 * (1.0 + dot)))
                    k = _math.sqrt(max(0.0, 0.5 * (1.0 - dot))) / cross_len
                    orient = Gf.Quatf(w_q, k * fwd_z, 0.0, k * (-fwd_x))

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
            wheel_root_env = stage.GetPrimAtPath(
                f"/World/envs/env_{env_id}/Robot/mushr_nano"
            )
            if not wheel_root_env.IsValid():
                continue
            for prim in Usd.PrimRange(wheel_root_env):
                if not prim.IsA(UsdGeom.Mesh):
                    continue
                if prim.GetParent().GetName() not in wheel_link_names:
                    continue
                UsdShade.MaterialBindingAPI.Apply(prim).Bind(
                    rubber_mat,
                    UsdShade.Tokens.strongerThanDescendants,
                    "physics",
                )

        self.scene.articulations["robot"] = self.robot
        self.scene.sensors["camera_left"] = self.camera_left
        self.scene.sensors["camera_right"] = self.camera_right
        for link_name, sensor in self._wall_contact_sensors.items():
            self.scene.sensors[f"wall_contact_{link_name}"] = sensor

        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    # ------------------------------------------------------------------
    # Wall geometry helper
    # ------------------------------------------------------------------

    def _extract_wall_segments_xy(self):
        """Extract 2D line segments from /World/Walls for proximity ray-cast.

        Read-only pass over the same wall mesh used for termination.  Each triangle
        edge is projected to the XY plane; we keep only edges whose endpoints have
        nearly equal Z so vertical wall edges (which collapse to a single point in
        XY) are discarded and the wall's footprint is what remains.

        Visibility, collision, and termination logic on /World/Walls are not modified.
        Returns a list of [[ax, ay], [bx, by]] entries (world frame), or [] on failure.
        """
        try:
            import omni.usd
            from pxr import Gf, Usd, UsdGeom

            stage = omni.usd.get_context().get_stage()
            wall_root = stage.GetPrimAtPath("/World/Walls")
            if not wall_root.IsValid():
                print(
                    "[ConeTrackEnv] /World/Walls not found — proximity ray-cast disabled"
                )
                return []

            xf_cache = UsdGeom.XformCache()
            segments: list = []

            for prim in Usd.PrimRange(wall_root):
                if not prim.IsA(UsdGeom.Mesh):
                    continue
                mesh = UsdGeom.Mesh(prim)
                points_attr = mesh.GetPointsAttr().Get()
                if points_attr is None:
                    continue
                counts_attr = mesh.GetFaceVertexCountsAttr().Get()
                indices_attr = mesh.GetFaceVertexIndicesAttr().Get()
                if counts_attr is None or indices_attr is None:
                    continue

                world_xf = xf_cache.GetLocalToWorldTransform(prim)
                world_pts = [
                    world_xf.Transform(Gf.Vec3d(float(p[0]), float(p[1]), float(p[2])))
                    for p in points_attr
                ]

                idx_ptr = 0
                for fc in counts_attr:
                    face = [int(indices_attr[idx_ptr + i]) for i in range(fc)]
                    idx_ptr += fc
                    # Fan-triangulate, walk each triangle's 3 edges.
                    for k in range(1, fc - 1):
                        tri = (face[0], face[k], face[k + 1])
                        pts3 = (world_pts[tri[0]], world_pts[tri[1]], world_pts[tri[2]])
                        for e in range(3):
                            a3 = pts3[e]
                            b3 = pts3[(e + 1) % 3]
                            # Drop vertical edges — they project to a point in XY.
                            if abs(a3[2] - b3[2]) > 0.05:
                                continue
                            dx = b3[0] - a3[0]
                            dy = b3[1] - a3[1]
                            if dx * dx + dy * dy < 1e-4:
                                continue
                            segments.append(
                                [
                                    [float(a3[0]), float(a3[1])],
                                    [float(b3[0]), float(b3[1])],
                                ]
                            )
            return segments
        except Exception as e:
            print(f"[ConeTrackEnv] Wall segment extraction failed: {e}")
            return []

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
        """Convert actions → Ackermann commands (effort-controlled DC motor, AWD).

        Throttle (actions[:,0]) ∈ [-1,1] is mapped to a per-wheel drive/brake torque
        (forward-only; see the zone description below) applied via set_joint_effort_target.

        Latency model: raw actions are pushed into _action_delay_buf; actuators
        receive the command from delay_steps ticks ago.
        """
        # On the very first physics step all asset sets are still visible (RTX has
        # already built every BVH during sim init). Now hide all but set 0.
        if not self._assets_warmup_done:
            self._set_active_assets(0)
            self._assets_warmup_done = True

        # Round-robin asset cycling — each set appears exactly twice per PPO iteration.
        # Sequence: 0,1,2,3,4,0,1,2,3,4,... advancing every asset_randomize_every_n_steps.
        self._asset_step_counter += 1
        if self._asset_step_counter >= self.cfg.asset_randomize_every_n_steps:
            self._asset_step_counter = 0
            self._asset_round_robin_idx += 1
            self._set_active_assets(self._asset_round_robin_idx % NUM_ASSET_SETS)

        raw = actions[:, :2].clamp(-1.0, 1.0)  # [N, 2]  — undelayed policy output

        # Push into delay FIFO and read the oldest (delayed) entry
        self._action_delay_buf = self._action_delay_buf.roll(-1, dims=1)
        self._action_delay_buf[:, -1] = raw
        delayed = self._action_delay_buf[:, 0]  # [N, 2]  — what the actuators act on

        # ── Throttle → per-wheel torque (forward-only, effort-controlled DC motor) ──
        # Three zones (the car can NEVER be driven in reverse):
        #   throttle > 0  DRIVE: T = drive_torque · clamp(1 − ω/(throttle·ω_noload), 0, 1)
        #                 — DC torque-speed curve, full torque at standstill tapering to
        #                   0 at the throttle-scaled no-load speed (full throttle → max_velocity_m_s).
        #   throttle = 0  COAST: T = 0.
        #   throttle < 0  BRAKE: T = −brake_torque_gain · |throttle|, opposing rotation,
        #                 faded out below brake_release_omega so it decelerates to a stop
        #                 and then coasts (never applies a reverse-driving torque).
        # Wheel slip is left to PhysX: when |T|·(N wheels)/r exceeds μ·m·g the tyres break
        # traction (matches the real car only slipping under the hardest braking).
        u = delayed[:, 0].unsqueeze(-1)  # [N,1] throttle
        wheel_omega = self.robot.data.joint_vel[:, self._all_throttle_ids].clamp(
            min=0.0
        )  # [N,4] forward spin

        # DRIVE branch (throttle > 0)
        omega_noload = (
            u.clamp(min=1e-3) * self.cfg.max_velocity_m_s / WHEEL_RADIUS
        )  # [N,1]
        drive_rolloff = (1.0 - wheel_omega / omega_noload).clamp(0.0, 1.0)  # [N,4]
        drive_torque = self.cfg.drive_torque * drive_rolloff  # [N,4]

        # BRAKE branch (throttle < 0): baseline + graded torque, opposes motion, fades to 0
        # near standstill so it stops the car and then coasts (never drives in reverse).
        # |throttle| is remapped through brake_throttle_scale so policy −1 = the slip-ceiling
        # brake level (no point braking harder once the tyres slip).
        brake_level = (-u).clamp(
            min=0.0
        ) * self.cfg.brake_throttle_scale  # [N,1] >0 when braking
        brake_fade = (wheel_omega / self.cfg.brake_release_omega).clamp(
            0.0, 1.0
        )  # [N,4] 1 when moving, →0 at rest
        brake_torque = (
            -(self.cfg.brake_torque_base + self.cfg.brake_torque_gain * brake_level)
            * brake_fade
        )  # [N,4] ≤ 0

        # throttle > 0 → drive, throttle < 0 → brake, throttle == 0 → coast (zero torque)
        self._wheel_torque = torch.where(
            u > 0.0,
            drive_torque,
            torch.where(u < 0.0, brake_torque, torch.zeros_like(brake_torque)),
        )

        # Steering: direct passthrough, no LPF (matches real hardware — no filter on servo)
        self._filtered_steer = delayed[:, 1]
        steer_ang = self._filtered_steer * self.cfg.max_steer_rad
        self._steer_tan = torch.tan(steer_ang)

        # Track the last two RAW (undelayed) steering/throttle commands for rate penalties.
        self._steer_cmd_prev = self._steer_cmd_curr
        self._steer_cmd_curr = raw[:, 1]
        self._throttle_cmd_prev = self._throttle_cmd_curr
        self._throttle_cmd_curr = raw[:, 0]

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
        self._wall_ever_touched |= wall_hit_now & ~in_grace

        wall_mask = self._wall_ever_touched
        wheel_torque = self._wheel_torque.clone()
        wheel_torque[wall_mask] = (
            0.0  # cut drive torque after a wall hit (car coasts to a stop)
        )
        self.robot.set_joint_effort_target(
            wheel_torque, joint_ids=self._all_throttle_ids
        )

        steer_positions = self._steer_tan.unsqueeze(-1).expand(-1, 2).clone()
        steer_positions[wall_mask] = 0.0
        self.robot.set_joint_position_target(steer_positions, joint_ids=self._steer_ids)

    # ------------------------------------------------------------------
    # Depth camera cache  (called once per step, shared by obs + rewards)
    # ------------------------------------------------------------------

    def _process_single_depth(self, camera) -> torch.Tensor:
        """Shared depth pipeline for one TiledCamera → [N, policy_h*policy_w].

        Noise effects are applied at the camera's full 240×180 resolution before
        cropping and policy-side resizing, matching the real ArduCam TOF sensor pipeline.
        """
        import torch.nn.functional as F

        depth_raw = camera.data.output["distance_to_image_plane"]  # [N, H, W, 1]
        depth = depth_raw[..., 0]  # [N, H, W]
        depth = torch.where(
            torch.isfinite(depth),
            depth,
            torch.tensor(self.cfg.depth_max_m, device=depth.device),
        )

        # Edge weights computed on the FULL-RESOLUTION CLEAN signal so that real
        # depth discontinuities (not noise spikes) drive the mixed-pixel scatter.
        if self.cfg.depth_edge_scatter_prob > 0.0:
            _d4 = depth.unsqueeze(1)  # [N,1,H,W]
            _px = F.pad(_d4, (1, 1, 0, 0), mode="replicate")  # [N,1,H,W+2]
            _py = F.pad(_d4, (0, 0, 1, 1), mode="replicate")  # [N,1,H+2,W]
            _dx = (_px[:, :, :, 2:] - _px[:, :, :, :-2]) * 0.5  # centred ∂/∂x
            _dy = (_py[:, :, 2:, :] - _py[:, :, :-2, :]) * 0.5  # centred ∂/∂y
            _edge_w = (
                (_dx.abs() + _dy.abs()).squeeze(1)  # L1 grad [N,H,W]
                / self.cfg.depth_edge_gradient_threshold_m
            ).clamp(0.0, 1.0)

        # ── Noise passes at full 240×180 sensor resolution ────────────────────
        if self.cfg.depth_noise_sigma_m > 0.0:
            depth = depth + torch.randn_like(depth) * self.cfg.depth_noise_sigma_m

        if self.cfg.depth_invalid_pixel_prob > 0.0:
            invalid = torch.rand_like(depth) < self.cfg.depth_invalid_pixel_prob
            depth = depth.masked_fill(invalid, self.cfg.depth_max_m)

        if self.cfg.depth_color_noise_prob > 0.0:
            color_mask = torch.rand_like(depth) < self.cfg.depth_color_noise_prob
            depth = torch.where(
                color_mask,
                torch.rand_like(depth) * self.cfg.depth_max_m,
                depth,
            )

        if self.cfg.depth_edge_scatter_prob > 0.0:
            scatter_mask = torch.rand_like(depth) < (
                self.cfg.depth_edge_scatter_prob * _edge_w
            )
            scatter_vals = torch.where(
                torch.rand_like(depth) > 0.5,
                torch.full_like(depth, self.cfg.depth_max_m),
                torch.zeros_like(depth),
            )
            depth = torch.where(scatter_mask, scatter_vals, depth)

        if (
            self.cfg.depth_edge_blur_sigma_m > 0.0
            and self.cfg.depth_edge_scatter_prob > 0.0
        ):
            depth = depth + torch.randn_like(depth) * (
                self.cfg.depth_edge_blur_sigma_m * _edge_w
            )

        # ── Crop AFTER noise (removes top/bottom rows at full resolution) ─────
        h = depth.shape[1]
        top = int(self.cfg.depth_crop_top_fraction * h)
        bot = h - int(self.cfg.depth_crop_bottom_fraction * h)
        depth = depth[:, top:bot, :]  # [N, H', W]

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
        self._camera_left_obs = self._process_single_depth(self.camera_left)
        self._camera_right_obs = self._process_single_depth(self.camera_right)

    # ------------------------------------------------------------------
    # Observations
    # ------------------------------------------------------------------

    def _get_observations(self) -> dict:
        self._update_depth_cache()

        parts = [self._camera_left_obs, self._camera_right_obs]
        if self.cfg.include_wheel_vel:
            wheel_omega_mean = (
                self.robot.data.joint_vel[:, self._all_throttle_ids].abs().mean(dim=-1)
            )
            wheel_vel_norm = (
                wheel_omega_mean * WHEEL_RADIUS / self.cfg.max_velocity_m_s
            ).clamp(0.0, 1.0)
            parts.append(wheel_vel_norm.unsqueeze(-1))  # [N, 1]

        obs = torch.cat(parts, dim=-1)
        return {"policy": obs}

    # ------------------------------------------------------------------
    # Rewards
    # ------------------------------------------------------------------

    def _get_rewards(self) -> torch.Tensor:
        self._update_depth_cache()

        lin_vel_b = self.robot.data.root_lin_vel_b[:, 0]  # m/s forward
        dt = self.cfg.decimation * self.cfg.sim.dt  # ≈ 0.0333 s

        # Distance covered this step
        r_distance = (
            self.cfg.distance_weight * lin_vel_b / self.cfg.max_velocity_m_s * dt
        )

        # Episode average-speed bonus: rewards sustained pace, not just current speed.
        forward_speed = lin_vel_b.clamp(min=0.0)
        self._cumulative_forward_vel += forward_speed
        speed_sample_count = self.episode_length_buf.float().clamp(min=1.0)
        avg_speed_norm = (
            self._cumulative_forward_vel
            / speed_sample_count
            / self.cfg.max_velocity_m_s
        ).clamp(0.0, 1.0)
        r_average_speed = self.cfg.average_speed_weight * avg_speed_norm

        pos_xy = self.robot.data.root_pos_w[:, :2]

        # ── Steering deadzone penalty ────────────────────────────────────────────
        steer_norm = torch.abs(self._filtered_steer).clamp(0.0, 1.0)
        steer_excess = (steer_norm - self.cfg.steer_deadzone).clamp(min=0.0)
        steer_excess_norm = steer_excess / max(1.0 - self.cfg.steer_deadzone, 1e-6)
        r_steer_shape = -self.cfg.steer_deadzone_weight * steer_excess_norm

        # ── Steering-rate (smoothness) penalty ───────────────────────────────────
        # Penalise the change in commanded steering between the two most recent
        # policy steps to suppress jerky chatter.
        steer_rate = (self._steer_cmd_curr - self._steer_cmd_prev).abs()
        r_steer_rate = -self.cfg.steer_rate_weight * steer_rate

        # ── Throttle-rate (smoothness) penalty ───────────────────────────────────
        throttle_rate = (self._throttle_cmd_curr - self._throttle_cmd_prev).abs()
        r_throttle_rate = -self.cfg.throttle_rate_weight * throttle_rate

        # ── Wheel slip penalty ───────────────────────────────────────────────────
        # Penalty 1.0 means the wheel is spinning at max velocity but car isn't moving (or vice versa)
        wheel_omega_mean = (
            self.robot.data.joint_vel[:, self._all_throttle_ids].abs().mean(dim=-1)
        )
        wheel_vel = wheel_omega_mean * WHEEL_RADIUS
        slip_ratio = (wheel_vel - forward_speed).abs() / self.cfg.max_velocity_m_s
        r_slip = -self.cfg.slip_weight * slip_ratio.clamp(0.0, 1.0)

        r_alive = torch.full(
            (self.num_envs,), self.cfg.alive_weight, device=self.device
        )

        # ── Spawn-escape bonus ────────────────────────────────────────────────
        # One-time reward scaled by speed at the moment of escape.  Full bonus only
        # if the car is already moving at max speed when it crosses the radius.
        spawn_dist = torch.linalg.vector_norm(pos_xy - self._spawn_pos, dim=-1)
        just_escaped = (~self._spawn_escaped) & (
            spawn_dist > self.cfg.spawn_escape_radius_m
        )
        self._spawn_escaped |= just_escaped
        speed_frac = (lin_vel_b / self.cfg.max_velocity_m_s).clamp(0.0, 1.0)
        r_spawn_escape = (
            self.cfg.spawn_escape_reward * just_escaped.float() * speed_frac
        )

        # ── Proximity speed penalty (ray-cast against /World/Walls) ──────────────
        # Cast a virtual ray from the car's XY position along its world-frame
        # linear-velocity vector and find the closest hit on the (invisible) wall
        # geometry.  Walls remain hidden / termination-only — this only reads the
        # cached 2D footprint built once at startup.
        #
        # Closeness = 1 − dist / proximity_lookahead_m, clamped to [0, 1].  Walls
        # beyond the lookahead give zero signal; walls right ahead give 1.0.
        # Penalty = -weight × speed_norm × closeness.  Strongest exactly when the
        # car is fast AND a wall is close along the heading of motion (corner entry).
        if self.cfg.proximity_speed_weight > 0.0 and self._wall_seg_a is not None:
            vel_w = self.robot.data.root_lin_vel_w[:, :2]  # [N, 2]
            speed_w = torch.linalg.vector_norm(vel_w, dim=-1)  # [N]
            moving = speed_w > 0.5
            dir_w = vel_w / speed_w.clamp(min=1e-3).unsqueeze(-1)  # [N, 2]

            # Vectorised 2D ray-vs-segment intersection across [N envs, M segs].
            # Ray:     P(t) = origin + t · dir_w,        t ≥ 0
            # Segment: Q(u) = a + u · (b − a),           0 ≤ u ≤ 1
            # With s = b − a:
            #     t = ((a − origin) × s)   / (dir × s)
            #     u = ((a − origin) × dir) / (dir × s)
            # where × is the 2D pseudo-cross (a × b = a_x*b_y − a_y*b_x).
            a = self._wall_seg_a.unsqueeze(0)  # [1, M, 2]
            b = self._wall_seg_b.unsqueeze(0)  # [1, M, 2]
            s = b - a  # [1, M, 2]
            r = dir_w.unsqueeze(1)  # [N, 1, 2]
            ap = a - pos_xy.unsqueeze(1)  # [N, M, 2]

            cross_rs = r[..., 0] * s[..., 1] - r[..., 1] * s[..., 0]  # [N, M]
            cross_ap_s = ap[..., 0] * s[..., 1] - ap[..., 1] * s[..., 0]
            cross_ap_r = ap[..., 0] * r[..., 1] - ap[..., 1] * r[..., 0]

            denom = torch.where(
                cross_rs.abs() > 1e-8,
                cross_rs,
                torch.full_like(cross_rs, 1e-8),
            )
            t_hit = cross_ap_s / denom
            u_hit = cross_ap_r / denom

            hit = (
                (cross_rs.abs() > 1e-8)
                & (t_hit > 0.0)
                & (u_hit >= 0.0)
                & (u_hit <= 1.0)
            )
            t_masked = torch.where(hit, t_hit, torch.full_like(t_hit, float("inf")))
            nearest_dist = t_masked.min(dim=1).values  # [N]

            closeness = 1.0 - (nearest_dist / self.cfg.proximity_lookahead_m).clamp(
                0.0, 1.0
            )
            speed_norm = (speed_w / self.cfg.max_velocity_m_s).clamp(0.0, 1.0)
            r_proximity_speed = (
                -self.cfg.proximity_speed_weight
                * closeness
                * speed_norm
                * moving.float()
            )
        else:
            r_proximity_speed = torch.zeros(self.num_envs, device=self.device)

        # ── Lap completion reward ─────────────────────────────────────────────
        # State machine: 0 = at-start, 1 = away-from-start, lap closes on 1→0.
        # The 1→0 transition requires *crossing the start line* — the line through
        # the spawn point perpendicular to the spawn heading — while inside
        # `lap_return_radius_m`.  Crossing is detected as a sign flip of the car's
        # signed forward position (pos − spawn) · heading_unit between this step
        # and the last.  The radius gate disambiguates: a round track intersects
        # the start line twice (near spawn and on the far side); only the spawn-
        # side crossing counts.
        pos_rel = pos_xy - self._spawn_pos
        forward_now = (
            pos_rel[:, 0] * self._spawn_heading_cos
            + pos_rel[:, 1] * self._spawn_heading_sin
        )
        spawn_dist_for_lap = torch.linalg.vector_norm(pos_rel, dim=-1)
        in_grace_lap = self.episode_length_buf < self.cfg.spawn_grace_steps

        crossed_start_line = (self._lap_prev_forward * forward_now) < 0.0
        in_return_radius = spawn_dist_for_lap < self.cfg.lap_return_radius_m

        # Timer runs continuously.  `_lap_start_step` is the step the *current* lap began:
        #   - at reset:           0 (timer counts from the moment of spawn)
        #   - on lap completion:  the completion step (next lap restarts immediately)
        # State machine sequences the lap:
        #   - 0 → 1 when the agent leaves the start circle (timer keeps running)
        #   - 1 → 0 when the start line is crossed while back inside the circle
        #           (lap closes here, new lap's timer starts on the same step)
        # Multiple completions per episode are supported — each closure logs a lap.
        leaving = (self._lap_state == 0) & (~in_return_radius) & (~in_grace_lap)
        self._lap_state = torch.where(
            leaving, torch.ones_like(self._lap_state), self._lap_state
        )

        completing = (self._lap_state == 1) & crossed_start_line & in_return_radius
        self._lap_state = torch.where(
            completing, torch.zeros_like(self._lap_state), self._lap_state
        )

        # Save signed forward for next-step crossing detection.
        self._lap_prev_forward = forward_now.detach()

        lap_steps_elapsed = (
            (self.episode_length_buf - self._lap_start_step).float().clamp(min=1.0)
        )
        lap_time_s = lap_steps_elapsed * dt
        lap_time_clamped = lap_time_s.clamp(
            self.cfg.lap_min_time_s, self.cfg.lap_max_time_s
        )
        r_lap = torch.where(
            completing,
            self.cfg.lap_reward_weight * self.cfg.lap_max_time_s / lap_time_clamped,
            torch.zeros_like(lap_time_s),
        )

        # Restart the timer on the same step a lap closes, so the next lap measures from here.
        self._lap_start_step = torch.where(
            completing, self.episode_length_buf.clone(), self._lap_start_step
        )

        # Lap metrics → RSL-RL extras → tensorboard.
        # rsl_rl aggregates per iteration via torch.mean(torch.cat(values_across_steps))
        # (on_policy_runner.py: see the ep_infos loop).  To keep iter-means meaningful
        # we avoid padding with zeros across the thousands of non-event steps:
        #   - lap_time_s    : empty on non-completion steps, vector of completed
        #                     lap times on completion steps.  Iter-mean = average
        #                     lap time across completions during the iteration.
        #   - fastest_lap_s : empty on non-completion steps, the (updated) global
        #                     scalar on completion steps.  Iter-mean is monotonic
        #                     non-increasing as the global improves and equals the
        #                     global exactly on iterations where it didn't change.
        #   - laps_per_min  : kept as a per-step value (k * 60/dt, k = laps closed
        #                     this step).  Iter MEAN = total_laps / iter_minutes,
        #                     summed across envs.  Padding zeros are REQUIRED here
        #                     for that rate interpretation.
        # rsl_rl drops any key absent in ep_infos[0]; an empty tensor still counts
        # as "present" and contributes nothing to torch.cat.
        empty = torch.tensor([], device=self.device)
        if completing.any():
            lap_times_done = lap_time_s[completing].detach()
            new_min = lap_times_done.min()
            self._fastest_lap_s_global = torch.where(
                self._fastest_lap_s_global > 0,
                torch.minimum(self._fastest_lap_s_global, new_min),
                new_min,
            )
            fastest_log = self._fastest_lap_s_global.detach().unsqueeze(0).clone()
        else:
            lap_times_done = empty
            fastest_log = empty

        laps_per_min = completing.float().sum().unsqueeze(0) * (60.0 / dt)

        self.extras["log"] = {
            "Lap/lap_time_s": lap_times_done,
            "Lap/fastest_lap_s": fastest_log,
            "Lap/laps_per_min": laps_per_min,
        }

        # ── Braking reward V2 ────────────────────────────────────────────────
        # Instantaneous, no session state: r = weight × v_forward² × |brake_input|.
        # The v² scaling means braking at low speed contributes almost nothing,
        # while braking at high speed dominates — strong incentive to brake
        # specifically when going fast (e.g., before corners) rather than always.
        if self.cfg.braking_v2_weight > 0.0:
            throttle_cmd_v2 = self._action_delay_buf[:, 0, 0]
            brake_input_v2 = (-throttle_cmd_v2).clamp(0.0, 1.0)  # 0 if not braking
            speed_fwd_v2 = lin_vel_b.clamp(min=0.0) / self.cfg.max_velocity_m_s
            r_braking_v2 = (
                self.cfg.braking_v2_weight * torch.pow(speed_fwd_v2, 2) * brake_input_v2
            )
        else:
            r_braking_v2 = torch.zeros(self.num_envs, device=self.device)

        total = (
            r_alive
            + r_distance
            + r_average_speed
            + r_steer_shape
            + r_steer_rate
            + r_throttle_rate
            + r_slip
            + r_spawn_escape
            + r_proximity_speed
            + r_braking_v2
            + r_lap
        )

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

        in_grace = self.episode_length_buf < self.cfg.spawn_grace_steps
        self._wall_ever_touched |= wall_terminated & ~in_grace
        wall_terminated = self._wall_ever_touched

        # Flip detection (suppressed during grace)
        q = self.robot.data.root_quat_w
        up_z = 1.0 - 2.0 * (q[:, 1] * q[:, 1] + q[:, 2] * q[:, 2])
        flipped = (up_z < 0.3) & ~in_grace

        # Near-stop termination — only fires if agent is essentially stationary
        # (slow_stop_speed_fraction × vel_cap, default 3% = 0.075 m/s at 2.5 m/s cap).
        # Grace period of 45 steps (~1.5 s) lets the car accelerate from spawn.
        lin_vel_b = self.robot.data.root_lin_vel_b[:, 0]
        stopped = (
            lin_vel_b < (self.cfg.slow_stop_speed_fraction * self.cfg.max_velocity_m_s)
        ) & (self.episode_length_buf > 45)

        terminated = wall_terminated | flipped | stopped
        time_out = self.episode_length_buf >= self.max_episode_length - 1
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
        sz = 0.002  # just above ground — constant for all spawn modes
        env_ids_t = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)

        pts = list(self.cfg.preset_spawn_points)
        if self.cfg.duplicate_with_flipped_yaw:
            pts = pts + [(x, y, theta + 180.0) for x, y, theta in pts]
        idx = torch.randint(0, len(pts), (n,), device="cpu").tolist()
        xs = torch.tensor(
            [pts[i][0] * self.cfg.spawn_x_scale for i in idx],
            dtype=torch.float32,
            device=self.device,
        )
        ys = torch.tensor(
            [pts[i][1] * self.cfg.spawn_y_scale for i in idx],
            dtype=torch.float32,
            device=self.device,
        )
        base_yaw = torch.tensor(
            [math.radians(pts[i][2]) for i in idx],
            dtype=torch.float32,
            device=self.device,
        )
        yaw = base_yaw.clone()
        if self.cfg.spawn_yaw_jitter_rad > 0.0:
            yaw = (
                yaw
                + (torch.rand(n, device=self.device) - 0.5)
                * 2.0
                * self.cfg.spawn_yaw_jitter_rad
            )
        spawn_xy = torch.stack([xs, ys], dim=-1)

        default_root_state[:, 0] = spawn_xy[:, 0]
        default_root_state[:, 1] = spawn_xy[:, 1]
        default_root_state[:, 2] = sz

        zeros = torch.zeros(n, device=self.device)
        dq = quat_from_euler_xyz(zeros, zeros, yaw)
        default_root_state[:, 3:7] = dq

        default_root_state[:, 7:] = 0.0
        default_root_state[:, 7] = speed_vec * torch.cos(yaw)
        default_root_state[:, 8] = speed_vec * torch.sin(yaw)

        self.robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self.robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)

        # Body velocity already seeded above via write_root_velocity_to_sim
        self._filtered_steer[env_ids] = 0.0
        self._steer_cmd_curr[env_ids] = 0.0
        self._steer_cmd_prev[env_ids] = 0.0
        self._throttle_cmd_curr[env_ids] = 0.0
        self._throttle_cmd_prev[env_ids] = 0.0
        self._action_delay_buf[env_ids] = 0.0  # clears latency FIFO

        self._wall_ever_touched[env_ids] = False
        self._cumulative_forward_vel[env_ids] = 0.0
        self._spawn_pos[env_ids_t] = spawn_xy
        self._spawn_escaped[env_ids_t] = False
        self._lap_state[env_ids_t] = 0
        self._lap_start_step[env_ids_t] = 0
        self._spawn_heading_cos[env_ids_t] = torch.cos(base_yaw)
        self._spawn_heading_sin[env_ids_t] = torch.sin(base_yaw)
        self._lap_prev_forward[env_ids_t] = 0.0
        # Note: _fastest_lap_s_global is intentionally NOT reset — it tracks the best ever.

        joint_pos = self.robot.data.default_joint_pos[env_ids]
        joint_vel = torch.zeros_like(joint_pos)
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)
