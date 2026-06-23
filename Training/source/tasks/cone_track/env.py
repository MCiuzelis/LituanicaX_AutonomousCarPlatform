"""MuSHR nano v2 RC car — cone-track navigation environment (DirectRLEnv).

Observation space:
    [0:N]        flattened blue-mask image from center RGB camera, after HSV filtering
                 and downsampling.  1.0 = blue cone pixel, 0.0 = non-blue pixel.
    [N]          (optional) average wheel surface speed ∈ [0, 1], normalized by
                 max_velocity_m_s.  Only appended when include_wheel_vel=True.

Camera profile (center RGB camera):
    - Sensor: RGB, 848×480 at 120 fps
    - FOV: 120°(H) × 66.7°(V) pinhole
    - Blue-only HSV mask: only blue pixels pass; everything else is crushed to black.
      Shadows are suppressed by the mask, producing a clean high-contrast binary image.
    - Policy runs at 120 Hz (sim 480 Hz, decimation=4)

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

Why blue-mask RGB:
    A single center RGB camera with HSV blue filtering gives the policy a high-contrast
    binary image where only the blue track stripes / cones are visible.  Shadows and
    background are suppressed by the mask, providing a clean signal for RL training —
    the agent learns to follow blue features without being distracted by lighting or
    texture variation.

Termination conditions:
    - Wall contact: any horizontal normal force on robot links above wall_contact_force_thresh_n
    - Robot tilt > 72° from vertical (up_z < 0.3)
    - Slow driving: speed < 2% of vel_cap for > ~1.5 s after episode start

TUNING NOTES (first run):
    - All adjustable parameters are in ConeTrackEnvCfg in one place below.
    - Spawn position: set preset_spawn_points in cfg.
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

from .parameters import (
    CAM_H,
    CAM_W,
    DOWNSAMPLING_AMOUNT,
    INCLUDE_WHEEL_VEL,
    VISION_CROP_BOTTOM_FRACTION,
    VISION_CROP_TOP_FRACTION,
)

# ---------------------------------------------------------------------------
# Asset USD paths  (relative to this file → project root → assets/)
# ---------------------------------------------------------------------------
_ASSETS_DIR = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "../../../assets")
)
_REPO_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), "../../../../"))
_EXPORT_DIR = os.path.join(_REPO_ROOT, "CAD", "Track", "Export")

MUSHR_USD = os.path.join(_ASSETS_DIR, "mushr_nano_v2.usd")
TRACK_USD = os.path.join(_EXPORT_DIR, "Lines.usdc")
WALL_USD = os.path.join(_EXPORT_DIR, "Walls.usdc")

# ---------------------------------------------------------------------------
# Physical constants — MuSHR nano v2 (fixed by the USD / hardware)
# These match mushr_nano_v2.usd geometry and are NOT user-adjustable without
# also modifying the robot USD.
# ---------------------------------------------------------------------------
WHEEL_RADIUS = 0.037  # m  (82 mm diameter wheels)

TRACK_SCALE = 1.0

# ---------------------------------------------------------------------------
# Center RGB camera optics (IMX708, f=2.87mm, FOV 120°×66.7°)
# Focal length matches spec sheet; aperture ratio is what sets the FOV.
# For Isaac PinholeCameraCfg: aperture / focal_length = 2 * tan(FOV/2).
# ---------------------------------------------------------------------------
_RGB_FOCAL_MM = 2.87
_RGB_H_APERTURE_MM = 2.0 * _RGB_FOCAL_MM * math.tan(math.radians(120.0 / 2.0))
_RGB_V_APERTURE_MM = 2.0 * _RGB_FOCAL_MM * math.tan(math.radians(66.7 / 2.0))

_REMAINING_FRACTION = max(0.0, 1.0 - VISION_CROP_TOP_FRACTION - VISION_CROP_BOTTOM_FRACTION)
_CROPPED_CAM_H = max(1, int(CAM_H * _REMAINING_FRACTION))
_NEW_V_APERTURE = _RGB_V_APERTURE_MM * _REMAINING_FRACTION
_V_APERTURE_OFFSET = _RGB_V_APERTURE_MM * (VISION_CROP_BOTTOM_FRACTION - VISION_CROP_TOP_FRACTION) / 2.0

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
    CAMERA          — center RGB, 848×480, 120 fps, blue-mask pipeline
    SPAWN           — starting position, heading, and speed
    SPAWN SPEED     — initial episode velocity range
    DYNAMICS        — physics-based motor model, steering limit
    TERMINATION     — contact-force threshold, slow-speed detection
    OBSERVATION     — optional wheel-speed channel
    REWARD WEIGHTS  — all reward scale factors
    """

    # ── Core env parameters ────────────────────────────────────────────────
    policy_hz: int = 60  # policy running frequency (Hz)
    decimation: int = 4  # sim steps per policy step
    episode_length_s: float = 90.0  # episode duration in seconds
    action_space: int = 2
    # observation_space is patched in ConeTrackEnv.__init__ after computing
    # the exact cropped vision mask size.  The placeholder below keeps
    # @configclass happy; the real value is written before super().__init__().
    observation_space: int = 1  # patched in __init__
    state_space: int = 0

    # sim.dt and sim.render_interval are set in __post_init__ from policy_hz & decimation
    sim: SimulationCfg = SimulationCfg(
        physx=sim_utils.PhysxCfg(
            gpu_heap_capacity=2**23,                  # 8 MB (default 67 MB)
            gpu_collision_stack_size=2**23,            # 8 MB (default 67 MB)
            gpu_found_lost_pairs_capacity=2**19,       # 0.5 MB (default 2M)
            gpu_found_lost_aggregate_pairs_capacity=2**20,  # 1 MB (default 34 MB)
            gpu_total_aggregate_pairs_capacity=2**19,  # 0.5 MB (default 2M)
            gpu_max_rigid_contact_count=2**20,         # 1M (default 8M)
            gpu_temp_buffer_capacity=2**21,            # 2 MB (default 16M)
            gpu_max_num_partitions=4,                  # 4 partitions (default 8)
        ),
        render=sim_utils.RenderCfg(
            enable_translucency=False,
            enable_reflections=False,
            enable_global_illumination=False,
            enable_dl_denoiser=False,
            enable_direct_lighting=True,
            enable_shadows=False,
            enable_ambient_occlusion=False,
            samples_per_pixel=1,
            rendering_mode="performance",
        ),
    )

    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=26,
        env_spacing=0.0,
        replicate_physics=True,
    )

    # ════════════════════════════════════════════════════════════════════════
    # CAMERA — center RGB (848×480, 120 fps, 120°×66.7° FOV)
    # ════════════════════════════════════════════════════════════════════════
    # The raw RGB frame is HSV-filtered for blue pixels → binary mask (white =
    # blue detected, black = everything else).  Shadows are suppressed by the
    # value-channel lower bound.  The binary mask is then cropped (top/bottom
    # fractions) and downsampled to (W/downsampling_amount) × (H/downsampling_amount).

    # Crop fractions: fraction of raw pixel rows discarded before downsampling.
    #   top    → removes sky / distant background
    #   bottom → removes near-ground clutter and the robot's own body
    vision_crop_top_fraction: float = VISION_CROP_TOP_FRACTION
    vision_crop_bottom_fraction: float = VISION_CROP_BOTTOM_FRACTION

    # ── Policy image downsampling ──────────────────────────────────────────
    # After HSV filtering and cropping, the binary mask is downsampled by this
    # factor in both dimensions before being flattened into the observation vector.
    downsampling_amount: int = DOWNSAMPLING_AMOUNT

    # ════════════════════════════════════════════════════════════════════════
    # SPAWN — initial robot position, heading, and speed
    # ════════════════════════════════════════════════════════════════════════
    # Grace period after each reset: wall / flip / stuck terminations are suppressed
    # for this many policy steps so that PhysX contact forces can settle.
    # 10 steps ≈ 83 ms at 120 Hz — enough for any spawn-physics transient.
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
    # milliseconds before reaching the wheels/steering.  At 120 Hz each step ≈ 8.33 ms,
    # so 100 ms ≈ 3 policy steps.  Set to 0.0 to disable.
    action_delay_ms: float = 17.0

    # ════════════════════════════════════════════════════════════════════════
    # TERMINATION THRESHOLDS
    # ════════════════════════════════════════════════════════════════════════
    # Horizontal contact force on any robot link that counts as a wall hit.
    wall_contact_force_thresh_n: float = 1.0  # Newtonsgz

    # slow_stop_speed_fraction — episode terminates when speed falls below this
    # (catches truly stuck / reversed agents; keep low so normal braking is fine)
    slow_stop_speed_fraction: float = 0.04  # terminate if speed < 4 % of vel_cap

    # ════════════════════════════════════════════════════════════════════════
    # OBSERVATION EXTRAS
    # ════════════════════════════════════════════════════════════════════════
    # Append the current average wheel surface speed to the observation.
    # Normalized: 0 = stopped, 1 = max_velocity_m_s.
    # This is the primary speed signal — essential for braking-before-corners.
    # On real hardware: read wheel encoder RPM, average all 4, multiply by WHEEL_RADIUS,
    # divide by max_velocity_m_s, clamp to [0, 1].
    include_wheel_vel: bool = INCLUDE_WHEEL_VEL

    # ════════════════════════════════════════════════════════════════════════
    # REWARD WEIGHTS
    # ════════════════════════════════════════════════════════════════════════
    # Constant reward per step the agent stays alive (not crashed / stuck).
    alive_weight: float = 0.015

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
    slip_weight: float = 0.5

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
    lap_reward_weight: float = 2.0
    lap_return_radius_m: float = 0.3  # start line while back inside closes the lap
    lap_min_time_s: float = 20.0  # lower clamp (caps maximum reward per lap)
    lap_max_time_s: float = 50.0  # upper clamp & numerator anchor

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
        """Derive timing and drivetrain actuators from the scalar config fields."""
        # ── Timing: sim dt and camera update from policy_hz & decimation ──────
        self.sim.dt = 1.0 / (self.decimation * self.policy_hz)
        self.sim.render_interval = self.decimation
        self.camera_center_cfg.update_period = 1.0 / self.policy_hz

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

    # Center RGB camera (replaces the two depth cameras):
    #   Position: 185 mm forward of base_link, centered (0 mm lateral), 20 mm above base_link.
    #   Orientation: forward, no rotation.
    camera_center_cfg: TiledCameraCfg = TiledCameraCfg(
        prim_path="/World/envs/env_.*/Robot/mushr_nano/base_link/camera_center",
        # update_period is set in __post_init__ from policy_hz
        height=_CROPPED_CAM_H,
        width=CAM_W,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=_RGB_FOCAL_MM,
            horizontal_aperture=_RGB_H_APERTURE_MM,
            vertical_aperture=_NEW_V_APERTURE,
            vertical_aperture_offset=_V_APERTURE_OFFSET,
            clipping_range=(0.01, 100.0),
        ),
        offset=TiledCameraCfg.OffsetCfg(
            pos=(
                0.155,
                0.0,
                0.055,
            ),
            rot=(-0.5, 0.5, -0.5, 0.5),  # No rotation, looking forward
            convention="ros",
        ),
        debug_vis=False,
    )

    # Live debug view in training — use scripts/rsl_rl/visualize.py instead.
    show_camera_debug: bool = False
    camera_debug_env_id: int = 0
    camera_debug_interval: int = 2


# ===========================================================================
# Environment Class
# ===========================================================================


class ConeTrackEnv(DirectRLEnv):
    cfg: ConeTrackEnvCfg

    def __init__(self, cfg: ConeTrackEnvCfg, render_mode: str | None = None, **kwargs):
        # ContactSensor dict must exist before super().__init__() calls _setup_scene().
        self._wall_contact_sensors: dict[str, ContactSensor] = {}

        # Compute exact vision observation size.
        # 1. Crop: remove top/bottom rows from the raw frame.
        # 2. Downsample: bilinear resize by downsampling_amount in both axes.
        cam_h = cfg.camera_center_cfg.height
        cam_w = cfg.camera_center_cfg.width
        
        # The camera image is already cropped by Isaac Sim
        cropped_h = cam_h
        cropped_w = cam_w

        ds = cfg.downsampling_amount
        self._policy_depth_w = max(1, cropped_w // ds)
        self._policy_depth_h = max(1, cropped_h // ds)

        vision_obs_size = self._policy_depth_h * self._policy_depth_w

        extra_obs = 0
        if cfg.include_wheel_vel:
            extra_obs += 1

        cfg.observation_space = vision_obs_size + extra_obs  # 1 camera

        print(
            f"[ConeTrackEnv] Vision obs: raw {cam_w}×{cam_h} "
            f"→ crop {cropped_w}×{cropped_h} "
            f"→ policy {self._policy_depth_w}×{self._policy_depth_h} "
            f"= {vision_obs_size} px  |  "
            f"+ veloc={cfg.include_wheel_vel}  |  "
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
        policy_dt = cfg.decimation * cfg.sim.dt  # ≈ 1/policy_hz s
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
        # Vision observation caches (policy resolution after color filtering + downsample)
        _cam_flat = self._policy_depth_h * self._policy_depth_w
        self._camera_center_obs = torch.zeros(
            self.num_envs, _cam_flat, device=self.device, dtype=torch.float16
        )
        self._last_camera_update_step = -1

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

        self.camera_center = TiledCamera(self.cfg.camera_center_cfg)
        for link_name in WALL_CONTACT_LINKS:
            sensor_cfg = ContactSensorCfg(
                prim_path=f"/World/envs/env_.*/Robot/mushr_nano/{link_name}",
                filter_prim_paths_expr=["/World/Walls"],
                history_length=1,
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
        self.scene.sensors["camera_center"] = self.camera_center
        for link_name, sensor in self._wall_contact_sensors.items():
            self.scene.sensors[f"wall_contact_{link_name}"] = sensor

        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

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
    # Vision cache  (called once per step, shared by obs + rewards)
    # ------------------------------------------------------------------

    def _process_single_rgb(self, camera) -> torch.Tensor:
        """GPU vision pipeline for RGB camera → [N, policy_h*policy_w] binary mask.

        Pipeline: raw RGB → blue HSV mask → morphological noise removal → downsample.
        Entirely GPU-based: no CPU copies or per-env Python loops.
        """
        import torch.nn.functional as F

        rgb = camera.data.output["rgb"][..., :3]  # [N, H, W, 3]
        N, H, W, _ = rgb.shape

        # Normalise to [0, 1]; handle both uint8 and float inputs
        if rgb.dtype == torch.uint8:
            rgb = rgb.float() / 255.0
        else:
            rgb = rgb.float().clamp(0.0, 1.0)

        # ── RGB → HSV on GPU (standard formula, H in [0, 360], SV in [0, 1]) ──
        r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]

        cmax, _ = rgb.max(dim=-1)
        cmin, _ = rgb.min(dim=-1)
        delta = cmax - cmin

        # Hue
        h = torch.zeros_like(cmax)
        valid = delta > 1e-8

        # Cmax == R
        mask_r = (cmax == r) & valid
        h[mask_r] = ((g[mask_r] - b[mask_r]) / delta[mask_r]) % 6.0 * 60.0

        # Cmax == G
        mask_g = (cmax == g) & valid & ~mask_r
        h[mask_g] = ((b[mask_g] - r[mask_g]) / delta[mask_g] + 2.0) * 60.0

        # Cmax == B
        mask_b = valid & ~mask_r & ~mask_g
        h[mask_b] = ((r[mask_b] - g[mask_b]) / delta[mask_b] + 4.0) * 60.0

        h = h % 360.0

        # Saturation
        s = torch.where(cmax > 0, delta / cmax, torch.zeros_like(cmax))

        # Value
        v = cmax

        # ── Blue inRange mask (OpenCV-compatible thresholds) ────────────────
        # Original OpenCV thresholds: lower=[90,30,50], upper=[150,255,255] in
        # OpenCV HSV (H/2, S*255, V*255).  Converted to standard HSV ranges:
        in_h_low, in_h_high = 180.0, 300.0
        in_s_low = 30.0 / 255.0
        in_v_low = 50.0 / 255.0

        mask = (
            (h >= in_h_low) & (h <= in_h_high)
            & (s >= in_s_low)
            & (v >= in_v_low)
        ).float()

        # ── Morphological opening: remove noise (replaces CPU connected-component filter) ──
        k = 3
        p = k // 2
        # Erosion →  dilation
        mask = -F.max_pool2d(-mask.unsqueeze(1), k, stride=1, padding=p)
        mask = F.max_pool2d(mask, k, stride=1, padding=p)
        mask = mask.squeeze(1)

        # ── Downsample to policy resolution ──────────────────────────────────
        mask_4d = F.interpolate(
            mask.unsqueeze(1),
            size=(self._policy_depth_h, self._policy_depth_w),
            mode="bilinear",
            align_corners=False,
        )
        return mask_4d.squeeze(1).reshape(self.num_envs, -1).to(torch.float16)

    def _update_vision_cache(self) -> None:
        """Process center camera and write self._camera_center_obs."""
        current_step = int(self.episode_length_buf.max().item())
        if self._last_camera_update_step == current_step:
            return
        self._last_camera_update_step = current_step
        self._camera_center_obs = self._process_single_rgb(self.camera_center)

    # ------------------------------------------------------------------
    # Observations
    # ------------------------------------------------------------------

    def _get_observations(self) -> dict:
        self._update_vision_cache()

        parts = [self._camera_center_obs]
        if self.cfg.include_wheel_vel:
            wheel_omega_mean = (
                self.robot.data.joint_vel[:, self._all_throttle_ids].abs().mean(dim=-1)
            )
            wheel_vel_norm = (
                wheel_omega_mean * WHEEL_RADIUS / self.cfg.max_velocity_m_s
            ).clamp(0.0, 1.0).to(torch.float16)
            parts.append(wheel_vel_norm.unsqueeze(-1))  # [N, 1]

        obs = torch.cat(parts, dim=-1)
        return {"policy": obs}

    # ------------------------------------------------------------------
    # Rewards
    # ------------------------------------------------------------------

    def _get_rewards(self) -> torch.Tensor:
        self._update_vision_cache()

        lin_vel_b = self.robot.data.root_lin_vel_b[:, 0]  # m/s forward
        dt = self.cfg.decimation * self.cfg.sim.dt  # ≈ 1/policy_hz s

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
