"""MuSHR nano v2 RC car — cone-track navigation environment (DirectRLEnv).

This module holds BOTH halves of the task:
    ConeTrackEnvCfg — every tunable number (camera, spawn, drivetrain, rewards).
    ConeTrackEnv    — the simulation itself (scene, actuation, observations, rewards).
PPO hyper-parameters live next door in ppo_cfg.py.

Observation space (flat vector, laid out in this order):
    [0:N]        flattened depth image from the LEFT ArduCam TOF camera (+27° yaw),
                 after crop and downsample.  Normalized so 1.0 = object at 0 m
                 (danger) and 0.0 = object at depth_max_m (clear).
    [N:2N]       same for the RIGHT camera (−27° yaw).
    [2N]         average wheel surface speed ∈ [0, 1], normalized by max_velocity_m_s
                 (mean |joint_vel| over the 4 throttle joints × WHEEL_RADIUS).
                 Only present when include_wheel_vel=True.  Matches real hardware:
                 wheel encoders give rotation rate directly.
    [2N+1: ]     the last prev_action_history_len (throttle, steer) commands,
                 oldest first, newest last.  The newest pair is the command issued
                 this step — the one the actuators apply next step, given
                 action_delay_ms.  Omitted when prev_action_history_len = 0.
    The trailing scalars are the "proprio tail"; their count MUST match
    ConeTrackActorCriticCfg.num_proprio_obs, which is where the policy network
    splits the vector to route them through their own encoder.

Camera profile (ArduCam TOF sensor for Raspberry Pi):
    - Sensor: Time-of-Flight depth, 240×180 pixels at 30 fps
    - Diagonal FOV: 70°  →  HFOV ≈ 58.5°, VFOV ≈ 45.6°  (4:3 aspect, pinhole)
    - Depth accuracy: ±2 cm (simulated via Gaussian noise σ ≈ 0.007 m)
    - Policy runs at 30 Hz, synced with camera (sim 120 Hz, decimation=4)

Action space (2):
    [0]  throttle command in [-1, 1] — forward-only (the car never reverses).  It is a
         PWM DUTY command, exactly what the deployed car writes to the motor driver:
           (0, 1]  → DRIVE  torque = drive_torque × (throttle − ω/ω_noload_full)
                     — a real brushed-DC torque-speed line: STALL TORQUE SCALES WITH
                     DUTY, and the zero-torque speed is throttle × max_velocity_m_s.
            0      → COAST  (zero torque, free roll)
           [-1, 0) → BRAKE  torque = −(brake_torque_base + brake_torque_gain × level),
                     level = |throttle| × brake_throttle_scale (policy −1 = the slip
                     ceiling).  Opposes motion, fades to 0 near standstill → stops then
                     coasts, never spinning the wheels backward.
    [1]  steering command in [-1, 1]  → scaled to [-max_steer_rad, +max_steer_rad]

Actuation (Ackermann / AWD, effort-controlled DC motor):
    - All 4 wheels receive a torque (Nm) via set_joint_effort_target.  Drive follows the
      brushed-DC torque-speed line T = k(V_duty − k_e·ω): full drive_torque only at full
      throttle and zero speed, scaling down linearly with duty.  Braking is an
      independent, throttle-graded reverse torque.  Wheel slip occurs naturally when the
      commanded torque exceeds the PhysX contact-friction limit (μ·N).
    - Front wheel steer receives position target (rad) = steer · max_steer_rad.
      This was tan(steer · max_steer_rad) until 2026-08-02, described as matching a
      "tan-convention" in mushr_nano_v2.usd — but the steer joints are plain
      PhysicsRevoluteJoints fed the same value on both sides (parallel steering, no
      Ackermann linkage), so Isaac Lab applied that tangent as an angle and the sim
      steered 30.4° while every comment said 28°.
    - No rise-rate ramp is modelled here on purpose: the car's
      THROTTLE_SLEW_UP_PER_SEC limiter is being made switchable on the deployment side
      (config.THROTTLE_SLEW_ENABLED) so that the commanded duty reaches the motor
      unshaped, which is what this model assumes.

Why depth camera vs RGB cone mask:
    The depth camera gives the policy direct metric information about wall and obstacle
    proximity.  A shallow corner looks identical in RGB (just more cone pixels) but
    in depth the wall is visibly close.  This makes braking-before-corners learnable:
    the agent can correlate "high depth values filling the frame → wall approaching →
    reduce throttle" without needing to understand color or stripe classification.

Termination conditions (each ends the episode; the first two are also priced into the
reward so that no exit is cheaper than continuing to drive — see crash_penalty_weight
and stall_penalty_weight):
    - Wall contact: any horizontal normal force on robot links above
      wall_contact_force_thresh_n, charged in proportion to the impact severity
    - Slow driving: speed below slow_stop_speed_fraction of the cap after
      slow_stop_grace_steps, charged a flat stall penalty
    - Robot tilt > 72° from vertical (up_z < 0.3)

Where to change what (everything below lives in ConeTrackEnvCfg):
    - Spawn position / heading  : preset_spawn_points, spawn_yaw_jitter_rad
    - Camera crop and noise     : depth_crop_*_fraction, depth_*_noise_*
    - Acceleration / braking    : drive_torque, brake_torque_base, brake_torque_gain
    - Top speed                 : max_velocity_m_s (also the reward normaliser)
    - Grip / suspension         : ground_*_friction, suspension_stiffness
    - Behaviour                 : the REWARD WEIGHTS section
"""

from __future__ import annotations

import math
import os
from collections.abc import Sequence
from typing import Any, cast

import isaaclab.sim as sim_utils
import numpy as np
import torch
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensor, ContactSensorCfg, TiledCamera, TiledCameraCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import quat_from_euler_xyz

# ---------------------------------------------------------------------------
# Asset USD paths — everything ships from Training/assets/Export, alongside the
# textures/ folder the .usdc files reference relatively.
#   this file → Training/source/cone_track/env.py, so ../.. is Training/
# ---------------------------------------------------------------------------
_EXPORT_DIR = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "../..", "assets", "Export")
)

MUSHR_USD = os.path.join(_EXPORT_DIR, "mushr_nano_v2.usd")
TRACK_USD = os.path.join(_EXPORT_DIR, "Track.usdc")
WALL_USD = os.path.join(_EXPORT_DIR, "Walls.usdc")

# Decorative asset sets — purely visual clutter the depth cameras see, cycled
# during training so the policy never learns one fixed arrangement of scenery.
# Index NUM_ASSET_SETS means "no decorations" (all sets hidden).
ASSET_SET_USDS = [
    os.path.join(_EXPORT_DIR, f"Assets_{i}.usdc") for i in range(1, 6)
]
NUM_ASSET_SETS = len(ASSET_SET_USDS)

# ---------------------------------------------------------------------------
# Physical constants — MuSHR nano v2 (fixed by the USD / hardware)
# These match mushr_nano_v2.usd geometry and are NOT user-adjustable without
# also modifying the robot USD.
# ---------------------------------------------------------------------------
WHEEL_RADIUS = 0.037  # m  (74 mm diameter wheels, after ROBOT_SCALE)

# Uniform spawn scale applied to mushr_nano_v2.usd.  Chosen so the USD's 97.6 mm wheels
# come out at the real car's 74 mm: 0.0488 m radius × 0.755 = 0.0368 m ≈ WHEEL_RADIUS.
ROBOT_SCALE = 0.755

# Axle spacing authored in the USD, before and after that scale.  The suspension joints
# put the front axle at x = +0.1385 and the rear at x = −0.158 in base_link's frame.
# Matching the wheels by scaling the whole robot necessarily scaled the wheelbase too,
# which is why it lands at 224 mm against the real car's 187 mm — see cfg.wheelbase_m.
USD_WHEELBASE_M = 0.1385 + 0.158  # 0.2965 m, unscaled
NATIVE_WHEELBASE_M = USD_WHEELBASE_M * ROBOT_SCALE  # 0.2239 m as spawned

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

# The four wheels, i.e. the subset of WALL_CONTACT_LINKS whose GROUND contact is
# read back for the wheel-lift penalty.  No extra sensors are needed for that: a
# ContactSensor always reports the body's total contact force in `net_forces_w`,
# and `filter_prim_paths_expr` only adds the per-filter breakdown on top of it.
# So the wall sensors already carry the ground reaction; the wall pair-force is
# read from force_matrix_w, the ground support from net_forces_w[..., 2].
WHEEL_LINKS = (
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
    CAMERA          — sensor resolution, FOV, depth range, crop, noise, downsampling
    SPAWN           — starting position, heading, and post-reset grace period
    DYNAMICS        — DC-motor drivetrain, grip, steering limit, action latency
    TERMINATION     — contact-force threshold, slow-speed detection
    OBSERVATION     — the optional proprioceptive channels
    REWARD WEIGHTS  — all reward scale factors
    SCENE DRESSING  — decorative asset sets the depth cameras can see
    HARDWARE CONFIG — robot articulation and the two camera sensors
    """

    # ── Core env parameters ────────────────────────────────────────────────
    decimation: int = 4  # policy at 30 Hz  (sim 120 Hz / 4)
    episode_length_s: float = 30.0  # 30 s × 30 Hz = 900 steps (≈2 laps at 13 s)
    action_space: int = 2
    # observation_space is patched in ConeTrackEnv.__init__ after computing
    # the exact cropped depth image size.  The placeholder below keeps
    # @configclass happy; the real value is written before super().__init__().
    observation_space: int = 1  # patched in __init__
    state_space: int = 0

    # render_interval is counted in SIM steps, so it must equal decimation for the
    # RTX sensors to be rendered exactly once per policy step (at 30 Hz).
    sim: SimulationCfg = SimulationCfg(dt=1.0 / 120.0, render_interval=4)

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
    depth_crop_top_fraction: float = 0.35
    depth_crop_bottom_fraction: float = 0.45

    # Gaussian depth noise: σ ≈ 0.007 m gives a ±2 cm (3σ) error band,
    # matching the ArduCam TOF spec.  Set to 0.0 to disable.
    depth_noise_sigma_m: float = 0.007

    # Random invalid-pixel dropout: each pixel is independently zeroed with this
    # probability, replicating the black-spec speckle noise visible in real ArduCam
    # TOF output.  Affected pixels are set to depth_max_m (treated as far/safe by the
    # policy, matching the real camera's low-confidence masking behaviour).
    # Visible as black dots in both windows of visualize.py.  Set to 0.0 to disable.
    # NOTE: this is the small iid floor only.  The dropout that actually dominates the
    # real sensor is structured, not iid — see the CONFIDENCE MODEL below.
    depth_invalid_pixel_prob: float = 0.006

    # ── ToF CONFIDENCE MODEL — the dominant real-sensor effect ───────────────────────────
    # [2026-08-02 sim2real fix]  Measured on the car: 79 % (left) / 69 % (right) of the
    # pixels inside the policy crop band fail the deployment's CONF_THRESHOLD=30 and are
    # forced to "far" by process_for_policy() in Deployment/.../modules/tof_camera.py.
    # The sim was dropping 0.6 % iid + 0.5 % speckle.  The policy therefore trained on a
    # dense depth image and is flown on a sparse one.
    #
    # The dropout is not random — it is signal-to-noise.  A ToF pixel returns usable phase
    # only if the reflected power clears the noise floor, and that power goes as
    #     P ∝ ρ · cos(incidence) / range²
    # so the two things that kill a pixel are DISTANCE and GRAZING INCIDENCE.  Both are
    # visible in the log:
    #
    #   • Range.  Confidence pass rate by measured distance inside the band (left/right):
    #       0.0-0.3 m  4%/12%   0.3-0.6 m  27%/50%   0.6-0.9 m  38%/50%
    #       0.9-1.2 m 35%/35%   1.2-1.5 m  34%/21%   1.5-1.8 m  12%/11%
    #     (the 0-0.3 m bucket is not real geometry — it is the dropouts themselves, which
    #     the sensor reports as 0 mm; 20 % of the band reads exactly 0.)  From 0.75 m out
    #     the fall-off is clean inverse-square: 0.50 → 0.115 over a 2.2× range increase is
    #     a factor 4.35 against the 4.84 that 1/r² predicts.
    #   • Incidence.  The real floor simply vanishes past ~1 m: the camera sits 65 mm up
    #     and sees the ground at 2-3°, so cos(incidence) ≈ 0.03 and nothing comes back.
    #     The sim renders that same floor as a crisp depth ramp — its bottom crop row
    #     averages 0.433 where the real ones average 0.145 (left) / 0.292 (right).  That
    #     ramp is the strongest and most consistent feature in the sim image and it does
    #     not exist on the car.
    #
    # Implementation (_apply_confidence_dropout): cos(incidence) is recovered from the
    # image-space depth gradient — one pixel turns the view ray by 1/f_px rad, so for a
    # locally planar surface tan(incidence) = |∇d| · f_px / d — and the pixel survives with
    #     p = clamp(max_pass · min(1, gain · cos(inc)^exponent / d²), floor, max_pass)
    # Survivors keep their depth; the rest are set to depth_max_m, exactly as the car does.
    #
    # TUNING.  Set depth_conf_report_every_n_steps > 0 and read the printout: it gives the
    # invalid fraction over the band, the fraction among IN-RANGE pixels (the meaningful
    # one — pixels already beyond depth_max_m read "far" either way), and the per-row
    # profile to line up against the real ones quoted in the printout.  Raise
    # depth_conf_gain_m2 to keep more pixels, lower it to keep fewer.
    depth_conf_enable: bool = True
    # Fitted at cos(inc)=1 against the pass-vs-range curve above.  gain/d² saturates for
    # d ≲ 0.87 m, then the model gives 0.354 at 1.05 m, 0.214 at 1.35 m and 0.143 at
    # 1.65 m, against measured 0.35/0.35, 0.34/0.21 and 0.12/0.11 (left/right).
    depth_conf_gain_m2: float = 0.75
    # Ceiling on the pass rate — the real sensor never returns a fully dense frame even at
    # point-blank range on a good surface (best measured bucket was ~0.50).
    depth_conf_max_pass: float = 0.52
    # Exponent on cos(incidence).  1.0 = Lambertian.  Raising it culls grazing surfaces
    # harder (and thins the near floor below the crop band, which the real sensor still
    # sees at 0.2-0.5 m); lowering it lets the sim's ground ramp survive into the band,
    # which is the specific artefact this whole model exists to remove.
    depth_conf_incidence_exponent: float = 1.0
    # Floor so no pixel is deterministically dead — the real sensor occasionally returns
    # a reading off a surface that should give nothing.
    depth_conf_floor_pass: float = 0.02
    # Counted in dropout calls, i.e. two per policy step (one per camera), so the printout
    # alternates left/right.  0 disables the diagnostic entirely (the default for training).
    depth_conf_report_every_n_steps: int = 0

    # Random-depth (coloured) speckle: each pixel is set to a uniform random depth in
    # [0, depth_max_m] with this probability.  Models stray reflections and multi-path
    # interference that appear as bright or mid-range dots (unlike depth_invalid_pixel_prob
    # which always sets the pixel to the far/black depth_max_m value).
    # Set to 0.0 to disable.
    depth_color_noise_prob: float = 0.005

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

    # ── Policy image downsampling ──────────────────────────────────────────
    # After cropping the raw depth frame (240×≈40 at defaults) the image is
    # bilinearly downsampled to this resolution before being flattened into
    # the observation vector.  Smaller = faster MLP convergence.
    # policy_depth_height < 0 → auto-compute from cropped aspect ratio.
    policy_depth_width: int = 120
    policy_depth_height: int = -1

    # ── Camera extrinsic randomisation ───────────────────────────────────────────────────
    # [2026-08-02 sim2real fix]  The sim mounts both cameras at exactly 65.5 mm with zero
    # pitch (base_link 71.5 mm − 6.0 mm offset, after the 0.755 spawn scale).  Fitting the
    # ground plane in the real frames gives left 79 mm @ 1.15° nose-down and right 63.5 mm
    # @ 1.32° nose-down — the two cameras are neither level nor equal to each other, and
    # the chassis adds its own pitch on top.  At this mounting height 1° moves the horizon
    # by 3.7 rows of the 36-row crop band, so a fixed-perfect calibration is a real gap.
    # Resampled per environment on every reset; set either to 0.0 to pin that axis.
    camera_pitch_jitter_deg: float = 1.5  # ± about the vehicle's lateral axis, + = nose-down
    camera_height_jitter_m: float = 0.010  # ± along the vehicle's vertical axis, in WORLD metres
    # The measured 8.9 ms skew between the two cameras' capture instants is deliberately
    # NOT modelled: RTX renders every sensor once per render_interval, so both TiledCameras
    # necessarily sample the same sim instant, and there is no phase offset to give them.
    # Representing it would need render_interval=2 (60 Hz rendering, ~2× the sensor cost)
    # plus per-sensor update phases.  At 100 °/s yaw it is 0.9° of view difference, well
    # inside the pitch jitter above, so it is left as a known unmodelled term.

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
        # (0.756, 5.0, 20.8),
        # (3.91, 3.31, -156.0),
        # (3.62, 1.97, -154.48),
        (0.0,  0.0, 0.0),
        (0.0,  -5.5, 0.0)
        # (-5.46, 3.7, 0.0),
        # (-3.6, 4.7, -217.86),
        # (-5.0,  0.0, 180.0),
        # (3.4,  0.0, 0.0),
    ]
    # Coordinate transform applied to every spawn point at runtime.
    # Blender exports X/Y inverted relative to Isaac Sim's frame, so both are -1.
    spawn_x_scale: float = -TRACK_SCALE
    spawn_y_scale: float = -TRACK_SCALE

    duplicate_with_flipped_yaw: bool = True  # if True, each (x, y, θ) spawns two episodes: one with +180deg rotation around z
    spawn_yaw_jitter_rad: float = 0.1  # ± heading noise applied in both modes

    # ════════════════════════════════════════════════════════════════════════
    # DYNAMICS — effort-controlled DC-motor drivetrain (AWD), tuned to real car
    #
    # Each step the policy throttle ∈ [-1, 1] is turned into a per-wheel TORQUE
    # (applied via set_joint_effort_target).  Three zones — forward-only:
    #
    #   throttle > 0  DRIVE: T = drive_torque · clamp(throttle − ω/ω_noload_full, 0, 1)
    #                 where ω_noload_full = max_velocity_m_s / r is FIXED (full throttle).
    #                 This is the brushed-DC torque-speed line T = k_t(V·duty − k_e·ω)/R:
    #                 stall torque is PROPORTIONAL TO DUTY, and the line still crosses
    #                 zero torque at throttle · max_velocity_m_s, so top speed per
    #                 throttle level is unchanged.
    #
    #                 [2026-08-02 sim2real fix]  The old form was
    #                     T = drive_torque · clamp(1 − ω/(throttle·ω_noload_full), 0, 1)
    #                 which handed out the FULL stall torque at any throttle > 0, so the
    #                 sim accelerated just as hard at throttle 0.4 as at 1.0 (5.79 m/s²
    #                 either way) while the real car managed 2.38 m/s² at duty 0.4.
    #                 Replaying the logged commands from the 2026-08-02 track run through
    #                 the old model gave a mean +2.65 m/s² where the car measured +0.07.
    #                 The corrected line reproduces the measured response at every duty:
    #                     real  a₀ = 5.95·duty m/s²   (from real_car_acceleration_log.txt,
    #                                                  v(t) = 6.7·duty·(1 − e^(−t/1.126 s)))
    #                     sim   a₀ = 4·drive_torque·u / r / m
    #                             = 4·0.068·u / 0.037 / 1.27 = 5.79·u m/s²
    #                 — a 3 % match across the whole throttle range instead of at one point.
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
    # ── Tyre–ground friction (sets traction / wheel-slip onset) ──────────────────────────
    # These combine with the bound wheel-rubber material (μ=1.0) under PhysX's "multiply"
    # rule, and because MULTIPLY outranks the rubber material's default AVERAGE in PhysX's
    # combine-mode precedence, the pair mode is MULTIPLY and the EFFECTIVE μ is simply
    # ground_μ × 1.0 = ground_μ.  So the number written here is the μ the tyres actually get.
    #
    # [2026-08-02 sim2real fix]  These were 2.0 / 1.6 — an effective μ of 2.0, i.e. a car
    # that can corner at ~2 g and put down any torque the motor can make.  The real car,
    # measured from the track run by differentiating the depth-image yaw rate, never
    # exceeded 0.63 g lateral (0.34 g mean) and was visibly sliding while doing it.  The
    # policy was therefore learning corner entry speeds that do not exist on the real
    # floor.  0.95 / 0.80 is rubber on a smooth indoor floor and puts the grip ceiling at
    # ~0.95 g, just above what the car was measured doing.
    #
    # ⚠  RE-VALIDATE THE DRIVETRAIN AFTER CHANGING THESE.  drive_torque and the brake
    #    terms were fitted when traction was effectively infinite; at μ≈0.95 the tyres
    #    now limit hard braking and standing starts, so the numbers below no longer
    #    reproduce the real car by construction.  Run
    #        uv run python source/cone_track/acceleration_test.py --headless
    #    and compare against Training/real_car_acceleration_log.txt:
    #        accel  0 → 4.7 m/s in 1.36 s over 3.45 m   (throttle 1.0)
    #        brake  from ~4.6 m/s: 2.61 m @ duty 0.1, 1.77 m @ 0.2,
    #                              1.35 m @ 0.3, 0.86 m @ 0.5 (0.5 visibly slipped)
    #    Expect the brake distances to grow first; if they do, the friction is now the
    #    binding constraint rather than brake_torque_gain, which is the intended physics.
    ground_static_friction: float = 0.95  # static μ (grip before slip); traction ≈ μ·m·g
    ground_dynamic_friction: float = 0.80  # sliding μ (once a wheel breaks traction)
    suspension_stiffness: float = (
        1e6  # suspension-joint lock stiffness (rigid; stops energy-sapping travel)
    )

    # ── Steering geometry ────────────────────────────────────────────────────────────────
    # What the vehicle actually does is set by the CURVATURE it can pull, κ = tan(δ)/L, so
    # max_steer_rad only means something alongside the wheelbase.  Both were wrong:
    #
    #   wheelbase   The MuSHR USD is spawned at scale 0.755 to bring its wheels to the real
    #               74 mm diameter, which scaled the wheelbase along with them: 296.5 mm ×
    #               0.755 = 224 mm.  The real car's wheelbase is 187 mm.  wheelbase_m below
    #               re-spaces the axles at load time (see _apply_wheelbase).
    #   max_steer   MEASURED WITH A PROTRACTOR (2026-08-02, HardwareTests/steering_angle_test
    #               with the wheels off the ground).  Pulse offset → road-wheel angle:
    #                   200 µs → 16.0°   250 µs → 18.4°   300 µs → 23.5°   350 µs → 29.0°
    #               At the deployed SERVO_AMPLITUDE = 0.35 a full-lock command is 350 µs, so
    #               δ_max ≈ 28–29°.  The linkage is very close to linear in between (a linear
    #               steer→angle map is within ~2° everywhere, which is protractor noise), so
    #               the sim's linear mapping needs no correction curve.
    #
    # δ_max = 28.5° = 0.497 rad at L = 0.187 m gives a KINEMATIC κ of 2.90 1/m (R = 0.34 m).
    #
    # ⚠  Do NOT tune this down to match the curvature the car is observed to pull on track.
    #    Optical flow on the logged run put the ACHIEVED κ at only 1.1–1.7 1/m — 38–59 % of
    #    kinematic — but that is understeer, not geometry: the front tyres are sliding.  (It
    #    is also a lower bound: the policy swings the steering at several Hz and the yaw
    #    response is low-passed by the 100 ms actuation lag, so a regression of instantaneous
    #    yaw on the command under-reads the steady-state gain.)  The distinction matters
    #    because the two have opposite speed dependence — understeer grows with v²/R while a
    #    steering-angle cap does not — so capping the angle would make the sim unable to turn
    #    at parking speed, where the real car pulls very nearly its full kinematic radius.
    #    The tyre/friction model above is what reproduces the sliding; this number is pure
    #    geometry.  Tune ground_*_friction against the skid-pad measurement, not this.
    wheelbase_m: float = 0.187  # real car, measured.  0.0 → leave the USD spacing alone
    max_steer_rad: float = 0.497  # 28.5° measured at SERVO_AMPLITUDE 0.35 → kinematic κ 2.90 1/m

    # Action latency simulation — all actuator commands are delayed by this many
    # milliseconds before reaching the wheels/steering.  At 30 Hz each step ≈ 33 ms,
    # so 100 ms ≈ 3 policy steps.  Set to 0.0 to disable.
    #
    # [2026-08-02 sim2real fix]  Was 33.0 (one policy step).  Measured on the real car by
    # correlating the steering command against the yaw rate recovered from depth-image
    # optical flow: the correlation is ~0 at zero lag and peaks sharply at 3 policy steps,
    # 100 ms (|corr| 0.61).  That is the STEERING loop — camera exposure and readout
    # (23 ms / 14 ms of frame age at inference for left / right), 4 ms of tick, the 50 Hz
    # servo PWM update, and the servo's own mechanical response.  The throttle path was
    # not separately identifiable from the log (the encoder's own filter and the
    # drivetrain time constant dominate it), so a single delay is applied to both
    # channels; if the throttle path is later measured and differs materially, split
    # _action_delay_buf into two FIFOs rather than compromising between them.
    action_delay_ms: float = 100.0

    # ════════════════════════════════════════════════════════════════════════
    # TERMINATION THRESHOLDS
    # ════════════════════════════════════════════════════════════════════════
    # Horizontal contact force on any robot link that counts as a wall hit.
    # ANY contact above this ends the episode — touching a wall is failure, full
    # stop.  The crash penalty is still scaled by impact angle and speed (see
    # crash_penalty_weight), so a tangential graze and a head-on hit cost very
    # different amounts, but both end the run.
    wall_contact_force_thresh_n: float = 1.0  # Newtons

    # slow_stop_speed_fraction — episode terminates when speed falls below this
    # (catches truly stuck / reversed agents; keep low so normal braking is fine)
    slow_stop_speed_fraction: float = 0.02  # terminate if speed < 3 % of vel_cap
    # Steps after reset before the slow-stop check arms, so the car can accelerate
    # away from a standstill spawn without being killed for it.
    slow_stop_grace_steps: int = 45  # ≈1.5 s at 30 Hz

    # ════════════════════════════════════════════════════════════════════════
    # OBSERVATION EXTRAS
    # ════════════════════════════════════════════════════════════════════════
    # Append the current average wheel surface speed to the observation.
    # Normalized: 0 = stopped, 1 = max_velocity_m_s.
    # This is the primary speed signal — essential for braking-before-corners.
    # On real hardware: read wheel encoder RPM, average all 4, multiply by WHEEL_RADIUS,
    # divide by max_velocity_m_s, clamp to [0, 1].
    include_wheel_vel: bool = True

    # Append the last N (throttle, steer) commands the policy issued, oldest first,
    # so the tail of the observation reads [a_{t-N+1} … a_t].  0 disables.
    #
    # Not a nicety: action_delay_ms puts one policy step between a command and the
    # actuators, so without this the network cannot know what steering angle is
    # currently applied, and a memoryless MLP physically cannot produce smooth
    # closed-loop control — it shows up as servo chatter and sliding.
    #
    # Beyond the first entry, the history is the only temporal context the policy
    # has: the actor is a plain MLP with no recurrence, and each depth frame is a
    # snapshot, so a window of past commands is what lets it infer what its own
    # inputs have been doing — whether it is mid-way through a steering sweep or
    # holding a line, and whether it has been on the brakes for one step or five.
    # At 30 Hz, 5 steps is ~167 ms of context, comfortably longer than the 33 ms
    # actuator delay it has to compensate for.
    #
    # Free on the real car: these are just the commands already sent.  Changing this
    # changes the observation width, so ConeTrackActorCriticCfg.num_proprio_obs must
    # be updated to match (= include_wheel_vel + 2 × prev_action_history_len), and
    # any exported policy must be fed the same layout on the Pi.
    prev_action_history_len: int = 5

    # ════════════════════════════════════════════════════════════════════════
    # REWARD WEIGHTS
    # ════════════════════════════════════════════════════════════════════════
    # ── Crash penalty ────────────────────────────────────────────────────────
    # One-off cost charged on the step a wall contact is latched (the same step the
    # episode terminates), scaled by how hard the car actually hit:
    #     r_crash = -crash_penalty_weight × (v_normal / max_velocity_m_s) ** crash_penalty_exponent
    # v_normal = |v · n̂| is the component of the car's world-frame velocity along the
    # normal of the wall it touched, taken from the nearest segment of the cached wall
    # footprint.  This makes the penalty a pure function of impact severity:
    #     head-on at top speed   → v ∥ n̂ → ratio 1.0   → full crash_penalty_weight
    #     tangential graze       → v ⟂ n̂ → ratio ≈ 0   → ≈ 0
    # so brushing a barrier while running alongside it costs almost nothing while
    # ploughing into one is expensive.  The exponent controls how sharply the two are
    # separated: 2.0 tracks kinetic energy, 1.0 tracks momentum and bites harder at
    # the low impact speeds (~0.8 m/s) the car currently achieves.  Revisit once it
    # is fast enough for the quadratic to matter.  Set the weight to 0.0 to disable.
    crash_penalty_weight: float = 3.0
    crash_penalty_exponent: float = 1.0

    # ── Stall penalty ────────────────────────────────────────────────────────
    # Charged when the episode ends via the slow-stop rule.  MANDATORY companion to
    # the crash penalty: without it, "stop moving" is a FREE exit.  Standing still
    # earns nothing, but it also risks nothing, and it ends the episode before any
    # crash penalty can be charged — so the moment crashing became expensive,
    # sitting at the spawn became the highest-return policy available, and the agent
    # took it.  This makes that exit strictly worse than driving: a stalled episode
    # returns -stall_penalty_weight, while even a clumsy lap banks distance reward.
    stall_penalty_weight: float = 3.0

    # Distance: rewards physical metres covered each step (v × dt).
    distance_weight: float = 4.0

    # Steering deadzone penalty: only penalises steering beyond the deadzone.
    steer_deadzone_weight: float = 0.005
    steer_deadzone: float = 0.05  # fraction of max_steer_rad that is free

    # Steering-rate (smoothness) penalty: penalises the change in steering command
    # between consecutive policy steps, r = -weight × |steer_t − steer_{t-1}|.
    # Directly suppresses the jerky servo chatter; encourages smooth, committed lines.
    steer_rate_weight: float = 0.04

    # Throttle-rate penalty: penalises the change in throttle/brake command
    # between consecutive policy steps to suppress banging between full throttle and brake.
    throttle_rate_weight: float = 0.015

    # Wheel slip penalty — ONE-SIDED: charges wheelspin (wheels turning faster than
    # the car) but not lockup (wheels slower than the car).  It used to be symmetric,
    # |wheel_vel − forward_speed|, which taxed hard braking at exactly the same rate
    # as wheelspin.  Braking hard is precisely what the car must do on corner entry,
    # and brake_throttle_scale=0.4 sits above the ~0.3 slip onset, so the whole top
    # of the brake range was penalised — a second of hard braking cost ~11x the
    # explicit crash penalty.  That is a large part of "enters corners too fast,
    # then slides".
    slip_weight: float = 1.0

    # ── Lateral slip (sideslip / oversteer) penalty ──────────────────────────
    # Charges the car for travelling at an angle to where it points — i.e. sliding.
    # Zero in a straight line BY CONSTRUCTION, whatever the wheels are doing, so it
    # can be made firm without taxing straight-line acceleration or braking the way
    # the longitudinal slip_weight does.
    #
    #   r = -lateral_slip_weight × ((beta - deadzone) / (pi - deadzone))**2
    #
    # Calibrated against the measured distance reward at the current 2.7 m/s average
    # (Rew/distance ≈ 0.054 per step), so a normal cornering slip angle is cheap and
    # a deliberate slide is not worth what it buys:
    #
    #     sideslip     penalty/step    vs distance reward
    #       8.6°          0.000              free (deadzone)
    #      20°            0.0066             12 %
    #      30°            0.023              43 %
    #      45°            0.068             126 %   <- sliding now costs more than
    #      90°            0.34              630 %      the progress it makes
    #
    # If the agent still flicks the rear out, raise the weight before touching the
    # deadzone — shrinking the deadzone starts penalising legitimate fast cornering.
    lateral_slip_weight: float = 1
    lateral_slip_deadzone_rad: float = 0.15  # ≈8.6°, free: normal tyre slip angle
    lateral_slip_min_speed_m_s: float = 0.5  # below this the angle is meaningless

    # ── Wheel-lift penalty ───────────────────────────────────────────────────
    # Charges the car for any wheel that is not carrying load.  This is the ROOT
    # CAUSE of the corner-slide exploit rather than one of its symptoms: the car
    # enters too fast, lateral load transfer unloads the inside wheels, and once a
    # wheel is off the ground the remaining contact patches cannot hold the corner,
    # so the rear steps out.  Some cars catch it and land back on four wheels, some
    # tip over.  The lateral-slip penalty above only sees the slide AFTER it starts;
    # this one prices the condition that causes it, one step earlier.
    #
    #   r = -wheel_lift_weight × (wheels_off / 4)**wheel_lift_exponent
    #
    # The exponent matters more than the weight.  One unloaded wheel is the normal
    # consequence of cornering genuinely hard and should cost something but stay
    # survivable; TWO wheels off is a rollover already in progress and must be far
    # worse than twice as bad.  Squaring gives exactly that shape:
    #
    #     wheels off    penalty/step    vs distance reward @2.7 m/s (0.054/step)
    #        1             0.25              460 %
    #        2             1.00             1850 %
    #        3             2.25             4170 %
    #        4             4.00             7410 %
    #
    # In episode terms a 0.5 s (15-step) tip-up on one wheel costs 3.75, i.e. more
    # than a full-speed crash (3.0); on two wheels it costs 15, roughly a whole lap
    # bonus.  Cornering that needs a wheel in the air is therefore never profitable.
    #
    # Detection is the vertical ground reaction on each wheel link, peak-held over
    # the 4 physics substeps of the policy step, so a single-substep contact dropout
    # cannot produce a false accusation.  Static load is 1.27 kg × 9.81 / 4 ≈ 3.1 N
    # per wheel, so the 0.5 N threshold fires at ~16 % of static load — slightly
    # BEFORE the tyre physically leaves the ground, which is deliberate: a wheel
    # carrying 0.4 N generates no useful grip either way, and the earlier signal is
    # easier to act on.  Suppressed during spawn grace, when the car is still
    # dropping the 2 mm onto the ground and the sensor history is post-reset zeros.
    wheel_lift_weight: float = 3.0
    wheel_lift_exponent: float = 2.0
    wheel_ground_force_thresh_n: float = 0.5  # vertical load below this = unloaded

    # ── Lap completion reward ────────────────────────────────────────────────
    # Sparse reward fired when the agent crosses the start line (the line through
    # the spawn point perpendicular to the spawn heading) while back inside the
    # `lap_return_radius_m` start circle, after having left that circle.
    #
    #   r_lap = lap_reward_weight × (lap_reward_ref_time_s / lap_time)**lap_reward_exponent
    #
    # ACCELERATING payoff: the exponent makes each further second of lap time worth
    # progressively MORE, which is the point — shaving 15 s → 14 s is easy and should
    # pay little, while 10 s → 9 s is hard and should pay a lot.  The marginal reward
    # per second saved is  p·r_lap/T, so it grows as the lap time falls:
    #
    #     lap time    r_lap    marginal reward per 1 s saved
    #      15.0 s     10.5              1.4
    #      13.0 s     14.0              2.2      <- current operating point
    #      11.4 s     18.3              3.2
    #      10.0 s     23.7              4.7
    #       8.0 s     36.9              9.2
    #
    # The reference time anchors the scale: a lap at exactly lap_reward_ref_time_s
    # pays exactly lap_reward_weight.  It is set to the lap time the agent is
    # currently achieving, and the weight to what the old linear form paid there
    # (6 × 30/13 ≈ 13.9), so switching to the exponent changes the SHAPE of the
    # incentive without changing the scale of the return.  That matters: at
    # lap_reward_weight=15 the old form put +35 into a single step, the critic could
    # not fit the spike (value loss 3.6 → 13.9) and the resulting high-variance
    # advantages pinned the LR to its floor.  Raise the weight only alongside the
    # value-loss trace.
    lap_reward_weight: float = 16.0  # reward for a lap at exactly the reference time
    lap_reward_ref_time_s: float = 13.0  # calibration anchor (current typical lap)
    lap_reward_exponent: float = 2.0  # 1.0 = the old ~linear form; 2.0 = quadratic
    lap_return_radius_m: float = (
        0.3  # start circle: leaving it starts the lap timer; crossing the
    )
    # start line while back inside closes the lap
    lap_min_time_s: float = 5.0  # lower clamp (caps maximum reward per lap)
    lap_max_time_s: float = 30.0  # upper clamp (floors the reward for a crawl)

    # ════════════════════════════════════════════════════════════════════════
    # SCENE DRESSING — decorative asset sets (visual only, no collision role)
    # ════════════════════════════════════════════════════════════════════════
    # Exactly one asset set is visible at a time; every this many policy steps the
    # scene swaps to the next one in the round-robin, so over an iteration the
    # policy sees the track dressed several different ways instead of memorising
    # one fixed arrangement of scenery.
    #
    # Set to num_steps_per_env / (NUM_ASSET_SETS * 2) so every set appears exactly
    # twice per PPO iteration.  At num_steps_per_env=1200: 1200 / (5*2) = 120.
    asset_randomize_every_n_steps: int = 120

    # ════════════════════════════════════════════════════════════════════════
    # HARDWARE CONFIG  (robot articulation + camera sensor)
    # ════════════════════════════════════════════════════════════════════════

    robot_cfg: ArticulationCfg = ArticulationCfg(
        prim_path="/World/envs/env_.*/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=MUSHR_USD,
            scale=(ROBOT_SCALE, ROBOT_SCALE, ROBOT_SCALE),
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
        update_period=4.0
        / 120.0,  # render only at policy steps (decimation=4, sim=120Hz)
        # Render at the sensor's native 240x180 to match the real ArduCam TOF.
        # All noise effects are applied at this full resolution before cropping
        # and downsampling, mirroring how the real camera produces noisy depth.
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
        update_period=4.0
        / 120.0,  # render only at policy steps (decimation=4, sim=120Hz)
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

        # Proprioceptive tail, appended after the two depth images.  Its width must
        # match ConeTrackActorCriticCfg.num_proprio_obs, which is the boundary the
        # policy network splits on to route these through their own encoder.
        self._action_history_len = max(0, cfg.prev_action_history_len)
        extra_obs = 0
        if cfg.include_wheel_vel:
            extra_obs += 1  # scalar: avg wheel surface speed normalized to [0, 1]
        # 2 entries (throttle, steer) per remembered command
        extra_obs += 2 * self._action_history_len
        cfg.observation_space = depth_obs_size * 2 + extra_obs  # left + right cameras
        self._num_proprio_obs = extra_obs

        print(
            f"[ConeTrackEnv] Depth obs per camera: raw {cam_w}×{cam_h} "
            f"→ crop {cropped_w}×{cropped_h} "
            f"→ policy {self._policy_depth_w}×{self._policy_depth_h} "
            f"= {depth_obs_size} px  |  2 cameras  |  "
            f"+{extra_obs} proprio ({self._action_history_len} action(s) of history)"
            f"  |  total obs = {cfg.observation_space}"
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
        self._steer_angle = torch.zeros(self.num_envs, device=self.device)

        # Precomputed tensor of all 4 driven wheel joint indices [rear_L, rear_R, front_L, front_R]
        self._all_throttle_ids: Any = torch.tensor(
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

        # Rolling history of the last _action_history_len raw (throttle, steer)
        # commands, oldest at index 0 and newest at index -1.  Fed back as an
        # observation so the network knows what the actuators are applying and what
        # it has been asking for over the preceding few steps.
        self._action_history = torch.zeros(
            self.num_envs, self._action_history_len, 2, device=self.device
        )

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
        # Fixed (NOT throttle-scaled): the torque-speed LINE pivots about this point while
        # its stall end scales with duty, which is how a PWM-driven brushed motor behaves.
        self._omega_noload_full = cfg.max_velocity_m_s / WHEEL_RADIUS
        _a0_full = 4.0 * cfg.drive_torque / WHEEL_RADIUS / cfg.car_mass_kg
        print(
            f"[ConeTrackEnv] Drivetrain: effort-controlled DC motor — "
            f"no-load {cfg.max_velocity_m_s:.2f} m/s ({self._omega_noload_full:.1f} rad/s/wheel), "
            f"drive_torque {cfg.drive_torque:.3f} Nm/wheel, brake_gain {cfg.brake_torque_gain:.3f} Nm/wheel/|u|"
        )
        print(
            f"[ConeTrackEnv]   standing-start accel = {_a0_full:.2f}·throttle m/s² "
            f"(real car: 5.95·duty, from real_car_acceleration_log.txt)"
        )

        # Steering: report the KINEMATIC CURVATURE, since κ = tan(δ)/L is what the vehicle
        # actually does and neither δ nor L means anything without the other.  Compare
        # against the real car's 2.90 1/m kinematic (28.5° protractor-measured at L=187 mm).
        # The curvature the car PULLS on track is lower — 1.1–1.7 1/m by optical flow — but
        # that gap is tyre slip and belongs to ground_*_friction, not here.
        _applied_steer = cfg.max_steer_rad
        _wheelbase = cfg.wheelbase_m if cfg.wheelbase_m > 0.0 else NATIVE_WHEELBASE_M
        _kappa = math.tan(_applied_steer) / _wheelbase
        print(
            f"[ConeTrackEnv] Steering: max_steer {cfg.max_steer_rad:.3f} rad "
            f"→ applied {math.degrees(_applied_steer):.1f}° at wheelbase {_wheelbase * 1000:.0f} mm "
            f"→ curvature {_kappa:.2f} 1/m (min turn radius {1.0 / _kappa:.2f} m)"
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
        # Contact / termination state.
        #   _wall_touched      — latched on the first wall contact of the episode.
        #                        Terminates the episode and, on that same step,
        #                        charges the crash penalty; the reset that follows
        #                        clears it, so the penalty can never double-charge.
        #   _crash_impact_speed — severity (m/s along the wall normal) of that hit,
        #                        resolved once per step by _resolve_crash_impact_speed.
        self._wall_touched = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self._crash_impact_speed = torch.zeros(self.num_envs, device=self.device)
        # Set by _get_dones each step, consumed by _get_rewards for the stall penalty.
        self._stalled = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        # World-frame XY velocity and position captured the instant a wall hit is
        # latched — i.e. before the collision has bled the car's speed off.  The
        # impact ANGLE is resolved against the wall geometry once per policy step in
        # _get_rewards, which keeps the nearest-segment search off the substep path.
        self._crash_impact_vel = torch.zeros(self.num_envs, 2, device=self.device)
        self._crash_impact_pos = torch.zeros(self.num_envs, 2, device=self.device)

        # World XY position at reset — the origin of the lap start line.
        self._spawn_pos = torch.zeros(self.num_envs, 2, device=self.device)
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

        # Depth observation caches (policy resolution after crop + downsample)
        _cam_flat = self._policy_depth_h * self._policy_depth_w
        self._camera_left_obs = torch.zeros(
            self.num_envs, _cam_flat, device=self.device
        )
        self._camera_right_obs = torch.zeros(
            self.num_envs, _cam_flat, device=self.device
        )
        self._last_camera_update_step = -1
        # Counts calls into the confidence dropout so the diagnostic can be throttled.
        self._conf_report_counter = 0
        # Base camera extrinsics, captured before any jitter is written over them, so the
        # jitter is always applied relative to the nominal mount rather than compounding.
        self._camera_base_offsets = {
            "camera_left": (
                tuple(cfg.camera_left_cfg.offset.pos),
                tuple(cfg.camera_left_cfg.offset.rot),
            ),
            "camera_right": (
                tuple(cfg.camera_right_cfg.offset.pos),
                tuple(cfg.camera_right_cfg.offset.rot),
            ),
        }
        # Latched False after the first failure so a USD write path that does not work on
        # this Isaac version degrades to "no jitter" instead of spamming every reset.
        self._camera_jitter_ok = (
            cfg.camera_pitch_jitter_deg > 0.0 or cfg.camera_height_jitter_m > 0.0
        )

        # 2D wall footprint, read once from the same /World/Walls mesh used for
        # termination (visibility / collision untouched) and cached on-device.
        # Consumed by _wall_normal_at to get the impact angle for the crash penalty.
        _wall_segs = self._extract_wall_segments_xy()
        if _wall_segs:
            _seg_arr = torch.from_numpy(np.asarray(_wall_segs, dtype=np.float32)).to(
                self.device
            )
            self._wall_seg_a = _seg_arr[:, 0, :].contiguous()  # [M, 2]
            self._wall_seg_b = _seg_arr[:, 1, :].contiguous()  # [M, 2]
            print(
                f"[ConeTrackEnv] Wall footprint: cached {_seg_arr.shape[0]} segments"
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

        # Re-space the axles to the real car's wheelbase before anything else touches the
        # robot, so every downstream consumer (collision setup, cloning) sees final geometry.
        self._apply_wheelbase(stage)

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

        # Walls are collision-only: exact ("none" approximation) triangle collision so
        # contacts are reported against the true geometry, kinematic so they never move,
        # and invisible so the depth cameras see the track rather than a boundary box.
        wall_root = stage.GetPrimAtPath("/World/Walls")
        if wall_root.IsValid():
            for prim in Usd.PrimRange(wall_root):
                if not prim.IsA(UsdGeom.Mesh):
                    continue
                wall_collision = UsdPhysics.CollisionAPI.Apply(prim)
                wall_collision.GetCollisionEnabledAttr().Set(True)
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
                cross_len = math.sqrt(fwd_z**2 + fwd_x**2)
                if cross_len < 1e-6:
                    orient = (
                        Gf.Quatf(1.0, 0.0, 0.0, 0.0)
                        if dot > 0
                        else Gf.Quatf(0.0, 1.0, 0.0, 0.0)
                    )
                else:
                    w_q = math.sqrt(max(0.0, 0.5 * (1.0 + dot)))
                    k = math.sqrt(max(0.0, 0.5 * (1.0 - dot))) / cross_len
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
    # Camera extrinsics
    # ------------------------------------------------------------------

    def _randomize_camera_extrinsics(self, env_ids) -> None:
        """Give each env a slightly different camera mount, resampled on reset.

        The sim's cameras are perfect: both at 65.5 mm, both exactly level, identical to
        each other in every environment.  The real pair is not — fitting the ground plane
        in the recorded frames puts the left at 79 mm / 1.15° nose-down and the right at
        63.5 mm / 1.32° nose-down, and the chassis pitches on top of that.  1° moves the
        horizon 3.7 rows of a 36-row crop band, so training on a perfect mount teaches the
        policy to trust a row index that means something different on the car.

        Implemented by rewriting the camera prims' local xform ops — the same attributes
        TiledCameraCfg.OffsetCfg writes at spawn — so nothing downstream has to know: both
        the renderer and Camera._update_poses read the prim transform every frame.

        Positions are authored in the robot's UNSCALED frame (ROBOT_SCALE is applied above
        them in the hierarchy), so a world-metre height jitter is divided by ROBOT_SCALE
        on the way in.  Pitch is pre-multiplied, i.e. applied in the vehicle frame about
        its lateral axis, which is where a mounting or chassis-attitude error actually
        lives — the same place the cfg's yaw term is applied.
        """
        if not self._camera_jitter_ok:
            return
        try:
            import omni.usd
            from pxr import Gf, UsdGeom
            from isaaclab.utils.math import (
                convert_camera_frame_orientation_convention,
            )

            stage = omni.usd.get_context().get_stage()
            pitch_max = math.radians(self.cfg.camera_pitch_jitter_deg)
            height_max = self.cfg.camera_height_jitter_m / ROBOT_SCALE

            for env_id in (int(i) for i in env_ids):
                for cam_name, (base_pos, base_rot) in self._camera_base_offsets.items():
                    prim = stage.GetPrimAtPath(
                        f"/World/envs/env_{env_id}/Robot/mushr_nano/base_link/{cam_name}"
                    )
                    if not prim.IsValid():
                        raise RuntimeError(f"camera prim missing for env {env_id}")

                    pitch = (torch.rand(1).item() * 2.0 - 1.0) * pitch_max
                    dz = (torch.rand(1).item() * 2.0 - 1.0) * height_max

                    # q_pitch about the vehicle's +Y (nose-down for +pitch), pre-multiplied
                    # onto the nominal ROS-convention mount orientation.
                    half = pitch * 0.5
                    qp = torch.tensor(
                        [math.cos(half), 0.0, math.sin(half), 0.0], dtype=torch.float32
                    )
                    qb = torch.tensor(base_rot, dtype=torch.float32)
                    w1, x1, y1, z1 = qp.tolist()
                    w2, x2, y2, z2 = qb.tolist()
                    q_ros = torch.tensor(
                        [
                            [
                                w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
                                w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                                w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                                w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
                            ]
                        ],
                        dtype=torch.float32,
                    )
                    q_gl = convert_camera_frame_orientation_convention(
                        q_ros, origin="ros", target="opengl"
                    )[0].tolist()

                    xf = UsdGeom.Xformable(prim)
                    for op in xf.GetOrderedXformOps():
                        name = op.GetOpName()
                        if name.endswith("translate"):
                            op.Set(
                                Gf.Vec3d(
                                    base_pos[0], base_pos[1], base_pos[2] + dz
                                )
                            )
                        elif name.endswith("orient"):
                            quat_type = type(op.Get())
                            imag_type = type(op.Get().GetImaginary())
                            op.Set(
                                quat_type(
                                    q_gl[0], imag_type(q_gl[1], q_gl[2], q_gl[3])
                                )
                            )
        except Exception as e:  # noqa: BLE001 — degrade to a fixed mount, never crash training
            self._camera_jitter_ok = False
            print(
                f"[ConeTrackEnv] Camera extrinsic randomisation disabled ({e}). "
                f"Training continues with the nominal mount; the sim will be optimistic "
                f"about how well-calibrated the real cameras are."
            )

    # ------------------------------------------------------------------
    # Chassis geometry
    # ------------------------------------------------------------------

    def _apply_wheelbase(self, stage) -> None:
        """Re-space the front/rear axles so the spawned wheelbase is cfg.wheelbase_m.

        WHY.  mushr_nano_v2.usd is spawned at ROBOT_SCALE=0.755 to bring its wheels down
        to the real car's 74 mm, and a uniform scale shrinks the wheelbase with them:
        296.5 mm × 0.755 = 224 mm against the real car's measured 187 mm.  A 20 % long
        wheelbase understates the curvature the car gets from a given steer angle, so it
        has to be corrected alongside max_steer_rad or the two errors partly mask each
        other (they did: the sim pulled 2.4-2.6 1/m against the real car's 1.1-1.7).

        HOW.  The wheelbase enters the *physics* in exactly one place — the X component of
        the four suspension joints' localPos0, which position each corner's chain relative
        to base_link.  Everything below (steer joint, throttle joint) is coincident with
        zero local offset, and every wheel mesh is authored about its own link origin, so
        nothing else has to move.  The link Xform translates are updated too: PhysX derives
        body poses from the joint chain and would correct them anyway, but leaving them
        stale would put the visuals in the wrong place for the first frame and in the
        viewport before play.

        Both axles are scaled about base_link's origin (x = 0) rather than about the axle
        midpoint, which keeps each axle's distance from the chassis in proportion and so
        preserves the front/rear static weight split that car_mass_kg was matched to.

        Raises on any missing attribute rather than silently applying a partial edit — a
        half-moved axle is a broken articulation that would be very hard to diagnose from
        training curves.
        """
        target = self.cfg.wheelbase_m
        if target <= 0.0:
            print(
                f"[ConeTrackEnv] Wheelbase: left at the USD's native "
                f"{NATIVE_WHEELBASE_M * 1000:.0f} mm (cfg.wheelbase_m = 0)"
            )
            return

        # Work in unscaled USD units; ROBOT_SCALE is applied on top at spawn.
        factor = (target / ROBOT_SCALE) / USD_WHEELBASE_M
        if abs(factor - 1.0) < 1e-6:
            return

        root = "/World/envs/env_0/Robot/mushr_nano"
        corners = ("front_left", "front_right", "back_left", "back_right")
        # Every prim whose X carries the axle position, per corner.  The front corners
        # have a steer link in the chain, the rear ones do not.
        link_suffixes = {
            "front": ("_wheel_suspension_link", "_wheel_steer_link", "_wheel_link"),
            "back": ("_wheel_suspension_link", "_wheel_link"),
        }

        moved = []
        for corner in corners:
            joint_prim = stage.GetPrimAtPath(
                f"{root}/base_link/{corner}_wheel_suspension"
            )
            if not joint_prim.IsValid():
                raise RuntimeError(
                    f"[ConeTrackEnv] Wheelbase: suspension joint for '{corner}' not found "
                    f"at {root}/base_link/{corner}_wheel_suspension — the robot USD "
                    f"changed shape; fix _apply_wheelbase before training."
                )
            attr = joint_prim.GetAttribute("physics:localPos0")
            if not attr.IsValid() or attr.Get() is None:
                raise RuntimeError(
                    f"[ConeTrackEnv] Wheelbase: '{corner}' suspension joint has no "
                    f"physics:localPos0 to rescale."
                )
            pos = attr.Get()
            old_x = float(pos[0])
            new_x = old_x * factor
            attr.Set(type(pos)(new_x, pos[1], pos[2]))
            moved.append((corner, old_x, new_x))

            for suffix in link_suffixes[corner.split("_")[0]]:
                link = stage.GetPrimAtPath(f"{root}/{corner}{suffix}")
                if not link.IsValid():
                    raise RuntimeError(
                        f"[ConeTrackEnv] Wheelbase: link {corner}{suffix} not found."
                    )
                t = link.GetAttribute("xformOp:translate")
                if not t.IsValid() or t.Get() is None:
                    raise RuntimeError(
                        f"[ConeTrackEnv] Wheelbase: {corner}{suffix} has no "
                        f"xformOp:translate to rescale."
                    )
                v = t.Get()
                t.Set(type(v)(float(v[0]) * factor, v[1], v[2]))

        front = next(x for c, _, x in moved if c.startswith("front"))
        back = next(x for c, _, x in moved if c.startswith("back"))
        achieved = (front - back) * ROBOT_SCALE
        print(
            f"[ConeTrackEnv] Wheelbase: {NATIVE_WHEELBASE_M * 1000:.0f} mm → "
            f"{achieved * 1000:.0f} mm (target {target * 1000:.0f} mm, axles scaled "
            f"×{factor:.4f} about base_link, {len(moved)} corners moved)"
        )

    # ------------------------------------------------------------------
    # Wall geometry helper
    # ------------------------------------------------------------------

    def _extract_wall_segments_xy(self):
        """Extract the 2D footprint of /World/Walls as a list of line segments.

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
                    "[ConeTrackEnv] /World/Walls not found — crash-impact angle "
                    "unavailable, every hit will be scored as head-on"
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
        """Show asset set set_idx; hide all others.

        set_idx == NUM_ASSET_SETS (or any out-of-range index) matches no set and
        therefore hides every decorative asset.
        """
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
        # already built every BVH during sim init). Now hide all but the first set.
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
        #   throttle > 0  DRIVE: T = drive_torque · clamp(throttle − ω/ω_noload_full, 0, 1)
        #                 — the brushed-DC torque-speed LINE.  Stall torque is proportional
        #                   to duty and the line hits zero torque at throttle·max_velocity.
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

        # DRIVE branch (throttle > 0).  The roll-off is (duty − normalised speed), NOT
        # (1 − speed/duty): the old form gave full stall torque at any duty, so the sim
        # launched as hard on 40 % throttle as on 100 % while the real car managed 40 % of
        # the shove.  See the DYNAMICS block for the measurements behind this.
        drive_rolloff = (
            u - wheel_omega / self._omega_noload_full
        ).clamp(0.0, 1.0)  # [N,4]
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
        #
        # The joint target is the road-wheel angle in RADIANS.  This used to be
        # tan(steer·max_steer_rad), justified in the comments as "the tan-convention of
        # mushr_nano_v2.usd" — but front_*_wheel_steer are plain PhysicsRevoluteJoints and
        # both get the same value (parallel steering, not an Ackermann linkage), so Isaac
        # Lab applied that tangent as an angle.  At max_steer_rad = 0.488 that quietly
        # steered 30.4° where every comment in this file said 28°.  The tan() is gone and
        # max_steer_rad now means exactly what it says.
        self._filtered_steer = delayed[:, 1]
        self._steer_angle = self._filtered_steer * self.cfg.max_steer_rad

        # Track the last two RAW (undelayed) steering/throttle commands for rate penalties.
        self._steer_cmd_prev = self._steer_cmd_curr
        self._steer_cmd_curr = raw[:, 1]
        self._throttle_cmd_prev = self._throttle_cmd_curr
        self._throttle_cmd_curr = raw[:, 0]

        # Push into the action history FIFO; the newest entry is this step's command,
        # which is what the actuators apply next step (action_delay_ms ≈ one tick).
        if self._action_history_len > 0:
            self._action_history = self._action_history.roll(-1, dims=1)
            self._action_history[:, -1] = raw

    def _wall_normal_at(self, pos_xy: torch.Tensor) -> torch.Tensor | None:
        """Unit XY normal of the wall segment nearest to each of the given positions.

        Uses the wall footprint cached at startup.  Only the
        handful of envs that just crashed are passed in, so the [K, M] point-to-segment
        distance matrix stays small.  The sign of the returned normal is arbitrary
        (either face of the wall); callers must take |v · n̂|.

        Returns [K, 2], or None if no wall geometry was extracted at startup.
        """
        if self._wall_seg_a is None or self._wall_seg_b is None:
            return None
        a = self._wall_seg_a.unsqueeze(0)  # [1, M, 2]
        ab = (self._wall_seg_b - self._wall_seg_a).unsqueeze(0)  # [1, M, 2]
        ap = pos_xy.unsqueeze(1) - a  # [K, M, 2]
        # Closest point on each segment: clamp the projection parameter to [0, 1].
        t = ((ap * ab).sum(-1) / ab.pow(2).sum(-1).clamp(min=1e-12)).clamp(0.0, 1.0)
        delta = ap - t.unsqueeze(-1) * ab  # [K, M, 2] point → segment
        nearest = delta.pow(2).sum(-1).argmin(dim=1)  # [K]
        seg = ab[0, nearest]  # [K, 2] direction of the hit wall
        normal = torch.stack([-seg[:, 1], seg[:, 0]], dim=-1)  # rotate 90°
        return normal / torch.linalg.vector_norm(
            normal, dim=-1, keepdim=True
        ).clamp(min=1e-6)

    def _update_wall_contact(self) -> torch.Tensor:
        """Poll the wall contact sensors, latch new hits and their pre-collision state.

        Called from both _apply_action (inside the physics substep loop) and _get_dones
        so that whichever poll first sees the contact captures the car's velocity while
        it is still the pre-collision one — a step later the wall has already absorbed
        it and every crash would look gentle.

        Any contact ends the episode, so the latch only ever fires once; the reset
        that follows clears it.  Severity is not judged here — contact force cannot
        tell a graze from a head-on hit (stopping 1.27 kg from 3 m/s inside one
        1/120 s substep is ~450 N either way).  _resolve_crash_impact_speed turns
        the latched pre-collision state into a severity for the crash penalty.

        Returns the accumulated, grace-filtered wall-contact mask.
        """
        peak_force = torch.zeros(self.num_envs, device=self.device)
        for sensor in self._wall_contact_sensors.values():
            fm = sensor.data.force_matrix_w
            if fm is not None:
                h = torch.linalg.vector_norm(fm[..., :2], dim=-1)
                peak_force = torch.maximum(
                    peak_force, h.reshape(self.num_envs, -1).amax(dim=1)
                )

        in_grace = self.episode_length_buf < self.cfg.spawn_grace_steps
        touching = peak_force.gt(self.cfg.wall_contact_force_thresh_n) & ~in_grace

        # Branchless latch of the pre-collision state: this runs on every physics
        # substep, so it must not read a mask back to the host (a `.any()` here would
        # add a GPU sync per substep).
        newly_hit = (touching & ~self._wall_touched).unsqueeze(-1)
        self._crash_impact_vel = torch.where(
            newly_hit, self.robot.data.root_lin_vel_w[:, :2], self._crash_impact_vel
        )
        self._crash_impact_pos = torch.where(
            newly_hit, self.robot.data.root_pos_w[:, :2], self._crash_impact_pos
        )

        self._wall_touched |= touching
        return self._wall_touched

    def _resolve_crash_impact_speed(self, crashed: torch.Tensor) -> torch.Tensor:
        """Impact severity of this step's wall hits, in m/s along the wall normal.

        Resolves the latched pre-collision velocity against the normal of the
        nearest cached wall segment, so a tangential graze reads ≈0 and a head-on
        hit reads the full closing speed.  Runs once per policy step and only for
        the envs that actually crashed, which keeps the nearest-segment search off
        the physics substep path.
        """
        self._crash_impact_speed = torch.zeros(self.num_envs, device=self.device)
        idx = crashed.nonzero(as_tuple=False).squeeze(-1)
        vel_xy = self._crash_impact_vel[idx]
        normal = self._wall_normal_at(self._crash_impact_pos[idx])
        if normal is None:
            # No wall footprint cached — no angle information available, so treat
            # every hit as head-on rather than silently under-penalising crashes.
            self._crash_impact_speed[idx] = torch.linalg.vector_norm(vel_xy, dim=-1)
        else:
            self._crash_impact_speed[idx] = (vel_xy * normal).sum(dim=-1).abs()
        return self._crash_impact_speed

    def _apply_action(self) -> None:
        wall_mask = self._update_wall_contact()
        wheel_torque = self._wheel_torque.clone()
        wheel_torque[wall_mask] = (
            0.0  # cut drive torque after a wall hit (car coasts to a stop)
        )
        self.robot.set_joint_effort_target(
            wheel_torque, joint_ids=self._all_throttle_ids
        )

        steer_positions = self._steer_angle.unsqueeze(-1).expand(-1, 2).clone()
        steer_positions[wall_mask] = 0.0
        self.robot.set_joint_position_target(steer_positions, joint_ids=self._steer_ids)

    # ------------------------------------------------------------------
    # Depth camera cache  (called once per step, shared by obs + rewards)
    # ------------------------------------------------------------------

    def _process_single_depth(self, camera) -> torch.Tensor:
        """Shared depth pipeline for one TiledCamera → [N, policy_h*policy_w].

        Noise effects are applied at the camera's full 240×180 resolution before
        cropping and downsampling, matching the real ArduCam TOF sensor pipeline.
        """
        import torch.nn.functional as F

        depth_raw = camera.data.output["distance_to_image_plane"]  # [N, H, W, 1]
        depth = depth_raw[..., 0]  # [N, H, W]
        depth = torch.where(
            torch.isfinite(depth),
            depth,
            torch.tensor(self.cfg.depth_max_m, device=depth.device),
        )

        # Kept before any noise pass: the confidence model needs the true surface geometry
        # to recover incidence angles, and Gaussian noise on a grazing plane would swamp
        # the very gradient that identifies it.
        clean_depth = depth

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

        # Structured confidence dropout — the dominant real-sensor effect.  Runs on the
        # UNCLIPPED depth so that grazing surfaces still carry the steep gradient that
        # identifies them; clipping to depth_max_m first would flatten the far ground into
        # a constant and make it look like a head-on wall.
        if self.cfg.depth_conf_enable:
            depth = self._apply_confidence_dropout(depth, clean_depth)

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

    def _apply_confidence_dropout(
        self, depth: torch.Tensor, clean_depth: torch.Tensor
    ) -> torch.Tensor:
        """Invalidate pixels the real ArduCam would return no confidence for.

        A ToF pixel resolves phase only when the returned power clears the noise floor,
        and that power goes as  ρ · cos(incidence) / range².  So a pixel dies either
        because it is far away or because it is looking along a surface rather than at it.
        Both terms are needed: range alone would keep the sim's crisp ground ramp (it is
        only 1-1.5 m away in the crop band) which the real sensor never returns, because
        at a 65 mm mounting height the floor there is seen at 2-3° of incidence.

        cos(incidence) is recovered from the depth image itself.  One pixel of image
        displacement turns the view ray by 1/f_px radians; if that step changes the range
        by |∇d| then the surface tilts away from the ray by
            tan(incidence) = |∇d| / (d / f_px)
        which is exact for a locally planar surface and needs no normals buffer.

        Survivors keep their depth, the rest are set to depth_max_m — the same "unmeasured
        reads as far" convention as process_for_policy() on the car, so a dropout can never
        be mistaken for an obstacle at the lens.
        """
        cfg = self.cfg
        # Angular pixel pitch (rad/px) from the configured optics, so this stays correct if
        # the FOV or resolution is ever changed.
        f_px = (
            math.sqrt(
                cfg.camera_left_cfg.width**2 + cfg.camera_left_cfg.height**2
            )
            / 2.0
            / math.tan(math.radians(_TOF_DIAG_FOV_DEG / 2.0))
        )

        d = clean_depth.clamp(min=0.05)
        # Central differences, replicate-padded, matching the edge-weight pass above.
        d4 = d.unsqueeze(1)
        px = torch.nn.functional.pad(d4, (1, 1, 0, 0), mode="replicate")
        py = torch.nn.functional.pad(d4, (0, 0, 1, 1), mode="replicate")
        gx = (px[:, :, :, 2:] - px[:, :, :, :-2]) * 0.5
        gy = (py[:, :, 2:, :] - py[:, :, :-2, :]) * 0.5
        grad = torch.sqrt(gx * gx + gy * gy).squeeze(1)

        tan_inc = grad * f_px / d
        cos_inc = torch.rsqrt(1.0 + tan_inc * tan_inc)

        snr = (
            cfg.depth_conf_gain_m2
            * cos_inc.clamp(min=1e-4).pow(cfg.depth_conf_incidence_exponent)
            / d.clamp(min=0.05).pow(2)
        )
        p_valid = (cfg.depth_conf_max_pass * snr.clamp(max=1.0)).clamp(
            cfg.depth_conf_floor_pass, cfg.depth_conf_max_pass
        )

        dropped = torch.rand_like(depth) > p_valid
        out = torch.where(
            dropped, torch.full_like(depth, cfg.depth_max_m), depth
        )

        if cfg.depth_conf_report_every_n_steps > 0:
            self._conf_report_counter += 1
            if self._conf_report_counter >= cfg.depth_conf_report_every_n_steps:
                self._conf_report_counter = 0
                self._report_confidence(dropped, clean_depth, out)
        return out

    def _report_confidence(
        self, dropped: torch.Tensor, clean_depth: torch.Tensor, out: torch.Tensor
    ) -> None:
        """Print the dropout statistics to line up against the numbers measured on the car.

        Only the crop band is reported — the rows above and below it never reach the policy,
        and including them would flatter the numbers with sky and near floor.
        """
        h = clean_depth.shape[1]
        top = int(self.cfg.depth_crop_top_fraction * h)
        bot = h - int(self.cfg.depth_crop_bottom_fraction * h)
        band_drop = dropped[:, top:bot, :]
        band_clean = clean_depth[:, top:bot, :]
        in_range = band_clean < self.cfg.depth_max_m

        obs = (1.0 - out[:, top:bot, :].clamp(0.01, self.cfg.depth_max_m) / self.cfg.depth_max_m)
        rows = torch.nn.functional.interpolate(
            obs.unsqueeze(1),
            size=(self._policy_depth_h, self._policy_depth_w),
            mode="bilinear",
            align_corners=False,
        ).squeeze(1).mean(dim=(0, 2))
        far = torch.nn.functional.interpolate(
            band_drop.float().unsqueeze(1),
            size=(self._policy_depth_h, self._policy_depth_w),
            mode="bilinear",
            align_corners=False,
        ).squeeze(1).mean(dim=(0, 2))

        def fmt(t):
            return " ".join(f"{v:.3f}" for v in t.tolist())

        print("[ConeTrackEnv][ToF confidence] crop band statistics")
        print(
            f"    dropped overall {band_drop.float().mean():.3f} | "
            f"dropped among in-range pixels {band_drop[in_range].float().mean():.3f} "
            f"| in-range pixels {in_range.float().mean():.3f}"
        )
        print(f"    dropped per row (top→bottom): {fmt(far)}")
        print(
            "      real left : 0.878 0.867 0.848 0.827 0.810 0.792 0.769 0.750 0.733 "
            "0.718 0.705 0.696 0.684 0.665 0.645 0.630 0.617 0.614"
        )
        print(
            "      real right: 0.656 0.637 0.632 0.619 0.607 0.596 0.579 0.580 0.589 "
            "0.599 0.589 0.580 0.575 0.571 0.573 0.563 0.530 0.461"
        )
        print(f"    obs mean per row (top→bottom): {fmt(rows)}")
        print(
            "      real left  0.039 → 0.145 | real right 0.182 → 0.292 "
            "| sim before this model 0.133 → 0.433"
        )

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

        # Layout is [depth_left | depth_right | proprio...], and the proprio tail
        # MUST stay last and match ConeTrackActorCriticCfg.num_proprio_obs — the
        # policy network splits the flat vector on that boundary to route the
        # scalars through their own encoder branch.
        parts = [self._camera_left_obs, self._camera_right_obs]
        if self.cfg.include_wheel_vel:
            wheel_omega_mean = (
                self.robot.data.joint_vel[:, self._all_throttle_ids].abs().mean(dim=-1)
            )
            wheel_vel_norm = (
                wheel_omega_mean * WHEEL_RADIUS / self.cfg.max_velocity_m_s
            ).clamp(0.0, 1.0)
            parts.append(wheel_vel_norm.unsqueeze(-1))  # [N, 1]
        if self._action_history_len > 0:
            # Flattened oldest-first as [thr_{t-N+1}, str_{t-N+1}, …, thr_t, str_t].
            # The last pair is the command issued this step, i.e. the control state
            # the policy is about to be acting from; the earlier pairs are the only
            # temporal context a memoryless MLP gets.
            parts.append(self._action_history.flatten(1))  # [N, 2 * N_hist]

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

        forward_speed = lin_vel_b.clamp(min=0.0)
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
        # One-sided: only wheelspin (wheels outrunning the car) is charged.  Lockup
        # under braking is left free — see slip_weight for why taxing it was
        # actively harmful.
        slip_ratio = (wheel_vel - forward_speed).clamp(min=0.0) / (
            self.cfg.max_velocity_m_s
        )
        r_slip = -self.cfg.slip_weight * slip_ratio.clamp(0.0, 1.0)

        # ── Lateral slip (sideslip / oversteer) penalty ──────────────────────────
        # Sideslip angle = the angle between where the car is GOING and where it is
        # POINTING, from body-frame velocity:
        #     beta = atan2(|v_lateral|, v_forward)      0 = tracking straight
        #                                            pi/2 = travelling pure sideways
        #                                              pi = spun fully backwards
        # This is the term that targets the corner exploit — brake in, then power the
        # rear loose to rotate the car — and it is deliberately NOT the longitudinal
        # slip above.  The two are independent by construction: in a straight line
        # v_lateral is zero, so accelerating hard enough to spin the wheels or braking
        # hard enough to lock them costs exactly nothing here no matter how large the
        # longitudinal slip gets.  Only rotating the car away from its velocity vector
        # is charged.  That separation is why longitudinal slip_weight previously had
        # to be cranked to 1.2 to suppress the flick: it was a blunt proxy that taxed
        # every straight-line throttle and brake input to reach the cornering
        # behaviour it was actually aimed at.
        v_fwd_b = self.robot.data.root_lin_vel_b[:, 0]
        v_lat_b = self.robot.data.root_lin_vel_b[:, 1]
        planar_speed = torch.linalg.vector_norm(
            self.robot.data.root_lin_vel_b[:, :2], dim=-1
        )
        slip_angle = torch.atan2(v_lat_b.abs(), v_fwd_b)
        # Deadzone: tyres only generate cornering force by running at some slip angle,
        # so committed fast cornering must stay free — only a genuine slide is charged.
        slip_excess = (slip_angle - self.cfg.lateral_slip_deadzone_rad).clamp(min=0.0)
        slip_excess_norm = slip_excess / max(
            math.pi - self.cfg.lateral_slip_deadzone_rad, 1e-6
        )
        # Squared, so the penalty stays near zero just past the deadzone and climbs
        # steeply into a real slide, and speed-gated because the angle is numerically
        # meaningless when the car is barely moving (a car creeping at 0.05 m/s with
        # 0.04 m/s of drift reads 39°, which is noise, not a slide).
        slip_moving = (planar_speed > self.cfg.lateral_slip_min_speed_m_s).float()
        r_lateral_slip = (
            -self.cfg.lateral_slip_weight * slip_excess_norm.pow(2) * slip_moving
        )
        # Same gate on the diagnostic: report the angle averaged over the envs the
        # penalty actually applies to, otherwise stationary envs dominate the mean
        # with meaningless angles and the trace can't be read.
        slip_angle_deg = torch.rad2deg(slip_angle)
        slip_moving_n = slip_moving.sum().clamp(min=1.0)
        slip_angle_deg_gated = (slip_angle_deg * slip_moving).sum() / slip_moving_n
        slip_angle_deg_peak = (slip_angle_deg * slip_moving).max()

        # ── Wheel-lift penalty ───────────────────────────────────────────────────
        # Counts wheels carrying no vertical load, from the ground reaction the wall
        # sensors already report in net_forces_w (see WHEEL_LINKS for why no extra
        # sensor is needed).  The z component isolates ground support specifically:
        # the ground is a flat plane so its reaction is essentially world +z, while a
        # wall reaction is horizontal and cannot be mistaken for support.
        #
        # The history buffer holds one entry per physics substep (history_length =
        # decimation), and the peak over it is what decides contact, so a wheel counts
        # as DOWN if it carried load at any point during the policy step.  That errs
        # toward not charging the penalty, which is the right direction for a term
        # this heavy.
        wheels_down = torch.zeros(self.num_envs, device=self.device)
        for link_name in WHEEL_LINKS:
            forces = cast(
                torch.Tensor, self._wall_contact_sensors[link_name].data.net_forces_w_history
            )
            fz_peak = (
                forces[..., 2].abs().reshape(self.num_envs, -1).amax(dim=1)
            )
            wheels_down += fz_peak.gt(self.cfg.wheel_ground_force_thresh_n).float()

        n_wheels = float(len(WHEEL_LINKS))
        wheels_off = n_wheels - wheels_down
        lift_frac = wheels_off / n_wheels
        # Grace gate is mandatory, not cosmetic: _reset_idx zeroes the sensor history,
        # and the car spawns 2 mm above the ground, so without it every episode would
        # open with several steps of a full four-wheels-airborne charge.
        lift_active = (self.episode_length_buf >= self.cfg.spawn_grace_steps).float()
        r_wheel_lift = (
            -self.cfg.wheel_lift_weight
            * lift_frac.pow(self.cfg.wheel_lift_exponent)
            * lift_active
        )
        # Incidence rate (fraction of env-steps with at least one wheel unloaded) is
        # the readable trace; the mean count alone hides whether a rare 3-wheel event
        # or a constant 1-wheel one is driving it.
        lift_incidence = ((wheels_off > 0).float() * lift_active).mean()

        # ── Crash penalty ─────────────────────────────────────────────────────
        # Charged once, on the step the wall hit is latched — which is also the step
        # _get_dones terminates the episode, so _wall_touched is cleared by the reset
        # that immediately follows and the penalty can never be double-charged.
        # The angle is resolved here (once per policy step, only for the envs that
        # actually crashed) from the state latched in _update_wall_contact.
        crashed_now = self._wall_touched
        any_crash = bool(crashed_now.any())  # single host sync, reused for logging
        r_crash = torch.zeros(self.num_envs, device=self.device)
        impact_speed = self._crash_impact_speed
        if any_crash:
            impact_speed = self._resolve_crash_impact_speed(crashed_now)
            if self.cfg.crash_penalty_weight > 0.0:
                idx = crashed_now.nonzero(as_tuple=False).squeeze(-1)
                impact_norm = (impact_speed / self.cfg.max_velocity_m_s).clamp(
                    0.0, 1.0
                )
                r_crash[idx] = -(
                    self.cfg.crash_penalty_weight
                    * impact_norm[idx].pow(self.cfg.crash_penalty_exponent)
                )

        # ── Stall penalty ─────────────────────────────────────────────────────
        # Mirrors the crash penalty on the other exit from the episode.  _get_dones
        # has already run this step, so _stalled is current.
        r_stall = -self.cfg.stall_penalty_weight * self._stalled.float()

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
            self.cfg.lap_reward_weight
            * (self.cfg.lap_reward_ref_time_s / lap_time_clamped).pow(
                self.cfg.lap_reward_exponent
            ),
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

        # Crash metrics follow the same "empty on non-event steps" convention as the
        # lap keys above, so the iteration mean reads as the average over actual
        # crashes rather than being diluted by thousands of zero-padded steps.
        self.extras["log"] = {
            "Lap/lap_time_s": lap_times_done,
            "Lap/fastest_lap_s": fastest_log,
            "Lap/laps_per_min": laps_per_min,
            "Crash/impact_speed_m_s": impact_speed[crashed_now].detach()
            if any_crash
            else empty,
            "Crash/penalty": r_crash[crashed_now].detach() if any_crash else empty,
            # Event rates — zero-padded per-step MEANS over envs, so the iteration
            # mean is a per-env-step rate that is comparable across runs.  These used
            # to be sums, which silently scaled with num_envs and made runs at 12 and
            # 16 envs look different when they were behaving identically.
            "Term/wall": crashed_now.float().mean().unsqueeze(0),
            "Term/stall": self._stalled.float().mean().unsqueeze(0),
            # ── Reward decomposition ──────────────────────────────────────────
            # Per-env-step means, so they sum to Train/mean_reward / mean_episode_length
            # and each one's PER-EPISODE contribution is (its value × mean_episode_length).
            # The sparse terms (lap, crash, stall) are zero on almost every step, so read
            # them per-episode; the dense ones are meaningful per-step as they stand.
            # These exist because the split between distance, slip and the smoothness
            # penalties is otherwise only recoverable by arithmetic on the total, which
            # is how slip_weight went unnoticed at 100x every other coefficient.
            "Rew/distance": r_distance.mean().unsqueeze(0),
            "Rew/lap": r_lap.mean().unsqueeze(0),
            "Rew/crash": r_crash.mean().unsqueeze(0),
            "Rew/stall": r_stall.mean().unsqueeze(0),
            "Rew/slip": r_slip.mean().unsqueeze(0),
            "Rew/lateral_slip": r_lateral_slip.mean().unsqueeze(0),
            "Rew/wheel_lift": r_wheel_lift.mean().unsqueeze(0),
            "Rew/steer_deadzone": r_steer_shape.mean().unsqueeze(0),
            "Rew/steer_rate": r_steer_rate.mean().unsqueeze(0),
            "Rew/throttle_rate": r_throttle_rate.mean().unsqueeze(0),
            # Raw magnitudes behind the penalties, so a weight change can be reasoned
            # about without re-deriving the underlying quantity from its product.
            "Diag/slip_ratio": slip_ratio.mean().unsqueeze(0),
            # Sideslip angle over the moving envs only, mean and peak.  The peak is
            # the one to watch for the corner-flick: a slide that only happens at
            # corner entry is averaged away over the straights where the angle is ~0.
            "Diag/slip_angle_deg": slip_angle_deg_gated.unsqueeze(0),
            "Diag/slip_angle_deg_max": slip_angle_deg_peak.unsqueeze(0),
            # Wheels off the ground: mean count over all envs, and the fraction of
            # env-steps with at least one wheel unloaded.  The incidence is the one
            # that should collapse toward 0 as the penalty takes effect.
            "Diag/wheels_off_ground": (wheels_off * lift_active)
            .mean()
            .unsqueeze(0),
            "Diag/wheel_lift_incidence": lift_incidence.unsqueeze(0),
            "Diag/steer_rate_abs": steer_rate.mean().unsqueeze(0),
            "Diag/throttle_rate_abs": throttle_rate.mean().unsqueeze(0),
            "Diag/fwd_speed_m_s": forward_speed.mean().unsqueeze(0),
        }

        return (
            r_crash
            + r_stall
            + r_distance
            + r_steer_shape
            + r_steer_rate
            + r_throttle_rate
            + r_slip
            + r_lateral_slip
            + r_wheel_lift
            + r_lap
        )

    # ------------------------------------------------------------------
    # Termination
    # ------------------------------------------------------------------

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        # Polls the sensors once more: the final sim substep runs after the last
        # _apply_action, so a contact made on it is first seen here.
        wall_terminated = self._update_wall_contact()
        in_grace = self.episode_length_buf < self.cfg.spawn_grace_steps

        # Flip detection (suppressed during grace)
        q = self.robot.data.root_quat_w
        up_z = 1.0 - 2.0 * (q[:, 1] * q[:, 1] + q[:, 2] * q[:, 2])
        flipped = (up_z < 0.3) & ~in_grace

        # Near-stop termination — only fires if agent is essentially stationary
        # (slow_stop_speed_fraction × max_velocity_m_s), and only after
        # slow_stop_grace_steps so the car can accelerate away from a standstill spawn.
        lin_vel_b = self.robot.data.root_lin_vel_b[:, 0]
        stopped = (
            lin_vel_b < (self.cfg.slow_stop_speed_fraction * self.cfg.max_velocity_m_s)
        ) & (self.episode_length_buf > self.cfg.slow_stop_grace_steps)
        # Handed to _get_rewards (which runs immediately after) to charge the stall
        # penalty; without that cost, stalling out here is a free way to end the
        # episode while dodging all crash risk.
        self._stalled = stopped

        terminated = wall_terminated | flipped | stopped
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        return terminated, time_out

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = cast(Sequence[int], self.robot._ALL_INDICES)

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

        # Every episode starts from standstill (linear + angular velocity zeroed).
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
        self._action_history[env_ids] = 0.0

        self._wall_touched[env_ids] = False
        self._crash_impact_speed[env_ids] = 0.0
        self._crash_impact_vel[env_ids] = 0.0
        self._crash_impact_pos[env_ids] = 0.0
        self._spawn_pos[env_ids_t] = spawn_xy
        self._lap_state[env_ids_t] = 0
        self._lap_start_step[env_ids_t] = 0
        self._spawn_heading_cos[env_ids_t] = torch.cos(base_yaw)
        self._spawn_heading_sin[env_ids_t] = torch.sin(base_yaw)
        self._lap_prev_forward[env_ids_t] = 0.0
        # Note: _fastest_lap_s_global is intentionally NOT reset — it tracks the best ever.

        joint_pos = self.robot.data.default_joint_pos[env_ids]
        joint_vel = torch.zeros_like(joint_pos)
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

        # New episode, new camera calibration — see _randomize_camera_extrinsics.
        self._randomize_camera_extrinsics(env_ids)
