"""Drivetrain bench test — how fast the simulated car accelerates and stops.

Runs the real ConeTrackEnv (same physics, same actuator model) with collisions and
terminations disabled, holds a constant throttle until the target speed is reached,
then holds a constant brake until the car stops.  Reports top speed, phase times,
distance travelled, and how far the wheels rotated — the gap between the last two
is tyre slip.

This is the loop for matching the sim drivetrain to the measurements in
real_car_acceleration_log.txt: run it, compare, then adjust drive_torque /
brake_torque_base / brake_torque_gain / ground_*_friction in env.py.

Usage:
    uv run python acceleration_test.py --headless
    uv run python acceleration_test.py --headless --accel_throttle 0.5 --brake_power 0.5
"""

import argparse
import os

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="MuSHR nano v2 acceleration / braking test.")
parser.add_argument("--accel_throttle", type=float, default=1.0,
    help="Constant throttle during acceleration phase [0, 1].")
parser.add_argument("--brake_power", type=float, default=1.0,
    help="Constant braking magnitude during deceleration phase [0, 1].")
parser.add_argument("--max_speed", type=float, default=4.5,
    help="Speed (m/s) at which the acceleration phase is considered done.")
parser.add_argument("--max_time", type=float, default=20.0,
    help="Max simulation seconds before abort.")
parser.add_argument("--hide_obstacles", action="store_true", default=True,
    help="Disable walls/track/assets so car accelerates freely (default: on).")
parser.add_argument("--debug", action="store_true",
    help="Print per-step debug info.")

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ── Post-launch imports ───────────────────────────────────────────────────────
from typing import cast

import numpy as np
import torch
import gymnasium as gym
import cone_track  # noqa: F401  — registers the "ConeTrack" gym id

from cone_track.env import ConeTrackEnv, ConeTrackEnvCfg, WHEEL_RADIUS


def main():
    env_cfg = ConeTrackEnvCfg()
    env_cfg.scene.num_envs = 1
    # Isolate the drivetrain test to "ground plane + car" only:
    #   • walls/track/assets collision is disabled below (hide_obstacles)
    #   • wall-contact termination off (force threshold = ∞)
    #   • slow-stop termination off (negative fraction) — otherwise the episode
    #     "terminates" the instant the car brakes to a halt, which the loop below
    #     used to mis-report as "[WALL HIT] during braking".  With it disabled the
    #     braking phase ends cleanly via the velocity-based stop detector.
    env_cfg.wall_contact_force_thresh_n = float('inf')
    env_cfg.slow_stop_speed_fraction    = -1.0

    env = gym.make("ConeTrack", cfg=env_cfg)
    mushr_env = cast(ConeTrackEnv, env.unwrapped)
    device    = mushr_env.device

    # ── Hide obstacles so the car accelerates freely ──────────────────────────
    if args_cli.hide_obstacles:
        try:
            import omni.usd
            from pxr import Usd, UsdPhysics, UsdGeom
            stage = omni.usd.get_context().get_stage()
            for prim_path in ("/World/Walls", "/World/Track") + tuple(
                f"/World/Assets/Set{i}" for i in range(5)
            ):
                root = stage.GetPrimAtPath(prim_path)
                if root and root.IsValid():
                    for prim in Usd.PrimRange(root):
                        if prim.HasAPI(UsdPhysics.CollisionAPI):
                            col = UsdPhysics.CollisionAPI(prim)
                            col.GetCollisionEnabledAttr().Set(False)
                        if prim.HasAPI(UsdPhysics.MeshCollisionAPI):
                            prim.RemoveAPI(UsdPhysics.MeshCollisionAPI)
                        img = UsdGeom.Imageable(prim)
                        img.MakeInvisible()
            print("[HIDE] Walls, track, and assets disabled")
        except Exception as e:
            print(f"[HIDE] Failed to disable obstacles: {e}")

    # Read motor parameters directly from the env cfg (single source of truth)
    drive_torque      = mushr_env.cfg.drive_torque
    brake_torque_gain = mushr_env.cfg.brake_torque_gain
    max_vel_cfg       = mushr_env.cfg.max_velocity_m_s

    max_vel = args_cli.max_speed
    accel_throttle = float(np.clip(args_cli.accel_throttle, 0.0, 1.0))
    brake_power    = float(np.clip(args_cli.brake_power, 0.0, 1.0))

    print("─" * 60)
    print("MuSHR nano v2  —  Acceleration / Braking Test")
    # Query wheel body mass and inertia from PhysX
    try:
        wheel_body_names = [
            "back_left_wheel_link", "back_right_wheel_link",
            "front_left_wheel_link", "front_right_wheel_link",
        ]
        all_masses = mushr_env.robot.root_physx_view.get_masses()[0]
        all_body_names = mushr_env.robot.body_names
        print(f"  [TOTAL ARTICULATION MASS] = {all_masses.sum().item():.4f} kg "
              f"across {len(all_body_names)} bodies")
        for bn, bm in zip(all_body_names, all_masses.tolist()):
            print(f"      {bn:32s} {bm:.4f} kg")
        body_ids, _ = mushr_env.robot.find_bodies(wheel_body_names)
        if body_ids:
            masses    = mushr_env.robot.root_physx_view.get_masses()[0, body_ids]
            inertias9 = mushr_env.robot.root_physx_view.get_inertias()[0, body_ids]
            fn = "   Body"
            for name, m, i9 in zip(wheel_body_names, masses, inertias9):
                # inertia stored as [Ixx, Iyx, Izx, Ixy, Iyy, Izy, Ixz, Iyz, Izz]  (row-major 3×3)
                Ixx, Iyy, Izz = i9[0], i9[4], i9[8]
                print(f"  {fn} '{name}': mass={m.item():.4f} kg, "
                      f"Ixx={Ixx.item():.8f}, Iyy={Iyy.item():.8f}, Izz={Izz.item():.8f} kg·m²")
                fn = "         "
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"  [WARN] Could not query wheel inertias: {e}")

    print("─" * 60)
    print(f"  accel_throttle         = {accel_throttle:.2f}")
    print(f"  brake_power            = {brake_power:.2f}")
    print(f"  max_velocity_m_s       = {max_vel:.2f}  (cfg: {max_vel_cfg})")
    print(f"  car_mass_kg            = {mushr_env.cfg.car_mass_kg}")
    print(f"  drive_torque           = {drive_torque:.4f} Nm/wheel (stall)")
    print(f"  brake_torque_gain      = {brake_torque_gain:.4f} Nm/wheel per unit |throttle|")
    print(f"  no-load speed          = {max_vel_cfg:.2f} m/s")
    print(f"  WHEEL_RADIUS           = {WHEEL_RADIUS} m")
    print("─" * 60)

    all_ids = mushr_env._all_throttle_ids
    dt_env  = 1.0 / 30.0
    max_env_steps = int(args_cli.max_time * 30.0)

    env.reset()

    top_speed   = 0.0
    env_step_count = 0

    # ── ACCELERATION PHASE ────────────────────────────────────────────────────
    print(f"\n[ACCEL] Throttle = {accel_throttle:.2f}  target = {max_vel:.2f} m/s …")

    action = torch.tensor([[accel_throttle, 0.0]], device=device)
    wall_hit = False

    last_wv   = mushr_env.robot.data.joint_vel[0, all_ids].clone()
    start_pos = mushr_env.robot.data.root_pos_w[0, :2].clone()
    last_pos  = start_pos.clone()
    wheel_dist_accel = torch.zeros(4, device=device)

    while simulation_app.is_running():
        if env_step_count >= max_env_steps:
            print(f"[TIMEOUT] Aborting after {env_step_count * dt_env:.1f} s")
            break

        _, _, terminated, truncated, _ = env.step(action)
        env_step_count += 1

        wheel_dist_accel += last_wv.abs() * dt_env * WHEEL_RADIUS

        if terminated or truncated:
            wall_hit = True
            break

        last_wv  = mushr_env.robot.data.joint_vel[0, all_ids].clone()
        last_pos = mushr_env.robot.data.root_pos_w[0, :2].clone()
        fwd      = mushr_env.robot.data.root_lin_vel_b[0, 0].item()
        top_speed = max(top_speed, fwd)

        if args_cli.debug and env_step_count % 25 == 0:
            jv = mushr_env.robot.data.joint_vel[0, all_ids].tolist()
            pv = mushr_env.robot.root_physx_view
            try:
                applied = pv.get_dof_actuation_forces()[0, all_ids].tolist()
            except Exception:
                applied = []
            try:
                tgt = mushr_env._wheel_torque[0].tolist()
            except Exception:
                tgt = []
            print(f"  step{env_step_count:4d}: fwd={fwd:.3f}  "
                  f"jv={[f'{v:.1f}' for v in jv]}  "
                  f"cmdNm={[f'{v:.4f}' for v in tgt]}  "
                  f"appliedNm={[f'{v:.4f}' for v in applied]}  "
                  f"pos=({last_pos[0]:.2f},{last_pos[1]:.2f})")

        if top_speed >= max_vel * 0.99:
            break

    accel_steps   = env_step_count
    accel_time    = accel_steps * dt_env
    accel_end_pos = last_pos.clone()

    if top_speed < max_vel * 0.99 and not wall_hit:
        print(f"[WARN] Never reached target speed ({top_speed:.3f} / {max_vel:.2f} m/s)")
    if wall_hit:
        print(f"[WALL HIT] at step {accel_steps}  "
              f"top speed = {top_speed:.3f} m/s  "
              f"after {accel_time:.2f}s  (walls disabled — unexpected)")

    accel_car_dist = (accel_end_pos - start_pos).norm().item()
    print(f"  ── Top speed = {top_speed:.3f} m/s in {accel_time:.2f} s  "
          f"(car dist ≈ {accel_car_dist:.2f} m)")

    # ── BRAKING PHASE ─────────────────────────────────────────────────────────
    print(f"\n[BRAKE] brake_power = {brake_power:.2f}  stopping …")

    last_wv         = mushr_env.robot.data.joint_vel[0, all_ids].clone()
    brake_start_pos = mushr_env.robot.data.root_pos_w[0, :2].clone()
    wheel_dist_brake = torch.zeros(4, device=device)
    brake_steps     = 0
    stopped_streak  = 0
    brake_wall_hit  = False

    action_brake = torch.tensor([[-brake_power, 0.0]], device=device)

    while simulation_app.is_running():
        if accel_steps + brake_steps >= max_env_steps:
            print("[TIMEOUT] Aborting during braking")
            break

        _, _, terminated, truncated, _ = env.step(action_brake)
        brake_steps += 1

        wheel_dist_brake += last_wv.abs() * dt_env * WHEEL_RADIUS

        if terminated or truncated:
            brake_wall_hit = True
            break

        last_wv = mushr_env.robot.data.joint_vel[0, all_ids].clone()
        fwd     = mushr_env.robot.data.root_lin_vel_b[0, 0].item()

        if fwd <= 0.01:
            stopped_streak += 1
        else:
            stopped_streak = 0

        if stopped_streak >= 5:
            break

    if brake_wall_hit:
        print("[WALL HIT] during braking")

    brake_time = brake_steps * dt_env
    brake_end_pos = mushr_env.robot.data.root_pos_w[0, :2].clone()
    brake_car_dist = (brake_end_pos - brake_start_pos).norm().item()
    print(f"  ── Stopped after {brake_time:.2f} s  "
          f"(car dist ≈ {brake_car_dist:.2f} m)")

    # ── RESULTS ───────────────────────────────────────────────────────────────
    avg_wheel_accel = wheel_dist_accel.mean().item()
    avg_wheel_brake = wheel_dist_brake.mean().item()
    total_wheel     = avg_wheel_accel + avg_wheel_brake
    total_car       = accel_car_dist + brake_car_dist

    slip_frac_a = 0.0
    if avg_wheel_accel > 1e-9:
        slip_frac_a = max(0.0, 1.0 - accel_car_dist / avg_wheel_accel)

    print("\n" + "=" * 60)
    print("  RESULTS")
    print("=" * 60)
    print(f"  Top speed                 = {top_speed:.3f}  m/s")
    print(f"  Max velocity target       = {max_vel:.3f}  m/s")
    print(f"  Speed ratio               = {top_speed / max_vel:.3f}")
    print()
    print("  ── Acceleration phase ──")
    print(f"  Duration                  = {accel_time:.3f}  s")
    print(f"  Steps                     = {accel_steps}")
    print(f"  Car distance              = {accel_car_dist:.3f}  m")
    print(f"  Avg wheel rotation dist   = {avg_wheel_accel:.3f}  m  (mean of 4 wheels)")
    print(f"  Slip (wheel − car)        = {avg_wheel_accel - accel_car_dist:.3f}  m")
    print(f"  Slip fraction             = {slip_frac_a:.3f}")
    if avg_wheel_accel > 1e-9:
        eff_r = accel_car_dist / avg_wheel_accel * WHEEL_RADIUS
        print(f"  Effective wheel radius    = {eff_r:.4f}  m")
    print()
    print("  ── Braking phase ──")
    print(f"  Duration                  = {brake_time:.3f}  s")
    print(f"  Steps                     = {brake_steps}")
    print(f"  Car distance              = {brake_car_dist:.3f}  m")
    print(f"  Avg wheel rotation dist   = {avg_wheel_brake:.3f}  m  (mean of 4 wheels)")
    if avg_wheel_brake > 1e-9:
        slip_frac_b = max(0.0, 1.0 - brake_car_dist / avg_wheel_brake) if avg_wheel_brake > brake_car_dist else 0.0
        print(f"  Slip fraction             = {slip_frac_b:.3f}")
    print()
    print("  ── Combined ──")
    print(f"  Total car distance        = {total_car:.3f}  m")
    print(f"  Total avg wheel distance  = {total_wheel:.3f}  m")
    print("=" * 60)

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
