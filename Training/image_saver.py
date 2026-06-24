"""Continuously save raw RGB and policy-input frames from a trained policy.

Spawns 1 agent, runs a loaded checkpoint, and every ~1 second writes:
  {out_dir}/raw_{counter:06d}.png      – raw RGB camera view
  {out_dir}/policy_{counter:06d}.png    – processed black-white policy input

Filenames use an auto‑incrementing counter so previous frames are never
overwritten.

Usage:
    python image_saver.py --checkpoint /path/to/model.pt
    python image_saver.py --checkpoint Training/logs/rsl_rl/.../model.pt
"""

import argparse
import os
import sys

# Reduce PyTorch CUDA allocator fragmentation
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

from isaaclab.app import AppLauncher

import cli_args  # isort: skip

parser = argparse.ArgumentParser(description="Save camera frames from a trained policy.")
parser.add_argument("--task", type=str, default="ConeTrack", help="Name of the task.")
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point", help="RL agent config entry point."
)
parser.add_argument("--seed", type=int, default=None, help="Seed for the environment.")
parser.add_argument(
    "--use_pretrained_checkpoint",
    action="store_true",
    help="Use the pre-trained checkpoint from Nucleus.",
)
parser.add_argument(
    "--out_dir",
    type=str,
    default="/tmp/mushr_frames",
    help="Output directory for saved frames.",
)
parser.add_argument(
    "--save_interval",
    type=float,
    default=1.0,
    help="Approximate seconds between frame saves (default: 1.0).",
)
parser.add_argument(
    "--max_steps",
    type=int,
    default=None,
    help="Stop after this many environment steps (default: run forever).",
)
parser.add_argument(
    "--vel_cap",
    type=float,
    default=5.0,
    help="Override velocity cap (m/s).",
)
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli, hydra_args = parser.parse_known_args()
# always enable cameras — required for TiledCamera in the environment
args_cli.enable_cameras = True
args_cli.num_envs = 1

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import cv2
import gymnasium as gym
import numpy as np
import torch
from rsl_rl.runners import DistillationRunner, OnPolicyRunner

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.assets import retrieve_file_path

from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper
import isaaclab_tasks  # noqa: F401
import tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

# Register custom CNN-ActorCritic so rsl_rl's eval("ConeTrackActorCritic") resolves
import rsl_rl.runners.on_policy_runner
from tasks.cone_track.agents.actor_critic_vision import ConeTrackActorCritic
rsl_rl.runners.on_policy_runner.ConeTrackActorCritic = ConeTrackActorCritic


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    """Run policy and save frames."""
    # grab task name for checkpoint path
    task_name = args_cli.task.split(":")[-1]
    train_task_name = task_name.replace("-Play", "")

    # override configurations with non-hydra CLI arguments
    agent_cfg: RslRlBaseRunnerCfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs

    # set the environment seed
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # specify directory for logging experiments
    log_root_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs", "rsl_rl", agent_cfg.experiment_name)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    if args_cli.use_pretrained_checkpoint:
        from isaaclab_rl.utils.pretrained_checkpoint import get_published_pretrained_checkpoint
        resume_path = get_published_pretrained_checkpoint("rsl_rl", train_task_name)
        if not resume_path:
            print("[INFO] Unfortunately a pre-trained checkpoint is currently unavailable for this task.")
            return
    elif args_cli.checkpoint:
        ckpt = args_cli.checkpoint
        if not os.path.isabs(ckpt) and not os.path.isfile(ckpt):
            project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            candidate = os.path.join(project_root, ckpt)
            if os.path.isfile(candidate):
                ckpt = candidate
        resume_path = retrieve_file_path(ckpt)
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    log_dir = os.path.dirname(resume_path)
    env_cfg.log_dir = log_dir

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg)

    # Hide wall geometry
    import omni.usd
    from pxr import UsdGeom
    _stage = omni.usd.get_context().get_stage()
    for _wall_path in ("/World/Walls", "/World/WallsMergedMesh"):
        _prim = _stage.GetPrimAtPath(_wall_path)
        if _prim.IsValid():
            UsdGeom.Imageable(_prim).MakeInvisible()

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    if agent_cfg.class_name == "OnPolicyRunner":
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    elif agent_cfg.class_name == "DistillationRunner":
        runner = DistillationRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    else:
        raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")
    runner.load(resume_path)

    policy = runner.get_inference_policy(device=env.unwrapped.device)

    try:
        policy_nn = runner.alg.policy
    except AttributeError:
        policy_nn = runner.alg.actor_critic

    dt = env.unwrapped.step_dt

    # ── Prepare output directory ──────────────────────────────────────────────
    out_dir = args_cli.out_dir
    os.makedirs(out_dir, exist_ok=True)
    print(f"[SAVER] Saving frames to {out_dir}/")

    _mushr_env = env.unwrapped

    # Determine save frequency in steps
    save_every_n = max(1, int(round(args_cli.save_interval / dt)))
    print(f"[SAVER] step_dt={dt:.4f}s  saving every {save_every_n} steps (~{args_cli.save_interval}s)")

    frame_count = 0
    step_count = 0

    # reset environment
    obs = env.get_observations()

    # simulate environment
    while simulation_app.is_running():
        with torch.inference_mode():
            actions = policy(obs)
            obs, _, dones, _ = env.step(actions)
            policy_nn.reset(dones)

        step_count += 1

        if step_count % save_every_n == 0:
            frame_count += 1

            # ── Raw RGB ───────────────────────────────────────────────────────
            rgb_tensor = _mushr_env.camera_center.data.output["rgb"][0, :, :, :3].cpu().numpy().astype(np.uint8)
            raw_bgr = cv2.cvtColor(rgb_tensor, cv2.COLOR_RGB2BGR)
            raw_path = os.path.join(out_dir, f"raw_{frame_count:06d}.png")
            cv2.imwrite(raw_path, raw_bgr)

            # ── Policy input (black-white mask) ───────────────────────────────
            pol_flat = _mushr_env._process_single_rgb(_mushr_env.camera_center)
            pol_2d = pol_flat[0].reshape(
                _mushr_env._policy_depth_h, _mushr_env._policy_depth_w,
            ).cpu().numpy()
            pol_bgr = cv2.cvtColor((pol_2d * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
            policy_path = os.path.join(out_dir, f"policy_{frame_count:06d}.png")
            cv2.imwrite(policy_path, pol_bgr)

            print(f"[SAVER] Frame {frame_count:06d}  step {step_count}")

        if args_cli.max_steps is not None and step_count >= args_cli.max_steps:
            print(f"[SAVER] Reached max_steps={args_cli.max_steps}, stopping.")
            break

    print(f"[SAVER] Done — saved {frame_count} frames ({step_count} steps).")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
