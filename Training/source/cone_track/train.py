# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Train the ConeTrack policy with PPO (RSL-RL) in Isaac Sim.

Launches Isaac Sim, builds the vectorised ConeTrack environment, runs PPO, and
starts a TensorBoard server on the outputs directory.  One run writes exactly
one directory, ``outputs/<timestamp>[_<run_name>]/``, holding the TensorBoard
events, ``model_*.pt`` checkpoints, ``params/`` (resolved env + agent configs),
``.hydra/`` + ``train.log`` (Hydra's record of the invocation), ``git/``, and
later the ``exported/`` policy that play.py writes there.

The environment is configured in ``env.py`` and the PPO hyper-parameters in
``ppo_cfg.py`` next to this file; the CLI only overrides the handful of
run-level options below (plus anything passed as Hydra ``key=value`` arguments,
which are applied on top of both configs).

Usage:
    train --num_envs 26 --headless          # helper installed by install.sh
    uv run python source/cone_track/train.py --num_envs 26 --headless
    uv run python source/cone_track/train.py --resume --load_run 2026-07-26_11-41-00
"""

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path

# Reduce PyTorch CUDA allocator fragmentation: returns unused blocks to the
# system in smaller segments instead of holding giant cached arenas.
# Must be set before any CUDA context is created (AppLauncher).
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

# Everything a run writes goes under Training/outputs/.  Spelled out here rather
# than imported from the package: importing cone_track pulls in torch/rsl_rl,
# which must not happen before AppLauncher runs.
OUTPUTS_DIR = Path(__file__).resolve().parents[2] / "outputs"

# Isaac Sim must be launched before anything that touches omni/pxr is imported,
# so the imports in this file are deliberately split around the AppLauncher call.
from isaaclab.app import AppLauncher

# local imports
# Imported as a bare sibling module (this file's directory is sys.path[0]), NOT as
# cone_track.cli_args: that would execute cone_track/__init__.py, which pulls in
# torch and rsl_rl before Isaac Sim has launched.
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings (in steps).")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default="ConeTrack", help="Name of the task.")
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--max_iterations", type=int, default=None, help="RL Policy training iterations.")
parser.add_argument(
    "--distributed", action="store_true", default=False, help="Run training with multiple GPUs or nodes."
)
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

# always enable cameras — required for TiledCamera in the environment
args_cli.enable_cameras = True

# This run's directory.  Fixed here, before Hydra starts, so that Hydra's own
# output (.hydra/ + train.log) lands inside it instead of in the separate
# outputs/<date>/<time>/ tree it would otherwise invent.
RUN_DIR = OUTPUTS_DIR / (
    datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + (f"_{args_cli.run_name}" if args_cli.run_name else "")
)

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args + [f"hydra.run.dir={RUN_DIR}", "hydra.job.name=train"]

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ─── Post-launch imports (safe now that Isaac Sim is up) ──────────────────────

import importlib.metadata as metadata
import re
import socket
import subprocess
import time
import webbrowser
from packaging import version

import gymnasium as gym
import torch
from rsl_rl.runners import OnPolicyRunner

from isaaclab.utils.dict import print_dict
from isaaclab.utils.io import dump_yaml

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

import cone_track  # noqa: F401  — registers the "ConeTrack" gym id
from cone_track.env import ConeTrackEnvCfg
from cone_track.ppo_cfg import ConeTrackPPORunnerCfg
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

# The PPO config targets the rsl-rl 3.x API (obs groups, RslRlBaseRunnerCfg).
RSL_RL_VERSION = "3.0.1"
_installed_rsl_rl = metadata.version("rsl-rl-lib")
if version.parse(_installed_rsl_rl) < version.parse(RSL_RL_VERSION):
    print(
        f"[ERROR] rsl-rl-lib {_installed_rsl_rl} is too old; this project needs "
        f">= {RSL_RL_VERSION}.\n        Fix with:  uv pip install 'rsl-rl-lib=={RSL_RL_VERSION}'"
    )
    exit(1)

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False


def _launch_tensorboard(log_root_path: str) -> None:
    """Start TensorBoard in the background and print (and open) its URL."""
    port = 6006
    for candidate in range(6006, 6020):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if s.connect_ex(("localhost", candidate)) != 0:
                port = candidate
                break

    tb_exe = os.path.join(os.path.dirname(sys.executable), "tensorboard")
    
    # Tensorboard will listen on all interfaces (0.0.0.0)
    subprocess.Popen(
        [tb_exe, "--logdir", log_root_path, "--port", str(port), "--host", "0.0.0.0"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    time.sleep(2)

    url = f"http://localhost:{port}"
    print(f"\n[TensorBoard] Watching : {log_root_path}")
    print(f"[TensorBoard] Open at  : {url}\n")
    try:
        webbrowser.open(url)
    except Exception:
        pass


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ConeTrackEnvCfg, agent_cfg: ConeTrackPPORunnerCfg):
    """Build the env, restore an optional checkpoint, and run PPO to completion."""
    # override configurations with non-hydra CLI arguments (mutates agent_cfg in place)
    cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    agent_cfg.max_iterations = (
        args_cli.max_iterations if args_cli.max_iterations is not None else agent_cfg.max_iterations
    )

    # set the environment seed
    # note: certain randomizations occur in the environment initialization so we set the seed here
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    # check for invalid combination of CPU device with distributed training
    if args_cli.distributed and args_cli.device is not None and "cpu" in args_cli.device:
        raise ValueError(
            "Distributed training is not supported when using CPU device. "
            "Please use GPU device (e.g., --device cuda) for distributed training."
        )

    # multi-gpu training configuration
    if args_cli.distributed:
        env_cfg.sim.device = f"cuda:{app_launcher.local_rank}"
        agent_cfg.device = f"cuda:{app_launcher.local_rank}"

        # set seed to have diversity in different threads
        seed = agent_cfg.seed + app_launcher.local_rank
        env_cfg.seed = seed
        agent_cfg.seed = seed

    # everything this run writes goes into the single directory fixed above
    log_dir = str(RUN_DIR)
    # The Ray Tune workflow extracts experiment name using the logging line below, hence, do not
    # change it (see PR #2346, comment-2819298849)
    print(f"Exact experiment name requested from command line: {RUN_DIR.name}")
    print(f"[INFO] Logging this run in directory: {log_dir}")

    # TensorBoard watches the whole outputs tree, so every past run is comparable
    _launch_tensorboard(str(OUTPUTS_DIR))

    # tell the environment where to write its own artefacts
    env_cfg.log_dir = log_dir

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # resolve the checkpoint to resume from.  Hydra has already created THIS run's
    # (still empty) directory under outputs/, so the default ".*" run filter has to
    # be narrowed to exclude it — otherwise it would resolve to the new run itself.
    if agent_cfg.resume:
        load_run = agent_cfg.load_run
        if load_run == ".*":
            load_run = rf"(?!{re.escape(RUN_DIR.name)}$).*"
        resume_path = get_checkpoint_path(str(OUTPUTS_DIR), load_run, agent_cfg.load_checkpoint)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "train"),
            "step_trigger": lambda step: step % args_cli.video_interval == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    start_time = time.time()

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)  # pyright: ignore[reportArgumentType]

    # create runner from rsl-rl
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)  # pyright: ignore[reportAttributeAccessIssue]
    # write git state to logs
    runner.add_git_repo_to_log(__file__)
    # load the checkpoint
    if agent_cfg.resume:
        print(f"[INFO]: Loading model checkpoint from: {resume_path}")
        runner.load(resume_path)

    # dump the configuration into log-directory
    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)

    # run training
    runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)

    print(f"Training time: {round(time.time() - start_time, 2)} seconds")

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function (hydra_task_config injects env_cfg / agent_cfg)
    main()  # pyright: ignore[reportCallIssue]
    # close sim app
    simulation_app.close()