# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to train RL agent with RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse
import base64
import io
import os
import sys
import tarfile

# LD_PRELOAD must be set before any shared libraries load, which means before Python
# itself initialises its C extensions. Setting os.environ inside an already-running
# interpreter is too late for the current process. Re-exec with the flag in place so
# the dynamic linker picks up conda's libstdc++ (which has CXXABI_1.3.15) on the
# very first dlopen call.  The sentinel variable prevents infinite recursion.
if "ISAACLAB_LIBSTDCPP_FIXED" not in os.environ:
    _conda_prefix = os.environ.get("CONDA_PREFIX")
    if _conda_prefix:
        _libstdcpp = os.path.join(_conda_prefix, "lib", "libstdc++.so.6")
        if os.path.exists(_libstdcpp):
            os.environ["ISAACLAB_LIBSTDCPP_FIXED"] = "1"
            os.environ["LD_PRELOAD"] = _libstdcpp
            os.execv(sys.executable, [sys.executable] + sys.argv)

# ─────────────────────────────────────────────────────────────────────────────
# RunPod remote-training support
#
# Usage:
#   export RUNPOD_IMAGE=docker.io/you/isaaclab-training:latest
#   train --task ConeTrack --runpod [--num_envs N] [--max_iterations N] [--run_name v1]
#
# The Docker image must have Isaac Sim 5.1 + Isaac Lab 2.3 already installed
# (run Training/install.sh inside the container once, then push to a registry).
# Local source files (env.py, ppo_cfg.py, etc.) are packed and overlaid on the
# remote workspace every run, so edits are always reflected without rebuilding.
# TensorBoard event files are streamed back every ~20 s for live local viewing.
# ─────────────────────────────────────────────────────────────────────────────

_RUNPOD_IMAGE  = os.environ.get("RUNPOD_IMAGE", "")
_RUNPOD_POLL_S = 20


def _rp_pack_source() -> str:
    """Tar-gz Training/source + tasks + cli_args.py → base64 string."""
    root = os.path.dirname(os.path.abspath(__file__))
    buf  = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for name in ("source", "tasks", "cli_args.py", "pyproject.toml", "uv.lock"):
            p = os.path.join(root, name)
            if os.path.exists(p):
                tar.add(p, arcname=name)
    return base64.b64encode(buf.getvalue()).decode()


def _rp_forward_args(args) -> list:
    """Reconstruct CLI flags to pass to the remote train.py (--runpod excluded)."""
    fwd = []
    if getattr(args, "num_envs", None) is not None:
        fwd += ["--num_envs", str(args.num_envs)]
    if getattr(args, "seed", None) is not None:
        fwd += ["--seed", str(args.seed)]
    if getattr(args, "max_iterations", None) is not None:
        fwd += ["--max_iterations", str(args.max_iterations)]
    if getattr(args, "run_name", None):
        fwd += ["--run_name", args.run_name]
    return fwd


async def _train_on_runpod(args) -> None:
    """Submit training to a RunPod RTX 4090 and stream TensorBoard events locally."""
    import asyncio, socket, subprocess, time, webbrowser
    from datetime import datetime
    from runpod_flash import Endpoint, GpuType

    if not _RUNPOD_IMAGE:
        print(
            "[RunPod] ERROR: set RUNPOD_IMAGE to a Docker image with Isaac Sim + Isaac Lab.\n"
            "Example:  export RUNPOD_IMAGE=docker.io/yourname/isaaclab-training:latest"
        )
        return

    @Endpoint(
        name="cone-track-training",
        gpu=GpuType.NVIDIA_GEFORCE_RTX_4090,
        base_image=_RUNPOD_IMAGE,
    )
    def _remote(payload: dict):
        """Runs ON RunPod: unpack source overlay, train headless, stream TensorBoard chunks."""
        import base64, glob, io, os, subprocess, tarfile, time

        ws = "/workspace/Training"
        # Overlay local source edits onto the workspace (base image already has the install).
        with tarfile.open(fileobj=io.BytesIO(base64.b64decode(payload["src"])),
                          mode="r:gz") as tar:
            tar.extractall(ws)

        proc = subprocess.Popen(
            ["uv", "run", "python", os.path.join(ws, "train.py"),
             "--task", "ConeTrack", "--headless"]
            + payload.get("extra_args", []),
            cwd=ws,
        )

        log_root = os.path.join(ws, "logs", "rsl_rl", "cone_track")
        ev_file, offset = None, 0

        while proc.poll() is None:
            if ev_file is None:
                for path in glob.glob(f"{log_root}/*/events.out.*"):
                    ev_file = path
                    break
            if ev_file and os.path.exists(ev_file):
                with open(ev_file, "rb") as fh:
                    fh.seek(offset)
                    chunk = fh.read()
                if chunk:
                    yield {"events": base64.b64encode(chunk).decode()}
                    offset += len(chunk)
            time.sleep(payload.get("poll_s", 20))

        yield {"done": True, "exit_code": proc.returncode}

    # ── local side ─────────────────────────────────────────────────────────
    run_name  = getattr(args, "run_name", None) or "runpod"
    log_root  = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "logs", "rsl_rl", "cone_track")
    local_dir = os.path.join(log_root,
                             datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                             + f"_{run_name}_runpod")
    os.makedirs(local_dir, exist_ok=True)

    # Start local TensorBoard
    _tb_port = 6006
    for _p in range(6006, 6020):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as _s:
            if _s.connect_ex(("localhost", _p)) != 0:
                _tb_port = _p
                break
    _tb_exe = os.path.join(os.path.dirname(sys.executable), "tensorboard")
    subprocess.Popen([_tb_exe, "--logdir", log_root, "--port", str(_tb_port)],
                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    time.sleep(2)
    _tb_url = f"http://localhost:{_tb_port}"
    print(f"\n[TensorBoard] Watching : {log_root}")
    print(f"[TensorBoard] Open at  : {_tb_url}\n")
    try:
        webbrowser.open(_tb_url)
    except Exception:
        pass

    print(f"[RunPod] Job submitted — streaming events to {local_dir}")
    ev_local = os.path.join(local_dir, "events.out.tfevents.runpod")
    open(ev_local, "wb").close()  # seed empty file so TensorBoard registers the run

    async for msg in _remote({"src": _rp_pack_source(),
                               "extra_args": _rp_forward_args(args),
                               "poll_s": _RUNPOD_POLL_S}):
        if "events" in msg:
            data = base64.b64decode(msg["events"])
            with open(ev_local, "ab") as fh:
                fh.write(data)
            print(f"[RunPod] +{len(data):,} B → TensorBoard updated")
        if msg.get("done"):
            print(f"[RunPod] Training finished (exit {msg.get('exit_code', '?')})")
            break


# Intercept --runpod BEFORE importing Isaac Sim or starting AppLauncher locally.
if "--runpod" in sys.argv:
    import asyncio as _asyncio

    _rp_parser = argparse.ArgumentParser(add_help=False)
    _rp_parser.add_argument("--runpod",        action="store_true")
    _rp_parser.add_argument("--num_envs",       type=int, default=None)
    _rp_parser.add_argument("--seed",           type=int, default=None)
    _rp_parser.add_argument("--max_iterations", type=int, default=None)
    _rp_parser.add_argument("--run_name",       type=str, default=None)
    _rp_args, _ = _rp_parser.parse_known_args()

    _asyncio.run(_train_on_runpod(_rp_args))
    sys.exit(0)

# ─── Local path: Isaac Sim starts here ────────────────────────────────────────
from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--runpod", action="store_true", default=False,
                    help="Run training on a RunPod RTX 4090 (requires RUNPOD_IMAGE env-var).")
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
parser.add_argument("--export_io_descriptors", action="store_true", default=False, help="Export IO descriptors.")
parser.add_argument(
    "--ray-proc-id", "-rid", type=int, default=None, help="Automatically configured by Ray integration, otherwise None."
)
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

# always enable cameras — required for TiledCamera in the environment
args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Check for minimum supported RSL-RL version."""

import importlib.metadata as metadata
import platform

from packaging import version

# check minimum supported rsl-rl version
RSL_RL_VERSION = "3.0.1"
installed_version = metadata.version("rsl-rl-lib")
if version.parse(installed_version) < version.parse(RSL_RL_VERSION):
    if platform.system() == "Windows":
        cmd = [r".\isaaclab.bat", "-p", "-m", "pip", "install", f"rsl-rl-lib=={RSL_RL_VERSION}"]
    else:
        cmd = ["./isaaclab.sh", "-p", "-m", "pip", "install", f"rsl-rl-lib=={RSL_RL_VERSION}"]
    print(
        f"Please install the correct version of RSL-RL.\nExisting version is: '{installed_version}'"
        f" and required version is: '{RSL_RL_VERSION}'.\nTo install the correct version, run:"
        f"\n\n\t{' '.join(cmd)}\n"
    )
    exit(1)

"""Rest everything follows."""

import logging
import os
import socket
import subprocess
import time
import webbrowser
from datetime import datetime

import gymnasium as gym
import torch
from rsl_rl.runners import DistillationRunner, OnPolicyRunner

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.dict import print_dict
from isaaclab.utils.io import dump_yaml

from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper

import isaaclab_tasks  # noqa: F401
import tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

# import logger
logger = logging.getLogger(__name__)

# PLACEHOLDER: Extension template (do not remove this comment)

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
    subprocess.Popen(
        [tb_exe, "--logdir", log_root_path, "--port", str(port)],
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
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    """Train with RSL-RL agent."""
    # override configurations with non-hydra CLI arguments
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
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

    # specify directory for logging experiments
    log_root_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs", "rsl_rl", agent_cfg.experiment_name)
    print(f"[INFO] Logging experiment in directory: {log_root_path}")
    # specify directory for logging runs: {time-stamp}_{run_name}
    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    # The Ray Tune workflow extracts experiment name using the logging line below, hence, do not
    # change it (see PR #2346, comment-2819298849)
    print(f"Exact experiment name requested from command line: {log_dir}")
    if agent_cfg.run_name:
        log_dir += f"_{agent_cfg.run_name}"
    log_dir = os.path.join(log_root_path, log_dir)

    _launch_tensorboard(log_root_path)

    # set the IO descriptors export flag if requested
    if isinstance(env_cfg, ManagerBasedRLEnvCfg):
        env_cfg.export_io_descriptors = args_cli.export_io_descriptors
    else:
        logger.warning(
            "IO descriptors are only supported for manager based RL environments. No IO descriptors will be exported."
        )

    # set the log directory for the environment (works for all environment types)
    env_cfg.log_dir = log_dir

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # save resume path before creating a new log_dir
    if agent_cfg.resume or agent_cfg.algorithm.class_name == "Distillation":
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

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
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    # create runner from rsl-rl
    if agent_cfg.class_name == "OnPolicyRunner":
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
    elif agent_cfg.class_name == "DistillationRunner":
        runner = DistillationRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
    else:
        raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")
    # write git state to logs
    runner.add_git_repo_to_log(__file__)
    # load the checkpoint
    if agent_cfg.resume or agent_cfg.algorithm.class_name == "Distillation":
        print(f"[INFO]: Loading model checkpoint from: {resume_path}")
        # load previously trained model
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
    # run the main function
    main()
    # close sim app
    simulation_app.close()
