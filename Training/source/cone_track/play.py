# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Run a trained ConeTrack checkpoint in Isaac Sim (inference only).

Loads the latest (or a named) checkpoint from ``outputs/``, exports it to
TorchScript + ONNX into ``exported/`` next to the checkpoint, then drives the car
with the policy until you quit.  The ONNX export is what Deployment/autopilot.py
runs on the Raspberry Pi.  Everything this writes — the export, any video, and
Hydra's ``play.log`` — stays inside the run directory the checkpoint came from.

Two viewing modes:
    default   the Isaac Sim viewport, following the car.
    --fpv     forces a single environment and opens four live depth windows —
              raw and policy-pipeline views for each camera — in a separate
              process (Qt cannot share this one with Omniverse Kit).
Walls are hidden here: they exist only to terminate training episodes.

Usage:
    play --num_envs 4                          # helper installed by install.sh
    play --fpv --real-time
    play --checkpoint outputs/<run>/model_500.pt
    play --fpv --video                         # record the left depth stream
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

# Reduce PyTorch CUDA allocator fragmentation: returns unused blocks to the
# system in smaller segments instead of holding giant cached arenas.
# Must be set before any CUDA context is created (AppLauncher).
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

# Training runs live in Training/outputs/<timestamp>/ (see train.py).  Spelled out
# here rather than imported from the package: importing cone_track pulls in torch
# and rsl_rl, which must not happen before Isaac Sim has launched.
OUTPUTS_DIR = Path(__file__).resolve().parents[2] / "outputs"

from isaaclab.app import AppLauncher

# local imports
# Imported as a bare sibling module (this file's directory is sys.path[0]), NOT as
# cone_track.cli_args, for the same reason.
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Play an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default="ConeTrack", help="Name of the task.")
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--real-time", action="store_true", default=False, help="Throttle the loop to wall-clock time.")
parser.add_argument(
    "--fpv",
    action="store_true",
    default=False,
    help="First-person view: force 1 env and open the live raw/policy depth windows.",
)
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli, hydra_args = parser.parse_known_args()
# cameras are needed both for viewport recording and for the FPV depth windows
if args_cli.video or args_cli.fpv:
    args_cli.enable_cameras = True
if args_cli.fpv:
    args_cli.num_envs = 1

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ─── Post-launch imports (safe now that Isaac Sim is up) ──────────────────────

import time

# ── FPV display subprocess ────────────────────────────────────────────────────
# Same approach as visualize.py: Omniverse Kit owns the Qt event loop in this
# process, so cv2.imshow cannot run here.  A child process instead polls the PNG
# files this process writes atomically and shows them.  Its sys.path drops Kit's
# bundled paths so it picks up the GUI-capable opencv-python from .venv rather
# than Isaac Sim's headless build.
_FPV_DISPLAY_SCRIPT = r"""
import sys, os
sys.path = [p for p in sys.path if "pip_prebundle" not in p and "omni" not in p]
import cv2, time, signal
signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
OUT_DIR = "/tmp/mushr_debug"
FILES = [
    (os.path.join(OUT_DIR, "raw_left.png"),    "Left Raw Depth"),
    (os.path.join(OUT_DIR, "policy_left.png"), "Left Policy Depth"),
    (os.path.join(OUT_DIR, "raw_right.png"),   "Right Raw Depth"),
    (os.path.join(OUT_DIR, "policy_right.png"),"Right Policy Depth"),
]
while not os.path.exists(os.path.join(OUT_DIR, "raw_left.png")):
    time.sleep(0.05)
while True:
    for fpath, name in FILES:
        img = cv2.imread(fpath)
        if img is not None:
            cv2.imshow(name, img)
    if cv2.waitKey(33) & 0xFF == ord("q"):
        break
cv2.destroyAllWindows()
"""

from typing import cast

import cv2
import gymnasium as gym
import numpy as np
import torch
from rsl_rl.runners import OnPolicyRunner

from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, export_policy_as_jit, export_policy_as_onnx
import cone_track  # noqa: F401  — registers the "ConeTrack" gym id
from cone_track.env import ConeTrackEnv, ConeTrackEnvCfg
from cone_track.ppo_cfg import ConeTrackPPORunnerCfg
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config


def _resolve_checkpoint() -> str:
    """Find the checkpoint to play — before Hydra starts.

    Resolved this early so Hydra can be pointed at the run directory the checkpoint
    came from, instead of creating a directory of its own next to the training runs.
    The trade-off is that only the CLI flags are visible here: pick a run with
    ``--checkpoint`` / ``--load_run``, not with a Hydra ``agent.load_run=`` override.
    """
    if not args_cli.checkpoint:
        return get_checkpoint_path(str(OUTPUTS_DIR), args_cli.load_run or ".*", "model_.*.pt")

    ckpt = args_cli.checkpoint
    if not os.path.isabs(ckpt) and not os.path.isfile(ckpt):
        # The play alias cds to Training/; if the user gave a path relative to the
        # project root (e.g. Training/outputs/<run>/model_500.pt) try resolving it
        # one directory up so both forms work without needing an absolute path.
        candidate = os.path.join(OUTPUTS_DIR.parent.parent, ckpt)
        if os.path.isfile(candidate):
            ckpt = candidate
    return retrieve_file_path(ckpt)


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ConeTrackEnvCfg, agent_cfg: ConeTrackPPORunnerCfg, resume_path: str):
    """Load a checkpoint, export it, and run it in closed loop."""
    # override configurations with non-hydra CLI arguments (mutates agent_cfg in place)
    cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs

    # set the environment seed
    # note: certain randomizations occur in the environment initialization so we set the seed here
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # the run directory the checkpoint came from — everything below writes into it
    log_dir = os.path.dirname(resume_path)

    # set the log directory for the environment (works for all environment types)
    env_cfg.log_dir = log_dir

    # create isaac environment
    # When --fpv --video: cameras are already enabled by the fpv flag; render_mode not needed.
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if (args_cli.video and not args_cli.fpv) else None)

    # wrap for video recording (Isaac viewport — skipped when --fpv, which records the raw camera instead)
    if args_cli.video and not args_cli.fpv:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording the Isaac Sim viewport.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)  # pyright: ignore[reportArgumentType]

    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)  # pyright: ignore[reportAttributeAccessIssue]
    runner.load(resume_path)

    # obtain the trained policy for inference
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    # the raw network, for the exporters and for resetting recurrent state
    policy_nn = runner.alg.policy

    # extract the normalizer
    if hasattr(policy_nn, "actor_obs_normalizer"):
        normalizer = policy_nn.actor_obs_normalizer
    elif hasattr(policy_nn, "student_obs_normalizer"):
        normalizer = policy_nn.student_obs_normalizer
    else:
        normalizer = None

    # export policy to onnx/jit
    export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")
    export_policy_as_jit(policy_nn, normalizer=normalizer, path=export_model_dir, filename="policy.pt")
    export_policy_as_onnx(policy_nn, normalizer=normalizer, path=export_model_dir, filename="policy.onnx")

    dt = env.unwrapped.step_dt

    # ── FPV setup ─────────────────────────────────────────────────────────────
    _fpv_proc   = None
    _fpv_writer = None
    _fpv_out_dir = "/tmp/mushr_debug"
    if args_cli.fpv:
        os.makedirs(_fpv_out_dir, exist_ok=True)
        _mushr_env = cast(ConeTrackEnv, env.unwrapped)
        if os.environ.get("DISPLAY"):
            _display_env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
            _display_env["QT_QPA_FONTDIR"] = "/usr/share/fonts"
            _fpv_proc = subprocess.Popen(
                [sys.executable, "-c", _FPV_DISPLAY_SCRIPT], env=_display_env
            )
            print(f"[FPV] Display subprocess started (PID {_fpv_proc.pid})")
        else:
            print("[FPV] No $DISPLAY — writing frames to /tmp/mushr_debug/ (use feh --reload 0.25)")

        if args_cli.video:
            _cam_h = _mushr_env.cfg.camera_left_cfg.height
            _cam_w = _mushr_env.cfg.camera_left_cfg.width
            _fps   = round(1.0 / (_mushr_env.cfg.decimation * _mushr_env.cfg.sim.dt))
            os.makedirs(os.path.join(log_dir, "videos", "play"), exist_ok=True)
            _video_path = os.path.join(
                log_dir, "videos", "play",
                f"fpv_{time.strftime('%Y%m%d_%H%M%S')}.mp4",
            )
            _fpv_writer = cv2.VideoWriter(
                _video_path,
                cv2.VideoWriter_fourcc(*"mp4v"),  # pyright: ignore[reportAttributeAccessIssue]
                _fps,
                (_cam_w, _cam_h),
            )
            print(f"[FPV] Recording to {_video_path}  ({_cam_w}×{_cam_h} @ {_fps} fps)")

    def _write_atomic(path: str, img: np.ndarray) -> None:
        base, ext = os.path.splitext(path)
        tmp = base + "_tmp" + ext
        cv2.imwrite(tmp, img)
        os.replace(tmp, path)

    def _raw_depth_bgr(camera) -> np.ndarray:
        """Full-resolution sensor depth, false-coloured (blue = far, red = close)."""
        depth_np = camera.data.output["distance_to_image_plane"][0, :, :, 0].cpu().numpy()
        depth_max = _mushr_env.cfg.depth_max_m
        depth_np = np.where(np.isfinite(depth_np), depth_np, depth_max)
        raw_norm = 1.0 - np.clip(depth_np / depth_max, 0.0, 1.0)
        return cv2.applyColorMap(
            np.clip(raw_norm * 255, 0, 255).astype(np.uint8), cv2.COLORMAP_JET
        )

    def _write_camera(camera, suffix: str) -> None:
        """Write raw_{suffix}.png (sensor output) and policy_{suffix}.png (what the policy sees)."""
        _write_atomic(os.path.join(_fpv_out_dir, f"raw_{suffix}.png"), _raw_depth_bgr(camera))

        # Policy window: exact same pipeline as training (_process_single_depth).
        pol_flat = _mushr_env._process_single_depth(camera)   # [1, ph*pw], values [0,1]
        pol_2d   = pol_flat[0].reshape(
            _mushr_env._policy_depth_h, _mushr_env._policy_depth_w,
        ).cpu().numpy()
        pol_bgr  = cv2.applyColorMap(
            np.clip(pol_2d * 255, 0, 255).astype(np.uint8), cv2.COLORMAP_JET
        )
        _write_atomic(os.path.join(_fpv_out_dir, f"policy_{suffix}.png"), pol_bgr)

    def _fpv_step() -> None:
        """Refresh all four depth windows, and append a video frame when recording."""
        _write_camera(_mushr_env.camera_left,  "left")
        _write_camera(_mushr_env.camera_right, "right")
        if _fpv_writer is not None:
            _fpv_writer.write(_raw_depth_bgr(_mushr_env.camera_left))
    # ──────────────────────────────────────────────────────────────────────────

    # reset environment
    obs = env.get_observations()
    timestep = 0
    # simulate environment
    while simulation_app.is_running():
        start_time = time.time()
        # run everything in inference mode
        with torch.inference_mode():
            # agent stepping
            actions = policy(obs)
            # env stepping
            obs, _, dones, _ = env.step(actions)
            # reset recurrent states for episodes that have terminated
            policy_nn.reset(dones)
        if args_cli.fpv:
            _fpv_step()
            if args_cli.video:
                timestep += 1
                if timestep >= args_cli.video_length:
                    break
        elif args_cli.video:
            timestep += 1
            # Exit the play loop after recording one video
            if timestep == args_cli.video_length:
                break

        # time delay for real-time evaluation
        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)

    # close the simulator
    env.close()
    if _fpv_writer is not None:
        _fpv_writer.release()
        print("[FPV] Video saved.")
    if args_cli.fpv and _fpv_proc is not None and _fpv_proc.poll() is None:
        _fpv_proc.terminate()
        _fpv_proc.wait()


if __name__ == "__main__":
    _resume_path = _resolve_checkpoint()
    # Point Hydra at the run directory being played: it writes play.log there and,
    # with output_subdir disabled, leaves the run's own .hydra/ record untouched.
    sys.argv += [
        f"hydra.run.dir={os.path.dirname(_resume_path)}",
        "hydra.job.name=play",
        "hydra.output_subdir=null",
    ]
    # run the main function (hydra_task_config injects env_cfg / agent_cfg)
    main(_resume_path)  # pyright: ignore[reportCallIssue]
    # close sim app
    simulation_app.close()
