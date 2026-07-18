# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to play a checkpoint of an RL agent from RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse
import os
import subprocess
import sys

# Reduce PyTorch CUDA allocator fragmentation: returns unused blocks to the
# system in smaller segments instead of holding giant cached arenas.
# Must be set before any CUDA context is created (AppLauncher).
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Play an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default="ConeTrack", help="Name of the task.")
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument(
    "--use_pretrained_checkpoint",
    action="store_true",
    help="Use the pre-trained checkpoint from Nucleus.",
)
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")
parser.add_argument(
    "--fpv",
    action="store_true",
    default=False,
    help="First-person view: force 1 env and open live RGB + bbox overlay windows.",
)
parser.add_argument(
    "--vel_cap",
    type=float,
    default=5.0,
    help=(
        "Override the velocity cap used during play (m/s).  "
        "Sets both initial_vel_cap_m_s and vel_cap_max_m_s so the curriculum cannot "
        "accidentally lower the cap during the play session.  Default: 7.0 m/s."
    ),
)
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli, hydra_args = parser.parse_known_args()
# always enable cameras — required for TiledCamera in the environment
args_cli.enable_cameras = True
if args_cli.fpv:
    args_cli.num_envs = 1

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import time

# ── FPV display subprocess ────────────────────────────────────────────────────
_FPV_DISPLAY_SCRIPT = r"""
import sys, os
sys.path = [
    *[p for p in sys.path if "pip_prebundle" not in p and "omni" not in p],
]
import cv2, time, signal
signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
os.environ.setdefault("QT_LOGGING_RULES", "qt.qpa.fonts.warning=false")
OUT_DIR = "/tmp/mushr_debug"
FILES = [
    (os.path.join(OUT_DIR, "raw_center.png"),    "Center Raw RGB"),
    (os.path.join(OUT_DIR, "bbox_center.png"),   "Center BBox Overlay"),
]
while not os.path.exists(os.path.join(OUT_DIR, "raw_center.png")):
    time.sleep(0.05)
while True:
    for fpath, name in FILES:
        if os.path.exists(fpath):
            img = cv2.imread(fpath)
            if img is not None:
                cv2.imshow(name, img)
    if cv2.waitKey(33) & 0xFF == ord("q"):
        break
cv2.destroyAllWindows()
"""

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
from isaaclab.utils.dict import print_dict

from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper, export_policy_as_jit, export_policy_as_onnx
import isaaclab_tasks  # noqa: F401
import tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

# PLACEHOLDER: Extension template (do not remove this comment)


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    """Play with RSL-RL agent."""
    # grab task name for checkpoint path
    task_name = args_cli.task.split(":")[-1]
    train_task_name = task_name.replace("-Play", "")

    # override configurations with non-hydra CLI arguments
    agent_cfg: RslRlBaseRunnerCfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs

    # set the environment seed
    # note: certain randomizations occur in the environment initialization so we set the seed here
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # specify directory for logging experiments (absolute path, matches train.py)
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
            # The play alias cds to Training/ before running; if the user gave a path
            # relative to the project root (e.g. Training/logs/...) try resolving it
            # one directory up so both forms work without needing an absolute path.
            project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            candidate = os.path.join(project_root, ckpt)
            if os.path.isfile(candidate):
                ckpt = candidate
        resume_path = retrieve_file_path(ckpt)
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    log_dir = os.path.dirname(resume_path)

    # set the log directory for the environment (works for all environment types)
    env_cfg.log_dir = log_dir

    # create isaac environment
    # When --fpv --video: cameras are already enabled by the fpv flag; render_mode not needed.
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if (args_cli.video and not args_cli.fpv) else None)

    # Hide wall geometry in play mode — walls are only used during training to
    # penalise boundary contact; they serve no purpose during inference.
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

    # wrap for video recording (Isaac viewport — skipped when --fpv, which records the raw camera instead)
    if args_cli.video and not args_cli.fpv:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    # load previously trained model
    if agent_cfg.class_name == "OnPolicyRunner":
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    elif agent_cfg.class_name == "DistillationRunner":
        runner = DistillationRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    else:
        raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")
    runner.load(resume_path)

    # obtain the trained policy for inference
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    # extract the neural network module
    # we do this in a try-except to maintain backwards compatibility.
    try:
        # version 2.3 onwards
        policy_nn = runner.alg.policy
    except AttributeError:
        # version 2.2 and below
        policy_nn = runner.alg.actor_critic

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
        _mushr_env = env.unwrapped          # DirectRLEnv (ConeTrackEnv)
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
            import time as _time
            _cam_h = _mushr_env.cfg.camera_center_cfg.height
            _cam_w = _mushr_env.cfg.camera_center_cfg.width
            _fps   = round(1.0 / (env.unwrapped.cfg.decimation * env.unwrapped.cfg.sim.dt))
            os.makedirs(os.path.join(log_dir, "videos", "play"), exist_ok=True)
            _video_path = os.path.join(
                log_dir, "videos", "play",
                f"fpv_{_time.strftime('%Y%m%d_%H%M%S')}.mp4",
            )
            _fpv_writer = cv2.VideoWriter(
                _video_path,
                cv2.VideoWriter_fourcc(*"mp4v"),
                _fps,
                (_cam_w, _cam_h),
            )
            print(f"[FPV] Recording to {_video_path}  ({_cam_w}×{_cam_h} @ {_fps} fps)")

    def _write_atomic(path: str, img: np.ndarray) -> None:
        base, ext = os.path.splitext(path)
        tmp = base + "_tmp" + ext
        cv2.imwrite(tmp, img)
        os.replace(tmp, path)

    def _fpv_step() -> None:
        """Grab vision frame from center camera, write raw + bbox PNGs."""
        rgb_tensor = _mushr_env.camera_center.data.output["rgb"][0, :, :, :3].cpu().numpy().astype(np.uint8)
        raw_bgr = cv2.cvtColor(rgb_tensor, cv2.COLOR_RGB2BGR)
        _write_atomic(os.path.join(_fpv_out_dir, "raw_center.png"), raw_bgr)

        # ── Bbox overlay with YOLO-like filtering ────────────────────────────
        bbox_bgr = raw_bgr.copy()
        annotator_data = _mushr_env._rep_annotators[0].get_data()
        raw = annotator_data.get("data")
        if isinstance(raw, (list, tuple)):
            raw_data = raw[0] if raw else None
        else:
            raw_data = raw

        depth_map = (
            _mushr_env.camera_center.data.output["distance_to_camera"][0, :, :, 0]
            .cpu()
            .numpy()
        )

        cfg = _mushr_env.cfg
        if raw_data is not None:
            filtered = _mushr_env.filter_bboxes_numpy(
                raw_data, depth_map,
                max_dist=cfg.max_bbox_distance,
                min_area=cfg.min_bbox_area,
                min_side=cfg.min_bbox_side,
                occlusion_tol=cfg.bbox_occlusion_tol,
            )
            for i in range(len(filtered)):
                x1 = int(filtered[i, 0])
                y1 = int(filtered[i, 1])
                x2 = int(filtered[i, 2])
                y2 = int(filtered[i, 3])
                cv2.rectangle(bbox_bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)

        _write_atomic(os.path.join(_fpv_out_dir, "bbox_center.png"), bbox_bgr)

        if _fpv_writer is not None:
            _fpv_writer.write(raw_bgr)
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
        print(f"[FPV] Video saved.")
    if args_cli.fpv and _fpv_proc is not None and _fpv_proc.poll() is None:
        _fpv_proc.terminate()
        _fpv_proc.wait()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
