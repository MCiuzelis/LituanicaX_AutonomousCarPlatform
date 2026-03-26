"""Visualize MuSHR ArduCam TOF depth stream from Isaac Sim without running any policy.

The robot is spawned at its start position.  WASD keys drive it manually
(requires pynput — install with: pip install pynput).

A separate display subprocess (spawned BEFORE Isaac Sim starts) owns the
cv2.imshow windows.  Isaac Sim runs its own Qt application internally; creating
a second QApplication in the same process silently blocks imshow.  Running the
display in a child process avoids the conflict entirely.

Two windows are shown:
  "Raw Depth"      — full-resolution depth frame, false-coloured with COLORMAP_JET
                     (blue = far / safe, red = close / danger)
  "Policy Depth"   — cropped + noised depth at exactly the resolution the policy
                     network receives, also false-coloured

Frames are exchanged via PNG files written to /tmp/mushr_debug/.
Press  Ctrl+C  in this terminal  OR  Q  in any display window to stop.

Usage:
    python .../visualize.py --task Mushr
    python .../visualize.py --task Mushr --crop_top 0.2 --crop_bottom 0.15
"""

import argparse
import os
import subprocess
import sys
import threading

# Re-exec with conda's libstdc++ in LD_PRELOAD
if "ISAACLAB_LIBSTDCPP_FIXED" not in os.environ:
    _conda_prefix = os.environ.get("CONDA_PREFIX")
    if _conda_prefix:
        _libstdcpp = os.path.join(_conda_prefix, "lib", "libstdc++.so.6")
        if os.path.exists(_libstdcpp):
            os.environ["ISAACLAB_LIBSTDCPP_FIXED"] = "1"
            os.environ["LD_PRELOAD"] = _libstdcpp
            os.execv(sys.executable, [sys.executable] + sys.argv)

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Visualize MuSHR TOF depth streams (no policy).")
parser.add_argument("--task", type=str, default="Mushr", help="Gym task name.")
parser.add_argument(
    "--crop_top", type=float, default=None,
    help="Override depth_crop_top_fraction (e.g. 0.15).",
)
parser.add_argument(
    "--crop_bottom", type=float, default=None,
    help="Override depth_crop_bottom_fraction (e.g. 0.10).",
)

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True

# ── display subprocess ─────────────────────────────────────────────────────────
_CONDA_SP = "/home/matasciuzelis/miniconda3/envs/isaac/lib/python3.11/site-packages"

_DISPLAY_SCRIPT = rf"""
import sys, os
sys.path = [
    "{_CONDA_SP}",
    *[p for p in sys.path if "pip_prebundle" not in p and "omni" not in p],
]
import cv2, time, signal
signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))

OUT_DIR = "/tmp/mushr_debug"
FILES = [
    (os.path.join(OUT_DIR, "raw.png"),  "Raw Depth"),
    (os.path.join(OUT_DIR, "policy.png"), "Policy Depth"),
]

while not os.path.exists(os.path.join(OUT_DIR, "raw.png")):
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

# ── keyboard state ─────────────────────────────────────────────────────────────
try:
    from pynput import keyboard as _pynput_kb
    _HAS_PYNPUT = True
except ImportError:
    _HAS_PYNPUT = False


class _KeyState:
    def __init__(self):
        self._down: set = set()
        self._lock = threading.Lock()

    def on_press(self, key):
        try:
            with self._lock:
                self._down.add(key.char.lower())
        except AttributeError:
            pass

    def on_release(self, key):
        try:
            with self._lock:
                self._down.discard(key.char.lower())
        except AttributeError:
            pass

    def get_actions(self) -> tuple[float, float]:
        with self._lock:
            throttle = 1.0 if "w" in self._down else -1.0
            steer = (1.0 if "a" in self._down else 0.0) - (1.0 if "d" in self._down else 0.0)
        return throttle, steer


display_proc = None
if os.environ.get("DISPLAY"):
    _display_env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    display_proc = subprocess.Popen(
        [sys.executable, "-c", _DISPLAY_SCRIPT], env=_display_env
    )
    print(f"[visualize] Display subprocess started (PID {display_proc.pid})")
else:
    print("[visualize] No $DISPLAY — disk output only.")
    print("[visualize] Live view:  feh --reload 0.25 /tmp/mushr_debug/")

# ── launch Isaac Sim ───────────────────────────────────────────────────────────
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ── post-launch imports ────────────────────────────────────────────────────────
import numpy as np
import torch
import gymnasium as gym
import cv2  # headless fine here — only imwrite used

import isaaclab_tasks  # noqa: F401
import lituanicaXsim   # noqa: F401

from lituanicaXsim.tasks.mushr_maze.mushr_maze_env import MushrMazeEnvCfg


def _depth_to_colormap(depth_float: np.ndarray) -> np.ndarray:
    """Convert a float32 depth array (values in [0,1]) to a BGR uint8 image.

    Uses COLORMAP_JET: blue = far/safe (0.0), red = close/danger (1.0).
    The normalised input should be 1 - depth/depth_max so that close=1.0.
    """
    img8 = np.clip(depth_float * 255.0, 0, 255).astype(np.uint8)
    return cv2.applyColorMap(img8, cv2.COLORMAP_JET)


def _write_atomic(path: str, img: np.ndarray) -> None:
    base, ext = os.path.splitext(path)
    tmp = base + "_tmp" + ext
    cv2.imwrite(tmp, img)
    os.replace(tmp, path)


def main():
    env_cfg = MushrMazeEnvCfg()
    env_cfg.scene.num_envs = 1
    env_cfg.show_camera_debug = False
    if args_cli.device is not None:
        env_cfg.sim.device = args_cli.device

    # CLI overrides for crop fractions
    if args_cli.crop_top is not None:
        env_cfg.depth_crop_top_fraction = args_cli.crop_top
    if args_cli.crop_bottom is not None:
        env_cfg.depth_crop_bottom_fraction = args_cli.crop_bottom

    cam_h = env_cfg.camera_cfg.height   # 180
    cam_w = env_cfg.camera_cfg.width    # 240
    depth_max = env_cfg.depth_max_m

    crop_top_frac = env_cfg.depth_crop_top_fraction
    crop_bot_frac = env_cfg.depth_crop_bottom_fraction
    pol_h = max(1, int(cam_h * (1.0 - crop_top_frac - crop_bot_frac)))
    pol_w = cam_w

    print("[visualize] Creating environment …")
    env = gym.make(args_cli.task, cfg=env_cfg)
    mushr_env = env.unwrapped
    device = mushr_env.device

    out_dir = "/tmp/mushr_debug"
    os.makedirs(out_dir, exist_ok=True)

    print(
        f"[visualize] Camera: {cam_w}×{cam_h}  |  "
        f"Policy depth: {pol_w}×{pol_h}  |  "
        f"depth_max={depth_max:.1f} m  |  "
        f"crop_top={crop_top_frac:.2f}  crop_bottom={crop_bot_frac:.2f}"
    )
    print(f"[visualize] Frames → {out_dir}/   |   Press Ctrl+C to stop.")

    key_state = _KeyState()
    kb_listener = None
    if _HAS_PYNPUT:
        kb_listener = _pynput_kb.Listener(
            on_press=key_state.on_press, on_release=key_state.on_release
        )
        kb_listener.start()
        print("[visualize] WASD controls: W=accel  S=brake  A=left  D=right\n")
    else:
        print("[visualize] pynput not found — robot stays stationary.")
        print("[visualize] Install with:  pip install pynput\n")

    env.reset()

    try:
        while simulation_app.is_running():
            throttle, steer = key_state.get_actions()
            actions = torch.tensor([[throttle, steer]], device=device)
            env.step(actions)

            # Grab raw depth frame [1, H, W, 1]
            depth_tensor = mushr_env.camera.data.output["distance_to_image_plane"]
            depth_np = depth_tensor[0, :, :, 0].cpu().numpy()  # [H, W] float32

            # Replace inf/nan with depth_max for display
            depth_np = np.where(np.isfinite(depth_np), depth_np, depth_max)

            # ── Raw depth window (full resolution, false-coloured) ──────────────
            # Normalise: close=1.0 (red), far=0.0 (blue)
            raw_norm = 1.0 - np.clip(depth_np / depth_max, 0.0, 1.0)
            raw_bgr  = _depth_to_colormap(raw_norm)
            _write_atomic(os.path.join(out_dir, "raw.png"), raw_bgr)

            # ── Policy depth window (cropped + noised, matches what policy sees) ─
            top  = int(crop_top_frac * cam_h)
            bot  = cam_h - int(crop_bot_frac * cam_h)
            pol_np = depth_np[top:bot, :]                       # [H_crop, W]
            # Apply same noise as the env (for accurate preview)
            if env_cfg.depth_noise_sigma_m > 0.0:
                pol_np = pol_np + np.random.normal(0.0, env_cfg.depth_noise_sigma_m, pol_np.shape).astype(np.float32)
            pol_np = np.clip(pol_np, 0.01, depth_max)
            pol_norm = 1.0 - pol_np / depth_max                 # close=1 red, far=0 blue
            pol_bgr  = _depth_to_colormap(pol_norm)

            # Overlay min/max depth stats
            min_d = float(pol_np.min())
            max_d_vis = float(pol_np.max())
            label = f"min={min_d:.2f}m  max={max_d_vis:.2f}m"
            cv2.putText(
                pol_bgr, label, (2, pol_h - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1, cv2.LINE_AA,
            )
            _write_atomic(os.path.join(out_dir, "policy.png"), pol_bgr)

    except KeyboardInterrupt:
        pass

    env.close()

    if kb_listener is not None:
        kb_listener.stop()

    if display_proc is not None and display_proc.poll() is None:
        display_proc.terminate()
        display_proc.wait()


if __name__ == "__main__":
    main()
    simulation_app.close()
