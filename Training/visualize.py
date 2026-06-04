"""Visualize MuSHR dual ArduCam TOF depth streams from Isaac Sim without running any policy.

The robot is spawned at its start position.  WASD keys drive it manually
(requires pynput — install with: pip install pynput).

A separate display subprocess owns the cv2.imshow windows (avoids Qt conflict).

Four windows are shown:
  "Left Raw Depth"      — full-resolution left-camera depth, false-coloured COLORMAP_JET
                          (clean Isaac Sim output, no simulated noise)
  "Left Policy Depth"   — EXACT policy pipeline: crop + all noise + downsample, upscaled
  "Right Raw Depth"     — full-resolution right-camera depth (clean)
  "Right Policy Depth"  — exact policy pipeline for right camera

Noise parameters are all overridable from the command line — change them live to tune.
The noise overlay text at the bottom of each policy window shows the active values.

Frames are exchanged via PNG files written to /tmp/mushr_debug/.
Press  Ctrl+C  in this terminal  OR  Q  in any display window to stop.

Usage:
    python .../visualize.py --task Mushr
    python .../visualize.py --task Mushr --crop_top 0.2 --crop_bottom 0.15
    python .../visualize.py --task Mushr --color_noise_prob 0.05 --edge_noise_sigma 0.08
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
parser.add_argument("--task", type=str, default="ConeTrack", help="Gym task name.")
parser.add_argument("--crop_top",    type=float, default=None,
    help="Override depth_crop_top_fraction (e.g. 0.15).")
parser.add_argument("--crop_bottom", type=float, default=None,
    help="Override depth_crop_bottom_fraction (e.g. 0.10).")
# Noise overrides — all optional; cfg defaults are used when omitted.
parser.add_argument("--noise_sigma",      type=float, default=None,
    help="Override depth_noise_sigma_m (broadband Gaussian, metres).")
parser.add_argument("--invalid_prob",     type=float, default=None,
    help="Override depth_invalid_pixel_prob (black-dot speckle, 0–1).")
parser.add_argument("--color_noise_prob", type=float, default=None,
    help="Override depth_color_noise_prob (random-depth coloured dots, 0–1).")
parser.add_argument("--edge_grad_thresh", type=float, default=None,
    help="Override depth_edge_gradient_threshold_m (gradient m/px that gives edge_weight=1).")
parser.add_argument("--edge_noise_sigma", type=float, default=None,
    help="Override depth_edge_blur_sigma_m (Gaussian std at full edges, metres).")

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True

# ── display subprocess ─────────────────────────────────────────────────────────
_conda_prefix = os.environ.get("CONDA_PREFIX", "")
_CONDA_SP = (
    os.path.join(_conda_prefix, "lib", f"python{sys.version_info.major}.{sys.version_info.minor}", "site-packages")
    if _conda_prefix else ""
)

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
    _display_env["QT_QPA_FONTDIR"] = "/usr/share/fonts"
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
import tasks  # noqa: F401

from tasks.cone_track.env import ConeTrackEnvCfg


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
    env_cfg = ConeTrackEnvCfg()
    env_cfg.scene.num_envs = 1
    env_cfg.show_camera_debug = False
    if args_cli.device is not None:
        env_cfg.sim.device = args_cli.device

    # CLI overrides for crop fractions
    if args_cli.crop_top    is not None: env_cfg.depth_crop_top_fraction    = args_cli.crop_top
    if args_cli.crop_bottom is not None: env_cfg.depth_crop_bottom_fraction = args_cli.crop_bottom
    # CLI overrides for noise params
    if args_cli.noise_sigma      is not None: env_cfg.depth_noise_sigma_m               = args_cli.noise_sigma
    if args_cli.invalid_prob     is not None: env_cfg.depth_invalid_pixel_prob           = args_cli.invalid_prob
    if args_cli.color_noise_prob is not None: env_cfg.depth_color_noise_prob             = args_cli.color_noise_prob
    if args_cli.edge_grad_thresh is not None: env_cfg.depth_edge_gradient_threshold_m    = args_cli.edge_grad_thresh
    if args_cli.edge_noise_sigma is not None: env_cfg.depth_edge_blur_sigma_m           = args_cli.edge_noise_sigma

    cam_h     = env_cfg.camera_left_cfg.height   # 180
    cam_w     = env_cfg.camera_left_cfg.width    # 240
    depth_max = env_cfg.depth_max_m

    print("[visualize] Creating environment …")
    env = gym.make(args_cli.task, cfg=env_cfg)
    mushr_env = env.unwrapped
    device    = mushr_env.device

    out_dir = "/tmp/mushr_debug"
    os.makedirs(out_dir, exist_ok=True)

    print(
        f"[visualize] Camera: {cam_w}×{cam_h}  "
        f"| Policy res: {mushr_env._policy_depth_w}×{mushr_env._policy_depth_h}  "
        f"| depth_max={depth_max:.1f} m  "
        f"| crop_top={env_cfg.depth_crop_top_fraction:.2f}  "
        f"crop_bottom={env_cfg.depth_crop_bottom_fraction:.2f}"
    )
    print(
        f"[visualize] Noise: "
        f"gaussian={env_cfg.depth_noise_sigma_m:.3f} m  "
        f"| invalid_prob={env_cfg.depth_invalid_pixel_prob:.3f}  "
        f"| color_prob={env_cfg.depth_color_noise_prob:.3f}  "
        f"| edge_sigma={env_cfg.depth_edge_blur_sigma_m:.3f} m  "
        f"| edge_thresh={env_cfg.depth_edge_gradient_threshold_m:.3f} m/px"
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

    def _write_camera(cam_obj, suffix: str) -> None:
        """Write raw (clean) and policy (fully-noised) depth frames to disk.

        Raw window   — direct Isaac Sim sensor output, no noise applied, full resolution.
        Policy window — calls mushr_env._process_single_depth() which runs the IDENTICAL
                        pipeline the policy uses (crop → edge detection → all noise passes
                        → clamp → normalise → downsample), then upscales for display.
        """
        # ── Raw depth: unprocessed sensor output ──────────────────────────────
        raw_4d   = cam_obj.data.output["distance_to_image_plane"]  # [1, H, W, 1]
        depth_np = raw_4d[0, :, :, 0].cpu().numpy()
        depth_np = np.where(np.isfinite(depth_np), depth_np, depth_max)
        beyond   = depth_np >= depth_max
        raw_norm = 1.0 - np.clip(depth_np / depth_max, 0.0, 1.0)
        raw_bgr  = _depth_to_colormap(raw_norm)
        raw_bgr[beyond] = (0, 0, 0)            # out-of-range → black (matches real ArduCam)
        _write_atomic(os.path.join(out_dir, f"raw_{suffix}.png"), raw_bgr)

        # ── Policy depth: exact env pipeline ──────────────────────────────────
        pol_flat = mushr_env._process_single_depth(cam_obj)       # [1, ph*pw], values [0,1]
        pol_2d   = pol_flat[0].reshape(
            mushr_env._policy_depth_h, mushr_env._policy_depth_w,
        ).cpu().numpy()
        pol_bgr  = _depth_to_colormap(pol_2d)
        # Far pixels have pol_2d ≈ 0.0 (depth = depth_max → normalised to 0).
        # COLORMAP_JET maps 0 → blue, not black.  Explicitly set them to black so
        # the policy window matches the raw window.  This is display-only — the policy
        # receives 0.0 for these pixels regardless of how they are coloured here.
        pol_bgr[pol_2d < (1.0 / 255.0)] = (0, 0, 0)
        # Upscale to full camera width with nearest-neighbour so individual policy
        # pixels are clearly visible without bilinear blurring.
        disp_h = max(1, round(cam_w * mushr_env._policy_depth_h / mushr_env._policy_depth_w))
        pol_bgr = cv2.resize(pol_bgr, (cam_w, disp_h), interpolation=cv2.INTER_NEAREST)
        # Active noise settings overlay at the bottom of each policy window.
        noise_txt = (
            f"g={env_cfg.depth_noise_sigma_m:.3f} "
            f"inv={env_cfg.depth_invalid_pixel_prob:.2f} "
            f"col={env_cfg.depth_color_noise_prob:.2f} "
            f"edge={env_cfg.depth_edge_blur_sigma_m:.3f}@{env_cfg.depth_edge_gradient_threshold_m:.2f}"
        )
        cv2.putText(
            pol_bgr, noise_txt,
            (2, disp_h - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1, cv2.LINE_AA,
        )
        _write_atomic(os.path.join(out_dir, f"policy_{suffix}.png"), pol_bgr)

    try:
        while simulation_app.is_running():
            throttle, steer = key_state.get_actions()
            actions = torch.tensor([[throttle, steer]], device=device)
            env.step(actions)
            _write_camera(mushr_env.camera_left,  "left")
            _write_camera(mushr_env.camera_right, "right")


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
