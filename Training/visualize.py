"""Visualize MuSHR center RGB camera + blue-mask policy pipeline from Isaac Sim.

The robot is spawned at its start position.  WASD keys drive it manually
(requires pynput — install with: pip install pynput).

A separate display subprocess owns the cv2.imshow windows (avoids Qt conflict).

Two windows are shown:
  "Center Raw RGB"      — full-resolution 848×480 RGB feed (clean Isaac Sim output)
  "Center Policy Mask"  — EXACT policy pipeline: blue-only HSV mask + downsample, upscaled

Frames are exchanged via PNG files written to /tmp/mushr_debug/.
Press  Ctrl+C  in this terminal  OR  Q  in any display window to stop.

Usage:
    python .../visualize.py --task Mushr
"""

import argparse
import os
import subprocess
import sys
import threading

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(
    description="Visualize MuSHR center RGB camera (no policy)."
)
parser.add_argument("--task", type=str, default="ConeTrack", help="Gym task name.")

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True

# ── display subprocess ─────────────────────────────────────────────────────────
_DISPLAY_SCRIPT = r"""
import sys, os
sys.path = [
    *[p for p in sys.path if "pip_prebundle" not in p and "omni" not in p],
]
import cv2, time, signal
signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))

OUT_DIR = "/tmp/mushr_debug"
FILES = [
    (os.path.join(OUT_DIR, "raw_center.png"),    "Center Raw RGB"),
    (os.path.join(OUT_DIR, "policy_center.png"), "Center Policy Mask"),
]

while not os.path.exists(os.path.join(OUT_DIR, "raw_center.png")):
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
            steer = (1.0 if "a" in self._down else 0.0) - (
                1.0 if "d" in self._down else 0.0
            )
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
import cv2  # headless fine here — only imwrite used
import gymnasium as gym
import isaaclab_tasks  # noqa: F401
import numpy as np
import tasks  # noqa: F401
import torch
from tasks.cone_track.env import ConeTrackEnvCfg


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

    # Don't terminate on zero speed during manual driving
    env_cfg.slow_stop_speed_fraction = 0.0
    env_cfg.spawn_grace_steps = 100000

    cam_h = env_cfg.camera_center_cfg.height
    cam_w = env_cfg.camera_center_cfg.width

    print("[visualize] Creating environment …")
    env = gym.make(args_cli.task, cfg=env_cfg)
    mushr_env = env.unwrapped
    device = mushr_env.device

    # ── debug sphere at camera position ────────────────────────────────────────
    try:
        import omni.usd
        from pxr import Gf, UsdGeom

        stage = omni.usd.get_context().get_stage()
        cam_prim = stage.GetPrimAtPath(
            mushr_env.cfg.camera_center_cfg.prim_path.replace("env_.*", "env_0")
        )
        sphere = UsdGeom.Sphere.Define(
            stage, cam_prim.GetPath().AppendChild("guide_sphere")
        )
        sphere.GetRadiusAttr().Set(0)
        print(f"[visualize] Camera guide sphere under {cam_prim.GetPath()}")
    except Exception as e:
        print(f"[visualize] Guide sphere failed: {e}")
    # ───────────────────────────────────────────────────────────────────────────

    # Never terminate / respawn during manual driving
    mushr_env._check_termination = lambda: (
        torch.zeros(mushr_env.num_envs, dtype=torch.bool, device=mushr_env.device),
        torch.zeros(mushr_env.num_envs, dtype=torch.bool, device=mushr_env.device),
    )

    out_dir = "/tmp/mushr_debug"
    os.makedirs(out_dir, exist_ok=True)

    print(
        f"[visualize] Camera: {cam_w}×{cam_h} RGB  "
        f"| Policy mask res: {mushr_env._policy_depth_w}×{mushr_env._policy_depth_h}"
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
        """Write raw RGB and policy blue-mask frames to disk."""
        rgb_tensor = (
            cam_obj.data.output["rgb"][0, :, :, :3].cpu().numpy().astype(np.uint8)
        )
        raw_bgr = cv2.cvtColor(rgb_tensor, cv2.COLOR_RGB2BGR)
        _write_atomic(os.path.join(out_dir, f"raw_{suffix}.png"), raw_bgr)

        # ── Policy vision ──────────────────────────────────
        pol_flat = mushr_env._process_single_rgb(cam_obj)
        pol_2d = (
            pol_flat[0]
            .reshape(
                mushr_env._policy_depth_h,
                mushr_env._policy_depth_w,
            )
            .cpu()
            .numpy()
        )

        pol_bgr = cv2.cvtColor((pol_2d * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)

        disp_h = max(
            1, round(cam_w * mushr_env._policy_depth_h / mushr_env._policy_depth_w)
        )
        pol_bgr = cv2.resize(pol_bgr, (cam_w, disp_h), interpolation=cv2.INTER_NEAREST)
        _write_atomic(os.path.join(out_dir, f"policy_{suffix}.png"), pol_bgr)

    try:
        while simulation_app.is_running():
            throttle, steer = key_state.get_actions()
            actions = torch.tensor([[throttle, steer]], device=device)
            env.step(actions)
            _write_camera(mushr_env.camera_center, "center")

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
