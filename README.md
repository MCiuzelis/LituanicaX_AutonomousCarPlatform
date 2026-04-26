# LituanicaX — Intelligent Robot Competition

RL-trained autonomous RC car that navigates a cone-track circuit using a depth camera.  
Policy trained in NVIDIA Isaac Sim 5.1 / Isaac Lab (PPO); deployed on a Raspberry Pi 5.

---

## Repository Structure

```
.
├── Deployment/               # Code that runs on the Raspberry Pi 5
│   ├── autopilot.py          # Main autonomous-driving loop
│   ├── hardware.py           # Motor, servo, and camera interface
│   └── Policy/               # Exported policy files (deploy these to the Pi)
│       ├── policy.pt         # TorchScript export
│       └── policy.onnx       # ONNX export (used by autopilot.py)
│
├── CAD/
│   ├── Car/                  # MuSHR nano v2 CAD (Blender + STL)
│   ├── PCB/                  # KiCAD schematic + layout for driver board
│   └── Track/                # Race-track CAD (Blender + FBX)
│
└── Training/                 # Isaac Sim RL training pipeline
    ├── train.py              # Entry point — launches Isaac Sim, runs PPO
    ├── play.py               # Loads a checkpoint, runs inference + FPV view
    ├── visualize.py          # Manual WASD drive + live depth-camera windows
    ├── assets/               # USD scene files (robot, track, walls)
    │   ├── mushr_nano_v2.usd
    │   ├── TRACK.usd         # Cone-track (visible to policy camera)
    │   ├── WALLS.usd         # Boundary walls (termination only, invisible)
    │   └── RAMPS.usd
    ├── logs/                 # Training outputs — excluded from git
    └── source/
        └── tasks/
            └── cone_track/
                ├── __init__.py   # gym.register("ConeTrack", ...)
                ├── env.py        # ConeTrackEnv + ConeTrackEnvCfg
                └── agents/
                    └── ppo_cfg.py  # ConeTrackPPORunnerCfg
```

---

## Hardware

| Component | Details |
|---|---|
| Robot | MuSHR nano v2 — Ackermann-steered, all-wheel-drive RC car |
| Onboard compute | Raspberry Pi 5 (4 GB) |
| Depth sensor | ArduCam TOF — 240×180 px @ 30 fps, 70° diagonal FOV, ±2 cm accuracy |
| Custom PCB | Motor driver + servo control + power management (CAD/PCB/) |
| Training GPU | Any NVIDIA GPU supported by Isaac Sim (RTX 3080+ recommended) |

---

## Installation (Training — Linux only)

**Supported OS: Ubuntu 24.04 LTS (x86-64).  Ubuntu 26.04 is not supported by Isaac Sim 5.1.**

### 1. Install `uv`

```bash
wget -qO- https://astral.sh/uv/install.sh | sh
```

### 2. Run the setup script

From the repository root:

```bash
./Training/install.sh
```

This will:
1. Initialize and update the `Training/IsaacLab` git submodule.
2. Pin the submodule to Isaac Lab `v2.3.0` (required for Isaac Sim 5.1 compatibility).
3. Create `Training/.venv` and install all dependencies with `uv sync`.

See `Training/README.md` for details.

---

## Terminal Shortcuts

Three shortcuts are installed to `~/.local/bin` so you can run training commands from any directory:

| Command | Equivalent |
|---|---|
| `train [args]` | `cd Training && uv run python train.py [args]` |
| `play [args]` | `cd Training && uv run python play.py [args]` |
| `visualize [args]` | `cd Training && uv run python visualize.py [args]` |

`~/.local/bin` must be on your `PATH` (it is by default on Ubuntu 24.04).  
Changes take effect in new terminal sessions (or run `source ~/.bashrc`).

---

## Training

```bash
# Basic run (default 40 envs, 3000 iterations)
train --task ConeTrack

# Common overrides
train --task ConeTrack \
    --num_envs 200 \
    --max_iterations 5000 \
    --seed 42 \
    --run_name my_experiment
```

Checkpoints are saved every 20 iterations to:
```
Training/logs/rsl_rl/cone_track/<YYYY-MM-DD_HH-MM-SS>_<run_name>/
```

Monitor training:
```bash
cd Training && uv run tensorboard --logdir logs/rsl_rl/cone_track/
```

---

## Inference / Playback

```bash
# Load the latest checkpoint automatically
play --task ConeTrack --num_envs 1

# Load a specific checkpoint
play --task ConeTrack --num_envs 1 \
    --checkpoint Training/logs/rsl_rl/cone_track/<run>/model_1000.pt

# First-person depth-camera view (requires $DISPLAY)
play --task ConeTrack --num_envs 1 --fpv
```

`play.py` exports the policy to `<checkpoint_dir>/exported/policy.pt` and `policy.onnx`
on every run — copy these to `Deployment/Policy/` for on-robot use.

---

## Depth Camera Visualisation (no policy)

```bash
visualize --task ConeTrack

# Tune crop and noise parameters live
visualize --task ConeTrack \
    --crop_top 0.30 --crop_bottom 0.25 \
    --noise_sigma 0.010 --invalid_prob 0.15
```

Opens four cv2 windows (left/right × raw/policy-processed depth).  
WASD to drive; `Q` to quit.

---

## Deployment on Raspberry Pi 5

### One-time hardware setup

Enable hardware PWM by adding to `/boot/firmware/config.txt`:
```
dtoverlay=pwm-2chan,pin=12,func=4,pin2=18,func2=4
```

### Running the autopilot

```bash
source ~/arducam_venv/bin/activate
python Deployment/autopilot.py
```

The autopilot loads `Deployment/Policy/policy.onnx` and drives the car autonomously.  
Hold **Spacebar** in the browser UI to activate the policy.

---

## Sim-to-Real Notes

- The policy observes **normalised depth** only (no RGB, no velocity).  
  Close objects → 1.0; far/clear → 0.0.
- Noise simulated during training: Gaussian σ ≈ 0.008 m, 20 % invalid-pixel dropout,
  5 % colour-speckle, and edge-blur at depth discontinuities.
- Action latency of 33 ms (1 policy step) is simulated in training to match hardware.
- `Deployment/autopilot.py` mirrors the same depth pipeline (`crop → clamp → normalise`)
  used in `env.py → _process_single_depth()`.
