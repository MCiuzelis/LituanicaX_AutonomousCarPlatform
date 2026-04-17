# LituanicaX — Intelligent Robot Competition

RL-trained autonomous RC car that navigates a cone-track circuit using a depth camera.  
Policy trained in NVIDIA Isaac Sim / Isaac Lab (PPO); deployed on a Raspberry Pi 5.

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
├── Hardware/
│   ├── Car/                  # MuSHR nano v2 CAD (Blender + STL)
│   ├── PCB/                  # KiCAD schematic + layout for driver board
│   └── Track/                # Race-track CAD (Blender + FBX)
│
└── Training/                 # Isaac Sim RL training pipeline
    ├── train.py              # Entry point — launches Isaac Sim, runs PPO
    ├── play.py               # Loads a checkpoint, runs inference + FPV view
    ├── visualize.py          # Manual WASD drive + live depth-camera windows
    ├── cli_args.py           # Shared RSL-RL CLI helpers
    ├── assets/               # USD scene files (robot, track, walls)
    │   ├── mushr_nano_v2.usd
    │   ├── CYLINDERS.usd     # Cone-track (visible to policy camera)
    │   ├── WALLS_no_ramps.usd# Boundary walls (termination only, invisible)
    │   └── RAMPS.usd
    ├── logs/                 # Training outputs — excluded from git
    └── source/
        ├── setup.py
        └── lituanicax/       # Isaac Lab external extension (Python package)
            ├── __init__.py
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
| Onboard compute | Raspberry Pi 5 (8 GB) |
| Depth sensor | ArduCam TOF — 240×180 px @ 30 fps, 70° diagonal FOV, ±2 cm accuracy |
| Custom PCB | Motor driver + servo control + power management (Hardware/PCB/) |
| Training GPU | Any NVIDIA GPU supported by Isaac Sim (RTX 3080+ recommended) |

---

## Software Dependencies

### Training (Linux PC)

| Package | Version | Notes |
|---|---|---|
| NVIDIA GPU driver | ≥ 535.54 | Required by Isaac Sim 4.x |
| NVIDIA Isaac Sim | 4.2+ | Installed via pip or Omniverse |
| Isaac Lab | 2.0+ | See installation below |
| rsl-rl-lib | ≥ 3.0.1 | PPO training backend (checked at startup) |
| conda / Python | 3.10–3.11 | Conda env named `isaac` |
| pynput | any | WASD keyboard control in `visualize.py` |
| packaging | any | Version check in training scripts |

### Deployment (Raspberry Pi 5)

| Package | Notes |
|---|---|
| Python 3.11 | System or virtualenv |
| arducam-tof-python | ArduCam TOF SDK (from ArduCam GitHub) |
| onnxruntime | Policy inference |
| RPi.GPIO / gpiozero | Motor and servo GPIO control |

---

## Isaac Sim & Isaac Lab Installation

### 1. Install Isaac Sim via pip (recommended, headless + GUI)

```bash
conda create -n isaac python=3.10
conda activate isaac

# Install Isaac Sim — pick the version matching your driver
pip install isaacsim==4.5.0 \
    isaacsim-extscache-physics==4.5.0 \
    isaacsim-extscache-kit==4.5.0 \
    isaacsim-extscache-kit-sdk==4.5.0 \
    --extra-index-url https://pypi.nvidia.com

# Verify
python -c "from isaaclab.app import AppLauncher; print('OK')"
```

> For the latest available versions: `pip index versions isaacsim --extra-index-url https://pypi.nvidia.com`

### 2. Install Isaac Lab

```bash
git clone https://github.com/isaac-sim/IsaacLab.git
cd IsaacLab
pip install -e source/isaaclab
pip install -e source/isaaclab_assets
pip install -e source/isaaclab_tasks
pip install -e source/isaaclab_rl
```

### 3. Install the LituanicaX extension

```bash
pip install -e /path/to/LituanicaX_InteligentRobotCompetition/Training/source
```

Re-run this after any changes to `env.py` or `ppo_cfg.py`.

### 4. Install remaining training dependencies

```bash
pip install rsl-rl-lib>=3.0.1 packaging pynput
```

---

## Training

```bash
conda activate isaac
cd Training

# Basic run (40 envs, 3000 iterations)
python train.py --task ConeTrack

# Override common parameters
python train.py --task ConeTrack \
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
tensorboard --logdir Training/logs/rsl_rl/cone_track/
```

---

## Inference / Playback

```bash
# Load the latest checkpoint automatically
python play.py --task ConeTrack --num_envs 1

# Load a specific checkpoint
python play.py --task ConeTrack --num_envs 1 \
    --checkpoint Training/logs/rsl_rl/cone_track/<run>/model_1000.pt

# First-person depth-camera view (requires $DISPLAY)
python play.py --task ConeTrack --num_envs 1 --fpv
```

`play.py` exports the policy to `<checkpoint_dir>/exported/policy.pt` and `policy.onnx`
on every run — copy these to `Deployment/Policy/` for on-robot use.

---

## Depth Camera Visualisation (no policy)

```bash
python visualize.py --task ConeTrack

# Tune crop and noise parameters live
python visualize.py --task ConeTrack \
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

---

## Sim-to-Real Notes

- The policy observes **normalised depth** only (no RGB, no velocity).  
  Close objects → 1.0; far/clear → 0.0.
- Noise simulated during training: Gaussian σ ≈ 0.008 m, 20 % invalid-pixel dropout,
  5 % colour-speckle, and edge-blur at depth discontinuities.
- Action latency of 33 ms (1 policy step) is simulated in training to match hardware.
- `Deployment/autopilot.py` mirrors the same depth pipeline (`crop → clamp → normalise`)
  used in `env.py → _process_single_depth()`.

---

## Asset Path Notes

All USD paths in `env.py` are relative to the `Training/` directory.  
If assets are moved or the repo is cloned to a different machine, no path changes are needed
as long as the directory structure is preserved.

`Training/Blender/curve_points.csv` — sector-gate centreline exported from Blender —
must be present for training.  If missing, the environment will error at startup.


Blender fbx export settings: scale 0.01, Z up, -y forwards only visible objects