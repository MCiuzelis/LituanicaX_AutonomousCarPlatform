# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

RL-trained autonomous RC car navigating a cone-track circuit using depth cameras only. Policy trained in NVIDIA Isaac Sim 5.1 / Isaac Lab 2.3 with PPO; deployed on a Raspberry Pi 5. The simulation uses a MuSHR nano v2 Ackermann-steered car with dual ArduCam TOF depth cameras.

## Commands

All training commands must be run from within `Training/` because the `uv` virtual environment lives there. Terminal shortcuts handle the `cd` automatically (see below).

### Terminal shortcuts (from any directory)

```bash
train --task ConeTrack                          # launch PPO training
train --task ConeTrack --num_envs 40 --run_name v1
play  --task ConeTrack --num_envs 1 --fpv       # inference + first-person view
visualize --task ConeTrack                      # WASD manual drive + depth windows
```

### Direct invocation (from `Training/`)

```bash
cd Training
uv run python train.py --task ConeTrack
uv run python play.py  --task ConeTrack --num_envs 1
uv run python visualize.py --task ConeTrack
```

### Fresh environment setup (Linux x86-64 only, Ubuntu 24.04 LTS)

```bash
pip install uv          # or: wget -qO- https://astral.sh/uv/install.sh | sh
./Training/install.sh   # syncs IsaacLab submodule to v2.3.0, runs uv sync
```

### TensorBoard

```bash
cd Training
uv run tensorboard --logdir logs/rsl_rl/cone_track/
```

## Architecture

### Training pipeline (`Training/`)

```
train.py          → gym.make("ConeTrack") → RslRlVecEnvWrapper → PPO (RSL-RL)
play.py           → load checkpoint → inference; exports policy.pt + policy.onnx
visualize.py      → no policy, WASD keyboard; tune depth pipeline params live
cli_args.py       → shared argparse helpers for RSL-RL (experiment, resume, checkpoint)
```

The gym task `"ConeTrack"` is registered in `source/tasks/cone_track/__init__.py` and maps to `ConeTrackEnv` / `ConeTrackEnvCfg` in `env.py`.

### Environment (`source/tasks/cone_track/env.py`)

Subclasses Isaac Lab's `DirectRLEnv`. Key design choices:

- **Observation (7681 dims):** left depth (3840) + right depth (3840) + avg wheel surface speed (1). No RGB. Action history is available (`include_prev_action`) but **off** by default; wheel velocity (`include_wheel_vel`) is **on**.
- **Depth normalization:** `1.0 - depth_m / 2.0` — value 1.0 = obstacle at 0 m, 0.0 = clear at 2 m.
- **Shared depth pipeline** (`_process_single_depth`): crop top 35% / bottom 30% → Gaussian noise + invalid/colour speckle + edge scatter/blur → clamp [0.01, 2.0] m → normalize → bilinear downsample to 120×32. The identical pipeline runs in `Deployment/autopilot.py` for sim-to-real transfer.
- **Action space (2 dims):** throttle ∈ [−1, 1] (three-zone: drive / coast / brake), steering ∈ [−1, 1] (scaled to ±0.488 rad).
- **Action delay:** `action_delay_ms = 0` by default (latency model disabled). Set non-zero to emulate hardware latency (33 ms ≈ 1 policy step).
- **Motor model:** direct AWD torque control with linear torque-speed curve derived from `car_mass_kg`, `wheel_grip_mu_s`, `slip_threshold`, and `max_velocity_m_s`; braking torque via `brake_strength`.
- **Wall-hit penalty:** one-time terminal penalty scaled by impact directionality — full for a head-on hit, ~0 for a glancing forward rub (`wall_hit_penalty_weight`).
- **Cameras:** ±30° yaw (not pitch), 122 mm forward, ±50 mm lateral, 70 mm up from `base_link`.

### PPO config (`source/tasks/cone_track/agents/ppo_cfg.py`)

- Actor: 512 → 256 → 128 → 2, ELU activation, obs normalization on. `init_noise_std=0.75`.
- Critic: 1024 → 512 → 256 → 1, obs normalization on.
- 1200 steps/env per update, 6 epochs × 24 mini-batches, γ=0.997, λ=0.95.
- `learning_rate=1e-4`, adaptive schedule with `desired_kl=0.02`, `entropy_coef=0.002`, `clip_param=0.2`, `max_grad_norm=0.75`.

### Deployment (`Deployment/`)

- `autopilot.py` — runs on RPi 5; loads `Policy/policy.onnx`; WebSocket server on `:8080`; 30 Hz policy rate, spacebar gate.
- `hardware.py` — hardware PWM (not GPIO PWM) for motor + servo; max throttle clamped to 6%.
- Policy files are exported by `play.py` and copied to `Deployment/Policy/`.

### Assets

Robot USD lives in `Training/assets/`; track geometry is exported to `CAD/Track/Export/` (paths resolved in `env.py`):
- `Training/assets/mushr_nano_v2.usd` — robot articulation
- `CAD/Track/Export/Track.usdc` — visible cone geometry (policy cameras see this)
- `CAD/Track/Export/Walls.usdc` — invisible boundary walls (termination only; hidden during `play.py`)
- `CAD/Track/Export/Assets1.usdc … Assets5.usdc` — 5 decorative asset sets cycled round-robin (`asset_randomize_every_n_steps`); purely visual domain randomization the depth cameras see
- `Training/assets/RAMPS.usd` — optional elevation geometry (currently disabled in `_setup_scene`)

### Key configuration values (in `ConeTrackEnvCfg`)

| Parameter | Default | Effect |
|---|---|---|
| `decimation` | 4 | Policy at ~30 Hz (sim 120 Hz) |
| `depth_max_m` | 2.0 | Matches ArduCam hardware range |
| `action_delay_ms` | 0 | Latency sim (disabled; 33 ms ≈ 1 step) |
| `max_velocity_m_s` | 6.0 | Soft top-speed limit (torque → 0 here) |
| `distance_weight` | 150.0 | Primary forward-progress reward |
| `braking_v2_weight` | 5.0 | Dense brake-at-speed reward (`v²·√‖brake‖`) |
| `spawn_escape_reward` | 150 | One-time bonus for leaving the spawn at speed |
| `wall_hit_penalty_weight` | 25.0 | Terminal penalty, scaled by head-on impact |
| `wall_contact_force_thresh_n` | 10.0 | Horizontal force counting as a wall hit |

## Dependency notes

- `pyproject.toml` locks Python to exactly 3.11 and Linux x86-64; `uv` enforces this.
- Isaac Sim 5.1 is installed as a pip package from `https://pypi.nvidia.com`.
- `IsaacLab` is a git submodule pinned to `v2.3.0`; `install.sh` handles the checkout.
- Ubuntu 26.04 is **not** supported by Isaac Sim 5.1 — use Ubuntu 24.04 LTS.
