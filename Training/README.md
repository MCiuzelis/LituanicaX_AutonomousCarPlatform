# Training Environment

Isaac Sim 5.1 / Isaac Lab training pipeline for the LituanicaX intelligent robot
competition.  A MuSHR nano v2 RC car learns to drive a cone track from two depth
cameras, using PPO (RSL-RL).  The exported ONNX policy is what runs on the
Raspberry Pi in `Deployment/`.

## Layout

```
Training/
├── source/cone_track/          # everything that is code
│   ├── __init__.py             # gym.register("ConeTrack", …)
│   ├── env.py                  # ConeTrackEnv + ConeTrackEnvCfg — ALL env tuning
│   ├── ppo_cfg.py              # ConeTrackPPORunnerCfg — ALL PPO tuning
│   ├── actor_critic.py         # depth trunk + proprio encoder network
│   ├── train.py                # train a policy (launches TensorBoard too)
│   ├── play.py                 # run a checkpoint; exports policy.pt / policy.onnx
│   ├── visualize.py            # WASD-drive the car, live depth windows, noise tuning
│   ├── acceleration_test.py    # drivetrain bench test (accel / braking / slip)
│   └── cli_args.py             # shared RSL-RL command-line arguments
├── assets/
│   ├── Track/                  # Blender sources
│   └── Export/                 # USD the sim loads: car, track, walls, prop sets
├── outputs/                    # one directory per run (git-ignored)
└── IsaacLab/                   # submodule, pinned to v2.3.0
```

Day to day you only touch `env.py` (robot, track, rewards), `ppo_cfg.py`
(learning), and the entry-point scripts.

## Run outputs

A training run writes exactly one directory, and everything it produces lives
there:

```
outputs/<YYYY-MM-DD_HH-MM-SS>[_<run_name>]/
├── events.out.tfevents.*   # TensorBoard
├── model_*.pt              # checkpoints, every save_interval iterations
├── exported/               # policy.pt + policy.onnx, written by play.py
├── params/                 # resolved env.yaml + agent.yaml
├── videos/                 # only with --video
├── git/                    # commit state at launch
├── .hydra/ + train.log     # the invocation, as Hydra resolved it
└── play.log                # last play session on this run
```

`train.py` points TensorBoard at `outputs/` as a whole, so every run shows up
side by side.  To open it by hand:

```bash
cd Training && uv run tensorboard --logdir outputs/
```

## Commands

`install.sh` puts `train`, `play`, `visualize` and `acceleration_test` helpers
in `~/.local/bin`; they `cd` here and run under `uv` for you.  `ConeTrack` is
the default task.

```bash
train --num_envs 26 --headless          # train; TensorBoard opens automatically
train --resume --load_run 2026-07-26_11-41-00
play  --num_envs 1 --real-time          # newest checkpoint of the newest run
play  --checkpoint outputs/<run>/model_500.pt
play  --fpv                             # live raw + policy depth windows
visualize --crop_top 0.2 --noise_sigma 0.01
acceleration_test --headless
```

Anything else is a plain `uv run python source/cone_track/<script>.py …` from
this directory.  `train.py` and `play.py` also accept Hydra-style `key=value`
overrides, which are applied on top of both configs — except for run selection
(`--load_run` / `--checkpoint`), which is resolved before Hydra starts so that
Hydra can write into the run directory.

## Fresh setup

From the repository root:

```bash
wget -qO- https://astral.sh/uv/install.sh | sh
./Training/install.sh
```

That command will:

1. Initialize and update the `Training/IsaacLab` git submodule.
2. Force the submodule to the Isaac Sim 5.1-compatible Isaac Lab tag `v2.3.0`.
3. Create or update the `Training/.venv` virtual environment with `uv sync`.
4. Install the `train` / `play` / `visualize` / `acceleration_test` helper scripts.

If the submodule is already present, you can also run `uv sync` directly inside
`Training/`, but `./Training/install.sh` is the safer fresh-machine command.

Notes:

- The virtual environment lives at `Training/.venv`; everything runs through `uv`.
- The Isaac Lab packages are installed editable from the submodule, and `source/`
  is on the path, so `cone_track` imports directly.

## Editor setup

`pyrightconfig.json` configures pyright (Zed, VS Code, …): it points at `.venv`
and adds the Isaac Lab submodule source trees to the import path.  That last
part is what fixes "`isaaclab` could not be resolved" — Isaac Lab is installed
as an editable package, which setuptools implements with an import hook that no
type checker can follow.

**Open `Training/` as the project root**, not the repository root; pyright only
reads this config from the workspace root.

`typings/` holds one-line stubs for `pxr` and `omni` (pyright's default
`stubPath`).  Both fill their namespaces from C++ at runtime — `pxr` via
`Tf.PreparePythonModule()`, `omni` via extension paths Kit injects on launch —
so without the stubs every `from pxr import UsdGeom` and `import omni.usd` in
`env.py` / `acceleration_test.py` is reported as an unresolved import.  They are
editor-only: nothing imports them at runtime, and deleting them changes nothing
except that pyright starts complaining.

## NVIDIA Driver Compatibility

Isaac Sim 5.1.0 requires an NVIDIA proprietary driver in the **550–580 series**.

| Driver series | Status |
|---|---|
| 550, 560, 570, 580 | ✅ Compatible (tested: 550.163.01, 580.159.03) |
| 590, 595 | ❌ Crashes in `librtx.scenedb.plugin.so` during Hydra/RTX init |

### Dual-GPU (iGPU + dGPU) systems

When displays are connected to an integrated GPU (Intel/AMD) and the NVIDIA GPU
has no active display, Vulkan defaults to the iGPU. The helper scripts in
`~/.local/bin/` (`train` / `play` / `visualize`) set
`__NV_PRIME_RENDER_OFFLOAD=1` automatically on dual-GPU systems to route
Vulkan rendering to the NVIDIA GPU.

### Installing / downgrading the driver

```bash
# Ubuntu 24.04
sudo apt-get install nvidia-driver-550-open   # or -570-open, -580-open
sudo reboot
```

If **Secure Boot** is enabled, the DKMS kernel module must be signed with a key
enrolled in the Machine Owner Key (MOK) database. After the first reboot:

1. The **MOK Manager** blue EFI screen appears — press any key to enter it.
2. Select **Enroll MOK** → **Continue** → **Yes** to enrol the key.
3. The system reboots again; the NVIDIA driver will now load.

Verify with:

```bash
nvidia-smi   # should show driver version and GPU
```
