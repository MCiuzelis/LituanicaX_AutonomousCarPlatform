# Training Environment

This directory contains the Isaac Sim / Isaac Lab training environment for the
LituanicaX intelligent robot competition.

## Fresh setup

From the repository root:

install uv 

```bash
wget -qO- https://astral.sh/uv/install.sh | sh
./Training/install.sh
```

That command will:

1. Initialize and update the `Training/IsaacLab` git submodule.
2. Force the submodule to the Isaac Sim 5.1-compatible Isaac Lab tag `v2.3.0`.
3. Create or update the `Training/.venv` virtual environment with `uv sync`.

If the submodule is already present, you can also run `uv sync` directly inside
`Training/`, but `./Training/install.sh` is the safer fresh-machine command.

## Notes

- The virtual environment lives at `Training/.venv`.
- The local Isaac Lab packages are installed from the submodule in editable mode.
- The local `tasks` package is installed from `Training/source`.

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
