#!/usr/bin/env bash

set -euo pipefail

# ─────────────────────────────────────────────────────────────
# Pre-flight checks
# ─────────────────────────────────────────────────────────────

# --- Required NVIDIA driver ---
# Isaac Sim 5.1.0 requires an NVIDIA driver between 550.x and 580.x inclusive.
# Driver >= 590.x (e.g. 595) causes a crash in librtx.scenedb.plugin.so during
# Hydra/RTX engine initialisation (known incompatibility).
if nvidia-smi &>/dev/null; then
    driver_ver="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1)"
    if [[ -n "$driver_ver" ]]; then
        major="${driver_ver%%.*}"
        if [[ "$major" -ge 590 ]]; then
            cat >&2 <<INCOMPAT

========================================================================
WARNING: NVIDIA driver ${driver_ver} detected.
Isaac Sim 5.1.0 is incompatible with driver series >= 590.

Working versions: 550, 560, 570, 580 series
        (tested: 550.163.01, 580.159.03)

To downgrade on Ubuntu 24.04:
  sudo apt-get install --reinstall nvidia-driver-550-open
  # Reboot after install.

On newer kernels (6.17+) the 550/570 meta-packages pull in the 580 kernel
module automatically – that is fine; only the userspace libraries matter.
========================================================================

INCOMPAT
        fi
    fi
else
    cat >&2 <<NODRIVER

========================================================================
ERROR: NVIDIA driver is not running.
Isaac Sim requires an NVIDIA GPU with a proprietary driver (550-580 series).

  Check:  nvidia-smi
  Install on Ubuntu 24.04:
    sudo apt-get install nvidia-driver-570-open
    # Reboot and (if Secure Boot is on) enrol the MOK key at the blue
    # EFI prompt that appears after reboot.
========================================================================

NODRIVER
    exit 1
fi

# --- Python 3.11 ---
if ! uv python find 3.11 &>/dev/null; then
    echo "[install.sh] Installing Python 3.11 via uv …"
    uv python install 3.11
fi

# ─────────────────────────────────────────────────────────────
# Isaac Lab submodule
# ─────────────────────────────────────────────────────────────
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/.." && pwd)"
isaaclab_dir="${script_dir}/IsaacLab"
isaaclab_ref="v2.3.0"

git -C "${repo_root}" submodule sync --recursive
git -C "${repo_root}" submodule update --init --recursive Training/IsaacLab

# Isaac Sim 5.1 resolves cleanly with Isaac Lab v2.3.0; newer Isaac Lab refs
# pull in a Starlette pin that conflicts with Isaac Sim's FastAPI stack.
git -C "${isaaclab_dir}" fetch --tags origin release/2.3.0 2>/dev/null || true
git -C "${isaaclab_dir}" checkout --detach "${isaaclab_ref}"

# ─────────────────────────────────────────────────────────────
# Python dependencies
# ─────────────────────────────────────────────────────────────
cd "${script_dir}"
uv venv --python 3.11
uv sync

# Isaac Sim pulls in opencv-python-headless (no imshow support).
# Force-replace it with the GUI-capable build; GTK3 is already present on
# Ubuntu 24.04.  We pin numpy to 1.x because Isaac Sim's compiled extensions
# ship with 1.26 and the ABI isn't guaranteed across the 1→2 boundary.
uv pip install --force-reinstall "opencv-python" "numpy<2.0.0"

# setuptools 82 removed pkg_resources which TensorBoard 2.x still imports.
# Pin to the last version that ships it (matches the build-system constraint).
uv pip install "setuptools<82.0.0"

# ─────────────────────────────────────────────────────────────
# Helper scripts in ~/.local/bin
# ─────────────────────────────────────────────────────────────
# On dual-GPU systems where displays are driven by the iGPU, Vulkan defaults
# to the Intel device.  Setting __NV_PRIME_RENDER_OFFLOAD=1 forces Vulkan to
# the NVIDIA GPU for Omniverse Kit.
local_bin="${HOME}/.local/bin"
mkdir -p "${local_bin}"
PRIME_WRAPPER='if nvidia-smi -L 2>/dev/null | grep -qi intel; then export __NV_PRIME_RENDER_OFFLOAD=1; fi'

for name in train play visualize; do
    cat > "${local_bin}/${name}" <<SCRIPT
#!/bin/bash
TRAINING_DIR="${script_dir}"
${PRIME_WRAPPER}
cd "\$TRAINING_DIR"
exec uv run python ${name}.py "\$@"
SCRIPT
    chmod +x "${local_bin}/${name}"
done

echo ""
echo "=============================="
echo "  Setup complete!"
echo "  - Virtual env:  .venv/"
echo "  - Isaac Lab:    IsaacLab/ @ v2.3.0"
echo "  - Isaac Sim:    5.1.0 (pip: nvidia index)"
echo ""
echo "  Commands:"
echo "    train --task ConeTrack --num_envs 10 --headless"
echo "    play  --task ConeTrack --num_envs 1"
echo "    visualize --task ConeTrack"
echo "=============================="
