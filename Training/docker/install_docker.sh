#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────────────
# LituanicaX – Isaac Sim / Isaac Lab training environment bootstrap
#
# Prerequisites (container / bare metal):
#   • Isaac Sim 5.1.0 installed at /isaac-sim  (binary install, NOT pip)
#   • uv installed (`pip install uv` or system package manager)
#   • git, cmake, build-essential
#
# What it does:
#   1. Initialises / repairs the IsaacLab git submodule (Training/IsaacLab)
#      and pins it to v2.3.0 — the release that matches Isaac Sim 5.1.
#   2. Creates a _isaac_sim symlink inside the submodule so that
#      `isaaclab.sh` (and its --install / -p flags) resolve correctly.
#   3. Installs all Isaac Lab Python packages (isaaclab, isaaclab_tasks,
#      isaaclab_assets, isaaclab_rl) as editable installs in Isaac Sim's
#      bundled Python — source edits are reflected without reinstall.
#   4. Pins rsl-rl-lib to 3.0.1 (the minimum version the PPO config targets).
#   5. Installs the project's `cone_track` package (source/cone_track) as editable.
#   6. Runs `uv sync` for local development tooling (ruff, mypy, …).
#   7. Replaces opencv-python-headless → opencv-python (GUI imshow),
#      pins setuptools < 82 (TensorBoard still needs pkg_resources).
# ──────────────────────────────────────────────────────────────────────────────

set -euo pipefail

# Prerequisite check
if [ ! -d /isaac-sim ]; then
    echo "[ERROR] Isaac Sim not found at /isaac-sim."
    echo "  Please install Isaac Sim 5.1.0 at /isaac-sim before running this script."
    echo "  See: https://docs.omniverse.nvidia.com/isaacsim/latest/installation.html"
    exit 1
fi

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
repo_root="$(cd "${script_dir}/.." && pwd)"
isaaclab_dir="${script_dir}/IsaacLab"

# ─── 1. Isaac Lab submodule (v2.3.0) ──────────────────────────────────────────

echo "=== 1/6 – Initialising Isaac Lab submodule (v2.3.0) ==="
git -C "${repo_root}" submodule sync Training/IsaacLab
if ! git -C "${repo_root}" submodule update --init Training/IsaacLab 2>/dev/null; then
    echo "[WARN] Submodule commit not found in the upstream repo. Re-initialising …"
    git -C "${repo_root}" submodule deinit -f Training/IsaacLab 2>/dev/null || true
    rm -rf "${isaaclab_dir}"
    git -C "${repo_root}" submodule update --init Training/IsaacLab
fi

git -C "${isaaclab_dir}" fetch --tags origin
git -C "${isaaclab_dir}" checkout v2.3.0

# ─── 2. _isaac_sim symlink (so isaaclab.sh can find Isaac Sim) ──────────────

echo "=== 2/6 – Symlinking Isaac Sim ==="
ln -sf /isaac-sim "${isaaclab_dir}/_isaac_sim"

# ─── 3. Isaac Lab packages (editable) in Isaac Sim's Python ─────────────────

echo "=== 3/6 – Installing Isaac Lab packages ==="
source /isaac-sim/setup_python_env.sh
"${isaaclab_dir}/isaaclab.sh" --install

# ─── 4. Pin rsl-rl-lib 3.0.1 ──────────────────────────────────────────────────
# The --install step above pulls in the latest rsl-rl via isaaclab_rl[all]; we
# need the exact version whose PPO config (actor_hidden_dims / critic_hidden_dims)
# is compatible with this project.

echo "=== 4/6 – Pinning rsl-rl-lib ==="
"${isaaclab_dir}/isaaclab.sh" -p -m pip install --quiet 'rsl-rl-lib==3.0.1'

# ─── 5. Custom tasks package ──────────────────────────────────────────────────

echo "=== 5/6 – Installing the cone_track package ==="
"${isaaclab_dir}/isaaclab.sh" -p -m pip install --quiet --no-deps -e "${script_dir}"

# ─── 6. Local tooling (uv) & patches ──────────────────────────────────────────

echo "=== 6/6 – Installing local development dependencies ==="
cd "${script_dir}"
uv sync

# Isaac Sim bundles opencv-python-headless (no imshow).
uv pip install --quiet --force-reinstall opencv-python

# setuptools ≥82 removed pkg_resources; TensorBoard still needs it.
uv pip install --quiet "setuptools<82.0.0"

# ──────────────────────────────────────────────────────────────────────────────
echo ""
echo "✔ Installation complete."
echo ""
echo "  Run training (headless):"
echo "    ${isaaclab_dir}/isaaclab.sh -p ${script_dir}/source/cone_track/train.py \\"
echo "      --task ConeTrack --headless --num_envs 10"
echo ""
echo "  Run training (with viewer, first ensure X11 / remote desktop):"
echo "    ${isaaclab_dir}/isaaclab.sh -p ${script_dir}/source/cone_track/train.py \\"
echo "      --task ConeTrack --num_envs 10"
echo ""