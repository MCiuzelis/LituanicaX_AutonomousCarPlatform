#!/usr/bin/env bash
# One-time setup of Sim/: Python 3.11 venv with Isaac Sim 5.1 + Isaac Lab 2.3 (pip wheels) via uv.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

if ! nvidia-smi &>/dev/null; then
    echo "NVIDIA driver not running - Isaac Sim needs an RTX GPU with driver 550-580." >&2; exit 1
fi
major="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1 | cut -d. -f1)"
if [[ "$major" -ge 590 ]]; then
    echo "WARNING: driver ${major}.x - Isaac Sim 5.1 crashes on >= 590 (use 550-580)." >&2
fi
command -v uv >/dev/null || { echo "install uv first: wget -qO- https://astral.sh/uv/install.sh | sh" >&2; exit 1; }

uv python install 3.11
uv sync
# Isaac Sim pulls in the headless OpenCV; the viewer window needs the GUI build. numpy stays 1.x
# (Isaac Sim's compiled extensions are built against it).
uv pip install --force-reinstall "opencv-python" "numpy<2.0.0"
echo "done - try: OMNI_KIT_ACCEPT_EULA=YES uv run python source/rc_sim/run.py --algo manual --real-time"
