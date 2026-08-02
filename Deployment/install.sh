#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
ROS_UNDERLAY="${ROS_UNDERLAY:-${HOME}/ros2_jazzy}"
ROS_SETUP="${ROS_UNDERLAY}/install/setup.bash"
WORKSPACE_DIR="${SCRIPT_DIR}/Ros2"
TOP_LEVEL_PACKAGE_SRC="${REPO_ROOT}/LituanicaX_RC_Car"
PACKAGE_DST="${WORKSPACE_DIR}/src/LituanicaX_RC_Car"
VENV_DIR="${SCRIPT_DIR}/.venv"

# The ROS workspace is optional: Deployment/Python runs the same stack on the
# venv alone. Only build it when an underlay is actually present, so a
# Python-only Pi can still install. SKIP_ROS=1 forces the Python-only path.
BUILD_ROS=1
if [ "${SKIP_ROS:-0}" = "1" ]; then
    BUILD_ROS=0
    echo "SKIP_ROS=1: installing the Python stack only."
elif [ ! -f "${ROS_SETUP}" ]; then
    BUILD_ROS=0
    echo "No ROS underlay at ${ROS_SETUP} — installing the Python stack only."
    echo "Set ROS_UNDERLAY=/path/to/ros2_jazzy if you also want the ROS workspace."
fi

if [ -d "${TOP_LEVEL_PACKAGE_SRC}" ]; then
    PACKAGE_SRC="${TOP_LEVEL_PACKAGE_SRC}"
else
    PACKAGE_SRC="${PACKAGE_DST}"
fi

if [ "${BUILD_ROS}" = "1" ]; then
    if [ ! -d "${PACKAGE_SRC}" ]; then
        echo "Package source not found at ${TOP_LEVEL_PACKAGE_SRC} or ${PACKAGE_DST}"
        exit 1
    fi
    echo "Using ROS underlay: ${ROS_UNDERLAY}"
fi

if ! command -v uv &>/dev/null; then
    echo "uv not found, installing..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="${HOME}/.local/bin:${PATH}"
fi

echo "Using uv: $(uv --version)"

echo "Installing system dependencies..."
sudo apt update
sudo apt install -y \
    build-essential \
    cmake \
    curl \
    liblgpio-dev \
    libopencv-dev \
    python3-aiohttp \
    python3-dev \
    python3-numpy \
    python3-opencv \
    python3-pip \
    python3-rpi-lgpio \
    python3-spidev \
    rsync \
    swig

if ! dpkg -l | grep -q arducam-tof-sdk-dev; then
    echo "Adding Arducam APT repository..."
    OS_VERSION="$(lsb_release -r | awk '{print $2}' | cut -d. -f1)"
    if [ "${OS_VERSION}" -ge 13 ]; then
        sudo rm -f /etc/apt/sources.list.d/arducam_list_files.list
        sudo curl -s --compressed -o /usr/share/keyrings/arducam-keyring.pgp "https://arducam.github.io/arducam_ppa/KEY.gpg"
        sudo curl -s --compressed -o /etc/apt/sources.list.d/arducam_list_files.sources "https://arducam.github.io/arducam_ppa/arducam_list_files.sources"
    else
        curl -s --compressed "https://arducam.github.io/arducam_ppa/KEY.gpg" | sudo apt-key add -
        sudo curl -s --compressed -o /etc/apt/sources.list.d/arducam_list_files.list "https://arducam.github.io/arducam_ppa/arducam_list_files.list"
    fi
    sudo apt update
fi

echo "Installing Arducam SDK..."
sudo apt install -y \
    arducam-config-parser-dev \
    arducam-evk-sdk-dev \
    arducam-tof-sdk-dev

echo "Creating Python runtime venv at ${VENV_DIR}..."
uv venv "${VENV_DIR}"

# The venv is created without system site-packages, so every import the stack
# makes has to come from this list — the python3-* APT packages above are not
# visible from inside it. rpi-lgpio supplies RPi.GPIO (drive) and pulls in
# lgpio; spidev is the ADC (monitoring); smbus2 is the AS5600 encoder.
echo "Installing Python runtime packages into the venv..."
uv pip install --python "${VENV_DIR}/bin/python" \
    ArducamDepthCamera \
    aiohttp \
    opencv-python \
    "numpy<2.0.0" \
    onnxruntime \
    rpi-lgpio \
    spidev \
    smbus2

echo "Verifying the venv can import everything the Python stack needs..."
"${VENV_DIR}/bin/python" - <<'PYCHECK'
import importlib, sys
missing = []
for name in ("numpy", "cv2", "aiohttp", "onnxruntime", "ArducamDepthCamera",
             "RPi.GPIO", "spidev", "smbus2", "lgpio"):
    try:
        importlib.import_module(name)
    except Exception as exc:
        missing.append(f"{name}: {type(exc).__name__}: {exc}")
if missing:
    print("Venv is incomplete:")
    for line in missing:
        print("  " + line)
    sys.exit(1)
print("All runtime imports OK.")
PYCHECK

if [ "${BUILD_ROS}" = "1" ]; then
    if [ "$(readlink -f "${PACKAGE_SRC}")" != "$(readlink -f "${PACKAGE_DST}")" ]; then
        echo "Synchronizing ROS package into ${PACKAGE_DST}..."
        mkdir -p "${WORKSPACE_DIR}/src"
        rsync -a --delete \
            --exclude "__pycache__/" \
            --exclude "*.egg-info/" \
            "${PACKAGE_SRC}/" \
            "${PACKAGE_DST}/"
    else
        echo "Using existing ROS package at ${PACKAGE_DST}"
    fi

    echo "Building deployment ROS workspace..."
    (
        set +u
        source "${ROS_SETUP}"
        set -u
        cd "${WORKSPACE_DIR}"
        colcon build --symlink-install
    )
fi

echo
echo "Done."
echo "Run the Python stack (no ROS, no sourcing):"
echo "  ${SCRIPT_DIR}/Python/run"

if [ "${BUILD_ROS}" = "1" ]; then
    echo
    echo "Or use the ROS workspace with:"
    echo "  source ${ROS_SETUP}"
    echo "  source ${WORKSPACE_DIR}/install/setup.bash"
    echo "  ros2 launch lituanicax_rc_car ros_autopilot.launch.py"
fi
