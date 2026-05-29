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

if [ ! -f "${ROS_SETUP}" ]; then
    echo "ROS underlay not found at ${ROS_SETUP}"
    echo "Set ROS_UNDERLAY=/path/to/ros2_jazzy or build ROS 2 first."
    exit 1
fi

if [ -d "${TOP_LEVEL_PACKAGE_SRC}" ]; then
    PACKAGE_SRC="${TOP_LEVEL_PACKAGE_SRC}"
else
    PACKAGE_SRC="${PACKAGE_DST}"
fi

if [ ! -d "${PACKAGE_SRC}" ]; then
    echo "Package source not found at ${TOP_LEVEL_PACKAGE_SRC} or ${PACKAGE_DST}"
    exit 1
fi

echo "Using ROS underlay: ${ROS_UNDERLAY}"

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

echo "Installing Python runtime packages into the venv..."
uv pip install --python "${VENV_DIR}/bin/python" \
    ArducamDepthCamera \
    aiohttp \
    opencv-python \
    "numpy<2.0.0" \
    onnxruntime

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

echo
echo "Done."
echo "Use the deployment workspace with:"
echo "  source ${ROS_SETUP}"
echo "  source ${WORKSPACE_DIR}/install/setup.bash"
echo "  ros2 launch lituanicax_rc_car ros_autopilot.launch.py"
