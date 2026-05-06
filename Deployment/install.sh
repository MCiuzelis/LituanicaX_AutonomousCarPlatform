#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${SCRIPT_DIR}/.venv"

# Install uv if not present
if ! command -v uv &>/dev/null; then
    echo "uv not found, installing..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="${HOME}/.local/bin:${PATH}"
fi

echo "Using uv: $(uv --version)"

# Install system dependencies
echo "Installing system dependencies..."
sudo apt update
sudo apt install -y cmake curl libopencv-dev swig liblgpio-dev

# Add Arducam APT repository if not present
if ! dpkg -l | grep -q arducam-tof-sdk-dev; then
    echo "Adding Arducam APT repository..."
    OS_VERSION=$(lsb_release -r | awk '{print $2}' | cut -d. -f1)
    if [ "$OS_VERSION" -ge 13 ]; then
        sudo rm -f /etc/apt/sources.list.d/arducam_list_files.list
        sudo curl -s --compressed -o /usr/share/keyrings/arducam-keyring.pgp "https://arducam.github.io/arducam_ppa/KEY.gpg"
        sudo curl -s --compressed -o /etc/apt/sources.list.d/arducam_list_files.sources "https://arducam.github.io/arducam_ppa/arducam_list_files.sources"
    else
        curl -s --compressed "https://arducam.github.io/arducam_ppa/KEY.gpg" | sudo apt-key add -
        sudo curl -s --compressed -o /etc/apt/sources.list.d/arducam_list_files.list "https://arducam.github.io/arducam_ppa/arducam_list_files.list"
    fi
    sudo apt update
fi

# Install Arducam SDK packages
echo "Installing Arducam SDK..."
sudo apt install -y arducam-config-parser-dev arducam-evk-sdk-dev arducam-tof-sdk-dev

# Create virtual environment
echo "Creating virtual environment at ${VENV_DIR}..."
uv venv "${VENV_DIR}"

# Install Python packages
echo "Installing Python packages..."
uv pip install --python "${VENV_DIR}/bin/python" \
    ArducamDepthCamera \
    rpi-hardware-pwm \
    opencv-python \
    "numpy<2.0.0" \
    onnxruntime \
    aiohttp \
    spidev \
    rpi-lgpio

echo "Done. Activate with: source ${VENV_DIR}/bin/activate"
