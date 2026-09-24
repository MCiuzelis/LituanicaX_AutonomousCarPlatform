#!/usr/bin/env bash
# Apply forced IMX708 overlay for Arducam 12MP IMX708 (no EEPROM -> not auto-detected on Pi 5).
# Usage: scripts/setup_camera_overlay.sh [cam0|cam1]   (reboot afterwards)
set -euo pipefail

PORT="${1:-cam0}"
CONFIG=/boot/firmware/config.txt

if [ "$PORT" != "cam0" ] && [ "$PORT" != "cam1" ]; then
  echo "Invalid port: $PORT (use cam0 or cam1)" >&2
  exit 1
fi

BACKUP="${CONFIG}.bak.$(date +%Y%m%d%H%M%S)"
sudo cp "$CONFIG" "$BACKUP"
echo "Backup saved: $BACKUP"

sudo sed -i 's/^camera_auto_detect=.*/camera_auto_detect=0/' "$CONFIG"
if ! grep -q '^dtoverlay=imx708' "$CONFIG"; then
  printf '\n# Added by rc_car (scripts/setup_camera_overlay.sh): Arducam 12MP IMX708 (no EEPROM -> force overlay)\ndtoverlay=imx708,%s\n' "$PORT" | sudo tee -a "$CONFIG" >/dev/null
else
  sudo sed -i "s/^dtoverlay=imx708.*/dtoverlay=imx708,$PORT/" "$CONFIG"
fi

echo "--- camera section of config.txt ---"
grep -nE 'camera_auto_detect|dtoverlay=imx708' "$CONFIG"
sync
echo "--- done; reboot required ---"