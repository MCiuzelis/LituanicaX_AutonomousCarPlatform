#!/usr/bin/env bash
# Switch /boot/firmware/config.txt from the generic spidev overlay to the native MCP3202 IIO
# driver on SPI0 CS0 (dtoverlay=mcp3202,spi0-0-present). Reboot afterwards.
set -e
CFG=/boot/firmware/config.txt
sudo cp "$CFG" "$CFG.bak.spidev0"
sudo sed -i '/^dtoverlay=spi0-2cs$/d' "$CFG"
if ! grep -q '^dtoverlay=mcp3202' "$CFG"; then
  printf '\n# Added by rc_car (scripts/setup_mcp3202_overlay.sh): native MCP3202 IIO driver on SPI0 CS0\ndtoverlay=mcp3202,spi0-0-present\n' | sudo tee -a "$CFG" >/dev/null
fi
echo "--- spi/mcp3202 lines now ---"
grep -nE 'spi|mcp3202' "$CFG"
echo "done"