#!/usr/bin/env bash
# SPI0 has two devices: the MCP3202 ADC on CE0 (kernel IIO driver, SPI mode 0 by default) and the
# BNO085 IMU on CE1 (spidev, mode 3 only). Alternating between them flips the clock idle level
# at the start of a transfer, which adds a clock edge: the next transfer is shifted by one bit
# (garbled BNO085 packets, and potentially bad ADC readings). The MCP3202 also supports mode 1,1,
# so this overlay puts it in mode 3 as well. It also lets the gpio group use realtime priority
# (the BNO085 driver runs its reader with SCHED_FIFO when allowed). Reboot afterwards.
set -e
CFG=/boot/firmware/config.txt
DTS=$(mktemp --suffix=.dts)
cat > "$DTS" <<'DTS'
/dts-v1/;
/plugin/;
/ {
	compatible = "brcm,bcm2712";
	fragment@0 {
		target-path = "/axi/pcie@1000120000/rp1/spi@50000/mcp3202@0";
		__overlay__ {
			spi-cpol;
			spi-cpha;
		};
	};
};
DTS
sudo dtc -@ -q -I dts -O dtb -o /boot/firmware/overlays/mcp3202-mode3.dtbo "$DTS"
rm -f "$DTS"
if ! grep -q '^dtoverlay=mcp3202-mode3' "$CFG"; then
  sudo cp "$CFG" "$CFG.bak.mode3"
  # must come after the mcp3202 overlay that creates the node
  sudo sed -i '/^dtoverlay=mcp3202,spi0-0-present/a # Added by scripts/setup_spi0_mode3.sh: MCP3202 in SPI mode 3 like the BNO085 on CE1\ndtoverlay=mcp3202-mode3' "$CFG"
fi
echo '@gpio - rtprio 50' | sudo tee /etc/security/limits.d/90-bno085-rtprio.conf >/dev/null
echo "--- spi/mcp3202 lines now ---"
grep -nE 'spi|mcp3202' "$CFG"
echo "done: reboot, then check: cat /proc/device-tree/axi/pcie@1000120000/rp1/spi@50000/mcp3202@0/spi-cpol (exists) and ulimit -r (50)"
