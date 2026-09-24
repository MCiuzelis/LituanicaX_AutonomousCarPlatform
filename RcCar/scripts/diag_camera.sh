#!/usr/bin/env bash
# Camera detection diagnostics.
echo "===== date ====="; date -Is
echo "===== user / groups ====="; id
echo "===== rpicam-hello --list-cameras ====="
rpicam-hello --list-cameras 2>&1 || true
echo "===== /dev/v4l-subdev* ====="; ls -la /dev/v4l-subdev* 2>/dev/null || true
echo "===== /dev/media* ====="; ls -la /dev/media* 2>/dev/null || true
echo "===== i2c adapters ====="; ls -la /sys/bus/i2c/devices/ 2>/dev/null || true
echo "===== dmesg camera lines ====="
sudo dmesg | grep -iE 'unicam|pisp|imx7|dwc2|csi|mipi|ad5398|iw709|dw97|cam[0-9]' | tail -80 || true
echo "===== chosen/overlays ====="
sudo hexdump -v -e '1/1 "%02x"' /proc/device-tree/chosen/overlays 2>/dev/null | xxd -r -p 2>/dev/null; echo
echo "===== config.txt camera lines ====="
grep -nE 'camera_auto_detect|dtoverlay' /boot/firmware/config.txt || true