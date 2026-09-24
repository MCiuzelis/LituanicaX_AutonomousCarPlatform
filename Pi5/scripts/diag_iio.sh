#!/usr/bin/env bash
# MCP3202 / IIO diagnostics: kernel messages, IIO devices + channels, spidev nodes.
echo "== dmesg mcp320/spi =="
sudo dmesg | grep -iE 'mcp320|spidev|spi0' | tail -12
echo "== iio devices =="
for d in /sys/bus/iio/devices/iio:device*; do
  [ -e "$d" ] || continue
  echo "-- $d --"
  cat "$d/name" 2>/dev/null
  echo "channels:"
  ls "$d" | grep -E '^in_voltage' || true
done
echo "== spidev =="
ls /dev/spidev* 2>/dev/null || echo "(none)"