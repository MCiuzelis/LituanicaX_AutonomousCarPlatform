#!/usr/bin/env bash
# i2cdetect scan of the camera (bus 11) and GPIO header (13/14) I2C buses.
SCAN=/usr/sbin/i2cdetect
for b in 11 13 14; do
  echo "-- bus $b --"
  if [ -e "/dev/i2c-$b" ]; then
    sudo "$SCAN" -y "$b" 2>&1 | head -18
  else
    echo "(no /dev/i2c-$b)"
  fi
done