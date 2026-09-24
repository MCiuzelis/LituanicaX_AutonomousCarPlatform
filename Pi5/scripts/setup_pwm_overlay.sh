#!/usr/bin/env bash
# Enable RP1 PWM0 on GPIO12 (PWM0_CHAN0) + GPIO13 (PWM0_CHAN1) persistently.
# Without this overlay only PWM1 (the fan block) exists; the header pins need PWM0.
# hardware/pwm.py loads the same overlay at runtime via `dtoverlay` if it is missing.
set -euo pipefail
CONFIG=/boot/firmware/config.txt
LINE='dtoverlay=pwm-2chan,pin=12,func=4,pin2=13,func2=4'

if grep -q "^${LINE}$" "$CONFIG"; then
  echo "already present: $LINE"
else
  BACKUP="${CONFIG}.bak.$(date +%Y%m%d%H%M%S)"
  sudo cp "$CONFIG" "$BACKUP"
  echo "Backup saved: $BACKUP"
  printf '\n# Added by rc_car (scripts/setup_pwm_overlay.sh): hardware PWM for RC servos on GPIO12/13 (RP1 PWM0)\n%s\n' "$LINE" | sudo tee -a "$CONFIG" >/dev/null
  echo "added: $LINE"
fi
# apply now as well (no reboot needed) if PWM0 is not already up
if ! ls -d /sys/devices/platform/1f00098000.pwm >/dev/null 2>&1; then
  sudo dtoverlay pwm-2chan,pin=12,func=4,pin2=13,func2=4 && echo "overlay loaded at runtime"
fi
grep -nE 'pwm' "$CONFIG"
for c in /sys/class/pwm/pwmchip*; do echo "$c -> $(readlink -f "$c/device")"; done
