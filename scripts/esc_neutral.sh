#!/usr/bin/env bash
# Put the ESC throttle pin (GPIO13 = RP1 PWM0 channel 1) at 50 Hz / 1500 us neutral.
# Run at boot by esc-neutral.service (/etc/systemd/system) so the ESC always sees a valid
# neutral signal, and hardware/esc.py never removes it (keep_on_close).
set -u
GPIO=13; CH=1; PERIOD=20000000; NEUTRAL=1500000
for _ in $(seq 1 50); do
  CHIP=$(for c in /sys/class/pwm/pwmchip*; do [ -e "$c/device" ] && readlink -f "$c/device" | grep -q 1f00098000 && echo "$c"; done | head -1)
  [ -n "$CHIP" ] && break
  sleep 0.2
done
[ -n "${CHIP:-}" ] || { echo "PWM0 chip not found (pwm-2chan overlay loaded?)" >&2; exit 1; }
[ -d "$CHIP/pwm$CH" ] || echo "$CH" > "$CHIP/export"
for _ in $(seq 1 20); do [ -w "$CHIP/pwm$CH/period" ] && break; sleep 0.1; done
P="$CHIP/pwm$CH"
# duty must never exceed period: shrink duty first if a larger period was left behind
cur_duty=$(cat "$P/duty_cycle" 2>/dev/null || echo 0)
[ "$cur_duty" -gt "$PERIOD" ] && echo 0 > "$P/duty_cycle"
echo "$PERIOD"  > "$P/period"
echo "$NEUTRAL" > "$P/duty_cycle"
echo 1          > "$P/enable"
pinctrl set "$GPIO" a0
echo "ESC neutral: $P period=$(cat "$P/period") duty=$(cat "$P/duty_cycle") enable=$(cat "$P/enable") $(pinctrl get "$GPIO")"
