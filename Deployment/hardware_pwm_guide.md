# Raspberry Pi 5 Native Hardware PWM on GPIO12/GPIO13

This document explains how to unlock native hardware-level PWM on GPIO12 and GPIO13 of a Raspberry Pi 5 running Pi OS (kernel 6.12+). Covers why standard approaches fail, how the Pi 5's RP1 southbridge behaves, and how to configure the system.

## The Problem

The Raspberry Pi 5 offloads I/O to an **RP1** southbridge (PCIe-connected to the BCM2712 SoC). The RP1 has two hardware PWM blocks:

| Block      | DTB Node      | Target Pins          | Default  |
|------------|---------------|----------------------|----------|
| **PWM0**   | `pwm@98000`   | GPIO12 & GPIO13      | Disabled |
| **PWM1**   | `pwm@9c000`   | GPIO45               | Enabled  |

PWM0 is **disabled** by default in the base device tree. If `dtparam=audio=on` is set, the audio driver additionally hijacks `chip0`, blocking userspace access. Software PWM wrappers or simple pin-mux tweaks will not work — signals drop or fall back to jittery CPU bit-banging.

## The Solution

Use the kernel's `pwm-2chan` device tree overlay. Passing explicit pin/function parameters lets the kernel handle clock gating, claim the pins, and set the correct alternate function (`alt0`). Then drive the PWM via the Linux sysfs interface (`/sys/class/pwm/pwmchip0/`).

## RPi.GPIO Interaction

Once the overlay claims these pins, they belong to the RP1 hardware timers. If an `RPi.GPIO` script calls `GPIO.cleanup()`, the pin mux resets to GPIO mode.

**Fix:** call `pinctrl set <pin> a0 pd` at startup to restore hardware routing.

---

## Step-by-Step Setup

### 1. Configure `/boot/firmware/config.txt`

```bash
sudo nano /boot/firmware/config.txt
```

- **Disable analogue audio:** find `dtparam=audio=on` and change it to `off` or comment it out:
  ```
  #dtparam=audio=on
  ```
- **Enable PWM0 via overlay:** add at the end of the file:
  ```
  dtoverlay=pwm-2chan,pin=12,func=4,pin2=13,func2=4
  ```

Save (`Ctrl+O`, `Enter`, `Ctrl+X`) and reboot.

### 2. Verify

```bash
# Confirm the sysfs node is live
ls -l /sys/class/pwm/pwmchip0

# Verify pins are in Alternate Function 0 (PWM0_CH0 / PWM0_CH1)
pinctrl get 12,13
```

Expected output:
```
12: a0    pd | lo   # GPIO12 = PWM0_CHAN0
13: a0    pd | lo   # GPIO13 = PWM0_CHAN1
```

### 3. Python Usage

No third-party libraries needed. The script below forces the pinmux, opens channels, and sweeps a sine wave on both pins:

```python
import time
import math
import os
import subprocess

PWM_CHIP_PATH = "/sys/class/pwm/pwmchip0"
PWM_FREQUENCY = 50
PERIOD_NS = int(1_000_000_000 / PWM_FREQUENCY)

SERVO_MIN_NS = 500_000
SERVO_MAX_NS = 2_500_000
SWEEP_SPEED_HZ = 1.5

def setup_channel(channel, pin):
    subprocess.run(["pinctrl", "set", str(pin), "a0", "pd"], check=True)
    path = f"{PWM_CHIP_PATH}/pwm{channel}"
    if not os.path.exists(path):
        with open(f"{PWM_CHIP_PATH}/export", "w") as f:
            f.write(str(channel))
        time.sleep(0.1)
    with open(f"{path}/enable", "w") as f:
        f.write("0")
    with open(f"{path}/period", "w") as f:
        f.write(str(PERIOD_NS))
    with open(f"{path}/duty_cycle", "w") as f:
        f.write(str(SERVO_MIN_NS))
    with open(f"{path}/enable", "w") as f:
        f.write("1")
    return path

try:
    ch12_path = setup_channel(channel=0, pin=12)
    ch13_path = setup_channel(channel=1, pin=13)
    print(f"Native Hardware PWM active on GPIO 12 & 13. Speed: {SWEEP_SPEED_HZ}Hz.")
except (FileNotFoundError, subprocess.CalledProcessError):
    print("Error: ensure audio is disabled and config.txt is updated.")
    exit(1)

try:
    start_time = time.time()
    while True:
        sine_val = math.sin(2 * math.pi * SWEEP_SPEED_HZ * (time.time() - start_time))
        pulse_ns = int(SERVO_MIN_NS + ((sine_val + 1.0) / 2.0) * (SERVO_MAX_NS - SERVO_MIN_NS))
        pulse_ns = max(SERVO_MIN_NS, min(SERVO_MAX_NS, pulse_ns))
        with open(f"{ch12_path}/duty_cycle", "w") as f:
            f.write(str(pulse_ns))
        with open(f"{ch13_path}/duty_cycle", "w") as f:
            f.write(str(pulse_ns))
        print(f"Pulse Width: {pulse_ns:,} ns | Sine: {sine_val:+.3f}", end="\r")
        time.sleep(0.01)
except KeyboardInterrupt:
    print("\nExiting...")
finally:
    for path in (ch12_path, ch13_path):
        try:
            with open(f"{path}/enable", "w") as f:
                f.write("0")
        except:
            pass
    print("Hardware channels deactivated.")
```

## Troubleshooting

| Symptom | Fix |
|---------|-----|
| `Permission denied` or `FileNotFoundError` on sysfs | Verify `config.txt`, reboot, ensure your user is in the `gpio` group |
| Script runs but servo/motor is still | `RPi.GPIO.cleanup()` stripped the pinmux — the embedded `pinctrl set` call handles this automatically. Manual fix: `pinctrl set 12 a0 pd && pinctrl set 13 a0 pd` |

## Architecture

```
┌──────────────────────────────────────────────────┐
│                  Raspberry Pi 5                   │
│  BCM2712 SoC ─── PCIe ─── RP1 I/O Controller     │
│                               │                   │
│                     ┌─────────┴──────────┐        │
│                     │  pwm@98000 (PWM0)  │ (unlocked via overlay)
│                     └────────────────────┘        │
│                               │                   │
│                ┌──────────────┴──────────────┐    │
│                ▼                             ▼    │
│      ┌──────────────────┐      ┌────────────────┐ │
│      │ PWM0 Channel 0   │      │ PWM0 Channel 1 │ │
│      │ (pwmchip0/pwm0)  │      │ (pwmchip0/pwm1)│ │
│      └──────────────────┘      └────────────────┘ │
│                │                             │    │
│                ▼                             ▼    │
│      ┌──────────────────┐      ┌────────────────┐ │
│      │ GPIO12 Out       │      │ GPIO13 Out     │ │
│      │ (Pure Hardware)  │      │ (Pure Hardware)│ │
│      └──────────────────┘      └────────────────┘ │
└──────────────────────────────────────────────────┘
```

## Files Referenced

| Path | Purpose |
|------|---------|
| `/boot/firmware/config.txt` | Disable audio, enable `pwm-2chan` overlay |
| `/sys/class/pwm/pwmchip0/pwm0` | Sysfs control for GPIO12 (Channel 0) |
| `/sys/class/pwm/pwmchip0/pwm1` | Sysfs control for GPIO13 (Channel 1) |
