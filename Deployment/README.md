# Raspberry Pi 5 Hardware PWM on GPIO12/GPIO13

This document explains how to get hardware-level PWM working on GPIO12 and GPIO13
of a Raspberry Pi 5 running Pi OS Lite (kernel 6.12+), what goes wrong, and why
each fix is necessary.

## The Problem

The Pi 5 uses the RP1 I/O controller (connected via PCIe to the BCM2712 SoC).
The RP1 has two hardware PWM blocks:

| Block      | DTB Node      | Kernel Driver | Default Status |
|------------|---------------|---------------|----------------|
| PWM0       | `pwm@98000`   | `rpi-pwm`     | **disabled**   |
| PWM1       | `pwm@9c000`   | `rpi-pwm`     | enabled        |

PWM0 (pwm@98000) is the block whose outputs are physically routed to GPIO12
(channel 0) and GPIO13 (channel 1) via the `alt0 (pwm0)` pin function.
However, the Raspberry Pi foundation **disables PWM0 in the base device tree**
by default, reserving it for other uses. Only PWM1 (pwm@9c000) is enabled,
and its output is routed exclusively to GPIO45.

This means that even if you set GPIO12/13 to `alt0 (pwm0)` via debugfs
pinmux-select, the PWM0 block itself is clock-gated and non-functional — the
pin is connected to a dead peripheral.

## The Solution: PIO-Based PWM (pwm-pio-rp1)

Instead of trying to re-enable the disabled PWM0 block (which would require a
custom device tree overlay), we use the RP1's **PIO (Programmable I/O) state
machines**. PIO is a dedicated microcontroller inside the RP1 that can generate
precise timing waveforms independently of the main CPU. It is the officially
supported mechanism for flexible PWM on arbitrary GPIO pins on Pi 5.

The `pwm-pio-rp1` kernel driver (by Phil Elwell) exposes PIO-based PWM channels
as standard Linux PWM chips via sysfs, compatible with the same `rpi_hardware_pwm`
Python library used for hardware block PWM.

## Side Effect

Once a pin is claimed by `pwm-pio-rp1`, `RPi.GPIO` can no longer access it
("GPIO not allocated" error). This is correct — the pin is owned by the PIO
driver. The `software_pwm.py` script will also fail on these pins for the same
reason.

## Step-by-Step Setup

### 1. Enable the pwm-pio overlays in config.txt

Add these two lines to `/boot/firmware/config.txt`:

```
dtoverlay=pwm-pio,gpio=12
dtoverlay=pwm-pio,gpio=13
```

Each line creates a separate PIO-PWM device node in the device tree. When the
kernel boots, it automatically loads the `pwm_pio_rp1` kernel module for each
node, creating:

- `/sys/class/pwm/pwmchip1` → GPIO13 (device `/pwm_pio@d`)
- `/sys/class/pwm/pwmchip2` → GPIO12 (device `/pwm_pio@c`)

(PWM0 from pwm@9c000 remains at pwmchip0.)

The chip numbers (1, 2) are assigned at probe time and could theoretically swap
between kernel versions, so auto-discovery is recommended (see code below).

### 2. Reset pad control registers at boot

`RPi.GPIO` (and similar libraries) can leave a stale value in the RP1 pad
control registers when they configure a pin as a GPIO output. The pad register
for GPIO12 at physical address `0x1f000d0130` gets set to `0x00000003` when
`GPIO.setup(12, GPIO.OUT)` is called. This value persists across `cleanup()`
and prevents the alternate-function output from driving the pin, even though
the pinmux shows the correct function.

Create `/usr/local/bin/setup-pwm-pins.sh`:

```bash
#!/bin/bash
python3 -c "
import mmap, os, struct
fd = os.open('/dev/mem', os.O_RDWR | os.O_SYNC)
base = 0x1f000d0000 & ~(4096-1)
m = mmap.mmap(fd, 4096*6, offset=base, flags=mmap.MAP_SHARED, prot=mmap.PROT_READ|mmap.PROT_WRITE)
for pin in (12, 13):
    off = 0x100 + pin*4 - (base - 0x1f000d0000)
    m.seek(off)
    val = struct.unpack('<I', m.read(4))[0]
    if val != 0:
        m.seek(off)
        m.write(struct.pack('<I', 0))
m.close()
os.close(fd)
"
```

**Pad register address math:**

The RP1 GPIO pad control registers live at a fixed offset from the RP1 base:

```
RP1 base address:    0x1f00000000
GPIO pad registers:  0x1f000d0000 + 0x100 + pin * 4
```

So GPIO12 is at `0x1f000d0000 + 0x100 + 12*4 = 0x1f000d0130`.

The value `0x00000000` means the pad uses its default electrical configuration
(drive strength, slew rate, pull disable) — which is what the alt-function PWM
output requires. The value `0x00000003` (set by RPi.GPIO) configures the pad
for GPIO output mode with a pull-up, which blocks the PWM alt-function signal.

Create a systemd oneshot service at `/etc/systemd/system/pwm-pins.service`:

```
[Unit]
Description=Reset PWM pad registers on GPIO12/13 for Pi 5
After=local-fs.target

[Service]
Type=oneshot
ExecStart=/usr/local/bin/setup-pwm-pins.sh
RemainAfterExit=yes

[Install]
WantedBy=multi-user.target
```

Enable it:

```bash
sudo systemctl enable pwm-pins.service
```

### 3. Auto-discover pwmchip numbers in your Python code

Since the pwmchip numbers are assigned dynamically, hardcoding them is brittle.
Instead, look up the correct chip by scanning the `OF_FULLNAME` in the device
uevent:

```python
import os

def _find_pwmchip(gpio_pin):
    for entry in os.listdir("/sys/class/pwm"):
        if not entry.startswith("pwmchip"):
            continue
        chip = int(entry[7:])
        try:
            with open(f"/sys/class/pwm/{entry}/device/uevent") as f:
                for line in f:
                    if line.startswith("OF_FULLNAME="):
                        addr = line.strip().split("@")[-1]
                        if int(addr, 16) == gpio_pin:
                            return chip
        except OSError:
            continue
    raise RuntimeError(f"No pwmchip found for GPIO{gpio_pin}")
```

Then use it with `rpi_hardware_pwm`:

```python
from rpi_hardware_pwm import HardwarePWM

motor = HardwarePWM(pwm_channel=0, hz=20000, chip=_find_pwmchip(12))
motor.start(50)
```

### 4. Controlling the motor driver

The motor driver (e.g., L293D or TB6612) requires both a PWM signal for speed
and two digital direction pins. In `PCB_test.py`:

- **PWM (GPIO12):** Speed control via `pwm-pio` driver, 20 kHz
- **INA (GPIO24):** Direction A, digital output via `RPi.GPIO`
- **INB (GPIO23):** Direction B, digital output via `RPi.GPIO`

The `rpi_hardware_pwm` library creates its own PWM export, so there's no need
to manually write to sysfs.

### 5. Do NOT use debugfs pinmux-select

The `pwm-pio` driver handles pinmux configuration internally via the standard
pinctrl API. Manually setting GPIO12/13 to `alt0` via
`/sys/kernel/debug/pinctrl/.../pinmux-select` is unnecessary and would conflict
with the PIO driver. The PIO driver sets the pins to `alt7 (pio)` function.

## Verification

After rebooting, check that the setup is correct:

```bash
# Both pwm-pio devices exist
ls /sys/class/pwm/pwmchip*/device/uevent

# Correct pin function (should show alt7/pio)
grep "pin 1[2-3]" /sys/kernel/debug/pinctrl/*rp1*/pins

# Pad registers are clean
sudo python3 -c "
import mmap, os, struct
fd = os.open('/dev/mem', os.O_RDWR | os.O_SYNC)
base = 0x1f000d0000 & ~(4096-1)
m = mmap.mmap(fd, 4096*6, offset=base, flags=mmap.MAP_SHARED, prot=mmap.PROT_READ)
for pin in (12, 13):
    off = 0x100 + pin*4 - (base - 0x1f000d0000)
    m.seek(off)
    print(f'GPIO{pin} pad: 0x{struct.unpack(\"<I\", m.read(4))[0]:08x}')
m.close()
os.close(fd)
"
```

Both pad registers should read `0x00000000`.

## Troubleshooting

### "GPIO not allocated" from RPi.GPIO

The `pwm-pio` overlay claims the pin. This is expected — you cannot use
`RPi.GPIO` to control a pin that is owned by the PIO driver. Use only
`HardwarePWM` (or direct sysfs writes) for these pins.

### No PWM signal on the pin

1. **Check pad register:** If it shows `0x00000003`, run
   `sudo systemctl restart pwm-pins.service` to reset it.

2. **Check pin function:** Look at `pinctrl-rp1/pins` — it should show
   `alt7 (pio)`, NOT `alt0 (pwm0)` or `gpio`.

3. **Check PWM sysfs:** Verify `enable=1`, `period` and `duty_cycle` have
   sensible values:
   ```
   cat /sys/class/pwm/pwmchip2/pwm0/{enable,period,duty_cycle}
   ```

4. **Wrong pwmchip:** Ensure you're using the correct chip number. Run the
   auto-discovery function above.

### pwm-pio driver not loaded

The driver `pwm_pio_rp1` auto-loads via device-tree modalias matching. If the
overlay is present in `config.txt` but no pwmchip appears, verify the overlay
was applied:

```bash
dmesg | grep -i "pwm.pio\|pwm_pio"
```

Manually loading the module may help diagnose:

```bash
sudo modprobe pwm_pio_rp1
```

### software_pwm.py fails after overlay is added

This is expected. The `pwm-pio` overlay claims GPIO12/13 for the PIO driver,
preventing `RPi.GPIO` from accessing them. If you need software PWM on these
pins, remove the overlays from `config.txt` and reboot.

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────┐
│                    Raspberry Pi 5                        │
│                                                         │
│  BCM2712 SoC ─── PCIe ─── RP1 I/O Controller           │
│                              │                          │
│                    ┌─────────┴──────────┐               │
│                    │  pwm@98000 (PWM0)  │  DISABLED     │
│                    │  (drives GPIO12/13)│               │
│                    └────────────────────┘               │
│                    ┌────────────────────┐               │
│                    │  pwm@9c000 (PWM1)  │  ENABLED      │
│                    │  (drives GPIO45)   │  (not used)   │
│                    └────────────────────┘               │
│                    ┌────────────────────┐               │
│                    │  PIO State Machine │  🔧 THIS IS   │
│                    │  (pwm-pio-rp1)     │  THE FIX      │
│                    │  drives GPIO12/13  │               │
│                    └────────────────────┘               │
│                              │                          │
│                    ┌─────────┴──────────┐               │
│                    │  GPIO12 → Motor PWM│               │
│                    │  GPIO13 → Servo PWM│               │
│                    └────────────────────┘               │
└─────────────────────────────────────────────────────────┘
```

## Files Referenced

| File | Purpose |
|------|---------|
| `/boot/firmware/config.txt` | Boot config; loads `pwm-pio` overlays |
| `/usr/local/bin/setup-pwm-pins.sh` | Boot script; resets pad registers |
| `/etc/systemd/system/pwm-pins.service` | Systemd service running the above |
| `PCB_test.py` | Main test script using auto-discovery |
| `software_pwm.py` | Reference — software PWM (will fail on PIO-claimed pins) |
| `/sys/class/pwm/pwmchip2/pwm0/` | PIO-PWM sysfs interface (GPIO12) |
| `/sys/class/pwm/pwmchip1/pwm0/` | PIO-PWM sysfs interface (GPIO13) |
