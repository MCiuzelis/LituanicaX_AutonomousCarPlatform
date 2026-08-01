import time
import math
import os
import subprocess

# --- HARDWARE CONFIGURATION ---
PWM_CHIP_PATH = "/sys/class/pwm/pwmchip0"
PWM_FREQUENCY = 50  # 50 Hz standard for RC servos
PERIOD_NS = int(1_000_000_000 / PWM_FREQUENCY)  # 20,000,000 ns

# --- SERVO TUNING PARAMETERS ---
# 1. AMPLITUDE: Expanded from [1ms - 2ms] to [0.5ms - 2.5ms] for maximum physical rotation
SERVO_MIN_NS = 500_000    # 0.5 ms pulse (Full Left / 0 degrees)
SERVO_MAX_NS = 2_500_000  # 2.5 ms pulse (Full Right / 180-270 degrees)

# 2. SPEED: Increased from 0.2 Hz (5 seconds per cycle) to 1.5 Hz (1.5 cycles per second)
# Scale this up or down to change the velocity!
SWEEP_SPEED_HZ = 1.5 

def setup_channel(channel, pin):
    subprocess.run(["pinctrl", "set", str(pin), "a0", "pd"], check=True)
    path = f"{PWM_CHIP_PATH}/pwm{channel}"
    if not os.path.exists(path):
        with open(f"{PWM_CHIP_PATH}/export", "w") as f: f.write(str(channel))
        time.sleep(0.1)
    
    with open(f"{path}/enable", "w") as f: f.write("0")
    with open(f"{path}/period", "w") as f: f.write(str(PERIOD_NS))
    with open(f"{path}/duty_cycle", "w") as f: f.write(str(SERVO_MIN_NS))
    with open(f"{path}/enable", "w") as f: f.write("1")
    return path

# Initialize channels (GPIO 12 -> Ch 0, GPIO 13 -> Ch 1)
try:
    ch12_path = setup_channel(channel=0, pin=12)
    ch13_path = setup_channel(channel=1, pin=13)
    print(f"Hardware PWM active on GPIO 12 & 13 at {SWEEP_SPEED_HZ}Hz. Press Ctrl+C to stop.")
except (FileNotFoundError, subprocess.CalledProcessError):
    print("Error: Ensure audio is disabled and your overlay is set in config.txt")
    exit(1)

try:
    start_time = time.time()
    while True:
        # Generate sine value (-1.0 to +1.0) multiplied by our new speed target
        sine_val = math.sin(2 * math.pi * SWEEP_SPEED_HZ * (time.time() - start_time))
        
        # Map to our new wide amplitude pulse range
        pulse_ns = int(SERVO_MIN_NS + ((sine_val + 1.0) / 2.0) * (SERVO_MAX_NS - SERVO_MIN_NS))
        pulse_ns = max(SERVO_MIN_NS, min(SERVO_MAX_NS, pulse_ns))
        
        # Write simultaneously to the RP1 hardware registers
        with open(f"{ch12_path}/duty_cycle", "w") as f: f.write(str(pulse_ns))
        with open(f"{ch13_path}/duty_cycle", "w") as f: f.write(str(pulse_ns))
        
        print(f"Pulse Width: {pulse_ns:,} ns | Sine: {sine_val:+.3f}", end="\r")
        time.sleep(0.01)  # Dropped loop sleep to 10ms so the target updates keep pace with high speeds

except KeyboardInterrupt:
    print("\nStopping PWM channels...")
finally:
    for path in (ch12_path, ch13_path):
        try:
            with open(f"{path}/enable", "w") as f: f.write("0")
        except: pass