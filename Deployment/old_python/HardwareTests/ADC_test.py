import time
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from hardware import Robot

robot = Robot()

print("=== ADC Telemetry ===")
print("Reading motor current and battery voltage only.")
print("Press CTRL+C to stop.\n")

try:
    while True:
        current = robot.get_motor_current()
        voltage = robot.get_battery_voltage()
        print(f"\rCurrent: {current:.3f}A | Battery: {voltage:.2f}V", end="", flush=True)
        time.sleep(0.05)
except KeyboardInterrupt:
    print("\n\nStopping...")
finally:
    robot.stop()
