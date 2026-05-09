import time, math, sys, os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from hardware import Robot

robot = Robot(max_throttle=1.0)

try:
    print("Running sine wave test. Press Ctrl+C to exit.")
    t = 0.0
    while True:
        motor_val = math.sin(t)

        robot.set_throttle(motor_val)
        print(f"Motor: {motor_val:>5.1f} | Throttle: {motor_val*100:>5.1f}%", end="\r")

        t += 0.05
        time.sleep(0.05)

except KeyboardInterrupt:
    print("\nInterrupt received — stopping motor immediately!")
finally:
    robot.stop()