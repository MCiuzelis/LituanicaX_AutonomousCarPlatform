import time
import sys
import termios
import tty
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from hardware import Robot

# -------- KEYBOARD INPUT (non-blocking) --------
def get_key():
    import select
    dr, _, _ = select.select([sys.stdin], [], [], 0)
    if dr:
        return sys.stdin.read(1)
    return None

# -------- SETUP --------
robot = Robot(
    servo_amplitude=0.5,
    servo_direction=1
)

# Tuning parameters
center_offset = 0      # in microseconds
amplitude = 0.5

OFFSET_STEP = 10       # µs per press
AMP_STEP = 0.05


def sweep_test():
    print("Running sweep test...")

    apply_servo(-1)
    time.sleep(1)

    apply_servo(1)
    time.sleep(1)

    apply_servo(0)
    time.sleep(1)


# -------- MAIN LOOP --------
def main():
    global center_offset, amplitude

    print("Servo tuning controls:")
    print("A/D: center offset | W/S: amplitude | T: test | C: center | Q: quit")

    # Setup terminal
    old_settings = termios.tcgetattr(sys.stdin)
    tty.setcbreak(sys.stdin.fileno())

    try:
        apply_servo(0)

        while True:
            key = get_key()

            if key:
                if key == 'a':
                    center_offset -= OFFSET_STEP
                elif key == 'd':
                    center_offset += OFFSET_STEP
                elif key == 'w':
                    amplitude += AMP_STEP
                elif key == 's':
                    amplitude -= AMP_STEP
                elif key == 't':
                    sweep_test()
                elif key == 'c':
                    apply_servo(0)
                elif key == 'q':
                    break

                # Clamp amplitude
                amplitude = max(0.0, min(2.0, amplitude))

                print(f"\rOffset: {center_offset} µs | Amplitude: {amplitude:.2f}   ", end="")

            time.sleep(0.02)

    finally:
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)

        # ✅ Stop via Robot (handles PWM + cleanup)
        robot.stop()

        print("\nStopped.")


if __name__ == "__main__":
    main()