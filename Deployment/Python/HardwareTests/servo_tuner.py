import time
import sys
import termios
import tty
import os
import RPi.GPIO as GPIO

PIN = 13
FREQ = 50
SERVO_MIN_US = 500
SERVO_MAX_US = 2500
SERVO_CENTER_US = 1500

GPIO.setmode(GPIO.BCM)
GPIO.setup(PIN, GPIO.OUT)
servo_pwm = GPIO.PWM(PIN, FREQ)
servo_pwm.start(0)

def set_servo(value):
    value = max(-1.0, min(1.0, value))
    span = (SERVO_MAX_US - SERVO_MIN_US) / 2.0
    pulse_us = SERVO_CENTER_US + offset_us + (value * amplitude * span)
    pulse_us = max(SERVO_MIN_US, min(SERVO_MAX_US, pulse_us))
    duty = (pulse_us / 20000.0) * 100.0
    servo_pwm.ChangeDutyCycle(duty)

def apply_servo(value):
    set_servo(value)

def sweep_test():
    print("Running sweep test...")
    apply_servo(-1)
    time.sleep(1)
    apply_servo(1)
    time.sleep(1)
    apply_servo(0)
    time.sleep(1)

offset_us = 0
amplitude = 0.5
OFFSET_STEP = 10
AMP_STEP = 0.05

def main():
    global offset_us, amplitude

    print("Servo tuning controls:")
    print("A/D: center offset | W/S: amplitude | T: test | C: center | Q: quit")

    old_settings = termios.tcgetattr(sys.stdin)
    tty.setcbreak(sys.stdin.fileno())

    try:
        apply_servo(0)

        while True:
            import select
            dr, _, _ = select.select([sys.stdin], [], [], 0)
            key = sys.stdin.read(1) if dr else None

            if key:
                if key == 'a':
                    offset_us -= OFFSET_STEP
                elif key == 'd':
                    offset_us += OFFSET_STEP
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

                amplitude = max(0.0, min(2.0, amplitude))

                print(f"\rOffset: {offset_us} us | Amplitude: {amplitude:.2f}   ", end="")

            time.sleep(0.02)

    finally:
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)
        servo_pwm.stop()
        del servo_pwm
        GPIO.cleanup()
        print("\nStopped.")

if __name__ == "__main__":
    main()
