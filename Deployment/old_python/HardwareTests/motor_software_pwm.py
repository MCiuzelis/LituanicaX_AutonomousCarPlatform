import time
import RPi.GPIO as GPIO

PIN_MOTOR_PWM = 12
PIN_INA = 24
PIN_INB = 23

FREQ = 1000

GPIO.setmode(GPIO.BCM)
GPIO.setup(PIN_MOTOR_PWM, GPIO.OUT)
GPIO.setup(PIN_INA, GPIO.OUT)
GPIO.setup(PIN_INB, GPIO.OUT)

pwm = GPIO.PWM(PIN_MOTOR_PWM, FREQ)
pwm.start(0)

def set_motor(speed):
    speed = max(-100.0, min(100.0, speed))
    if speed >= 0:
        GPIO.output(PIN_INA, GPIO.HIGH)
        GPIO.output(PIN_INB, GPIO.LOW)
    else:
        GPIO.output(PIN_INA, GPIO.LOW)
        GPIO.output(PIN_INB, GPIO.HIGH)
    pwm.ChangeDutyCycle(abs(speed))

try:
    print("Software PWM motor test. Press Ctrl+C to exit.\n")

    print("Forward ramp 0 → 100%")
    for duty in range(0, 101, 5):
        set_motor(duty)
        print(f"  Forward {duty:3d}%", end="\r")
        time.sleep(0.15)
    print()

    print("Coast (0%)")
    set_motor(0)
    time.sleep(1)

    print("Reverse ramp 0 → 100%")
    for duty in range(0, 101, 5):
        set_motor(-duty)
        print(f"  Reverse {duty:3d}%", end="\r")
        time.sleep(0.15)
    print()

    print("Coast (0%)")
    set_motor(0)
    time.sleep(1)

    print("Sine sweep at 0.5 Hz")
    t = 0.0
    while True:
        import math
        val = math.sin(2 * math.pi * 0.5 * t) * 100
        set_motor(val)
        print(f"Motor: {val:>6.1f}%", end="\r")
        t += 0.02
        time.sleep(0.02)

except KeyboardInterrupt:
    print("\nStopping motor...")
finally:
    set_motor(0)
    pwm.stop()
    GPIO.cleanup()
