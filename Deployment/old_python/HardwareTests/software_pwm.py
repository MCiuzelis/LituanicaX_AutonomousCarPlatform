import time
import RPi.GPIO as GPIO

PIN = 13
FREQ = 50

GPIO.setmode(GPIO.BCM)
GPIO.setup(PIN, GPIO.OUT)

pwm = GPIO.PWM(PIN, FREQ)
pwm.start(0)

try:
    while True:
        for duty in range(0, 101, 5):
            pwm.ChangeDutyCycle(duty)
            print(f"Duty: {duty:3d}%", end="\r")
            time.sleep(0.1)
        for duty in range(100, -1, -5):
            pwm.ChangeDutyCycle(duty)
            print(f"Duty: {duty:3d}%", end="\r")
            time.sleep(0.1)
except KeyboardInterrupt:
    print()

finally:
    pwm.stop()
    GPIO.cleanup()