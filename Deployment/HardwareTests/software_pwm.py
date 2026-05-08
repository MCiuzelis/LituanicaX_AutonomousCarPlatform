import time
import math
import RPi.GPIO as GPIO

# --- PIN CONFIGURATION ---
PIN_INA = 24
PIN_INB = 23
PIN_PWM = 14   # GPIO12

# --- PWM SETTINGS ---
PWM_FREQUENCY = 1000  # Software PWM frequency in Hz

# --- INITIALIZATION ---
GPIO.setmode(GPIO.BCM)

GPIO.setup(PIN_INA, GPIO.OUT)
GPIO.setup(PIN_INB, GPIO.OUT)
GPIO.setup(PIN_PWM, GPIO.OUT)

# Create software PWM on GPIO12
motor_pwm = GPIO.PWM(PIN_PWM, PWM_FREQUENCY)
motor_pwm.start(0)

try:
    print("Starting motor sine sweep. Press Ctrl+C to stop.")

    start_time = time.time()

    while True:
        elapsed = time.time() - start_time

        # Sine wave from -1.0 to +1.0
        motor_sine = math.sin(2 * math.pi * 0.2 * elapsed)

        # Set motor direction and PWM duty cycle
        if motor_sine >= 0:
            GPIO.output(PIN_INA, GPIO.HIGH)
            GPIO.output(PIN_INB, GPIO.LOW)
            duty = motor_sine * 100
        else:
            GPIO.output(PIN_INA, GPIO.LOW)
            GPIO.output(PIN_INB, GPIO.HIGH)
            duty = abs(motor_sine) * 100

        # Apply PWM
        motor_pwm.ChangeDutyCycle(duty)

        print(
            f"Time: {elapsed:.1f}s | "
            f"Sine: {motor_sine:+.3f} | "
            f"Duty: {duty:5.1f}%",
            end="\r"
        )

        time.sleep(0.02)

except KeyboardInterrupt:
    print("\nStopping motor...")

finally:
    motor_pwm.stop()
    GPIO.cleanup()