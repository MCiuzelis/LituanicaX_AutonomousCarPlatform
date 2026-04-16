import RPi.GPIO as GPIO
from rpi_hardware_pwm import HardwarePWM

class Robot:
    def __init__(
        self,
        # Hardware PWM uses channels, not just GPIO numbers
        # Pin 18 is typically PWM Channel 2 or 3 depending on overlay
        # Pin 12 is typically PWM Channel 0
        pwm_motor_channel=2, 
        dir_motor_pin=15,
        servo_channel=0,
        pwm_freq=1000,
        servo_freq=50,
        servo_min_us=500,
        servo_max_us=2500,
        servo_center_us=1500,
        servo_offset_us=-130,
        servo_amplitude=0.25,
        servo_direction=1,
        motor_direction=-1,
        max_throttle=0.06
    ):
        # Store config
        self.dir_motor_pin = dir_motor_pin
        self.pwm_freq = pwm_freq
        self.servo_freq = servo_freq

        self.servo_min_us = servo_min_us
        self.servo_max_us = servo_max_us
        self.servo_center_us = servo_center_us
        self.servo_offset_us = servo_offset_us

        self.servo_amplitude = servo_amplitude
        self.servo_direction = servo_direction
        self.motor_direction = motor_direction
        self.max_throttle = max(0.0, min(1.0, max_throttle))

        # GPIO setup for Direction pin (Standard GPIO)
        GPIO.setmode(GPIO.BCM)
        GPIO.setup(self.dir_motor_pin, GPIO.OUT)

        # Create Hardware PWM instances
        # HardwarePWM(channel, hz)
        self.motor_pwm = HardwarePWM(pwm_channel=pwm_motor_channel, hz=self.pwm_freq)
        self.servo_pwm = HardwarePWM(pwm_channel=servo_channel, hz=self.servo_freq)

        # Start PWM (0 duty initially)
        self.motor_pwm.start(0)
        self.servo_pwm.start(0)

    # ---------------- MOTOR ----------------
    def set_throttle(self, value):
        value = max(-1.0, min(1.0, value))
        value *= self.max_throttle
        value *= self.motor_direction

        direction = GPIO.HIGH if value >= 0 else GPIO.LOW
        speed = abs(value)

        GPIO.output(self.dir_motor_pin, direction)

        # rpi-hardware-pwm uses 0-100 for duty cycle
        self.motor_pwm.change_duty_cycle(speed * 100.0)

    # ---------------- SERVO ----------------
    def set_steering(self, value):
        value = max(-1.0, min(1.0, value))
        value = value * self.servo_direction * self.servo_amplitude

        span = (self.servo_max_us - self.servo_min_us) / 2
        pulse = (
            self.servo_center_us
            + self.servo_offset_us
            + (value * span)
        )

        pulse = max(self.servo_min_us, min(self.servo_max_us, pulse))

        # Convert µs → duty cycle (based on 50Hz / 20ms period)
        duty = (pulse / 20000.0) * 100.0

        self.servo_pwm.change_duty_cycle(duty)

    # ---------------- CLEANUP ----------------
    def stop(self):
        try:
            self.motor_pwm.stop()
            self.servo_pwm.stop()
        finally:
            GPIO.cleanup()